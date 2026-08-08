//! Reverse-mode autograd. `backward()` seeds the output with ones, walks the
//! graph in reverse-topological order, and accumulates gradients into a
//! [`GradStore`]. Only `Float` tensors participate.

use crate::{BinaryOp, Device, Float, GradStore, Op, ReduceOp, Shape, Tensor, TensorId, UnaryOp};
use std::collections::HashMap;

impl<D: Device> Tensor<D, Float> {
    /// Compute gradients of `self` w.r.t. all leaf tensors that require grad.
    pub fn backward(&self) -> crate::Result<GradStore<D>> {
        let _guard = crate::NoGradGuard::new();

        let sorted = self.sorted_nodes();
        let mut grads = GradStore::new();
        grads.insert(self, self.ones_like()?);

        for node in sorted.iter() {
            let op = match node.op() {
                None => {
                    debug_assert!(node.is_leaf());
                    continue;
                }
                Some(op) => op,
            };
            let grad = grads.remove(node).expect("grad not populated");
            backward_op(node, op, &grad, &mut grads)?;
        }
        Ok(grads)
    }

    /// Reverse-topological order of nodes reachable from grad-requiring leaves.
    pub fn sorted_nodes(&self) -> Vec<&Tensor<D, Float>> {
        fn walk<'a, D: Device>(
            node: &'a Tensor<D, Float>,
            mut nodes: Vec<&'a Tensor<D, Float>>,
            seen: &mut HashMap<TensorId, bool>,
        ) -> (bool, Vec<&'a Tensor<D, Float>>) {
            if let Some(&tg) = seen.get(&node.id()) {
                return (tg, nodes);
            }
            let mut track = false;
            nodes = if node.is_leaf() {
                track = true;
                nodes
            } else if let Some(op) = node.op() {
                for input in op_inputs(op) {
                    let (tg, n) = walk(input, nodes, seen);
                    track |= tg;
                    nodes = n;
                }
                nodes
            } else {
                nodes
            };
            seen.insert(node.id(), track);
            if track {
                nodes.push(node);
            }
            (track, nodes)
        }
        let (_tg, mut nodes) = walk(self, vec![], &mut HashMap::new());
        nodes.reverse();
        nodes
    }
}

/// Float inputs of an op that gradient can flow through.
fn op_inputs<D: Device>(op: &Op<D>) -> Vec<&Tensor<D, Float>> {
    match op {
        Op::Binary(a, b, _) | Op::Matmul(a, b) => vec![a, b],
        Op::BinaryScalarRhs(a, _, _) | Op::BinaryScalarLhs(_, a, _) => vec![a],
        Op::Unary(a, _)
        | Op::Reduce(a, _, _)
        | Op::Broadcast(a)
        | Op::Narrow(a, _, _, _)
        | Op::Slice(a, _, _, _, _)
        | Op::Reshape(a)
        | Op::Transpose(a, _, _)
        | Op::Permute(a, _)
        | Op::Copy(a)
        | Op::Cast(a)
        | Op::IndexSelect(a, _, _)
        | Op::Gather(a, _, _)
        | Op::Softmax(a, _) => vec![a],
        Op::IndexAdd(a, _, b, _) | Op::ScatterAdd(a, _, b, _) | Op::RmsNorm(a, b, _) => vec![a, b],
        Op::Cat(args, _) => args.iter().collect(),
        Op::Pick(_, tv, fv) => tv.iter().chain(fv.iter()).collect(),
        Op::Neg(a) | Op::Abs(a) | Op::Sign(a) | Op::Pow(a, _) | Op::Affine(a, _, _) | Op::Clamp(a, _, _) => vec![a],
    }
}

/// Build the keepdim shape: `arg` dims with each reduced axis set to 1.
fn keepdim_shape(arg_dims: &[usize], reduced: &[usize]) -> Shape {
    let mut dims = arg_dims.to_vec();
    for &d in reduced {
        dims[d] = 1;
    }
    Shape::from(dims)
}

fn backward_op<D: Device>(node: &Tensor<D, Float>, op: &Op<D>, grad: &Tensor<D, Float>, grads: &mut GradStore<D>) -> crate::Result<()> {
    match op {
        // ---- Binary tensor-tensor ----
        Op::Binary(lhs, rhs, BinaryOp::Add) => {
            grads.or_insert(lhs)?.impl_add_(grad)?;
            grads.or_insert(rhs)?.impl_add_(grad)?;
        }
        Op::Binary(lhs, rhs, BinaryOp::Sub) => {
            grads.or_insert(lhs)?.impl_add_(grad)?;
            grads.or_insert(rhs)?.impl_sub_(grad)?;
        }
        Op::Binary(lhs, rhs, BinaryOp::Mul) => {
            let lg = grad.mul(rhs)?;
            grads.or_insert(lhs)?.impl_add_(&lg)?;
            let rg = grad.mul(lhs)?;
            grads.or_insert(rhs)?.impl_add_(&rg)?;
        }
        Op::Binary(lhs, rhs, BinaryOp::Div) => {
            let lg = grad.div(rhs)?;
            grads.or_insert(lhs)?.impl_add_(&lg)?;
            // d/drhs (lhs/rhs) = -lhs/rhs^2
            let rg = grad.mul(lhs)?.div(&rhs.sqr()?)?;
            grads.or_insert(rhs)?.impl_sub_(&rg)?;
        }
        Op::Binary(lhs, rhs, BinaryOp::Maximum) | Op::Binary(lhs, rhs, BinaryOp::Minimum) => {
            let mask_lhs = node.eq(lhs)?.cast_float(node.dtype())?;
            let mask_rhs = node.eq(rhs)?.cast_float(node.dtype())?;
            // split the gradient where both equal the output (scale by 1/(mask+1)).
            let lg = mask_lhs.mul(grad)?.div(&mask_rhs.add_scalar(1.0)?)?;
            grads.or_insert(lhs)?.impl_add_(&lg)?;
            let rg = mask_rhs.mul(grad)?.div(&mask_lhs.add_scalar(1.0)?)?;
            grads.or_insert(rhs)?.impl_add_(&rg)?;
        }

        // ---- Binary scalar-rhs ----
        Op::BinaryScalarRhs(lhs, _, BinaryOp::Add) | Op::BinaryScalarRhs(lhs, _, BinaryOp::Sub) => {
            grads.or_insert(lhs)?.impl_add_(grad)?;
        }
        Op::BinaryScalarRhs(lhs, c, BinaryOp::Mul) => {
            let lg = grad.mul_scalar(*c)?;
            grads.or_insert(lhs)?.impl_add_(&lg)?;
        }
        Op::BinaryScalarRhs(lhs, c, BinaryOp::Div) => {
            let lg = grad.div_scalar(*c)?;
            grads.or_insert(lhs)?.impl_add_(&lg)?;
        }
        Op::BinaryScalarRhs(lhs, _, BinaryOp::Maximum) | Op::BinaryScalarRhs(lhs, _, BinaryOp::Minimum) => {
            let mask = node.eq(lhs)?.cast_float(node.dtype())?;
            let lg = mask.mul(grad)?;
            grads.or_insert(lhs)?.impl_add_(&lg)?;
        }

        // ---- Binary scalar-lhs ----
        Op::BinaryScalarLhs(_, rhs, BinaryOp::Add) => {
            grads.or_insert(rhs)?.impl_add_(grad)?;
        }
        Op::BinaryScalarLhs(_, rhs, BinaryOp::Sub) => {
            grads.or_insert(rhs)?.impl_sub_(grad)?;
        }
        Op::BinaryScalarLhs(c, rhs, BinaryOp::Mul) => {
            let rg = grad.mul_scalar(*c)?;
            grads.or_insert(rhs)?.impl_add_(&rg)?;
        }
        Op::BinaryScalarLhs(c, rhs, BinaryOp::Div) => {
            // y = c / x => dy/dx = -c / x^2
            let rg = grad.mul_scalar(-*c)?.div(&rhs.sqr()?)?;
            grads.or_insert(rhs)?.impl_add_(&rg)?;
        }
        Op::BinaryScalarLhs(_, rhs, BinaryOp::Maximum) | Op::BinaryScalarLhs(_, rhs, BinaryOp::Minimum) => {
            let mask = node.eq(rhs)?.cast_float(node.dtype())?;
            let rg = mask.mul(grad)?;
            grads.or_insert(rhs)?.impl_add_(&rg)?;
        }

        Op::Unary(arg, uop) => backward_unary(node, arg, *uop, grad, grads)?,

        // ---- elementwise ops ----
        Op::Neg(arg) => {
            grads.or_insert(arg)?.impl_add_(&grad.neg()?)?;
        }
        Op::Abs(arg) => {
            let g = grad.mul(&arg.sign()?)?;
            grads.or_insert(arg)?.impl_add_(&g)?;
        }
        Op::Sign(_) => {} // gradient is zero everywhere
        Op::Pow(arg, e) => {
            let g = grad.mul(&arg.pow(e - 1.0)?)?.mul_scalar(*e)?;
            grads.or_insert(arg)?.impl_add_(&g)?;
        }
        Op::Affine(arg, mul, _add) => {
            let g = grad.mul_scalar(*mul)?;
            grads.or_insert(arg)?.impl_add_(&g)?;
        }
        Op::Clamp(arg, min, max) => {
            let dtype = arg.dtype();
            let mut mask = arg.ones_like()?;
            if let Some(lo) = min {
                let t_lo = arg.zeros_like()?.add_scalar(*lo)?;
                mask = mask.mul(&arg.gt(&t_lo)?.cast_float(dtype)?)?;
            }
            if let Some(hi) = max {
                let t_hi = arg.zeros_like()?.add_scalar(*hi)?;
                mask = mask.mul(&arg.lt(&t_hi)?.cast_float(dtype)?)?;
            }
            let g = grad.mul(&mask)?;
            grads.or_insert(arg)?.impl_add_(&g)?;
        }

        // ---- Matmul ----
        Op::Matmul(lhs, rhs) => {
            grads.or_insert(lhs)?.add_matmul_(grad, &rhs.transpose_last()?)?;
            grads.or_insert(rhs)?.add_matmul_(&lhs.transpose_last()?, grad)?;
        }

        // ---- Reduce ----
        Op::Reduce(arg, rop, reduced) => backward_reduce(node, arg, *rop, reduced, grad, grads)?,

        // ---- Broadcast: sum grad over the broadcasted dims ----
        Op::Broadcast(arg) => {
            let arg_dims = arg.dims();
            let node_dims = node.dims();
            let left = node_dims.len() - arg_dims.len();
            let mut sum_dims: Vec<usize> = (0..left).collect();
            for (d, (nd, ad)) in node_dims[left..].iter().zip(arg_dims.iter()).enumerate() {
                if nd != ad {
                    sum_dims.push(d + left);
                }
            }
            let mut arg_grad = grad.clone();
            for &d in sum_dims.iter() {
                arg_grad = arg_grad.sum_keepdim(d)?;
            }
            for _ in 0..left {
                arg_grad = arg_grad.squeeze(0)?;
            }
            let g = arg_grad.broadcast_as(arg.shape().clone())?;
            grads.or_insert(arg)?.impl_add_(&g)?;
        }

        // ---- shape movements ----
        Op::Reshape(arg) => {
            let g = grad.reshape(arg.shape().clone())?;
            grads.or_insert(arg)?.impl_add_(&g)?;
        }
        Op::Transpose(arg, d1, d2) => {
            let g = grad.transpose(*d1, *d2)?;
            grads.or_insert(arg)?.impl_add_(&g)?;
        }
        Op::Permute(arg, dims) => {
            let mut inv = vec![0; dims.len()];
            for (i, &d) in dims.iter().enumerate() {
                inv[d] = i;
            }
            let g = grad.permute(inv)?;
            grads.or_insert(arg)?.impl_add_(&g)?;
        }
        Op::Narrow(arg, dim, start, len) => {
            let g = pad_grad_along(arg, grad, *dim, *start, *len)?;
            grads.or_insert(arg)?.impl_add_(&g)?;
        }
        Op::Cat(args, dim) => {
            let mut start = 0;
            for arg in args {
                let len = arg.dims()[*dim];
                let g = grad.narrow(*dim, start, len)?;
                grads.or_insert(arg)?.impl_add_(&g)?;
                start += len;
            }
        }
        Op::Copy(arg) => {
            grads.or_insert(arg)?.impl_add_(grad)?;
        }

        // ---- Cast: cast the gradient back to the input precision ----
        Op::Cast(arg) => {
            let g = grad.cast(arg.dtype())?;
            grads.or_insert(arg)?.impl_add_(&g)?;
        }

        // ---- indexing ----
        Op::IndexSelect(arg, indices, dim) => {
            // scatter grad back to the selected positions.
            let acc = grads.or_insert(arg)?;
            let updated = acc.index_add(indices, grad, *dim)?;
            *grads.or_insert(arg)? = updated;
        }
        Op::Gather(arg, indices, dim) => {
            let acc = grads.or_insert(arg)?;
            let updated = acc.scatter_add(indices, grad, *dim)?;
            *grads.or_insert(arg)? = updated;
        }
        Op::IndexAdd(init, indices, src, dim) => {
            grads.or_insert(init)?.impl_add_(grad)?;
            let src_grad = grad.index_select(indices, *dim)?;
            grads.or_insert(src)?.impl_add_(&src_grad)?;
        }
        Op::ScatterAdd(init, indices, src, dim) => {
            grads.or_insert(init)?.impl_add_(grad)?;
            let src_grad = grad.gather(indices, *dim)?;
            grads.or_insert(src)?.impl_add_(&src_grad)?;
        }

        // ---- pick: route grad through the mask to each branch ----
        Op::Pick(mask, tv, fv) => {
            if let Some(tv) = tv {
                let g = mask.pick_false(grad, 0.0)?;
                grads.or_insert(tv)?.impl_add_(&g)?;
            }
            if let Some(fv) = fv {
                let g = mask.pick_true(0.0, grad)?;
                grads.or_insert(fv)?.impl_add_(&g)?;
            }
        }

        // ---- softmax: g_in = y * (g - sum(g*y, dim, keepdim)) ----
        Op::Softmax(input, dim) => {
            let y = node; // softmax output
            let gy = grad.mul(y)?;
            let s = gy.sum_keepdim(*dim)?;
            let g = y.mul(&grad.sub(&s.broadcast_as(grad.shape().clone())?)?)?;
            grads.or_insert(input)?.impl_add_(&g)?;
        }

        // ---- slice: dilate (step>1) + pad back to arg shape ----
        Op::Slice(arg, dim, start, _end, step) => {
            let arg_dtype = arg.dtype();

            let body_grad: Tensor<D, Float>;
            let body_grad_ref = if *step == 1 {
                grad
            } else {
                let grad_len = grad.dims()[*dim];
                let span_len = if grad_len > 0 {
                    (grad_len - 1) * step + 1
                } else {
                    0
                };

                // Insert a dim of size 1 after `dim`, so that each grad element
                // can be interleaved with (step-1) zeros.
                let mut unsqueezed_shape = grad.dims().to_vec();
                unsqueezed_shape.insert(*dim + 1, 1);
                let grad_unsqueezed = grad.reshape(Shape::from(unsqueezed_shape))?;

                // Build the gap zeros alongside the unsqueezed dim.
                let mut zeros_shape = grad_unsqueezed.dims().to_vec();
                zeros_shape[*dim + 1] = step - 1;
                let zeros_gap =
                    Tensor::<D, Float>::zeros(Shape::from(zeros_shape), arg_dtype)?;

                // Interleave: cat along the new dim, then flatten back.
                let dilated =
                    Tensor::cat(&[&grad_unsqueezed, &zeros_gap], *dim + 1)?;
                let mut flattened_shape = grad.dims().to_vec();
                flattened_shape[*dim] = grad_len * step;
                let flattened = dilated.reshape(Shape::from(flattened_shape))?;

                body_grad = flattened.narrow(*dim, 0, span_len)?;
                &body_grad
            };

            let body_len = body_grad_ref.dims()[*dim];
            let arg_grad = pad_grad_along(arg, body_grad_ref, *dim, *start, body_len)?;
            grads.or_insert(arg)?.impl_add_(&arg_grad)?;
        }

        // ---- not yet wired ----
        Op::RmsNorm(..) => return Err(crate::Error::BackwardNotSupported("rms_norm")),
    }
    Ok(())
}

/// Pad `grad` with zeros along `dim` back to `arg`'s size (narrow backward).
fn pad_grad_along<D: Device>(
    arg: &Tensor<D, Float>,
    grad: &Tensor<D, Float>,
    dim: usize,
    start: usize,
    len: usize,
) -> crate::Result<Tensor<D, Float>> {
    let arg_dims = arg.dims();
    let make_pad = |size: usize| {
        let mut dims = arg_dims.to_vec();
        dims[dim] = size;
        Tensor::<D, Float>::zeros(Shape::from(dims), arg.dtype())
    };
    let right = arg_dims[dim] - start - len;
    let left_pad = if start != 0 {
        Some(make_pad(start)?)
    } else {
        None
    };
    let right_pad = if right != 0 {
        Some(make_pad(right)?)
    } else {
        None
    };
    match (left_pad, right_pad) {
        (None, None) => Ok(grad.clone()),
        (Some(l), None) => Tensor::cat(&[&l, grad], dim),
        (None, Some(r)) => Tensor::cat(&[grad, &r], dim),
        (Some(l), Some(r)) => Tensor::cat(&[&l, grad, &r], dim),
    }
}

/// Gradient of unary ops. Fused kernels in luma-core are expanded here into
/// composed tensor ops (correctness first; can be fused later).
fn backward_unary<D: Device>(
    node: &Tensor<D, Float>,
    arg: &Tensor<D, Float>,
    op: UnaryOp,
    grad: &Tensor<D, Float>,
    grads: &mut GradStore<D>,
) -> crate::Result<()> {
    // local gradient factor `local` such that d(arg) += grad * local
    let contrib = match op {
        UnaryOp::Exp => grad.mul(node)?,               // d/dx e^x = e^x = node
        UnaryOp::Ln => grad.div(arg)?,                 // 1/x
        UnaryOp::Sin => grad.mul(&arg.cos()?)?,        // cos x
        UnaryOp::Cos => grad.mul(&arg.sin()?)?.neg()?, // -sin x
        UnaryOp::Tanh => {
            // 1 - tanh^2 = 1 - node^2
            let factor = node.sqr()?.neg()?.add_scalar(1.0)?;
            grad.mul(&factor)?
        }
        UnaryOp::Sqr => grad.mul(arg)?.mul_scalar(2.0)?, // 2x
        UnaryOp::Sqrt => {
            // 1/(2 sqrt x) = 0.5 / node
            grad.mul_scalar(0.5)?.div(node)?
        }
        UnaryOp::Recip => {
            // -1/x^2 = -node^2
            grad.mul(&node.sqr()?)?.neg()?
        }
        UnaryOp::Relu => {
            // mask = arg > 0
            let mask = arg.gt(&arg.zeros_like()?)?.cast_float(arg.dtype())?;
            grad.mul(&mask)?
        }
        UnaryOp::LeakyRelu(slope) => {
            let pos = arg.gt(&arg.zeros_like()?)?.cast_float(arg.dtype())?;
            let neg = pos.neg()?.add_scalar(1.0)?.mul_scalar(slope)?;
            grad.mul(&pos.add(&neg)?)?
        }
        UnaryOp::Sigmoid => {
            // node * (1 - node)
            let factor = node.neg()?.add_scalar(1.0)?.mul(node)?;
            grad.mul(&factor)?
        }
        UnaryOp::Erf => {
            // 2/sqrt(pi) * exp(-x^2)
            let scale = 2.0 / std::f64::consts::PI.sqrt();
            grad.mul(&arg.sqr()?.neg()?.exp()?)?.mul_scalar(scale)?
        }
        UnaryOp::Silu => {
            // sig = sigmoid(x); silu = x*sig; d = sig*(1 - silu) + silu
            let sig = arg.sigmoid()?;
            let silu = arg.mul(&sig)?;
            let factor = sig.mul(&silu.neg()?.add_scalar(1.0)?)?.add(&silu)?;
            grad.mul(&factor)?
        }
        UnaryOp::Gelu => {
            // matches luma-core's tanh-approx gelu grad
            let c1 = 0.0356774;
            let c2 = 0.797885;
            let c3 = 0.0535161;
            let c4 = 0.398942;
            let x3 = arg.mul(&arg.sqr()?)?;
            let inner = x3.mul_scalar(c1)?.add(&arg.mul_scalar(c2)?)?;
            let tanh = inner.tanh()?;
            let dt = x3.mul_scalar(c3)?.add(&arg.mul_scalar(c4)?)?;
            // 0.5*tanh + dt*(1 - tanh^2) + 0.5
            let factor = tanh.mul_scalar(0.5)?.add(&dt.mul(&tanh.sqr()?.neg()?.add_scalar(1.0)?)?)?.add_scalar(0.5)?;
            grad.mul(&factor)?
        }
        UnaryOp::GeluErf => {
            // c1 * exp(-x^2/2) * x + erf(x/sqrt2)/2 + 0.5
            let c1 = 0.398942;
            let sqrt2 = std::f64::consts::SQRT_2;
            let neg_half_sq = arg.sqr()?.neg()?.div_scalar(2.0)?;
            let scaled_exp = neg_half_sq.exp()?.mul(arg)?.mul_scalar(c1)?;
            let erf_term = arg.div_scalar(sqrt2)?.erf()?.div_scalar(2.0)?;
            let factor = scaled_exp.add(&erf_term)?.add_scalar(0.5)?;
            grad.mul(&factor)?
        }
        UnaryOp::Floor | UnaryOp::Ceil | UnaryOp::Round => {
            return Err(crate::Error::BackwardNotSupported("floor/ceil/round"));
        }
    };
    grads.or_insert(arg)?.impl_add_(&contrib)?;
    Ok(())
}

/// Gradient of reductions. Reshapes grad to keepdim form then broadcasts back.
fn backward_reduce<D: Device>(
    node: &Tensor<D, Float>,
    arg: &Tensor<D, Float>,
    op: ReduceOp,
    reduced: &[usize],
    grad: &Tensor<D, Float>,
    grads: &mut GradStore<D>,
) -> crate::Result<()> {
    let keep = keepdim_shape(arg.dims(), reduced);
    match op {
        ReduceOp::Sum => {
            let g = grad.reshape(keep)?.broadcast_as(arg.shape().clone())?;
            grads.or_insert(arg)?.impl_add_(&g)?;
        }
        ReduceOp::Mean => {
            let n = arg.element_count() / node.element_count().max(1);
            let g = grad.reshape(keep)?.broadcast_as(arg.shape().clone())?.div_scalar(n as f64)?;
            grads.or_insert(arg)?.impl_add_(&g)?;
        }
        ReduceOp::Prod => {
            // d(prod)/dx_i = prod / x_i (if x_i != 0)
            let n_broadcast = node.reshape(keep.clone())?.broadcast_as(arg.shape().clone())?;
            let g = grad.reshape(keep)?.broadcast_as(arg.shape().clone())?.mul(&n_broadcast)?.div(arg)?;
            grads.or_insert(arg)?.impl_add_(&g)?;
        }
        ReduceOp::Max | ReduceOp::Min => {
            // route grad to the arg elements equal to the reduced value.
            let node_b = node.reshape(keepdim_shape(arg.dims(), reduced))?.broadcast_as(arg.shape().clone())?;
            let mask = node_b.eq(arg)?.cast_float(arg.dtype())?;
            let g = grad.reshape(keepdim_shape(arg.dims(), reduced))?.broadcast_as(arg.shape().clone())?.mul(&mask)?;
            grads.or_insert(arg)?.impl_add_(&g)?;
        }
    }
    Ok(())
}
