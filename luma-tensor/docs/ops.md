## `index_select`

```
index_select(x, dim, index)
```

对输入 x 的 dim 维度，只取出部分（根据 index 中的索引）。

最简单的：x 是 vec，dim 只能写 0，index 必须是 vec shape，然后去选择 x 中的元素。

`index_select([100, 200, 300], 0, [0, 2])` => `[100, 300]`

或者一个更常见的场景，对一批输入训练数据：x (batch, features)（batch 个样本）。我们需要选择其中特定的几个，就用 index_select，dim 为 0，index 是需要的样本索引。

也可以选择某些特定 feature，那 dim 就是 1。



## `gather`

```
gather(x, dim, index)
```

`index_select` 的 index 必须是 vec，每个索引值取的是“一批”数据，这个要根据输入 x 的其他维度的规模决定。

而 `gather` 想做的是：`index` 可以不仅是 vec，而是任意 shape，每个索引只取输入 `x` 的一个值（所以输出的 shape 和 `index` 一致）。为了做到这一点，即每个索引选择一个值，但 `x` 是多维的，一个索引值本身是不足以确定 `x` 中的具体一个值的，而我们的 `dim` 参数就是指定这个索引值所在的维度。那么问题就是：除了 `dim` 这个维度之外，其他维度的索引怎么办呢？

解决方法是约束 `index` 的 shape：`index` 和 `x` 必须有相同的维度数，并且除了 `dim` 之外，其他维度的大小不能超过 `x` 对应维度的大小。这样，`index` 中每个索引值所在的位置，就可以确定 `x` 在其他维度上的索引，而这个索引值本身则确定 `x` 在 `dim` 维度上的索引。这样一个 `index` 中的元素就可以唯一确定 `x` 中的一个具体值，同时也就能保证输出的 shape 和 `index` 一致。

我了解到最经典的用法是 LLM 预训练时，要从输出 logits (batch, seq_len, embed_dim) 中的每个 token（一共 batch * seq_len 个）的 embed_dim 个 logits 值取出一个 logits，输出 shape 为 (batch, seq_len)。这个情况就是用 gather 了。输入 index 的 shape 为 (batch, seq_len)，一个值对应一个 token，然后这个值就是这个 token 要选择的 logits 的索引！

