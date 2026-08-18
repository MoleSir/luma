use std::borrow::Cow;

use crate::{Device, Result};

pub trait MatrixFill: Clone + Sized {
    fn zero_val() -> Self;
    fn one_val() -> Self;
}

impl MatrixFill for f64 {
    fn zero_val() -> Self {
        0.0
    }
    fn one_val() -> Self {
        1.0
    }
}
impl MatrixFill for f32 {
    fn zero_val() -> Self {
        0.0_f32
    }
    fn one_val() -> Self {
        1.0_f32
    }
}
impl MatrixFill for i64 {
    fn zero_val() -> Self {
        0
    }
    fn one_val() -> Self {
        1
    }
}
impl MatrixFill for i32 {
    fn zero_val() -> Self {
        0
    }
    fn one_val() -> Self {
        1
    }
}
impl MatrixFill for u32 {
    fn zero_val() -> Self {
        0
    }
    fn one_val() -> Self {
        1
    }
}
impl MatrixFill for u8 {
    fn zero_val() -> Self {
        0
    }
    fn one_val() -> Self {
        1
    }
}
impl MatrixFill for bool {
    fn zero_val() -> Self {
        false
    }
    fn one_val() -> Self {
        true
    }
}

pub trait FloatFrom: bytemuck::Pod + Copy + MatrixFill + 'static {
    fn into_storage<'a, D: Device>(data: impl Into<Cow<'a, [Self]>>, device: &D) -> Result<D::FloatStorage>;
}

impl FloatFrom for f64 {
    fn into_storage<'a, D: Device>(data: impl Into<Cow<'a, [Self]>>, device: &D) -> Result<D::FloatStorage> {
        D::f_from_f64(data, device)
    }
}

impl FloatFrom for f32 {
    fn into_storage<'a, D: Device>(data: impl Into<Cow<'a, [Self]>>, device: &D) -> Result<D::FloatStorage> {
        D::f_from_f32(data, device)
    }
}

pub trait IntFrom: bytemuck::Pod + Copy + MatrixFill + 'static {
    fn into_storage<'a, D: Device>(data: impl Into<Cow<'a, [Self]>>, device: &D) -> Result<D::IntStorage>;
}

impl IntFrom for i64 {
    fn into_storage<'a, D: Device>(data: impl Into<Cow<'a, [Self]>>, device: &D) -> Result<D::IntStorage> {
        D::i_from_i64(data, device)
    }
}

impl IntFrom for i32 {
    fn into_storage<'a, D: Device>(data: impl Into<Cow<'a, [Self]>>, device: &D) -> Result<D::IntStorage> {
        D::i_from_i32(data, device)
    }
}

impl IntFrom for u32 {
    fn into_storage<'a, D: Device>(data: impl Into<Cow<'a, [Self]>>, device: &D) -> Result<D::IntStorage> {
        D::i_from_u32(data, device)
    }
}

impl IntFrom for u8 {
    fn into_storage<'a, D: Device>(data: impl Into<Cow<'a, [Self]>>, device: &D) -> Result<D::IntStorage> {
        D::i_from_u8(data, device)
    }
}

pub trait BoolFrom: Copy + MatrixFill + 'static {
    fn into_storage<'a, D: Device>(data: impl Into<Cow<'a, [Self]>>, device: &D) -> Result<D::BoolStorage>;
}

impl BoolFrom for bool {
    fn into_storage<'a, D: Device>(data: impl Into<Cow<'a, [Self]>>, device: &D) -> Result<D::BoolStorage> {
        D::b_from_bool(data, device)
    }
}
