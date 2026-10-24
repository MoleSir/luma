use std::{io::{Read, Write}, net::TcpStream};
use luma_tensor::{DType, Device, Shape, Tensor, dtype::DTypeKindT, ops::construct::BytesDTypeKind};
use crate::DistResult;

pub fn recv_tensor<Dev: Device, K: BytesDTypeKind<Dev>>(stream: &mut TcpStream, device: &Dev) -> DistResult<Tensor<Dev, K>> {
    // 接受：shape / dtype / data
    // 1. shape: [rank][xxx, xxx, ..]
    let mut buf = [0u8; 8];
    stream.read_exact(&mut buf)?;
    let dim_len = u64::from_be_bytes(buf) as usize;
    
    let mut buf = [0u8; 8];
    let mut dims = vec![];
    for _ in 0..dim_len {
        stream.read_exact(&mut buf)?;
        let dim = u64::from_be_bytes(buf) as usize;
        dims.push(dim);
    }
    let shape = Shape::new(dims);

    // 2. dtype
    let mut buf = [0u8; 1];
    stream.read_exact(&mut buf)?;
    let dtype: K::DType = byte_to_dtype(buf[0]).expect("invalid dtype").try_into()?;
    
    // 3. data
    let data_size = dtype.size_of() * shape.element_count();
    let mut data = vec![0u8; data_size];
    stream.read_exact(&mut data)?;

    let tensor = Tensor::<Dev, K>::from_bytes(data, shape, (device, dtype))?;
    Ok(tensor)
}

pub fn send_tensor<Dev: Device, K: BytesDTypeKind<Dev>>(stream: &mut TcpStream, x: &Tensor<Dev, K>) -> DistResult<()> {
    // 发送：shape / dtype / data
    // 1. shape: [rank][xxx, xxx, ..]
    let dims = x.dims().to_vec();

    let buf = (dims.len() as u64).to_be_bytes();
    stream.write_all(&buf)?;

    let mut buf = vec![];
    for &dim in dims.iter() {
        buf.extend((dim as u64).to_be_bytes());
    }
    assert_eq!(buf.len(), 8 * dims.len());
    stream.write_all(&buf)?;

    // 2. dtype
    let dtype = x.dtype();
    let byte = [dtype_to_byte(dtype.into())];
    stream.write_all(&byte)?;

    // 3. data
    let data = x.to_bytes()?;
    stream.write_all(&data)?;
    Ok(())
}

pub fn dtype_to_byte(dtype: DType) -> u8 {
    match dtype {
        DType::F32 => 0,
        DType::F64 => 1,
        DType::F16 => 2,
        DType::BF16 => 3,
        DType::I32 => 4,
        DType::U32 => 5,
        DType::U8 => 6,
        DType::Bool => 7,
    }
}

pub fn byte_to_dtype(byte: u8) -> Option<DType> {
    Some(match byte {
        0 => DType::F32,
        1 => DType::F64,
        2 => DType::F16,
        3 => DType::BF16,
        4 => DType::I32,
        5 => DType::U32,
        6 => DType::U8,
        7 => DType::Bool,
        _ => return None,
    })
}
