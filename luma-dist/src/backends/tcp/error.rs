use thiserror::Error;

#[derive(Debug, Error)]
pub enum TcpError {
    #[error(transparent)]
    Tensor(#[from] luma_tensor::Error),

    #[error("rank ack failed, expect {0}, but got {1}")]
    RankAck(usize, usize),

    #[error("bool tensor not support reduce")]
    BoolTensorNotSupportReduce,

    #[error("value {0} not sync")]
    SyncBytes(&'static str),

    #[error("no connection to peer {0}")]
    NoPeer(usize),

    #[error("tcp p2p only supports rank 0 as one endpoint (peer {0})")]
    UnsupportedPeer(usize),
}
