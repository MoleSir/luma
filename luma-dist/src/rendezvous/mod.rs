//! Rendezvous: how ranks find each other and exchange their bootstrap payload.
//!
//! A *rendezvous* is the meeting point where every rank of a group exchanges the
//! small blob it needs to start communicating — for NCCL that blob is the
//! 128-byte unique id produced by rank 0. The mechanism (TCP, shared file, an
//! external store, …) is orthogonal to the collective backend, so it lives
//! behind the [`Rendezvous`] trait and is chosen at the call site by passing a
//! concrete value:
//!
//! ```ignore
//! use luma_dist::{tcp, backends::nccl};
//!
//! let comm = nccl::init_process_group(tcp!("localhost:2333"), world_size, rank, device)?;
//! ```
//!
//! [`tcp!`](crate::tcp) is a thin sugar for [`Tcp::new`]; both are equivalent.
mod tcp;
pub use tcp::Tcp;
use crate::DistResult;

/// Exchanges a single bootstrap payload among all ranks of a group.
///
/// Implementations must agree on how rank 0 publishes its payload and how every
/// other rank retrieves it.
pub trait Rendezvous: Send + Sync + 'static {
    /// Publish `payload` from rank 0 and return it on every rank.
    ///
    /// `rank` 0 passes its payload; all other ranks pass a slice
    /// ([`&[]`](slice)) and receive rank 0's payload. This call blocks until
    /// every rank has taken part.
    fn exchange(&self, rank: usize, world_size: usize, payload: &[u8]) -> DistResult<Vec<u8>>;

    /// A `host:port` at which peers can reach rank 0, when the transport knows
    /// one (used by a backend to open its own data-plane connections).
    ///
    /// Defaults to `None`; transports that do not expose an address let the
    /// backend advertise one through [`exchange`](Self::exchange) instead.
    fn root_addr(&self) -> Option<&str> {
        None
    }
}

/// Construct a [`Tcp`] rendezvous: `tcp!("host:port")`.
///
/// Equivalent to [`Tcp::new("host:port")`](Tcp::new), but leaves room for extra
/// options (timeouts, …) without touching call sites.
#[macro_export]
macro_rules! tcp {
    ($addr:expr) => {
        $crate::rendezvous::Tcp::new($addr)
    };
}
