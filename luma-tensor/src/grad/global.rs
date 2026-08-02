//! Global autograd enable/disable switch, mirroring PyTorch's
//! `torch.set_grad_enabled` / `torch.no_grad()`.
//!
//! Controlled via [`NoGradGuard`] (RAII) or the [`no_grad!`] macro.

use std::cell::Cell;

thread_local! {
    static GRAD_ENABLED: Cell<bool> = Cell::new(true);
}

/// Set the global gradient enable flag.
pub fn set_grad_enabled(enabled: bool) {
    GRAD_ENABLED.with(|c| c.set(enabled));
}

/// Check whether gradient tracking is currently active.
pub fn is_grad_enabled() -> bool {
    GRAD_ENABLED.with(|c| c.get())
}

/// RAII guard that disables gradient tracking for the duration of its
/// lifetime, restoring the previous setting on drop.
///
/// # Example
/// ```ignore
/// {
///     let _guard = NoGradGuard::new();
///     // ops here don't build the computation graph
/// }
/// // grad tracking is restored
/// ```
pub struct NoGradGuard {
    prev: bool,
}

impl NoGradGuard {
    pub fn new() -> Self {
        let prev = is_grad_enabled();
        set_grad_enabled(false);
        Self { prev }
    }
}

impl Default for NoGradGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for NoGradGuard {
    fn drop(&mut self) {
        set_grad_enabled(self.prev);
    }
}

/// Temporarily disable gradient tracking within the current scope.
///
/// Equivalent to `{ let _guard = NoGradGuard::new(); ... }`.
///
/// # Example
/// ```ignore
/// no_grad!();
/// // .backward() will not traverse beyond this scope
/// ```
#[macro_export]
macro_rules! no_grad {
    () => {
        let _guard = $crate::NoGradGuard::new();
    };
}
