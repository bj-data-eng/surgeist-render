use super::{
    BackendErrorCode, Error, GpuFaultKind, Result, RuntimeOperation,
    backend::{DeviceSignal, DeviceTerminalSignal},
};
use std::sync::Arc;

/// Private ownership stage for a render-owned GPU operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GpuOperationStage {
    SurfaceCreate,
    RendererCreate,
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "Task 6 installs the presented resize transaction using this existing classification"
        )
    )]
    SurfaceConfigure,
    Render,
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "Task 6 installs the second presented submission transaction using this classification"
        )
    )]
    Present,
}

impl GpuOperationStage {
    const fn error_code(self) -> BackendErrorCode {
        match self {
            Self::SurfaceCreate => BackendErrorCode::SurfaceCreateFailed,
            Self::RendererCreate => BackendErrorCode::RendererCreateFailed,
            Self::SurfaceConfigure => BackendErrorCode::SurfaceConfigureFailed,
            Self::Render => BackendErrorCode::RenderFailed,
            Self::Present => BackendErrorCode::PresentFailed,
        }
    }

    fn classify_fault(self, kind: GpuFaultKind, message: impl Into<String>) -> Error {
        let message = message.into();
        let code = if kind == GpuFaultKind::OutOfMemory {
            BackendErrorCode::SurfaceOutOfMemory
        } else {
            self.error_code()
        };
        Error::new(code, message)
    }

    #[cfg(test)]
    pub(crate) fn classify_fault_for_test(self, kind: GpuFaultKind, message: &str) -> Error {
        self.classify_fault(kind, message)
    }
}

/// A value that can only be made visible by an explicit successful commit.
#[must_use = "GPU draft state must be committed or dropped"]
pub(crate) struct GpuOperationDraft<'a, T> {
    target: &'a mut Option<T>,
    value: Option<T>,
}

impl<'a, T> GpuOperationDraft<'a, T> {
    pub(crate) fn new(target: &'a mut Option<T>, value: T) -> Self {
        Self {
            target,
            value: Some(value),
        }
    }

    pub(crate) fn commit(mut self) {
        *self.target = self.value.take();
    }
}

/// Owns one active generation and clears only that generation on every exit.
#[must_use = "GPU operation leases must remain alive until scopes resolve"]
pub(crate) struct GpuOperationLease {
    signal: Arc<DeviceSignal>,
    generation: u64,
}

impl GpuOperationLease {
    pub(crate) fn begin(signal: Arc<DeviceSignal>, generation: u64) -> Self {
        signal.activate(generation);
        Self { signal, generation }
    }

    #[cfg(test)]
    pub(crate) fn begin_for_test(signal: &Arc<DeviceSignal>) -> Result<Self> {
        let generation = signal.next_test_generation()?;
        Ok(Self::begin(Arc::clone(signal), generation))
    }

    const fn generation(&self) -> u64 {
        self.generation
    }
}

impl Drop for GpuOperationLease {
    fn drop(&mut self) {
        self.signal.clear_active(self.generation);
    }
}

/// Nested WGPU scopes and the active-generation lease for one operation.
///
/// Scope fields are stored in reverse pop order. The explicit `Drop` below
/// preserves that order when an async operation future is canceled.
#[must_use = "GPU operation transactions must resolve scopes before publishing state"]
pub(crate) struct GpuOperationTransaction {
    validation: Option<wgpu::ErrorScopeGuard>,
    out_of_memory: Option<wgpu::ErrorScopeGuard>,
    internal: Option<wgpu::ErrorScopeGuard>,
    lease: GpuOperationLease,
    stage: GpuOperationStage,
}

impl GpuOperationTransaction {
    pub(crate) fn begin(
        device: &wgpu::Device,
        signal: Arc<DeviceSignal>,
        generation: u64,
        stage: GpuOperationStage,
    ) -> Self {
        let lease = GpuOperationLease::begin(signal, generation);
        let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
        let out_of_memory = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
        Self {
            validation: Some(validation),
            out_of_memory: Some(out_of_memory),
            internal: Some(internal),
            lease,
            stage,
        }
    }

    /// Resolves all error scopes before the caller may publish its draft state.
    pub(crate) async fn finish(mut self, operation: RuntimeOperation) -> Result<()> {
        let validation = self
            .validation
            .take()
            .expect("transaction validation scope must be present")
            .pop()
            .await;
        let out_of_memory = self
            .out_of_memory
            .take()
            .expect("transaction out-of-memory scope must be present")
            .pop()
            .await;
        let internal = self
            .internal
            .take()
            .expect("transaction internal scope must be present")
            .pop()
            .await;

        if let Some(terminal) = self.lease.signal.first_terminal() {
            return match terminal {
                DeviceTerminalSignal::Lost { .. } => Err(terminal.error(operation)),
                DeviceTerminalSignal::Faulted {
                    kind,
                    message,
                    operation_generation: Some(generation),
                } if generation == self.lease.generation() => {
                    Err(self.stage.classify_fault(kind, message))
                }
                DeviceTerminalSignal::Faulted { .. } => Err(terminal.error(operation)),
            };
        }

        if let Some(error) = [validation, out_of_memory, internal]
            .into_iter()
            .flatten()
            .next()
        {
            return Err(classify_captured_error(self.stage, error));
        }
        Ok(())
    }
}

impl Drop for GpuOperationTransaction {
    fn drop(&mut self) {
        drop(self.validation.take());
        drop(self.out_of_memory.take());
        drop(self.internal.take());
    }
}

fn classify_captured_error(stage: GpuOperationStage, error: wgpu::Error) -> Error {
    let kind = match error {
        wgpu::Error::Validation { .. } => GpuFaultKind::Validation,
        wgpu::Error::OutOfMemory { .. } => GpuFaultKind::OutOfMemory,
        wgpu::Error::Internal { .. } => GpuFaultKind::Internal,
    };
    stage
        .classify_fault(kind, error.to_string())
        .with_source(error)
}
