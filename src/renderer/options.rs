/// GPU-only renderer configuration that is fixed when a [`super::Renderer`] is created.
///
/// Defaults are [`Antialiasing::Area`], diagnostics disabled,
/// [`EffectQualityPolicy::RequireHighPrecision`], and a 64 MiB idle
/// [`ResourceCacheBudget`]. Options select quality and retention policy, never a
/// CPU renderer or CPU fallback.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Options {
    antialiasing: Antialiasing,
    debug: bool,
    effect_quality_policy: EffectQualityPolicy,
    resource_cache_budget: ResourceCacheBudget,
}

impl Options {
    /// Creates the default GPU-only renderer configuration.
    ///
    /// This is equivalent to [`Self::default`].
    #[must_use]
    pub const fn new() -> Self {
        Self {
            antialiasing: Antialiasing::Area,
            debug: false,
            effect_quality_policy: EffectQualityPolicy::RequireHighPrecision,
            resource_cache_budget: ResourceCacheBudget::DEFAULT,
        }
    }

    /// Returns the configured antialiasing method.
    #[must_use]
    pub const fn antialiasing(self) -> Antialiasing {
        self.antialiasing
    }

    /// Returns this configuration with a different antialiasing method.
    #[must_use]
    pub const fn with_antialiasing(mut self, antialiasing: Antialiasing) -> Self {
        self.antialiasing = antialiasing;
        self
    }

    /// Returns whether renderer diagnostics are enabled.
    #[must_use]
    pub const fn debug(self) -> bool {
        self.debug
    }

    /// Returns this configuration with renderer diagnostics enabled or disabled.
    #[must_use]
    pub const fn with_debug(mut self, debug: bool) -> Self {
        self.debug = debug;
        self
    }

    /// Returns the policy for effect precision when high precision is unavailable.
    #[must_use]
    pub const fn effect_quality_policy(self) -> EffectQualityPolicy {
        self.effect_quality_policy
    }

    /// Returns this configuration with a different effect precision policy.
    ///
    /// Allowing reduced precision changes only the GPU working format; it does
    /// not enable CPU execution.
    #[must_use]
    pub const fn with_effect_quality_policy(
        mut self,
        effect_quality_policy: EffectQualityPolicy,
    ) -> Self {
        self.effect_quality_policy = effect_quality_policy;
        self
    }

    /// Returns the maximum retained idle effect-resource cache budget.
    #[must_use]
    pub const fn resource_cache_budget(self) -> ResourceCacheBudget {
        self.resource_cache_budget
    }

    /// Returns this configuration with a different idle effect-resource cache budget.
    #[must_use]
    pub const fn with_resource_cache_budget(
        mut self,
        resource_cache_budget: ResourceCacheBudget,
    ) -> Self {
        self.resource_cache_budget = resource_cache_budget;
        self
    }
}

impl Default for Options {
    fn default() -> Self {
        Self::new()
    }
}

/// Policy for choosing high precision or reduced precision on a compatible GPU.
///
/// If the selected GPU cannot satisfy the policy, rendering returns a typed runtime
/// capability error. Neither policy permits CPU execution or fallback.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum EffectQualityPolicy {
    /// Require high-precision `Rgba16Float` effect execution.
    #[default]
    RequireHighPrecision,
    /// Prefer high precision and allow reduced-precision `Rgba8Unorm` only when unavailable.
    AllowReducedPrecision,
}

/// Byte budget for retaining idle, reusable GPU effect resources.
///
/// The budget never rejects resources required by the active frame. Allocation
/// and device-limit failures remain typed runtime failures.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResourceCacheBudget(u64);

impl ResourceCacheBudget {
    /// Disables idle effect-resource retention.
    pub const DISABLED: Self = Self(0);

    /// Retains up to 64 MiB of idle effect resources by default.
    pub const DEFAULT: Self = Self(64 * 1024 * 1024);

    /// Creates an idle effect-resource retention budget in bytes.
    #[must_use]
    pub const fn new(bytes: u64) -> Self {
        Self(bytes)
    }

    /// Returns this retention budget in bytes.
    #[must_use]
    pub const fn bytes(self) -> u64 {
        self.0
    }
}

impl Default for ResourceCacheBudget {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Antialiasing mode for the GPU fine-raster stage.
///
/// The default is area antialiasing.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Antialiasing {
    /// Use analytic area coverage.
    #[default]
    Area,
    /// Use eight-sample multisample antialiasing.
    Msaa8,
    /// Use sixteen-sample multisample antialiasing.
    Msaa16,
}
