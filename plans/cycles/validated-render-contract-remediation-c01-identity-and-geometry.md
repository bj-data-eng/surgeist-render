# P03-I01-S01-C01 Collision-Safe Identity And Finite Geometry

## 1 Header

- Cycle: `P03/I01/S01/C01`.
- Owning repository: `surgeist-render`.
- Status: `complete`.
- Cycle base: published/read-back initiative baseline
  `b02fa0c372472c88a511f45cb74b1ec0b356d181`.
- Specification: `plans/specs/validated-render-contract-remediation.md` at
  `2ee7b14d525519ae1f5c8a2756512ab786b10cea`, normalized SHA-256
  `573a7f869f349d38c560dd9455fb6deb878161df4d3213517725a162dde63d7a`;
  R01 items 1, 2, and 4; R02; R03.1, R03.2, R03.4; R04.1-R04.4,
  R04.6; R05 rows 1-9 and 11; R06.1, R06.2, R06.4; R07; and R08 items
  1-4 and 6-10. Specification review: `CLEAN`.
- Sequence: `plans/sequences/validated-render-contract-remediation.md` at
  `3ebac7d40858b44f9dcdd2db337621d060cd4716`, normalized SHA-256
  `348523b2ff1762050041519cf2020b2a1748048838f31f91d34c87383910a6a0`;
  `C01 Collision-Safe Identity And Finite Geometry`. Sequence review: `CLEAN`.
- Outcome: Peniko owns unique backend blob IDs; render-owned mask reuse and
  upload telemetry use exact content equality; public and canonical rectangle
  validation reject non-finite derived maxima; and normalized-command stats
  support is test-only without a dead-code allowance.

## 2 Boundary

- This cycle changes behavior only at two validated correctness boundaries:
  image identity under a deliberate 64-bit collision and rectangle construction
  whose finite components produce a non-finite maximum.
- Public `ImageId` remains a copyable `u64` newtype with `new` and `get`; it is
  documented as a compact fingerprint or caller handle, never a sole proof of
  byte equality or backend identity.
- A private exact content identity owns dimensions and shared RGBA8 bytes.
  Render-owned key equality and upload telemetry compare exact content after any
  compact hash prefilter. Peniko's `Blob::new` owns unique Vello-facing IDs.
- Existing mask quality, extend, physical-size, allocation, lease, retention,
  graph-import, and cache-budget semantics remain unchanged.
- `Rect::try_new` and `validate_rect` reject either non-finite derived maximum
  with `InvalidInput`. Finite zero-area rectangles remain valid, and private
  unchecked constructors remain available only for internal invalid fixtures.
- Only rustdoc required to describe the changed `ImageId`, `Image`, and `Rect`
  contracts is included. Complete current-surface documentation belongs to C02.
- Root, siblings, authored-style ownership, hierarchical front door, adapters,
  API artifacts, gitlinks, dependencies, features, scripts, generators, CI,
  permanent lints, source parsers, plan-closure tests, and unrelated cleanup are
  excluded.
- No production-visible test API, duplicate pixel buffer, broad lint allowance,
  plan identifier outside `plans/`, or owned `unsafe` is permitted.
- The clean current `main` is the landing worktree. Planning commits after the
  cycle base are part of the cycle range; each task uses the reviewed predecessor
  head and contributes one logical implementation commit.

## 3 Impacts

- Public API: behavior-correcting for `Rect::try_new`; documentation-additive
  for changed image and rectangle items; public names, representations,
  reexports, variants, defaults, and visibility otherwise unchanged.
- Dependencies/features/targets: unchanged; no software acquisition.
- Generated artifacts: none. Root-owned API artifacts are excluded.
- Docs/examples: focused rustdoc only; README and examples unchanged unless a
  changed-item link requires a factual repair.
- MSRV: installed Rust 1.97 remains the root integration compatibility floor.
- Root follow-up: after C01 publication, return the leaf candidate and corrected
  contract as C02's immutable base; do not edit root.
- Unsafe: owned Rust remains completely free of `unsafe` and allowances that
  permit it.

## 4 Tasks

### 4.1 T01 Make Backend And Render-Owned Image Identity Collision-Safe

- Files/area: `src/image.rs`; exact private consumers in `src/frame/graph.rs`,
  `src/resource/{manager,lease,test_support}.rs`, `src/stats.rs`,
  `src/renderer/{mod,publication,test_support}.rs`, and focused image/GPU tests.
- Intended behavior: `Image::from_rgba` gives Vello a Peniko-generated unique
  blob ID; render-owned resolved-mask and first-observation identity compares
  exact dimensions and shared bytes; quality/extend/size distinctions remain;
  identical exact content remains safely reusable.
- RED evidence: add condition-named tests that deliberately construct two
  same-sized images with different bytes and one forced public fingerprint, then
  observe distinct Peniko blob IDs, mask keys/resources, and first-upload
  telemetry. Before correction, the focused test must fail for the intended
  alias or misclassification, not for an unrelated setup error. The seam is
  private and `#[cfg(test)]` only.
- Acceptance: forced collisions cannot alias direct Vello residency, graph mask
  imports, idle retained uploads, or upload statistics; exact equal content and
  equal sampling facts retain reuse; no pixel copy is added beyond existing
  shared/Peniko ownership; changed identity rustdoc is accurate.
- Commands:

  ```sh
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render colliding_image_fingerprints
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render resolved_mask_upload_keys_include_identity_dimensions_and_sampling
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render warm_image_reuse_reports_cache_hit
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render image_color_filter_execution_changes_image_identity_when_bytes_change
  CARGO_NET_OFFLINE=true cargo fmt --check
  CARGO_NET_OFFLINE=true RUSTFLAGS="-D warnings" cargo check -p surgeist-render
  CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets -- -F unsafe-code -D warnings
  ```

- Dependency/intended commit: reviewed C01 planning head; one exact image
  identity and regression commit.

### 4.2 T02 Reject Rectangles With Non-Finite Derived Maxima

- Files/area: `src/geometry.rs`, `src/validation.rs`, focused model/frame tests,
  and only the rustdoc describing the changed rectangle contract.
- Intended behavior: public construction and canonical internal validation
  reject x-maximum and y-maximum overflow immediately with `InvalidInput`, while
  every finite maximum including zero-size boundaries remains accepted.
- RED evidence: add focused public-constructor and canonical-validation tests for
  both overflowing axes. Before correction, constructor cases must unexpectedly
  succeed or expose a non-finite maximum and the internal fixture must pass
  canonical validation. Add the condition-named
  `rect_constructor_accepts_finite_and_zero_area_boundaries` characterization
  before changing validation and keep it green before and after the correction.
- Acceptance: both boundaries enforce identical derived-max semantics and
  diagnostics; `TryFrom<kurbo::Rect>` inherits the rule; existing downstream
  bounds rejection remains defense in depth; ordinary geometry behavior and
  error typing remain green.
- Commands:

  ```sh
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render rect_constructor_rejects_non_finite_derived_maxima
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render canonical_rect_validation_rejects_non_finite_derived_maxima
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render rect_constructor_accepts_finite_and_zero_area_boundaries
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render signed_device_bounds_floor_minima_and_ceil_maxima
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render rect_try_from_kurbo_rejects_invalid_bounds
  CARGO_NET_OFFLINE=true cargo fmt --check
  CARGO_NET_OFFLINE=true RUSTFLAGS="-D warnings" cargo check -p surgeist-render
  CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets -- -F unsafe-code -D warnings
  ```

- Dependency/intended commit: clean reviewed T01 head; one rectangle invariant
  and regression commit.

### 4.3 T03 Isolate Normalized-Command Statistics As Test Support

- Files/area: `src/command.rs`; the 21 existing call sites in model, style, and
  Vello tests remain unchanged as verification inputs.
- Intended outcome: compile `RenderCommands::stats` and its direct
  `collect_render_stats` import only under `cfg(test)`, remove
  `#[allow(dead_code)]`, and preserve the active stateful statistics publication
  in renderer dispatch.
- RED evidence: transiently remove only the allowance and run Clippy to confirm
  the helper is reported as unused in non-test targets; do not commit that
  intermediate state. Isolating the method/import to test builds then makes the
  same gate green while all existing normalized-command call sites compile.
- Acceptance: non-test targets contain neither helper nor import; test builds
  retain the one existing helper and all 21 callers; no replacement helper or
  allowance exists; warm reuse, failed publication, normalization statistics,
  and active render statistics retain their current behavior.
- Commands:

  ```sh
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render warm_image_reuse_reports_cache_hit
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render failed_render_does_not_warm_image_reuse_stats
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render texture_lifecycle_accounting_is_separate_from_image_cache_stats
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render scene_normalization_preserves_stats
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render authored_shadow_normalization_preserves_order_and_typed_boundaries
  CARGO_NET_OFFLINE=true cargo fmt --check
  CARGO_NET_OFFLINE=true RUSTFLAGS="-D warnings" cargo check -p surgeist-render
  CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets -- -F unsafe-code -D warnings
  ```

- Dependency/intended commit: clean reviewed T02 head; one test-only statistics
  isolation commit.

## 5 Completion

After all three ordered task ranges are independently `CLEAN`, transition this
plan to `complete` in a status-only commit and run the final matrix before the
distinct holistic review:

```sh
CARGO_NET_OFFLINE=true cargo fmt --check
CARGO_NET_OFFLINE=true RUSTFLAGS="-D warnings" cargo check -p surgeist-render
CARGO_NET_OFFLINE=true RUSTFLAGS="-D warnings" cargo check -p surgeist-render --features render-window
CARGO_NET_OFFLINE=true RUSTFLAGS="-D warnings" cargo check -p surgeist-render --features render-web
CARGO_NET_OFFLINE=true RUSTFLAGS="-D warnings" cargo check -p surgeist-render --features render-window,render-web
CARGO_NET_OFFLINE=true cargo test -p surgeist-render
CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets -- -F unsafe-code -D warnings
CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-window
CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets --features render-window -- -F unsafe-code -D warnings
CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-web
CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets --features render-web -- -F unsafe-code -D warnings
CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-window,render-web
CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets --features render-window,render-web -- -F unsafe-code -D warnings
CARGO_NET_OFFLINE=true RUSTFLAGS="-D warnings" cargo check -p surgeist-render --target wasm32-unknown-unknown --features render-web --lib --tests
CARGO_NET_OFFLINE=true RUSTUP_OFFLINE=1 cargo +1.97.0 check -p surgeist-render --all-targets
CARGO_NET_OFFLINE=true RUSTUP_OFFLINE=1 cargo +1.97.0 check -p surgeist-render --all-targets --features render-window,render-web
CARGO_NET_OFFLINE=true RUSTDOCFLAGS="-D warnings" cargo doc -p surgeist-render --no-deps --features render-window,render-web
CARGO_NET_OFFLINE=true cargo run -p surgeist-render --example render_window_smoke --features render-window
CARGO_NET_OFFLINE=true cargo run -p surgeist-render --example render_window_smoke --features render-window,render-web
git diff --check b02fa0c372472c88a511f45cb74b1ec0b356d181..HEAD
test -z "$(git status --porcelain)"
```

Final source/diff inspection must additionally prove no dependency, feature,
root/sibling, generated-artifact, permanent-lint, source-parser, plan-identifier,
or owned-unsafe change. C01 completion requires three task-clean verdicts, final
matrix success, holistic `CLEAN`, landing on local `main`, authority-remote
publication/readback agreement, cleanup of agent-owned temporary resources, and
the immutable C02-base handoff. Missing installed tooling, graphical host
capability, credentials, or stable remote access is reported through the
canonical blocker contract; it is not converted into a passing skip.
