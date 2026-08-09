# P03-I01-S01-C02 Current Public Model Documentation

## 1 Header

- Cycle: `P03/I01/S01/C02`.
- Owning repository: `surgeist-render`.
- Status: `complete`.
- Cycle base: published and remotely read-back C01 candidate
  `6aeb5a325bc750888a59586d98eda96cd1373f6b`.
- Specification: `plans/specs/validated-render-contract-remediation.md` at
  `2ee7b14d525519ae1f5c8a2756512ab786b10cea`, normalized SHA-256
  `573a7f869f349d38c560dd9455fb6deb878161df4d3213517725a162dde63d7a`;
  R01 item 3; R02; R03.3; R04.5; R05 public-documentation row 10;
  R06.3; R07; and R08 items 5 and 7-10. Specification review: `CLEAN`.
- Sequence: `plans/sequences/validated-render-contract-remediation.md` at
  `3ebac7d40858b44f9dcdd2db337621d060cd4716`, normalized SHA-256
  `348523b2ff1762050041519cf2020b2a1748048838f31f91d34c87383910a6a0`;
  `C02 Current Public Model Documentation`. Sequence review: `CLEAN`.
- Outcome: every item exported through the current `src/lib.rs` front door has
  accurate proportional rustdoc, and strict missing-docs rustdoc succeeds for
  default and combined features without a committed lint or inventory artifact.

## 2 Boundary

- C01 is published and read back at the cycle base. Its corrected `ImageId`,
  exact-content reuse, `Rect`, validation, and test-only statistics contracts
  are fixed inputs to this documentation-only cycle.
- At the cycle base, strict missing-docs rustdoc exits 101 with 925 diagnostics
  under both default features and `render-window,render-web`. This compiler
  inventory is entry evidence only and does not become a repository artifact.
- Rustdoc depth is proportional: types name phase, role, units, and invariants;
  variants and public fields name their observable choice or value; fallible
  construction names rejection; behavior-bearing methods name effects,
  context, and results; obvious accessors remain concise.
- Current authored-style exports are documented as current model behavior. The
  docs do not claim permanent ownership or resolve the deferred cross-crate
  authored-style boundary.
- Existing accurate rustdoc is preserved unless a local correction is required
  for consistency with C01 or to repair a broken intra-doc link.
- Public names, visibility, reexports, representations, fields, variants,
  defaults, conversions, errors, diagnostics, rendering behavior, and examples
  remain unchanged.
- Root, siblings, gitlinks, root-owned API artifacts, hierarchical front-door
  work, dependencies, features, targets, tests, fixtures, scripts, generators,
  CI, permanent lints, source parsers, inventory tests, plan-closure tests, and
  unrelated cleanup are excluded.
- No plan identifier or planning provenance may appear outside `plans/`. No
  owned `unsafe`, broad lint allowance, or production-visible test surface is
  permitted.
- The clean current `main` is the sequential landing worktree. Each task starts
  from its reviewed predecessor and contributes one logical documentation
  commit; no separate worktree or temporary repository resource is created.

## 3 Impacts

- Public API: documentation-additive only. Names, visibility, type layouts,
  trait implementations, defaults, and behavior remain unchanged.
- Dependencies/features/targets: unchanged; no software acquisition.
- Generated artifacts: none. Root owns API generation and is excluded.
- Docs/examples: source rustdoc only; README and examples remain unchanged
  unless the smallest factual source-link repair is required.
- MSRV: installed Rust 1.97 remains the root integration compatibility floor.
- Root follow-up: after publication, return the fully documented leaf candidate;
  do not edit root, refresh root API artifacts, or promote its gitlink.
- Unsafe: all 95 currently owned Rust files and any newly tracked or non-ignored
  Rust file remain free of executable `unsafe` and unsafe-enabling allowances.

## 4 Tasks

### 4.1 T01 Document Core Render Models

- Files/area: `src/geometry.rs`, `src/image.rs`, `src/layer.rs`, `src/paint.rs`,
  `src/renderer/options.rs`, `src/scene.rs`, `src/shape.rs`, and `src/text.rs`.
- Intended outcome: every public core geometry, image, layer, paint, renderer
  option, scene, shape, and text item has rustdoc that describes its current
  phase, units, invariants, construction, defaults, failures, and behavior at
  the depth required by R04.5.
- Structural RED evidence: at the cycle base, the strict compiler command emits
  missing-documentation diagnostics in these files. This is exact product
  artifact evidence, not a behavioral test or source-text assertion.
- Acceptance criteria:
  - every exported item, public variant, public field, constructor, builder,
    conversion, default, and behavior-bearing method in the owned files is
    documented proportionally;
  - image identity docs preserve C01's non-unique public-fingerprint and exact
    backend/content-identity distinction, and rectangle docs preserve the
    finite-derived-maximum contract;
  - the transitional strict-rustdoc audit emits no missing-docs diagnostic from
    an owned file; failure remains expected only from later task areas;
  - no source item other than rustdoc comments changes semantically, and no
    public surface, test, lint, manifest, or artifact changes.
- Commands:

```sh
CARGO_NET_OFFLINE=true cargo fmt --check
CARGO_NET_OFFLINE=true RUSTDOCFLAGS="-D warnings" cargo doc -p surgeist-render --no-deps --features render-window,render-web
CARGO_NET_OFFLINE=true RUSTDOCFLAGS="-D warnings" cargo test -p surgeist-render --doc --features render-window,render-web
CARGO_NET_OFFLINE=true RUSTFLAGS="-D warnings" cargo check -p surgeist-render --features render-window,render-web
CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets --features render-window,render-web -- -F unsafe-code -D warnings
CARGO_NET_OFFLINE=true RUSTDOCFLAGS="-D missing_docs" cargo doc -p surgeist-render --no-deps --features render-window,render-web
git diff --check 6aeb5a325bc750888a59586d98eda96cd1373f6b..HEAD
```

- The final strict command is a transitional compiler audit and may exit 101
  only for files owned by T02-T04. Any T01-owned diagnostic blocks this task.
- Dependency: reviewed C02 plan.
- Intended commit: document core render models.

### 4.2 T02 Document Capability And Error Models

- Files/area: `src/capability.rs` and `src/error.rs`.
- Intended outcome: every public capability report, operation, availability
  reason, error code, diagnostic payload, field, constructor, conversion, and
  accessor states when it is produced and how callers distinguish it.
- Structural RED evidence: the cycle-base strict compiler command reports 223
  missing-docs diagnostics across these two source areas.
- Acceptance criteria:
  - all current public items in both files have proportional rustdoc, including
    every public variant and field;
  - capability docs distinguish compiled features from runtime capability facts,
    and error docs describe current typed failure meaning without promising new
    variants, ownership, or recovery behavior;
  - the transitional strict-rustdoc audit emits no missing-docs diagnostic from
    either owned file; failure remains expected only from T03-T04 areas;
  - no behavior, representation, public surface, test, lint, manifest, or
    generated artifact changes.
- Commands:

```sh
CARGO_NET_OFFLINE=true cargo fmt --check
CARGO_NET_OFFLINE=true RUSTDOCFLAGS="-D warnings" cargo doc -p surgeist-render --no-deps --features render-window,render-web
CARGO_NET_OFFLINE=true RUSTDOCFLAGS="-D warnings" cargo test -p surgeist-render --doc --features render-window,render-web
CARGO_NET_OFFLINE=true RUSTFLAGS="-D warnings" cargo check -p surgeist-render --features render-window,render-web
CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets --features render-window,render-web -- -F unsafe-code -D warnings
CARGO_NET_OFFLINE=true RUSTDOCFLAGS="-D missing_docs" cargo doc -p surgeist-render --no-deps --features render-window,render-web
git diff --check 6aeb5a325bc750888a59586d98eda96cd1373f6b..HEAD
```

- The final strict command is a transitional compiler audit and may exit 101
  only for files owned by T03-T04. Any T02-owned diagnostic blocks this task.
- Dependency: T01 task review `CLEAN`.
- Intended commit: document capability and error models.

### 4.3 T03 Document Image, Background, Clip, And Mask Style Models

- Files/area: `src/style/mod.rs`, `src/style/image.rs`,
  `src/style/background.rs`, `src/style/clip.rs`, and `src/style/mask.rs`.
- Intended outcome: every current public style image, background, clip, mask,
  source, placement, repeat, attachment, coordinate-space, and normalization
  item has accurate rustdoc for its authored, normalized, or resolved phase.
- Structural RED evidence: the cycle-base strict compiler command reports 273
  missing-docs diagnostics across these five source areas.
- Acceptance criteria:
  - every exported item, variant, public field, constructor, builder,
    conversion, and behavior-bearing method in the owned files is documented;
  - docs distinguish authored symbolic inputs, normalized values, resolved
    resources, and current unsupported diagnostics without claiming permanent
    authored-style ownership or future execution;
  - the transitional strict-rustdoc audit emits no missing-docs diagnostic from
    an owned file; failure remains expected only from T04 files;
  - no behavior, visibility, reexport, representation, test, lint, manifest, or
    generated artifact changes.
- Commands:

```sh
CARGO_NET_OFFLINE=true cargo fmt --check
CARGO_NET_OFFLINE=true RUSTDOCFLAGS="-D warnings" cargo doc -p surgeist-render --no-deps --features render-window,render-web
CARGO_NET_OFFLINE=true RUSTDOCFLAGS="-D warnings" cargo test -p surgeist-render --doc --features render-window,render-web
CARGO_NET_OFFLINE=true RUSTFLAGS="-D warnings" cargo check -p surgeist-render --features render-window,render-web
CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets --features render-window,render-web -- -F unsafe-code -D warnings
CARGO_NET_OFFLINE=true RUSTDOCFLAGS="-D missing_docs" cargo doc -p surgeist-render --no-deps --features render-window,render-web
git diff --check 6aeb5a325bc750888a59586d98eda96cd1373f6b..HEAD
```

- The final strict command is a transitional compiler audit and may exit 101
  only for `src/style/decoration.rs` and `src/style/filter.rs`. Any T03-owned
  diagnostic blocks this task.
- Dependency: T02 task review `CLEAN`.
- Intended commit: document image and clipping style models.

### 4.4 T04 Document Decoration And Filter Style Models

- Files/area: `src/style/decoration.rs` and `src/style/filter.rs`.
- Intended outcome: every current public border, outline, box-decoration,
  shadow, filter, filter-region, and filter-operation item has accurate rustdoc
  for its authored, normalized, or executable phase and current diagnostics.
- Structural RED evidence: the cycle-base strict compiler command reports 152
  missing-docs diagnostics across these two source areas.
- Acceptance criteria:
  - every exported item, variant, public field, constructor, builder,
    conversion, and behavior-bearing method in both files is documented;
  - docs state units, finite/range invariants, operation ordering, defaults,
    fallible construction, and supported-versus-diagnostic-only behavior without
    describing a new rendering algorithm or future ownership;
  - strict missing-docs rustdoc succeeds globally under default and combined
    features with zero missing item, broken-link, warning, or doctest failure;
  - no behavior, visibility, reexport, representation, test, lint, manifest, or
    generated artifact changes.
- Commands:

```sh
CARGO_NET_OFFLINE=true cargo fmt --check
CARGO_NET_OFFLINE=true RUSTDOCFLAGS="-D warnings -D missing_docs" cargo doc -p surgeist-render --no-deps
CARGO_NET_OFFLINE=true RUSTDOCFLAGS="-D warnings -D missing_docs" cargo doc -p surgeist-render --no-deps --features render-window,render-web
CARGO_NET_OFFLINE=true RUSTDOCFLAGS="-D warnings" cargo test -p surgeist-render --doc --features render-window,render-web
CARGO_NET_OFFLINE=true RUSTFLAGS="-D warnings" cargo check -p surgeist-render --features render-window,render-web
CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets --features render-window,render-web -- -F unsafe-code -D warnings
git diff --check 6aeb5a325bc750888a59586d98eda96cd1373f6b..HEAD
```

- Dependency: T03 task review `CLEAN`.
- Intended commit: document decoration and filter style models.

## 5 Completion

After all four ordered task ranges are independently `CLEAN`, transition this
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
CARGO_NET_OFFLINE=true RUSTDOCFLAGS="-D warnings -D missing_docs" cargo doc -p surgeist-render --no-deps
CARGO_NET_OFFLINE=true RUSTDOCFLAGS="-D warnings -D missing_docs" cargo doc -p surgeist-render --no-deps --features render-window,render-web
CARGO_NET_OFFLINE=true RUSTDOCFLAGS="-D warnings" cargo test -p surgeist-render --doc --features render-window,render-web
CARGO_NET_OFFLINE=true cargo run -p surgeist-render --example render_window_smoke --features render-window
CARGO_NET_OFFLINE=true cargo run -p surgeist-render --example render_window_smoke --features render-window,render-web
git diff --check 6aeb5a325bc750888a59586d98eda96cd1373f6b..HEAD
test -z "$(git status --porcelain)"
```

Final source/diff inspection must additionally prove that the complete range is
rustdoc-only outside `plans/`; public names, visibility, reexports,
representations, defaults, behavior, and manifests are unchanged; no root,
sibling, generated-artifact, permanent-lint, source-parser, inventory-test,
plan-closure-test, dependency, feature, or non-plan identifier change exists;
and the explicit owned-Rust manifest plus unsafe-forbidden Clippy contain no
executable `unsafe` or unsafe-enabling allowance.

C02 completion requires four task-clean verdicts, final matrix success, a clean
holistic review over the exact cycle range, an unchanged-head final rerun, a
fast-forward publication of the immutable candidate to authority `origin/main`,
and fresh readback proving local `main`, `origin/main`, and observed remote
`main` agree. Return the published leaf candidate to root with C01's behavioral
corrections, C02's documentation-only public effect, exact verification, and
the deferred authored-style boundary noted as excluded. Do not edit root.

The only genuine blockers are unowned dirty state, unavailable required tooling
with no installed equivalent, a documentation requirement that would contradict
current behavior or the reviewed boundary, owned executable `unsafe`, or remote
publication failure after the bounded reconciliation process.
