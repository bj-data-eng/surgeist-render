# P02-I02-S01-C01 Authored Style And Reference Domains

## 1 Header

- Cycle: `P02/I02/S01/C01`.
- Owning repository: `surgeist-render`.
- Status: `draft`.
- Cycle base: `d5ac5c23b3c66d3fa451bed6b751f1c82275b5d1`, the local reviewed
  P02-I02 planning head.
- Published prerequisite: `4f7fcb8b81d96f16b426f045b336aaba345c4cfa`
  (`P02/I01/S01/C04`), verified on local/remote history before P02-I02 planning.
- Specification: `plans/specs/cohesive-module-decomposition.md` at
  `3365399fa2411efd5cd8fcfdfe74d4b756cd6a79`, normalized SHA-256
  `892c5c1c2162a07c83bc124c3687e05ff62c6906930c154f25117792ec63d035`;
  sections M01-M05.5 and M06-M09.
- Sequence: `plans/sequences/cohesive-module-decomposition.md` at
  `d5ac5c23b3c66d3fa451bed6b751f1c82275b5d1`, normalized SHA-256
  `a45655293a54b5bf0986d508d6d2b278af68e5267dc373d8f5d56cb84e58c74b`;
  entry `C01 Authored Style And Reference Domains`.
- Outcome: current style and CPU-reference responsibilities move to explicit
  private children without changing the style front door, normalization,
  validation, reference pixels, or any public/feature/dependency surface.

## 2 Boundary

- Owned production input: `src/style.rs` and only import/visibility repairs
  required by its faithful move.
- Owned test-only input: `src/reference.rs` and only import/visibility repairs
  required by its faithful move.
- Required style output:
  `src/style/{mod,image,filter,clip,mask,decoration,background}.rs`.
- Required reference output:
  `src/reference/{mod,color,filter,mask}.rs` under the existing crate-level
  `#[cfg(test)]` boundary.
- Existing `src/lib.rs` declarations/reexports, public paths, signatures,
  constructors, validation order, diagnostics, normalized command order, image
  placement/repeat results, decoration geometry, and CPU-reference bytes remain
  unchanged.
- No semantic rename, algorithm change, oracle change, compatibility shim,
  `include!`, `#[path]`, generated concatenation, glob-reexport maze, generic
  helper module, source parser, inventory test, or size/count gate is allowed.
- Root, sibling, adapter, API artifact, gitlink, public hierarchical-front-door,
  dependency, feature, target, example, and manifest work is excluded.
- Commands use installed artifacts offline. No acquisition, installation,
  bootstrap, or update is authorized.

## 3 Impacts

- Public API and caller migration: none; the same style items remain reexported
  from the same crate-root paths.
- Behavior and diagnostics: none.
- Dependencies, features, targets, MSRV, docs, and examples: none.
- Test impact: import/module relocation only; test names, operations, inputs,
  assertions, and reference oracles remain unchanged.
- Generated artifacts: none in this leaf; root-owned artifacts untouched.
- Safety: no Surgeist-owned executable `unsafe` or unsafe-enabling allowance.

## 4 Ordered Tasks

### 4.1 T01 Decompose The Authored Style Domain

- Characterize the current image placement/repeat, background, decoration,
  filter, clip, and mask behavior before moving definitions.
- Convert `src/style.rs` to `src/style/mod.rs` and the six required style
  children. Move each type with its constructors, validation, conversions, and
  intrinsic helpers:
  - `image.rs`: resources, sources/layers, size/position, placement, repeat,
    attachment, and background areas;
  - `filter.rs`: filter values/lists, filtered image paint, backdrop input, and
    drop-shadow payload;
  - `clip.rs`: clip inputs, normalized clips, geometry/transforms, validation;
  - `mask.rs`: mask inputs, sources, modes, layers, stacks, composition;
  - `decoration.rs`: border, outline, radii, fragments, normalized decoration
    commands, decoration normalization;
  - `background.rs`: background layers/stacks/blends and normalized background
    commands.
- Keep only child declarations, explicit current-surface reexports, and genuine
  cross-child coordination in `style/mod.rs`.
- Use explicit sibling imports and the narrowest visibility; do not copy an
  item or add forwarding-only functions.
- Required focused pre/post commands, each of which must select at least one
  test and pass before and after the move:

  ```sh
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render style_reference_identifiers
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render image_placement
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render image_repeat_plan
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render background_
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render box_decoration_
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render border_
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render outlines_
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render filter_lists_
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render filtered_image_paint_
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render backdrop_filter_input_
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render clip_input
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render mask_input
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render mask_layer_stack
  ```

- Task acceptance commands after the focused repetition:

  ```sh
  test ! -e src/style.rs
  for required in src/style/mod.rs src/style/image.rs src/style/filter.rs \
    src/style/clip.rs src/style/mask.rs src/style/decoration.rs \
    src/style/background.rs; do test -f "$required"; done
  test -z "$(rg -n 'include!|#\s*\[\s*path\s*=' src/style || true)"
  test -z "$(git diff d5ac5c23b3c66d3fa451bed6b751f1c82275b5d1 -- src/lib.rs Cargo.toml)"
  CARGO_NET_OFFLINE=true cargo fmt --check
  CARGO_NET_OFFLINE=true cargo check -p surgeist-render
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render
  CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets -- -F unsafe-code -D warnings
  CARGO_NET_OFFLINE=true cargo check -p surgeist-render --features render-window,render-web
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-window,render-web
  CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets --features render-window,render-web -- -F unsafe-code -D warnings
  git diff --check
  ```

- Commit only the T01 range and return a complete move/visibility inventory for
  task review.

### 4.2 T02 Decompose CPU Reference Oracles And Reconcile C01

- Start only from the reviewed T01 head.
- Characterize current color/filter, blur/shadow, mask/composite, and image-
  conversion reference behavior before moving definitions.
- Convert `src/reference.rs` to `src/reference/mod.rs`; keep shared reference
  pixel/buffer facts in `mod.rs` and move:
  - straight/premultiplied conversion, color transforms, and compiled color-
    filter reference behavior to `color.rs`;
  - Gaussian blur, drop shadow, materialized filter pipeline, and extent
    planning to `filter.rs`;
  - mask sampling/extend, opacity, source-over, blend, and image-conversion
    composition behavior to `mask.rs`.
- Use explicit sibling imports and the narrowest visibility; retain the
  crate-level test-only compilation boundary and every oracle byte/result.
- Reconcile C01 module declarations/reexports/imports without reopening T01
  behavior or changing public source.
- Required focused pre/post commands, each of which must select at least one
  test and pass before and after the move:

  ```sh
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render reference_buffer_
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render reference_premultiplied_
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render reference_source_over_
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render reference_pixels_
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render reference_blends_
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render reference_alpha_masks_
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render reference_blur_
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render reference_color_filter_
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render compiled_color_filter_pipeline_
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render image_straight_rgba8_
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render materialized_drop_shadow_
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render materialized_image_filter_reference_
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render resolved_alpha_masks_match_reference_
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render direct_vello_blend_modes_match_reference_
  ```

- Task acceptance commands after the focused repetition:

  ```sh
  test ! -e src/reference.rs
  for required in src/reference/mod.rs src/reference/color.rs \
    src/reference/filter.rs src/reference/mask.rs; do test -f "$required"; done
  test -z "$(rg -n 'include!|#\s*\[\s*path\s*=' src/reference || true)"
  test -z "$(git diff d5ac5c23b3c66d3fa451bed6b751f1c82275b5d1 -- src/lib.rs Cargo.toml)"
  CARGO_NET_OFFLINE=true cargo fmt --check
  CARGO_NET_OFFLINE=true cargo check -p surgeist-render
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render
  CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets -- -F unsafe-code -D warnings
  CARGO_NET_OFFLINE=true cargo check -p surgeist-render --features render-window,render-web
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-window,render-web
  CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets --features render-window,render-web -- -F unsafe-code -D warnings
  git diff --check
  ```

- Run the complete C01 matrix on the exact T02 head.
- Commit only the T02 range and return the complete move/visibility inventory
  for task review.

## 5 Verification And Completion

Before each task, the worker records exact focused characterization commands and
results for every moved responsibility. After each task, the same operations and
oracles pass. Completion requires separate task-review `CLEAN` verdicts, a
status-only `complete` plan commit, a distinct holistic `CLEAN` review over the
exact cycle range, post-review repetition of the complete matrix, publication,
and authority-remote readback.

The coordinator runs this matrix at the completed cycle head before holistic
review and repeats it afterward:

```sh
set -euo pipefail
test ! -e src/style.rs
test ! -e src/reference.rs
for required in \
  src/style/mod.rs src/style/image.rs src/style/filter.rs src/style/clip.rs \
  src/style/mask.rs src/style/decoration.rs src/style/background.rs \
  src/reference/mod.rs src/reference/color.rs src/reference/filter.rs \
  src/reference/mask.rs; do
  test -f "$required"
done
test -z "$(rg -n 'include!|#\s*\[\s*path\s*=' src/style src/reference || true)"
test -z "$(git diff d5ac5c23b3c66d3fa451bed6b751f1c82275b5d1 -- src/lib.rs Cargo.toml README.md examples)"
CARGO_NET_OFFLINE=true cargo fmt --check
CARGO_NET_OFFLINE=true cargo check -p surgeist-render
CARGO_NET_OFFLINE=true cargo test -p surgeist-render
CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets -- -F unsafe-code -D warnings
CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-window
CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets --features render-window -- -F unsafe-code -D warnings
CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-web
CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets --features render-web -- -F unsafe-code -D warnings
CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-window,render-web
CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets --features render-window,render-web -- -F unsafe-code -D warnings
CARGO_NET_OFFLINE=true cargo run -p surgeist-render --example render_window_smoke --features render-window
CARGO_NET_OFFLINE=true cargo run -p surgeist-render --example render_window_smoke --features render-window,render-web
CARGO_NET_OFFLINE=true cargo check -p surgeist-render --target wasm32-unknown-unknown --features render-web --lib --tests
rustc +1.97.0 --version
CARGO_NET_OFFLINE=true cargo +1.97.0 check -p surgeist-render --all-targets
CARGO_NET_OFFLINE=true cargo +1.97.0 check -p surgeist-render --all-targets --features render-window,render-web
CARGO_NET_OFFLINE=true RUSTDOCFLAGS="-D warnings" cargo doc -p surgeist-render --no-deps --features render-window,render-web
CARGO_NET_OFFLINE=true cargo tree -p surgeist-render -e normal --depth 1
CARGO_NET_OFFLINE=true cargo tree -p surgeist-render -e dev --depth 1
CARGO_NET_OFFLINE=true cargo tree -p surgeist-render -e features -i bytemuck
CARGO_NET_OFFLINE=true cargo tree -p surgeist-render -e features -i vello_shaders
CARGO_NET_OFFLINE=true cargo tree -p surgeist-render --target wasm32-unknown-unknown --features render-web -e features -i getrandom@0.3.4
test -z "$(git ls-files -- Cargo.lock)"
owned_rust_files=("${(@f)$(
  {
    git ls-files -- '*.rs'
    git ls-files --others --exclude-standard -- '*.rs'
  } | sort -u
)}")
test "${#owned_rust_files[@]}" -gt 0
if rg -n --pcre2 '#\s*\[\s*(?:unsafe\s*\(|no_mangle\b|export_name\b)|\bunsafe\s*(?:\{|fn\b|trait\b|impl\b|extern\b)|\bstatic\s+mut\b|\bextern\s*(?:"[^"]*")?\s*\{' "${owned_rust_files[@]}"; then
  exit 1
else
  test "$?" -eq 1
fi
git diff --check d5ac5c23b3c66d3fa451bed6b751f1c82275b5d1..HEAD
test "$(git rev-parse HEAD)" = "$(git rev-parse main)"
test -z "$(git status --porcelain)"
```

The live smoke executables must render and exit on the native host. If the host
or an installed target/toolchain becomes unavailable, C01 is blocked rather
than substituting a weaker check. Every unsafe-scan match is classified and an
executable match blocks completion.

The publication head is immutable after holistic review. Publication uses an
exact authority-remote lease, followed by fetch/query/readback proving local
`main`, `origin/main`, and observed authority-remote `main` agree. Root
integration remains excluded.
