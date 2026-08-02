# P01-I03-S01-C14 Platform Evidence Documentation And Final Quality

## 1 Header

- Cycle: `P01/I03/S01/C14`.
- Owning repository: `surgeist-render`.
- Status: `complete`.
- Cycle base and published prerequisite:
  `2bd2d36638b8a3436b69cb99a905d37b36886d16` (`P01/I03/S01/C13`),
  verified on local `main`, `origin/main`, and the observed authority-remote
  `main` before this plan was written.
- Specification: `plans/specs/gpu-render-pipeline.md` at
  `e88e00bc8bd9325ae82ef1f1db2e4c72de44b28b`,
  `sha256:30dee50db5e8ad2f06df7cbd01ef34c61b5ccd037924a100932053a8507712af`;
  sections S02-S06B, S31-S37, and S38 items 1-11.
- Sequence: `plans/sequences/gpu-render-pipeline.md` at
  `8709e0d6ce1e04b646d41763c5efafdf9ecf7daf`,
  `sha256:7047172964b388c17e6d7fee43fb0be788dd9c5761d51dabc566c605d31a57d1`;
  entry `C14 Platform Evidence Documentation And Final Quality`.
- Outcome: document the final GPU-only public contract, add the tracked native
  presented direct/graph smoke example, close deterministic dependency,
  provenance, feature, target, Rust-1.97, and quality guards, and produce a
  leaf candidate whose remaining browser-host and root-facade work is an exact
  handoff rather than an unproved leaf claim.

## 2 Boundary

- C13 is the immutable implementation entry state. Direct scenes use one
  transaction-owned internal-Vello pass; supported effects use the closed GPU
  graph; public capabilities, statistics, and the `101/22/S30C` inventories are
  reconciled; CPU pixel execution is test-only.
- This cycle changes documentation, the tracked public native-window example,
  manifest example-target metadata when required, and test-owned final guards.
  It does not add a production route, public semantic type, dependency, Cargo
  feature, browser harness, generated artifact, fixture, shader, or fallback.
- Native default, `render-web`, `render-window`, and combined feature evidence
  is leaf-owned. Native `WebCanvas` remains its typed platform diagnostic.
- The supported wasm leaf claim is compile-only
  `wasm32-unknown-unknown + render-web --lib --tests`. Real browser canvas
  execution and presentation remain root-host integration work.
- Native presented evidence must create a live `surgeist-window::Handle`, render
  and present one direct frame and one GPU-graph frame, assert both public route
  observations, and exit deterministically. A missing graphical session is a
  blocker, never a passing skip or authority to acquire a display substitute.
- The already-installed environment provides active Rust 1.97.0, exact 1.97
  toolchains, and the `wasm32-unknown-unknown` target. Commands run offline; no
  acquisition, update, bootstrap, or dependency download is authorized.
- Current root facts are read-only handoff evidence: root
  `359322aae90afbaf68ba7c9afffd79fb57b383d6` declares `rust-version = "1.97"`,
  records the authoritative leaf URL in `.gitmodules`, and currently pins
  `fe58f35aebaf43177fd761b8222a67b3e8f11827`. Root facade, API artifacts, and
  gitlink promotion remain outside this leaf cycle.
- Structural Clippy signals such as `too_many_lines` are advisory rather than
  Boolean C14 failures. The final I03 cycle, C15, owns the separate sprawl review
  of the earlier 100-line lint experiment and every justified cohesion
  remediation; C14 neither repeats nor hides that work.

## 3 Impacts

- Public API: documentation-only clarification of the already-published C13
  surface; no name, signature, trait, default, error, or capability change.
- Dependencies and features: unchanged. `pollster` remains dev-only;
  `getrandom = 0.3.4` with only `wasm_js` remains the sole target-specific dev
  feature unifier; external `vello` and `glifo` remain absent.
- Cargo targets: add only the tracked `render_window_smoke` example target and
  its `render-window` requirement when Cargo needs explicit gating.
- Artifacts and fixtures: `NOTICE-VELLO.md`, both Vello license texts, custom
  WGSL, and the Ahem fixture/provenance remain source-owned and unchanged unless
  a guard identifies a factual defect. There is no leaf generator.
- Documentation and examples: `README.md`, crate docs, affected public-item
  docs, and `examples/render_window_smoke.rs` become the final leaf-facing
  explanation and executable native composition example.
- MSRV: source and all targets must check on installed Rust 1.97.x; no newer API
  may enter this cycle.
- Root follow-up: facade adaptation, browser direct/graph canvas execution, API
  artifact refresh, integration checks, and gitlink promotion.
- Safety: every tracked or non-ignored Surgeist-owned Rust file remains free of
  executable unsafe and unsafe-enabling lint allowances.

## 4 Ordered Tasks

### 4.1 T01 Document The Final GPU-Only Public Contract

- Area: `README.md`, crate docs in `src/lib.rs`, affected public-item docs in
  `src/{capability.rs,error.rs,image.rs,layer.rs,renderer.rs,stats.rs,surface.rs,text.rs}`,
  and focused documentation guards in `src/tests.rs`.
- Outcome: explain the direct and GPU-graph routes, no-production-CPU-fallback
  policy, high/reduced precision policy, semantic versus runtime capabilities,
  async operation/failure publication model, statistics meaning, explicit
  readback, native presented smoke command, and browser/root ownership at the
  public front door. Every initiative-changed public item documents its phase,
  units, defaults, failure semantics, and host boundary where applicable.
- RED:
  `c14_public_docs_describe_gpu_routes_precision_failures_and_host_boundaries`
  fails because the C13 README is a one-line summary and crate docs do not state
  the final contract;
  `c14_changed_public_items_have_semantic_documentation` fails on the exact
  undocumented initiative-owned public items rather than requiring blanket
  documentation for unrelated historical surface.
- Acceptance: both tests pass without weakening source checks; examples and
  prose use only public names that exist at C13; docs make no browser execution,
  broad-effect support, fallback, or root-ownership overclaim; rustdoc builds
  with warnings denied under the combined native feature surface.
- Commands:
  `CARGO_NET_OFFLINE=true cargo test -p surgeist-render tests::c14_public_docs_describe_gpu_routes_precision_failures_and_host_boundaries -- --exact`;
  `CARGO_NET_OFFLINE=true cargo test -p surgeist-render tests::c14_changed_public_items_have_semantic_documentation -- --exact`;
  run
  `CARGO_NET_OFFLINE=true RUSTDOCFLAGS="-D warnings" cargo doc -p surgeist-render --no-deps --features render-window,render-web`;
  run `C14-CHECK`.
- Depends on: none.
- Intended commit: `docs(api): describe final gpu render contract`.

### 4.2 T02 Add Presented Direct And Graph Smoke Evidence

- Area: `Cargo.toml`, `examples/render_window_smoke.rs`, and the focused source
  guard in `src/tests.rs`.
- Outcome: add one public-composition example that owns only example lifecycle,
  creates a live native window handle, constructs the renderer/surface, renders
  and presents a direct scene and a supported GPU-graph scene, asserts
  `RenderRoute::DirectVello` then `RenderRoute::GpuGraph`, and exits after both
  successful presentations.
- RED: `render_window_smoke_source_covers_direct_and_graph_routes` fails because
  the tracked example target and source do not exist at the C13 base.
- Acceptance: the example uses no private module, unsafe, CPU fallback,
  readback, arbitrary sleep, hidden skip, test-only API, or extra dependency;
  Cargo gates the example to `render-window`; it checks and runs with
  `render-window` and the additive `render-window,render-web` combination; an
  unavailable live graphical host returns the canonical blocker.
- Commands:
  `CARGO_NET_OFFLINE=true cargo test -p surgeist-render tests::render_window_smoke_source_covers_direct_and_graph_routes -- --exact`;
  run
  `CARGO_NET_OFFLINE=true cargo check -p surgeist-render --example render_window_smoke --features render-window`;
  run both required native smoke commands in Section 5; run `C14-CHECK`.
- Depends on: T01.
- Intended commit: `test(window): add presented route smoke`.

### 4.3 T03 Close Platform Dependency And Provenance Guards

- Area: `src/tests.rs` and only a directly falsified C14-owned documentation,
  example, manifest, or provenance artifact when required by RED evidence.
- Outcome: make the final S36-S38 contract mechanically finite: exact direct
  dependency roles and sources, feature/target combinations, Rust-1.97
  compatibility, example/docs inventory, Vello provenance/license integrity,
  test-only oracle ownership, no stale external-Vello/helper surface, and no
  unsupported leaf/browser claim.
- RED: `c14_dependency_feature_and_provenance_contract_is_final` and
  `c14_final_quality_contract_matches_published_gpu_architecture` first expose
  any missing final cross-surface assertion against the C13 base plus T01-T02;
  they must fail on a concrete missing guard, not on a fabricated production
  behavior change.
- Acceptance: both tests pass; every S36 direct dependency has its exact role;
  target-specific `getrandom/wasm_js` remains confined to wasm dev resolution;
  `bytemuck/derive` is external-only; `vello_shaders` is WGSL-only; all three
  Vello provenance/license artifacts and imported-file hashes remain exact;
  the native feature matrix, presented smoke, wasm target build, exact Rust-1.97
  checks, rustdoc, C13 inventories/routes/stats, production-path guards, and
  owned-Rust safety scan are green.
- Commands:
  `CARGO_NET_OFFLINE=true cargo test -p surgeist-render tests::c14_dependency_feature_and_provenance_contract_is_final -- --exact`;
  `CARGO_NET_OFFLINE=true cargo test -p surgeist-render tests::c14_final_quality_contract_matches_published_gpu_architecture -- --exact`;
  run every command in Section 5, including dependency trees and the canonical
  unsafe scan.
- Depends on: T02.
- Intended commit: `test(platform): close final compatibility guards`.

## 5 Verification And Completion

Implementation and final commands use only already-installed artifacts with
`CARGO_NET_OFFLINE=true`. `C14-CHECK` is:

```sh
set -euo pipefail
CARGO_NET_OFFLINE=true cargo fmt --check
CARGO_NET_OFFLINE=true cargo check -p surgeist-render
CARGO_NET_OFFLINE=true cargo test -p surgeist-render
CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets -- -F unsafe-code -D warnings
test -z "$(git ls-files -- Cargo.lock)"
```

The complete C14 final command set is:

```sh
set -euo pipefail
cycle_base=2bd2d36638b8a3436b69cb99a905d37b36886d16
cycle_head=$(git rev-parse HEAD)
CARGO_NET_OFFLINE=true cargo test -p surgeist-render tests::c14_public_docs_describe_gpu_routes_precision_failures_and_host_boundaries -- --exact
CARGO_NET_OFFLINE=true cargo test -p surgeist-render tests::c14_changed_public_items_have_semantic_documentation -- --exact
CARGO_NET_OFFLINE=true cargo test -p surgeist-render tests::render_window_smoke_source_covers_direct_and_graph_routes -- --exact
CARGO_NET_OFFLINE=true cargo test -p surgeist-render tests::c14_dependency_feature_and_provenance_contract_is_final -- --exact
CARGO_NET_OFFLINE=true cargo test -p surgeist-render tests::c14_final_quality_contract_matches_published_gpu_architecture -- --exact
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
owned_rs=("${(@f)$(
  {
    git ls-files -- '*.rs'
    git ls-files --others --exclude-standard -- '*.rs'
  } | sort -u
)}")
test "${#owned_rs[@]}" -gt 0
if rg -n --pcre2 '#\s*\[\s*(?:unsafe\s*\(|no_mangle\b|export_name\b)|\bunsafe\s*(?:\{|fn\b|trait\b|impl\b|extern\b)|\bstatic\s+mut\b|\bextern\s*(?:"[^"]*")?\s*\{' "${owned_rs[@]}"; then
  exit 1
else
  test "$?" -eq 1
fi
git diff --check "$cycle_base..$cycle_head"
test "$(git rev-parse HEAD)" = "$cycle_head"
test -z "$(git status --porcelain)"
```

The final gate also runs every applicable S32-S35 and C13 integrated guard plus
the dependency-role/source and provenance comparisons. The block above is a
`zsh` command set because the repository's configured shell supplies the
newline-safe `(@f)` array expansion used for the explicit owned-Rust manifest.
Structural advisories do not become Boolean failures.

Completion requires all three tasks to have fresh `CLEAN` task reviews and
coordinator acceptance, a separate status-only `complete` commit, the complete
command set on the exact completed head, a fresh `CLEAN` holistic review, a
post-review repeat of the complete command set, and canonical publication and
remote readback. The handoff records the immutable candidate, additive C13 API
surface, exact feature/MSRV/wasm/native evidence, and the required C15
sprawl-review entry state. Browser execution remains owed by root, as do the
later facade/API-artifact/gitlink changes after final I03 publication. Missing
installed tooling, graphical-host capability, credentials, or stable remote
history uses the canonical blocker contract; no unavailable requirement is
counted as green.
