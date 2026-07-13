# Vello provenance notice

## Pinned package

- Package: `vello` 0.9.0.
- Cargo package checksum: `261359dbef879f8110ef7e1c442246c838d33d3d91cb05e0ea9288d432760c9f`.
- Source URL: <https://github.com/linebender/vello>.
- License expression: `Apache-2.0 OR MIT`.

## T1 status

- Imported files: none. T1 characterizes the existing external Vello route and
  does not copy Vello source into this repository.
- Current adaptations: none.
- Later import boundary: a later C03 task may add rows for private
  `src/vello_engine/` imports. Each row must name the upstream file, its
  pre-adaptation SHA-256, and the exact adaptation or omission.

## T2 private scene/glyph lowering

| Local files | Upstream source | Pre-adaptation SHA-256 | Material adaptation |
| --- | --- | --- | --- |
| `src/vello_engine/scene.rs`, `src/vello_engine/glyph.rs` | `vello-0.9.0/src/scene.rs` | `7c225e73f56629b1b85e8e5cd296428176ec6e59a0813975e2d4123aaddd1718` | Retains the private glyph-run lowering boundary over `vello_encoding 0.9.0`; splits selected-glyph preflight into `glyph.rs`; replaces both `FontRef::from_index(...).unwrap()` sites and font-derived parse/length assumptions with fallible Skrifa validation and typed diagnostics; rejects omitted glyph paths rather than logging/continuing. |

T2 intentionally omits COLR and bitmap text lowering (BGRA, PNG, and packed-mask
images). After their selected data preflights successfully, `VelloScene` stops at
the explicit `RenderFailed` append boundary; it does not silently omit or fall
back from those paths.

The T2 import omits Vello's public `Scene` API, non-text draw helpers,
estimation, append/reset helpers, direct renderer/device integration, and all
CPU/debug/map/poll/submission paths. It introduces no WGPU resource or
submission authority. The private production boundary remains unused until the
later T7 cutover.

## T4 checked WGPU encoder and resource leases

| Local files | Upstream source | Pre-adaptation SHA-256 | Material adaptation |
| --- | --- | --- | --- |
| `src/vello_engine/shaders.rs` | `vello-0.9.0/src/shaders.rs` | `c1392afa0ce8d33873e43a26ba79e881adb0a53e2ed92a90201fac5592a0058e` | Retains the pinned shader-selection schedule and WGSL binding metadata, but creates every module and compute pipeline through checked WGPU scopes. It uses the external WGSL-only `vello_shaders 0.9.0` metadata rather than copying generated shader sources. |
| `src/vello_engine/encoder.rs`, `src/vello_engine/resources.rs` | `vello-0.9.0/src/wgpu_engine.rs` | `d2bbb8151f27d7fd4ff82abaa1438e05cb45468dab36034f48e54eefba183e7c` | Retains only symbolic-recording realization, upload, bind-group, and compute-pass concepts. It accepts transaction-borrowed device, queue, command encoder, and target state; returns an explicit pending resource lease with consuming commit/abort transitions; and maps malformed private recordings to `RenderFailed`. |

The T4 import retains the upstream SPDX and copyright headers in each derived
source file. It omits the public Vello engine and renderer APIs, resource pools,
pipeline caches, parallel initialization, CPU execution, hot reload, image
overrides/registration, debug layers, profiler integration, downloads, mapping,
polling, surface helpers, and submission ownership. Checked shader construction
is the only shader path; command encoding remains owned by the caller's active
transaction and T4 does not route a lease into transaction publication.

## License artifacts

`LICENSES/Vello-0.9.0-APACHE-2.0.txt` and
`LICENSES/Vello-0.9.0-MIT.txt` preserve the verbatim dual-license texts from
the pinned package. They are provenance/license artifacts for the planned
internalization boundary; they do not assert that any Vello source file has
already been imported at T1.
