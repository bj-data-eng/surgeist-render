# Vello 0.9.0 provenance

`surgeist-render` derives its private raster-engine modules from the pinned
Vello main crate described below. This notice records the source state before
local adaptation; it is not a copy of Vello's public crate or API.

## Pinned package

- Package: `vello` 0.9.0.
- Cargo package checksum: `261359dbef879f8110ef7e1c442246c838d33d3d91cb05e0ea9288d432760c9f`.
- Source URL: <https://github.com/linebender/vello>.
- License expression: `Apache-2.0 OR MIT`.

## Imported upstream source files

| Local derived file | Upstream source | Pre-adaptation SHA-256 | Material adaptations |
| --- | --- | --- | --- |
| `src/vello_engine/scene.rs` | `vello-0.9.0/src/scene.rs` | `7c225e73f56629b1b85e8e5cd296428176ec6e59a0813975e2d4123aaddd1718` | Keeps private Vello-compatible scene lowering for the crate's commands and moves selected-glyph work behind a fallible preflight boundary; removes the public `Scene` surface and upstream font `unwrap` paths. |
| `src/vello_engine/glyph.rs` | `vello-0.9.0/src/scene.rs` | `7c225e73f56629b1b85e8e5cd296428176ec6e59a0813975e2d4123aaddd1718` | Extracts selected-glyph font, outline, COLR, bitmap, and PNG validation from scene lowering; converts malformed or unsupported inputs to typed diagnostics before external encoding and never logs-and-omits a glyph. |
| `src/vello_engine/raster.rs` | `vello-0.9.0/src/render.rs` | `f75a73fae27085c870273b6e670f355455eea61f1d1dde9b102ab9ed2528e7ed` | Retains the coarse/fine recording schedule over `vello_encoding` while producing a WGPU-free prepared pass with symbolic target and resource intents instead of an upstream renderer graph. |
| `src/vello_engine/recording.rs` | `vello-0.9.0/src/recording.rs` | `3c760a7c7610274443efe06c2e9a37eb71471b14a6635d9f65ce92b39de98b3c` | Recasts upstream recording proxies as private handles, resource intents, and dispatch records for the fixed raster schedule; it owns no WGPU object, polling, or submission path. |
| `src/vello_engine/shaders.rs` | `vello-0.9.0/src/shaders.rs` | `c1392afa0ce8d33873e43a26ba79e881adb0a53e2ed92a90201fac5592a0058e` | Retains shader selection and WGSL binding metadata from external WGSL-only `vello_shaders`; creates modules and pipelines through checked WGPU error scopes rather than the trusted upstream creation path. |
| `src/vello_engine/encoder.rs` | `vello-0.9.0/src/wgpu_engine.rs` | `d2bbb8151f27d7fd4ff82abaa1438e05cb45468dab36034f48e54eefba183e7c` | Realizes symbolic recordings, uploads, bind groups, and compute passes into a transaction-borrowed encoder; returns a lease and cannot submit command buffers or own a renderer. |
| `src/vello_engine/resources.rs` | `vello-0.9.0/src/wgpu_engine.rs` | `d2bbb8151f27d7fd4ff82abaa1438e05cb45468dab36034f48e54eefba183e7c` | Replaces the upstream engine's resource ownership with checked per-device allocation and explicit lease commit, abort, and atlas-recovery state under Surgeist transaction control. |

Every derived file listed above retains its upstream copyright and SPDX header.
`src/vello_engine/mod.rs` is Surgeist-owned composition that wires the derived
private phases together, so it deliberately carries no upstream source header.

## Omitted upstream main-crate sources

| Upstream source | Rationale |
| --- | --- |
| `vello-0.9.0/src/lib.rs` | Defines Vello's public facade, public renderer/options/errors, reexports, and feature wiring; Surgeist retains its own crate front door and private engine boundary. |
| `vello-0.9.0/src/util.rs` | Owns Vello device and surface utilities, including a blocking device-poll helper; Surgeist owns WGPU lifecycle and does not permit blocking, map, or poll execution in the internal engine. |
| `vello-0.9.0/src/debug.rs` | Defines optional debug-layer control whose documented validation route requires CPU readback; it is outside the checked production raster path. |
| `vello-0.9.0/src/debug/renderer.rs` | Implements debug visualization, debug downloads, and mapped-buffer CPU validation; it is excluded with Vello debug/readback support. |
| `vello-0.9.0/src/debug/validate.rs` | Implements CPU line-soup validation and logging for the optional debug layer; it is not part of production rendering. |

The imported upstream files and these omissions account for every Rust source
file under the pinned package's `src/` directory.

## License artifacts

The tracked [Apache-2.0 license](LICENSES/Vello-0.9.0-APACHE-2.0.txt) and
[MIT license](LICENSES/Vello-0.9.0-MIT.txt) preserve the pinned package's
verbatim dual-license texts.
