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

The T2 import omits Vello's public `Scene` API, non-text draw helpers,
estimation, append/reset helpers, direct renderer/device integration, and all
CPU/debug/map/poll/submission paths. It introduces no WGPU resource or
submission authority. The private production boundary remains unused until the
later T7 cutover.

## License artifacts

`LICENSES/Vello-0.9.0-APACHE-2.0.txt` and
`LICENSES/Vello-0.9.0-MIT.txt` preserve the verbatim dual-license texts from
the pinned package. They are provenance/license artifacts for the planned
internalization boundary; they do not assert that any Vello source file has
already been imported at T1.
