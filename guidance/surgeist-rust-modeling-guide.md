# Surgeist Rust Modeling Guide

Use this guide when designing or reviewing Surgeist Rust code. Surgeist should
prefer semantic, typed models over broad runtime interpretation. A type is not
just storage; it names ownership, phase, invariants, and allowed behavior.

There is a cheap way to write Rust: put many meanings into one enum or struct,
match everywhere, and rely on comments to explain which combinations are valid.
Surgeist should choose the better path: make invalid states hard to express,
make transitions explicit, and put interpretation behind intentional typed
boundaries.

The goal is not more types for their own sake. A good model makes the next
feature easier to explain, test, and integrate.

## Core Standard

Before adding or extending an API, ask:

- What semantic distinction is this code trying to represent?
- Which crate owns that distinction?
- Is this value authored, normalized, resolved, runtime, backend, or test data?
- Can invalid combinations be constructed by ordinary callers?
- Are callers repeatedly matching, rejecting, or reinterpreting the same states?
- Can the behavior be tested in the crate that owns it?

If the answer depends on convention, comments, or scattered matches, the model
is probably too loose.

## Broad Enums Are A Smell

Enums are excellent for closed choices in one semantic domain. They are a poor
substitute for modeling unrelated phases, value kinds, commands, or backend
states.

An enum is suspect when:

- variants belong to different phases of the pipeline
- variants carry unrelated data shapes
- many callers reject most variants immediately
- validity depends on another field, flag, feature, or call order
- adding one feature requires editing matches across the crate
- the enum becomes a transport bag between modules
- some variants are only meaningful to tests, generators, or one backend
- documentation has to explain which variants are legal in which context

Before adding a variant, ask whether the enum is still one closed choice. If it
has become several concepts sharing a container, split it.

Prefer:

- newtypes for units, identifiers, handles, and semantic names
- small structs with private fields for values with invariants
- phase-specific types for authored, normalized, resolved, runtime, and backend
  data
- smart constructors or `TryFrom` implementations for validation
- typed conversion functions at explicit boundaries
- helper methods that centralize interpretation when an enum is still right

## Model Phases Explicitly

Do not use one type for every phase of a value's life unless the phases really
share the same invariants.

Common Surgeist phases:

- authored input, such as CSS syntax, user configuration, descriptors, or DSL
  declarations
- normalized crate input, such as layout-ready values, resolved style data,
  retained commands, render-ready draw data, or native command intents
- algorithm state, such as layout tracks, text runs, render passes, projection
  state, or scheduling state
- runtime state, such as retained identities, window registries, host handles,
  caches, revisions, or pending effects
- backend state, such as native windows, renderer resources, font engines,
  platform adapters, or accessibility bridges
- test and fixture state, such as fake hosts, parity XML, snapshots, generated
  reports, or oracle data

Strong models usually have different types at phase boundaries. For example,
authored CSS values should not be indistinguishable from layout-ready values,
and native backend handles should not be ordinary app-facing IDs.

## Use Newtypes Aggressively

Use newtypes when a primitive carries meaning that the compiler should protect.

Good candidates:

- pixels, points, CSS pixels, device pixels, percentages, ratios, and scales
- node IDs, window IDs, surface IDs, resource IDs, generation IDs, and revisions
- axis, side, lane, index, run, span, range, and offset values
- names, selectors, roles, feature flags, and backend capability keys

Newtypes are cheap, but they change the shape of the program. A `WindowId` is
not a `u64`, a `CssPx` is not a `f32`, and a `Revision` is not an arbitrary
counter. The type should make accidental mixing impossible.

## Keep Invariants At Construction

If a value has invariants, ordinary callers should not be able to build an
invalid value with a struct literal.

Prefer:

- private fields plus constructors
- `try_new` or `TryFrom` for fallible validation
- `NonZero*`, range-specific wrappers, or domain-specific bounded values
- constructors that normalize once rather than forcing every caller to sanitize
- methods that preserve invariants instead of exposing mutable internals

Avoid public structs whose validity depends on comments like "must not be empty"
or "only valid after layout." Encode that fact in the type or constructor.

## Use Typestates For Real State Machines

When call order matters, model the state instead of documenting it.

Use typestate-style modeling when a value moves through meaningful phases such
as:

- unvalidated to validated
- open to closed
- pending to committed
- authored to normalized to resolved
- allocated to bound to released

This does not mean every flag deserves a generic state parameter. Use typestates
when they remove real invalid call sequences from the public or crate-internal
API.

## Make Conversion Boundaries Narrow

Representation changes should happen at named chokepoints:

- parser to style model
- style model to layout input
- layout output to render input
- retained tree to app or accessibility projection
- app command to host command
- backend event to framework event
- generated fixture to test runner input

Healthy chokepoints are explicit, tested, and narrow. Risky chokepoints are
implicit, duplicated, or bypassed by convenience paths.

Prefer conversion APIs that make ownership and failure visible:

- `From` only for infallible, unsurprising conversions
- `TryFrom` for validation or lossy rejection
- named methods for contextual conversions, such as `resolve_against(...)`
- adapter types when conversion requires backend, feature, or fixture context

## Keep Symbolic Values Symbolic

Do not collapse symbolic values too early. If a value needs context to become
concrete, preserve enough structure for the owning layer to resolve it.

Examples:

- CSS calc values should stay symbolic until the layer with the right basis can
  resolve them.
- Font fallback should stay symbolic until text shaping has font and locale
  context.
- Renderer resources should remain handles until the backend owns the storage.
- Window capabilities should be queried or reported rather than guessed through
  scattered branch checks.

A resolver is useful when it lets the model remain honest about missing context.

## Prefer Semantic Errors

String errors are acceptable at process edges, but they should not be the
primary contract inside Surgeist crates.

Prefer error types that name:

- the rejected value or command
- the violated invariant
- the phase or boundary where validation failed
- the backend or capability constraint, when relevant
- whether the error is caller misuse, unsupported input, or backend failure

Typed errors make tests sharper and keep public APIs from becoming prose-only
contracts.

## Use Traits Deliberately

Traits are behavior contracts, not a way to avoid choosing a model.

Use a trait when multiple implementations share real behavior and consumers
should be generic over that behavior. Keep traits small, object-safe only when
trait objects are intended, and sealed when downstream implementations would
make the contract impossible to evolve.

Prefer concrete types when there is one owner and one valid model. Prefer enums
when the implementation set is closed and callers must distinguish variants.
Prefer traits when behavior is open and the contract is stable enough to expose.

## Normalize Commands Before Applying Them

Commands that cross module, crate, app, host, or backend boundaries should not
be interpreted ad hoc at every call site.

Consider a normalized command, patch, transaction, or change set when code must
agree on:

- validation
- deduplication
- expansion into lower-level operations
- state mutation order
- emitted events
- cache invalidation
- revision advancement
- backend capability handling

The normalization type should live in the crate that owns the semantics, not in
whichever caller first needed a shortcut.

## Keep Fake And Real Paths Aligned

Test fakes should exercise the same semantics as production paths. They may
store less data or skip real backend effects, but they should not invent a
different command language or lifecycle model.

Compare fake and real implementations:

- Do they apply the same normalized commands?
- Do they validate names, capabilities, handles, and lifecycle transitions the
  same way?
- Do they update equivalent observable state?
- Do they emit equivalent events?
- Do failed operations leave equivalent state?

If drift is likely, introduce a shared transition helper, patch type, event
sink, or capability report.

## Capability Checks Are Contracts

Backend limitations should not be scattered branch checks or string messages.

Use explicit capability reports or query APIs for:

- native window roles, modality, fullscreen, cursor grab, accessibility, or IME
- renderer support for filters, masks, blend modes, image formats, or surfaces
- text backend support for scripts, shaping features, fonts, locales, or
  selection geometry
- dialog host support for modality, ownership, focus, and platform conventions

Capabilities should be testable independently of the backend path that consumes
them.

## Public APIs Need Front Doors

Each crate should expose intentional front-door APIs from `lib.rs` and keep
internals private by default.

Avoid:

- sibling crates reaching through private module paths
- broad common crates that become dependency sinks
- helper modules whose names hide ownership
- public fields that make future invariants breaking changes
- feature flags that expose half-modeled states

If a public API requires a paragraph of caveats, the type boundary probably
needs more work.

## Crate Prompts

These are prompts, not prescriptions.

- `surgeist-css`: authored syntax ownership, parse errors, source spans,
  lowering into style values, unsupported CSS reporting.
- `surgeist-style`: authored versus resolved values, property metadata,
  validation, invalidation, adapter contracts into layout/text/retained.
- `surgeist-layout`: symbolic values, basis-dependent sizing, axis mapping,
  track and fragment invariants, oracle/parity fixture inputs.
- `surgeist-render`: backend capabilities, draw command normalization, resource
  handles, scene encoding, surface lifecycle.
- `surgeist-retained`: identity ownership, patch atomicity, projection
  transactions, revision semantics, event routing.
- `surgeist-text`: font and shaping capabilities, source identity, layout cache
  keys, selection/cursor geometry, render projection.
- `surgeist-window`: host capabilities, command normalization, lifecycle state
  transitions, fake-vs-real host drift, native event translation.
- `surgeist-dialog`: dialog role semantics, modality, cross-window coordination,
  host capability interaction.
- `surgeist-shape`: geometry invariants, path construction, stroke/dash
  normalization, render-facing primitive contracts.
- `surgeist-test`: fixture ownership, integration harness contracts, coverage
  reports, generated artifacts, cross-crate verification boundaries.

## Final Check

Before accepting a model, state why it reduces future coordination rather than
merely moving complexity. The correct Rust model should make the next feature
easier to explain, test, and integrate.
