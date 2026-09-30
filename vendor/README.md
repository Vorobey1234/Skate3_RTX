# Local Bevy patch

## Bindless material resource reuse

`bevy_pbr/src/material_bind_groups.rs` chooses a fitting slab requiring the fewest
new resource slots instead of first-fit allocation. It prefers already resident
textures, then fuller slabs, while keeping existing allocation/refcount/free logic
and capacity limits. This adds work during material allocation, not each frame.
The game's explicit Vulkan shader probe tests reuse and retirement using real GPU
texture identities. See `docs/cpu-followup-optimizations.md` for scope and limits.

## Conservative occlusion depth pyramid

`bevy_core_pipeline` is vendored from crates.io 0.18.1 with its original licenses.
Its `src/experimental/mip_generation/{mod.rs,downsample_depth.wgsl}` backports
[Bevy #22603](https://github.com/bevyengine/bevy/pull/22603), commit
`43f4552308a29f4130c3ad71ea65d48f77daea6e`. The upstream file paths changed
between releases; the algorithm is the upstream fix. Power-of-two pyramid
levels and conservative source sampling prevent false occlusion at arbitrary
viewport sizes, including scaled targets and MSAA. The associated meshlet
texture-dimension fix is included in `bevy_pbr/src/meshlet/meshlet_cull_shared.wgsl`.
The game uses ordinary meshes, not meshlets.

The late mesh preprocessing pass in 0.18.1 also binds its indirect dispatch
buffer as writable storage, which fails wgpu validation when occlusion is on.
`bevy_pbr/src/render/{gpu_preprocess.rs,mesh_preprocess.wgsl}` now declares that
binding read-only in the late pass. Its counters are plain `u32` in that shader
variant, preserving the same buffer layout; the early producer retains atomic
counters and writable storage. No dispatch count or culling decision changes.

Unused depth-pyramid mip slots previously shared one dummy storage texture.
When the pyramid was short enough that the ReadWrite SPD intermediate (`mips[5]`)
was also a dummy, wgpu rejected the dispatch: `STORAGE_WRITE_ONLY` and
`STORAGE_READ_WRITE` on the same texture in one usage scope. Padding now uses one
dummy texture per binding slot, dummies are 1×1 (not 0×0), zero-sized viewports
clamp safely, and pyramid side lengths are at least 32 so that intermediate mip
exists as a real subresource.

`bevy_pbr` is the unmodified crates.io 0.18.1 source except for the changes
described below. Its original MIT and Apache licenses are included. Cargo selects
it through the workspace `[patch.crates-io]` entry; the Cargo registry is untouched.

## Cached lightmap bindings

Bevy 0.18.1 rebuilds a bind group for every allocated lightmap slab and every
render phase on each frame, even when all bound GPU resources are unchanged.
On San Vanelona, mesh binding preparation measured roughly 10 ms per frame.

Local changes are restricted to:

- `src/lightmap/mod.rs`: each slab retains at most eight bindings. Lookup compares
  the actual wgpu buffer identity, binding offset and size, and layout identity.
  Texture insertion/removal clears the cache. FIFO eviction bounds retained old
  buffers when phases disappear or instance buffers resize.
- `src/render/mesh_bindings.rs`: reuse or create the binding through that cache.
  The lightmap slab parameter is now mutable.
- `src/lightmap/cache_tests.rs`: explicit GPU regression test.
- `src/render/mesh.rs`: retain complete lookup tables for both alternating
  instance buffers per phase. Keys include buffer identity, offset, size, layout,
  and the lightmap binding revision. A slab creation, texture upload, or removal
  changes that revision. Tables are shared by `Arc`, so a cache hit avoids
  rebuilding and cloning thousands of individual entries. Keep only two tables
  per phase, and discard caches for phases that disappear.
- `src/render/mesh.rs`: cache the IDs of meshes with morph targets when
  `RenderAssets<RenderMesh>` changes. Each phase still rebuilds its dynamic morph
  bindings from the current GPU resources, but skips scanning static city meshes.

Writing new instance data into an existing buffer does not require rebuilding a
bind group. Changing the buffer, bound range, layout, or texture does. The patch
does not skip extraction, uploads, animation, or visibility updates. It retains
the existing single-texture and binding-array rendering paths and shaders.

Run the Vulkan test explicitly:

```powershell
cargo test -p bevy_pbr --lib cache_tracks_binding_identity_and_texture_replacement --locked -- --ignored
```

It covers cloned versus replaced buffers, changed binding ranges/layouts, writes
to an existing buffer, texture insertion/removal, bounded eviction, and actual
bind-group construction/reuse, alternating-buffer table reuse, table revisions,
and morph-target additions/replacements/removals. Regular game tests and a city screenshot/benchmark
also exercise the patched renderer.

Upstream references reviewed September 6, 2026:

- [Bevy issue 23595](https://github.com/bevyengine/bevy/issues/23595) reports
  increasing `prepare_mesh_bind_groups` cost. That report points at mesh scanning;
  local measurements and source inspection identified lightmap binding recreation
  as the opportunity addressed here.
- [Bevy 0.19 rendering improvements](https://bevy.org/news/bevy-0-19/#render-big-scenes-faster)
  explain further GPU batching and CPU overhead reductions. An engine migration
  is separate from this small patch; 0.19.1 source still rebuilds lightmap groups.

When upgrading Bevy, reassess this patch and rerun the GPU test, stationary and
moving-camera benchmarks, and visual comparison. Do not assume an unchanged
private renderer implementation across engine versions.

## Solari sky and light selection (RTX fork)

`bevy_solari` is the unmodified crates.io 0.18.1 source, with its original MIT
and Apache licenses, except for these changes:

- `src/scene/raytracing_scene_bindings.wgsl`: adds `sky_radiance()`. Upstream
  treats a ray that leaves the scene as black, which leaves outdoor shadows lit
  only by bounce light. The sky is a fixed fraction of each directional light's
  illuminance, blue when the light is high and warm near the horizon, plus a
  small night floor, so it follows the game's time of day.
- `src/realtime/restir_gi.wgsl`: an escaped diffuse GI ray becomes a sky sample
  placed `SKY_SAMPLE_DISTANCE` along the ray and facing back, so temporal and
  spatial reuse still validate it.
- `src/realtime/world_cache_update.wgsl`: an escaped cache ray adds the sky. The
  cache ray stops at 50 m, so a second ray confirms the escape first.
- `src/realtime/specular_gi.wgsl`: an escaped glossy path adds the sky.
- `src/scene/binder.rs`: skips directional lights with zero illuminance. Light
  selection is uniform over sources, so the game's receiver-only shadow light
  would otherwise waste half of the direct-light samples.
- `src/scene/blas.rs`: when a mesh is re-extracted, drops its old entries from
  the compaction queue. A mesh changed before its BLAS finished compacting
  (skinned proxies change every frame) otherwise left two entries for one BLAS,
  and the second `prepare_compaction_async` call is a fatal wgpu error.
- `src/realtime/mod.rs`: orders the lighting node before `StartMainPass`, so
  the atmosphere and forward draws composite over lit pixels deterministically.
- `src/realtime/resolve_dlss_rr_textures.wgsl`: sky pixels get unit diffuse
  albedo, zero specular albedo and an up normal instead of zeros. Ray
  Reconstruction still blacks out empty-depth pixels, so the game also moves the
  sky pass after it (below).

## Atmosphere after DLSS Ray Reconstruction (RTX fork)

- `bevy_pbr/src/atmosphere/mod.rs`: re-exports `AtmosphereNode`, so the game can
  move the sky render pass from inside the main pass to after DLSS RR.
- `bevy_pbr/src/atmosphere/render_sky.wgsl`: reads depth by UV rather than by
  pixel, because after DLSS upscaling the colour target is larger than the depth
  buffer. At equal sizes this is the same texel.
