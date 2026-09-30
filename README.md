<p align="center">
  <img src="docs/images/skating-crab.png" alt="Rust crab riding a skateboard" width="480">
</p>

# Skate 3 Rust Engine

A Rust and Bevy skating project built from Skate 3 reverse-engineering research.
Includes skating, tricks, grinds, offboard movement, difficulty settings and
`.skate` map support. Gameplay parity is still a work in progress.

## RTX fork

This fork replaces the lighting of the retail renderer with real-time path
tracing, built on Bevy Solari and NVIDIA DLSS:

- **G-buffer instead of lightmaps.** Opaque world geometry writes albedo,
  shading normal, roughness and emission to a deferred G-buffer
  (`crates/skate-game/src/rtx_world_gbuffer.wgsl`). Baked lightmaps, shadow maps
  and the fixed sun term are no longer used for opaque surfaces.
- **Ray-traced direct light.** ReSTIR DI samples the scene sun, emissive world
  materials (signs and lamps) and the map's point and spot lights. Shadows are
  traced rays.
- **Ray-traced indirect light.** ReSTIR GI with a world radiance cache handles
  multi-bounce diffuse light. Glossy reflections are traced paths. Rays that
  leave the scene see a clear sky, added by a small patch in `vendor/bevy_solari`.
- **DLSS Ray Reconstruction.** DLSS RR denoises the path-traced signal and
  upscales it to the output resolution in a single pass.
- **Time of day.** Escape > Graphics > Day & night sets the hour and cycle
  speed. A sun and moon follow it, the sky is Bevy's physical atmosphere (blue
  noon, orange sunset, moonlit night), and exposure adapts. Output uses Bevy's
  filmic tonemapper instead of the retail tone curve.
- **Derived PBR.** Materials without authored normal or specular maps get bump
  from the diffuse texture's luminance and roughness from its local detail, so
  flat, smooth surfaces pick up glossy traced reflections.
- **DLSS settings.** Escape > Graphics > DLSS: Auto, DLAA, Quality, Balanced,
  Performance, Ultra Performance, or Off (undenoised path tracing).
- **Characters** keep their authored PBR materials. Each frame they are skinned
  on the CPU into a ray-tracing proxy, so they cast traced shadows and appear in
  reflections.

The ray-traced scene is in `crates/skate-game/src/rtx.rs`. GPUs without hardware
ray queries fall back to the original raster renderer. Set `SKATE_RTX=0` to
force that fallback.

**Frame generation is not included.** DLSS Frame Generation requires NVIDIA
Streamline to take over the swapchain, and the Bevy/wgpu DLSS integration
(`dlss_wgpu`) supports only Super Resolution and Ray Reconstruction. On RTX 40
and 50 series GPUs, driver-level frame generation (NVIDIA Smooth Motion in the
NVIDIA App) can be used instead, where the driver supports it for this game.

Known limits: cutout and alpha-blended materials (foliage, fences, water) are
not in the ray-traced scene, so they cast no traced shadows. Blended surfaces
still use their baked lighting. Secondary hits use each material's average
albedo, not its texture.

### RTX build requirements

In addition to the requirements under **Build**: an NVIDIA RTX GPU, the
[LunarG Vulkan SDK](https://vulkan.lunarg.com/) (`VULKAN_SDK` set) and LLVM
(for `libclang`). `BUILD.bat` downloads the
[NVIDIA DLSS SDK v310.4.0](https://github.com/NVIDIA/DLSS/tree/v310.4.0) (the version `dlss_wgpu` 2.0 binds) into `.local/DLSS` unless
`DLSS_SDK` is already set, and copies `nvngx_dlss.dll` and `nvngx_dlssd.dll`
beside the executable. Their use is subject to the NVIDIA DLSS SDK license.

## Play

[Download Experimental](https://github.com/SK8-ENGINE/skate-3-rust-engine/releases/tag/experimental).
Successful `main` builds replace this prerelease. Choose **Latest** in Updates
for experimental updates; **Stable** is the default.

Extract the Windows release ZIP and run `skate3rust.exe`. Select your Skate 3
Xbox 360 ISO, or select `default.xex` in an extracted game folder. Keep its
`data` folder alongside it. Setup prepares the skater, animations and all disc maps, then
starts University. The original scoring and session-marker HUD assets are also
exported automatically during setup. No Blender, Python or Rust installation is needed.
ISO extraction needs internet access. The first conversion can take a while.

Use an XInput controller to play. Escape opens graphics, difficulty and map
settings. Maps can be switched without restarting the game.

**Skate 3 assets are not included.** Your converted files stay in
the `data` folder beside your executable. Each freshly unpacked copy runs its
own setup; it does not adopt another installation. In-place updates refresh
only changed asset groups.

## Build

Requires Windows, Rust with the MSVC toolchain, and LLVM installed in its default
location. Run `BUILD.bat` to build, then `PLAY.bat` to launch the test world.
`PLAY.bat` opens your saved map (University by default); use the in-game menu to switch maps, or drag a `.skate` file onto `PLAY.bat`. An XInput controller is required for gameplay;
Escape opens difficulty and graphics settings.

Development builds use a prepared asset set in `assets/private/` or the
installed asset directory. `scripts/Build-Release.ps1` builds the portable Windows
package and requires Python 3.13. GitHub Actions builds `main` automatically;
numbered releases are published separately.

Custom animations and climbing support remain available, but no custom clips
are shipped. The included format-demo map is original procedural content.

Implementation notes are in [`docs/`](docs/). Patched Bevy dependencies and
their licenses are in [`vendor/`](vendor/). This is an unofficial project,
not affiliated with EA.

## Advanced diagnostics

Windows builds support opt-in [performance timeline capture](docs/performance-tracing.md)
through the `--trace` CLI option, including optional GPU pass diagnostics.
