//! Path-traced lighting for the RTX build.
//!
//! Opaque surfaces are written to a deferred G-buffer, then Bevy Solari computes
//! all of their light with ray queries: ReSTIR DI for direct light from the sun
//! and emissive surfaces, ReSTIR GI plus a world radiance cache for multi-bounce
//! indirect light, and traced glossy reflections. DLSS Ray Reconstruction
//! denoises that signal and upscales it to the output resolution. Retail baked
//! lightmaps and shadow maps no longer light opaque geometry.
//!
//! Solari traces a separate scene: `RaytracingMesh3d` entities with a
//! `StandardMaterial`. Rays only need geometry, an average albedo and emission
//! at secondary hits (the primary hit comes from the G-buffer), so this module
//! builds those proxies for the merged retail world, map lights, loose
//! `StandardMaterial` meshes, and CPU-skinned characters.
//!
//! The sun and moon follow the menu's time of day. The sky is Bevy's physical
//! atmosphere, lit by the same lights, and exposure follows the sun so nights
//! stay readable. Output uses Bevy's filmic tonemapper: the retail tone curve
//! was built for baked retail values, not physical light.
//!
//! GPUs without hardware ray queries keep the original raster renderer, as does
//! `SKATE_RTX=0`.
use std::sync::atomic::{AtomicBool, Ordering};

use bevy::{
    anti_alias::dlss::{Dlss, DlssPerfQualityMode, DlssProjectId, DlssRayReconstructionFeature, DlssRayReconstructionSupported},
    asset::{LoadState, RenderAssetUsages},
    camera::{CameraMainTextureUsages, Exposure},
    core_pipeline::tonemapping::Tonemapping,
    mesh::{Indices, PrimitiveTopology, VertexAttributeValues, skinning::{SkinnedMesh, SkinnedMeshInverseBindposes}},
    pbr::{Atmosphere, DefaultOpaqueRendererMethod, ScatteringMedium},
    platform::collections::HashMap,
    prelude::*,
    render::{render_resource::TextureUsages, renderer::RenderDevice, view::Hdr},
    solari::{SolariPlugins, prelude::{RaytracingMesh3d, SolariLighting}},
};
use skate_data::skate_map::SkateMap;

use crate::{
    map_render::{AssetSink, SceneCommands},
    retail_render::{MaterialTable, RenderClass},
};

/// NGX identifies the application by this ID. Any stable UUID works for
/// development builds; a shipped build should use one issued by NVIDIA.
const DLSS_PROJECT_ID: bevy::asset::uuid::Uuid =
    bevy::asset::uuid::uuid!("5b3d7c1e-8a2f-4e61-9c47-3f0d2b8e6a15");

/// Solari's emissive light sources hold at most 65535 triangles per mesh.
const MAX_PROXY_TRIANGLES: usize = 65_535;

/// See `rtx_world_gbuffer.wgsl`: unlit retail families become emitters.
const EMISSIVE_NITS: f32 = 1000.0;

/// Noon sun, as in the retail scene. Real sunlight is about ten times this;
/// exposure is calibrated to the lower value, and the sky scales with it.
const SUN_LUX: f32 = 11_000.0;
const MOON_LUX: f32 = 30.0;
/// Camera EV100 at full day and at night.
const DAY_EV100: f32 = 9.7;
const NIGHT_EV100: f32 = 4.5;
/// Ambient cd/m^2 per lux of sun and moon; see `day_cycle`.
const AMBIENT_PER_LUX: f32 = 0.04;

static ACTIVE: AtomicBool = AtomicBool::new(false);

/// Whether this run renders with path tracing. Settled in `RtxPlugin::finish`,
/// before any map is prepared.
pub(crate) fn active() -> bool {
    ACTIVE.load(Ordering::Relaxed)
}

/// Must run before `DefaultPlugins`: its DLSS init plugin reads the project ID
/// while registering Vulkan extensions.
pub(crate) fn insert_project_id(app: &mut App) {
    app.insert_resource(DlssProjectId(DLSS_PROJECT_ID));
}

pub(crate) struct RtxPlugin;

impl Plugin for RtxPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(SolariPlugins)
            .init_resource::<ConvertedMeshes>()
            .add_systems(Update, (day_cycle, apply_dlss_mode).run_if(|| active()))
            .add_systems(
                PostUpdate,
                (configure_cameras, disable_shadow_maps, proxy_standard_meshes, proxy_skinned_meshes, skin_proxies)
                    .chain()
                    .after(TransformSystems::Propagate)
                    .run_if(|| active()),
            );
    }

    fn finish(&self, app: &mut App) {
        let supported = app
            .world()
            .get_resource::<RenderDevice>()
            .is_some_and(|device| device.features().contains(SolariPlugins::required_wgpu_features()));
        let requested = std::env::var_os("SKATE_RTX").is_none_or(|v| v != "0");
        let enabled = supported && requested;
        ACTIVE.store(enabled, Ordering::Relaxed);
        if enabled {
            info!("SKATE_RTX: path tracing enabled (Solari ReSTIR DI/GI, DLSS Ray Reconstruction when available)");
            if app.world().contains_resource::<DlssRayReconstructionSupported>() {
                sky_after_ray_reconstruction(app);
            }
        } else {
            // Solari's plugin switches every `Auto` material to deferred. Without
            // it, keep the forward renderer the retail materials were built for.
            app.insert_resource(DefaultOpaqueRendererMethod::forward());
            info!("SKATE_RTX: path tracing disabled (supported={supported}, requested={requested}); using raster renderer");
        }
    }
}

/// DLSS Ray Reconstruction outputs black wherever the depth buffer is empty, so
/// a sky drawn in the main pass never reaches the screen. Move the atmosphere's
/// sky pass from inside the main pass to just after RR, before tonemapping.
/// That pass also adds aerial perspective to geometry, which is fine after RR.
fn sky_after_ray_reconstruction(app: &mut App) {
    use bevy::core_pipeline::core_3d::graph::{Core3d, Node3d};
    use bevy::pbr::AtmosphereNode;
    use bevy::render::{RenderApp, render_graph::RenderGraph};
    let render_app = app.sub_app_mut(RenderApp);
    let mut graph = render_app.world_mut().resource_mut::<RenderGraph>();
    let Some(core) = graph.get_sub_graph_mut(Core3d) else { return };
    let moved = core.remove_node_edge(Node3d::MainOpaquePass, AtmosphereNode::RenderSky).is_ok()
        && core.remove_node_edge(AtmosphereNode::RenderSky, Node3d::MainTransparentPass).is_ok();
    if moved {
        core.add_node_edge(Node3d::DlssRayReconstruction, AtmosphereNode::RenderSky);
        core.add_node_edge(AtmosphereNode::RenderSky, Node3d::Tonemapping);
    } else {
        warn!("SKATE_RTX: could not reorder the atmosphere sky pass; the sky may render black under DLSS");
    }
}

// ---------------------------------------------------------------------------
// Camera and lights
// ---------------------------------------------------------------------------

/// Also re-run after a map change, which restores the retail tone state.
fn configure_cameras(
    mut commands: Commands,
    cameras: Query<
        Entity,
        (
            With<crate::camera::GameplayCamera>,
            Or<(Without<SolariLighting>, Without<Hdr>, With<crate::retail_render::RetailTone>)>,
        ),
    >,
    ray_reconstruction: Option<Res<DlssRayReconstructionSupported>>,
    mut media: ResMut<Assets<ScatteringMedium>>,
    mut medium: Local<Option<Handle<ScatteringMedium>>>,
) {
    for camera in &cameras {
        let medium = medium.get_or_insert_with(|| media.add(ScatteringMedium::default())).clone();
        let mut camera = commands.entity(camera);
        camera.remove::<crate::retail_render::RetailTone>().insert((
            Hdr,
            SolariLighting::default(),
            Msaa::Off,
            CameraMainTextureUsages::default().with(TextureUsages::STORAGE_BINDING),
            Atmosphere::earthlike(medium),
            Exposure { ev100: DAY_EV100 },
            Tonemapping::TonyMcMapface,
        ));
        if ray_reconstruction.is_some() {
            camera.insert(Dlss::<DlssRayReconstructionFeature> {
                perf_quality_mode: default(),
                reset: true,
                _phantom_data: default(),
            });
        } else {
            warn_once!("SKATE_RTX: DLSS Ray Reconstruction unavailable; path-traced lighting is not denoised");
        }
    }
}

#[derive(Component)]
struct Celestial {
    moon: bool,
}

/// Moves the sun and moon along the menu's time of day, sets their colour and
/// intensity, and matches camera exposure. Other directional lights (the retail
/// scene sun and shadow helpers) are dimmed to zero, which Solari skips. The
/// painted retail sky dome is hidden; the atmosphere replaces it.
fn day_cycle(
    mut commands: Commands,
    time: Res<Time<Virtual>>,
    menu: Option<ResMut<crate::graphics_menu::Menu>>,
    mut celestial: Query<(&Celestial, &mut DirectionalLight, &mut Transform)>,
    mut others: Query<&mut DirectionalLight, Without<Celestial>>,
    mut cameras: Query<&mut Exposure, With<crate::camera::GameplayCamera>>,
    mut domes: Query<&mut Visibility, With<MeshMaterial3d<crate::retail_sky::SkyMaterial>>>,
    mut ambient: ResMut<GlobalAmbientLight>,
) {
    if celestial.is_empty() {
        for moon in [false, true] {
            commands.spawn((
                Name::new(if moon { "RTX moon" } else { "RTX sun" }),
                Celestial { moon },
                DirectionalLight { shadows_enabled: false, illuminance: 0., ..default() },
                Transform::default(),
            ));
        }
        return;
    }
    let hour = menu.map_or(12., |mut menu| menu.advance_day(time.delta_secs()));
    // 06:00 rises in +X, 12:00 is overhead (tilted toward +Z), 18:00 sets in -X.
    let angle = hour / 24. * std::f32::consts::TAU - std::f32::consts::FRAC_PI_2;
    let sun = Vec3::new(angle.cos(), angle.sin(), 0.25).normalize();
    for (body, mut light, mut transform) in &mut celestial {
        let direction = if body.moon { -sun } else { sun };
        *transform = Transform::default().looking_to(-direction, Vec3::Y);
        let height = direction.y.max(0.).powf(0.4);
        if body.moon {
            light.illuminance = MOON_LUX * height;
            light.color = Color::srgb(0.7, 0.8, 1.0);
        } else {
            light.illuminance = SUN_LUX * height;
            let warmth = smoothstep(0., 0.35, direction.y);
            light.color = Color::srgb(1.0, 0.5 + 0.46 * warmth, 0.25 + 0.65 * warmth);
        }
    }
    for mut light in &mut others {
        if light.illuminance != 0. {
            light.illuminance = 0.;
        }
    }
    // Forward-drawn surfaces (hair, glass) miss path-traced sky light and get
    // Bevy's ambient term instead. Follow the sky, at about half strength
    // because ambient ignores occlusion; a fixed value is blinding at night
    // exposure.
    let lux: f32 = celestial.iter().map(|(_, light, _)| light.illuminance).sum();
    ambient.brightness = AMBIENT_PER_LUX * lux + 0.05;
    let day = smoothstep(-0.1, 0.15, sun.y);
    for mut exposure in &mut cameras {
        exposure.ev100 = NIGHT_EV100 + (DAY_EV100 - NIGHT_EV100) * day;
    }
    for mut visibility in &mut domes {
        if *visibility != Visibility::Hidden {
            *visibility = Visibility::Hidden;
        }
    }
}

/// Applies the Graphics menu's DLSS preset (`DLSS_MODES` order). "Off" removes
/// Ray Reconstruction: the path-traced image is then shown undenoised.
fn apply_dlss_mode(
    mut commands: Commands,
    menu: Option<Res<crate::graphics_menu::Menu>>,
    mut cameras: Query<(Entity, Option<&mut Dlss<DlssRayReconstructionFeature>>), With<SolariLighting>>,
    ray_reconstruction: Option<Res<DlssRayReconstructionSupported>>,
) {
    let Some(menu) = menu else { return };
    let mode = match menu.dlss_mode() {
        6 => {
            for (camera, dlss) in &cameras {
                if dlss.is_some() {
                    commands.entity(camera).remove::<Dlss<DlssRayReconstructionFeature>>();
                }
            }
            return;
        }
        1 => DlssPerfQualityMode::Dlaa,
        2 => DlssPerfQualityMode::Quality,
        3 => DlssPerfQualityMode::Balanced,
        4 => DlssPerfQualityMode::Performance,
        5 => DlssPerfQualityMode::UltraPerformance,
        _ => DlssPerfQualityMode::Auto,
    };
    for (camera, dlss) in &mut cameras {
        match dlss {
            Some(mut dlss) if dlss.perf_quality_mode != mode => dlss.perf_quality_mode = mode,
            Some(_) => {}
            None if ray_reconstruction.is_some() => {
                commands.entity(camera).insert(Dlss::<DlssRayReconstructionFeature> {
                    perf_quality_mode: mode,
                    reset: true,
                    _phantom_data: default(),
                });
            }
            None => {}
        }
    }
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0., 1.);
    t * t * (3. - 2. * t)
}

/// Solari traces shadow rays; cascaded shadow maps would only cost time.
fn disable_shadow_maps(mut lights: Query<&mut DirectionalLight, Changed<DirectionalLight>>) {
    for mut light in &mut lights {
        if light.shadows_enabled {
            light.shadows_enabled = false;
        }
    }
}

// ---------------------------------------------------------------------------
// Retail world proxies
// ---------------------------------------------------------------------------

/// Adds the ray-traced counterpart of the merged world: one proxy per source
/// material, carrying its average albedo. Cutout and blended materials are left
/// out because Solari has no alpha testing; their quads would cast solid shadows.
pub(crate) fn spawn_world_proxies(
    map: &SkateMap,
    table: &MaterialTable,
    commands: &mut SceneCommands,
    meshes: &mut impl AssetSink<Mesh>,
    materials: &mut impl AssetSink<StandardMaterial>,
) {
    if !active() {
        return;
    }
    let mut triangles: HashMap<usize, Vec<u32>> = HashMap::default();
    for tri in map.geometry.indices.chunks_exact(3) {
        let Some(source) = (map.geometry.vertices[tri[0] as usize].material as usize).checked_sub(1) else {
            continue;
        };
        let Some(entry) = table.entry(source) else { continue };
        if matches!(entry.class, RenderClass::Opaque | RenderClass::OpaqueTwoSided) {
            triangles.entry(source).or_default().extend_from_slice(tri);
        }
    }
    let mut average: HashMap<u32, Vec3> = HashMap::default();
    let mut proxies = 0usize;
    for (source, indices) in triangles {
        let material = &map.materials[source];
        let definition = material
            .retail_definition
            .as_deref()
            .and_then(crate::retail_render::Definition::parse);
        let texture = definition
            .as_ref()
            .and_then(|d| d.bindings.get("diffuse"))
            .map_or(material.textures[0], |b| b.texture);
        let albedo = *average.entry(texture).or_insert_with(|| average_albedo(map, texture));
        let emissive = definition.as_ref().is_some_and(|d| matches!(d.family, 11 | 12));
        let material = materials.add(StandardMaterial {
            base_color: Color::linear_rgb(albedo.x, albedo.y, albedo.z),
            emissive: if emissive { LinearRgba::rgb(albedo.x, albedo.y, albedo.z) * EMISSIVE_NITS } else { LinearRgba::BLACK },
            // Matches the G-buffer's derived roughness for untextured detail.
            perceptual_roughness: 0.6,
            ..default()
        });
        for chunk in indices.chunks(MAX_PROXY_TRIANGLES * 3) {
            let mesh = proxy_mesh(chunk, |i| {
                let v = &map.geometry.vertices[i as usize];
                (v.position, v.normal, v.uv)
            });
            commands.spawn((
                Name::new(format!("rt proxy material {source}")),
                RaytracingMesh3d(meshes.add(mesh)),
                MeshMaterial3d(material.clone()),
                Transform::default(),
            ));
            proxies += 1;
        }
    }
    info!("SKATE_RTX: {proxies} world ray-tracing proxies");
}

/// Mean of the squared stored texel, which is how the retail shader linearises
/// albedo. A sparse grid is plenty for a single colour.
fn average_albedo(map: &SkateMap, texture: u32) -> Vec3 {
    let Some(texture) = (texture as usize).checked_sub(1).and_then(|i| map.textures.get(i)) else {
        return Vec3::splat(0.5);
    };
    let texels = (texture.width * texture.height) as usize;
    if texels == 0 || texture.rgba.len() < texels * 4 {
        return Vec3::splat(0.5);
    }
    let step = (texels / 4096).max(1);
    let (mut sum, mut n) = (Vec3::ZERO, 0.);
    for texel in texture.rgba[..texels * 4].chunks_exact(4).step_by(step) {
        let c = Vec3::new(texel[0] as f32, texel[1] as f32, texel[2] as f32) / 255.;
        sum += c * c;
        n += 1.;
    }
    sum / n
}

/// Builds a mesh in the one layout Solari accepts: position, normal, UV0 and
/// tangent, with 32-bit indices. Tangents only need to be a valid frame; proxies
/// carry no normal maps.
fn proxy_mesh(indices: &[u32], vertex: impl Fn(u32) -> ([f32; 3], [f32; 3], [f32; 2])) -> Mesh {
    let mut remap: HashMap<u32, u32> = HashMap::default();
    let (mut positions, mut normals, mut uvs, mut tangents) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let local: Vec<u32> = indices
        .iter()
        .map(|&i| {
            *remap.entry(i).or_insert_with(|| {
                let (p, n, uv) = vertex(i);
                positions.push(p);
                normals.push(n);
                uvs.push(uv);
                tangents.push(any_tangent(Vec3::from_array(n)));
                positions.len() as u32 - 1
            })
        })
        .collect();
    Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::RENDER_WORLD)
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
        .with_inserted_attribute(Mesh::ATTRIBUTE_TANGENT, tangents)
        .with_inserted_indices(Indices::U32(local))
}

fn any_tangent(normal: Vec3) -> [f32; 4] {
    let normal = normal.try_normalize().unwrap_or(Vec3::Y);
    normal.any_orthonormal_vector().extend(1.).to_array()
}

// ---------------------------------------------------------------------------
// Sea
// ---------------------------------------------------------------------------

/// The converted maps contain no sea or river surface: past the shoreline the
/// original renderer shows the clear colour, and the atmosphere's ground under
/// RTX. A flat, dark, glossy plane at sea level stands in, lit and reflected by
/// the path tracer like any other surface. Level is read from `SKATE_SEA_LEVEL`
/// for tuning, else `DEFAULT_SEA_LEVEL`.
const DEFAULT_SEA_LEVEL: f32 = -3.0;
const SEA_HALF_EXTENT: f32 = 8_000.0;

pub(crate) fn spawn_sea(
    commands: &mut SceneCommands,
    meshes: &mut impl AssetSink<Mesh>,
    materials: &mut impl AssetSink<StandardMaterial>,
) {
    if !active() {
        return;
    }
    let level = std::env::var("SKATE_SEA_LEVEL").ok().and_then(|v| v.parse().ok()).unwrap_or(DEFAULT_SEA_LEVEL);
    let plane = Plane3d::new(Vec3::Y, Vec2::splat(SEA_HALF_EXTENT)).mesh().build();
    let Some(mut mesh) = standard_to_proxy(&plane) else { return };
    mesh.asset_usage = RenderAssetUsages::default();
    let mesh = meshes.add(mesh);
    commands.spawn((
        Name::new("RTX sea"),
        Mesh3d(mesh.clone()),
        RaytracingMesh3d(mesh),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.02, 0.05, 0.06),
            perceptual_roughness: 0.06,
            reflectance: 0.35,
            ..default()
        })),
        Transform::from_xyz(0., level, 0.),
    ));
}

// ---------------------------------------------------------------------------
// Map lights
// ---------------------------------------------------------------------------

/// Solari samples directional lights and emissive triangles only, so a map's
/// point and spot lights become small emissive spheres in the ray-traced scene.
/// They have no raster mesh. A sphere emitting `lumens` uniformly has radiance
/// lumens / (4 * pi^2 * r^2). Spot lights emit in all directions.
pub(crate) fn spawn_light_proxy(
    commands: &mut SceneCommands,
    meshes: &mut impl AssetSink<Mesh>,
    materials: &mut impl AssetSink<StandardMaterial>,
    position: Vec3,
    color: Color,
    lumens: f32,
    radius: f32,
) {
    if !active() || lumens <= 0. {
        return;
    }
    let radius = radius.max(0.05);
    let radiance = lumens / (4. * std::f32::consts::PI.powi(2) * radius * radius);
    let sphere = Sphere::new(radius).mesh().ico(1).expect("fixed subdivision");
    let Some(mesh) = standard_to_proxy(&sphere) else { return };
    commands.spawn((
        Name::new("rt light proxy"),
        RaytracingMesh3d(meshes.add(mesh)),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::BLACK,
            emissive: color.to_linear() * radiance,
            ..default()
        })),
        Transform::from_translation(position),
    ));
}

// ---------------------------------------------------------------------------
// Loose StandardMaterial meshes: test world, custom models, mod graphics
// ---------------------------------------------------------------------------

#[derive(Resource, Default)]
struct ConvertedMeshes(HashMap<AssetId<Mesh>, Handle<Mesh>>);

/// Marks an entity this module has already considered.
#[derive(Component)]
struct RtxHandled;

fn proxy_standard_meshes(
    mut commands: Commands,
    candidates: Query<
        (Entity, &Mesh3d, &MeshMaterial3d<StandardMaterial>),
        (Without<RaytracingMesh3d>, Without<SkinnedMesh>, Without<RtxHandled>),
    >,
    materials: Res<Assets<StandardMaterial>>,
    server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut converted: ResMut<ConvertedMeshes>,
) {
    for (entity, mesh, material) in &candidates {
        let Some(material) = materials.get(&material.0) else { continue };
        if !matches!(material.alpha_mode, AlphaMode::Opaque) {
            commands.entity(entity).insert(RtxHandled);
            continue;
        }
        let proxy = if let Some(proxy) = converted.0.get(&mesh.id()) {
            Some(proxy.clone())
        } else if let Some(source) = meshes.get(&mesh.0) {
            let proxy = if solari_compatible(source) {
                Some(mesh.0.clone())
            } else {
                standard_to_proxy(source).map(|m| meshes.add(m))
            };
            if let Some(proxy) = &proxy {
                converted.0.insert(mesh.id(), proxy.clone());
            }
            proxy
        } else if matches!(server.get_load_state(&mesh.0), Some(LoadState::Loading | LoadState::NotLoaded)) {
            continue; // Still loading; try again next frame.
        } else {
            None // Render-world-only mesh: no CPU copy to convert.
        };
        let mut entity = commands.entity(entity);
        entity.insert(RtxHandled);
        if let Some(proxy) = proxy {
            entity.insert(RaytracingMesh3d(proxy));
        }
    }
}

fn solari_compatible(mesh: &Mesh) -> bool {
    mesh.primitive_topology() == PrimitiveTopology::TriangleList
        && mesh.enable_raytracing
        && matches!(mesh.indices(), Some(Indices::U32(_)))
        && mesh.attributes().map(|(a, _)| a.id).eq([
            Mesh::ATTRIBUTE_POSITION.id,
            Mesh::ATTRIBUTE_NORMAL.id,
            Mesh::ATTRIBUTE_UV_0.id,
            Mesh::ATTRIBUTE_TANGENT.id,
        ])
}

fn standard_to_proxy(mesh: &Mesh) -> Option<Mesh> {
    if mesh.primitive_topology() != PrimitiveTopology::TriangleList {
        return None;
    }
    let positions = mesh.attribute(Mesh::ATTRIBUTE_POSITION)?.as_float3()?;
    let normals = mesh.attribute(Mesh::ATTRIBUTE_NORMAL).and_then(|a| a.as_float3());
    let uvs = match mesh.attribute(Mesh::ATTRIBUTE_UV_0) {
        Some(VertexAttributeValues::Float32x2(uvs)) => Some(uvs),
        _ => None,
    };
    let indices: Vec<u32> = match mesh.indices() {
        Some(indices) => indices.iter().map(|i| i as u32).collect(),
        None => (0..positions.len() as u32).collect(),
    };
    Some(proxy_mesh(&indices, |i| {
        let i = i as usize;
        (
            positions[i],
            normals.map_or([0., 1., 0.], |n| n[i]),
            uvs.map_or([0., 0.], |uv| uv[i]),
        )
    }))
}

// ---------------------------------------------------------------------------
// Skinned characters
//
// Solari builds acceleration structures from static vertex buffers, so skinned
// meshes are skinned on the CPU into a proxy each frame. Changing the proxy mesh
// makes Solari rebuild its BLAS. The proxy is a child with an identity transform,
// so vertices are written in the parent's local space.
// ---------------------------------------------------------------------------

#[derive(Component)]
struct SkinProxy {
    proxy: Entity,
    mesh: Handle<Mesh>,
}

fn proxy_skinned_meshes(
    mut commands: Commands,
    candidates: Query<
        (
            Entity,
            &Mesh3d,
            Option<&MeshMaterial3d<StandardMaterial>>,
            Option<&MeshMaterial3d<crate::customiser_material::SkaterMaterial>>,
        ),
        (With<SkinnedMesh>, Without<SkinProxy>, Without<RtxHandled>),
    >,
    mut standard: ResMut<Assets<StandardMaterial>>,
    skater: Res<Assets<crate::customiser_material::SkaterMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    for (entity, mesh, material, skater_material) in &candidates {
        let base = match (material, skater_material) {
            (Some(material), _) => standard.get(&material.0).cloned(),
            (None, Some(material)) => skater.get(&material.0).map(|m| m.base.clone()),
            (None, None) => None,
        };
        let Some(base) = base else { continue };
        let Some(source) = meshes.get(&mesh.0) else { continue };
        if !matches!(base.alpha_mode, AlphaMode::Opaque) || source.attribute(Mesh::ATTRIBUTE_JOINT_INDEX).is_none() {
            commands.entity(entity).insert(RtxHandled);
            continue;
        }
        let Some(mut proxy) = standard_to_proxy(source) else {
            commands.entity(entity).insert(RtxHandled);
            continue;
        };
        // Rewritten every frame, so the CPU copy must stay.
        proxy.asset_usage = RenderAssetUsages::default();
        let proxy_mesh = meshes.add(proxy);
        // Textures are irrelevant at secondary hits; keep the flat colour only.
        let material = standard.add(StandardMaterial {
            base_color: base.base_color,
            perceptual_roughness: base.perceptual_roughness,
            metallic: base.metallic,
            ..default()
        });
        let proxy = commands
            .spawn((
                Name::new("rt skinned proxy"),
                RaytracingMesh3d(proxy_mesh.clone()),
                MeshMaterial3d(material),
                Transform::default(),
                ChildOf(entity),
            ))
            .id();
        commands.entity(entity).insert(SkinProxy { proxy, mesh: proxy_mesh });
    }
}

fn skin_proxies(
    mut commands: Commands,
    sources: Query<(&Mesh3d, &SkinnedMesh, &SkinProxy, &GlobalTransform, &InheritedVisibility)>,
    joints: Query<&GlobalTransform>,
    bindposes: Res<Assets<SkinnedMeshInverseBindposes>>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    for (mesh, skin, proxy, transform, visibility) in &sources {
        // Hidden characters must not cast ray-traced shadows.
        if !visibility.get() {
            commands.entity(proxy.proxy).try_remove::<RaytracingMesh3d>();
            continue;
        }
        commands.entity(proxy.proxy).try_insert(RaytracingMesh3d(proxy.mesh.clone()));
        let Some(bindposes) = bindposes.get(&skin.inverse_bindposes) else { continue };
        let local_from_world = transform.affine().inverse();
        let matrices: Vec<Mat4> = skin
            .joints
            .iter()
            .zip(bindposes.iter())
            .map(|(&joint, bindpose)| {
                let world = joints.get(joint).map_or(Mat4::IDENTITY, |j| j.to_matrix());
                Mat4::from(local_from_world) * world * *bindpose
            })
            .collect();
        let Some(source) = meshes.get(&mesh.0) else { continue };
        let (Some(positions), Some(VertexAttributeValues::Uint16x4(joint_ids)), Some(VertexAttributeValues::Float32x4(weights))) = (
            source.attribute(Mesh::ATTRIBUTE_POSITION).and_then(|a| a.as_float3()),
            source.attribute(Mesh::ATTRIBUTE_JOINT_INDEX),
            source.attribute(Mesh::ATTRIBUTE_JOINT_WEIGHT),
        ) else {
            continue;
        };
        let normals = source.attribute(Mesh::ATTRIBUTE_NORMAL).and_then(|a| a.as_float3());
        let indices: Vec<u32> = match source.indices() {
            Some(indices) => indices.iter().map(|i| i as u32).collect(),
            None => (0..positions.len() as u32).collect(),
        };
        let skinned: Vec<([f32; 3], [f32; 3])> = (0..positions.len())
            .map(|i| {
                let mut m = Mat4::ZERO;
                for k in 0..4 {
                    if let Some(joint) = matrices.get(joint_ids[i][k] as usize) {
                        m += *joint * weights[i][k];
                    }
                }
                let p = m.transform_point3(Vec3::from_array(positions[i]));
                let n = normals.map_or(Vec3::Y, |n| m.transform_vector3(Vec3::from_array(n[i])).normalize_or(Vec3::Y));
                (p.to_array(), n.to_array())
            })
            .collect();
        // Same vertex order as `standard_to_proxy`, which remapped by first use.
        let mut order = Vec::with_capacity(positions.len());
        let mut seen = vec![false; positions.len()];
        for &i in &indices {
            if !std::mem::replace(&mut seen[i as usize], true) {
                order.push(i as usize);
            }
        }
        let Some(proxy) = meshes.get_mut(&proxy.mesh) else { continue };
        proxy.insert_attribute(Mesh::ATTRIBUTE_POSITION, order.iter().map(|&i| skinned[i].0).collect::<Vec<_>>());
        proxy.insert_attribute(Mesh::ATTRIBUTE_NORMAL, order.iter().map(|&i| skinned[i].1).collect::<Vec<_>>());
        proxy.insert_attribute(
            Mesh::ATTRIBUTE_TANGENT,
            order.iter().map(|&i| any_tangent(Vec3::from_array(skinned[i].1))).collect::<Vec<_>>(),
        );
    }
}
