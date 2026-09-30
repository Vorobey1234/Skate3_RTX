// Deferred G-buffer stages for merged world geometry (RTX path).
//
// The retail fragment in `retail_world.wgsl` multiplies albedo by a baked
// lightmap and a fixed sun term. Under path tracing, Solari computes all direct
// and indirect light itself, so this pass writes only the surface: linear
// albedo, the shading normal, roughness/reflectance and emission. Lightmaps,
// cube reflections and the authored fog frame are not read here.
//
// Vertex layout matches `WorldMaterial::specialize`, which installs one layout
// for every pipeline of the material.
#import bevy_pbr::{
    mesh_functions,
    mesh_view_bindings::view,
    prepass_bindings,
    prepass_io::FragmentOutput,
    pbr_deferred_types,
    rgb9e5,
    utils::octahedral_encode,
    view_transformations::position_world_to_clip,
}
#import skate_retail::material_bindings as bindings

// Unlit retail families (signs, lamps) become emitters. Luminance in cd/m^2 for
// an albedo of one; about 1000 nits keeps them near their retail on-screen
// brightness at Bevy's default exposure, and lets Solari use them as lights.
const EMISSIVE_NITS: f32 = 1000.0;

struct Vertex {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) uv_b: vec2<f32>,
    @location(4) color: vec4<f32>,
    @location(5) material_index: u32,
    @location(6) tangent: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(1) previous_world_position: vec4<f32>,
    @location(2) world_normal: vec3<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) color: vec4<f32>,
    @location(5) @interpolate(flat) material_index: u32,
    @location(6) world_tangent: vec4<f32>,
}

@vertex
fn vertex(v: Vertex) -> VertexOutput {
    var out: VertexOutput;
    let world_from_local = mesh_functions::get_world_from_local(v.instance_index);
    out.world_position = mesh_functions::mesh_position_local_to_world(
        world_from_local, vec4<f32>(v.position, 1.0));
    out.clip_position = position_world_to_clip(out.world_position.xyz);
    let previous_world_from_local = mesh_functions::get_previous_world_from_local(v.instance_index);
    out.previous_world_position = mesh_functions::mesh_position_local_to_world(
        previous_world_from_local, vec4<f32>(v.position, 1.0));
    out.world_normal = mesh_functions::mesh_normal_local_to_world(v.normal, v.instance_index);
    out.uv = v.uv;
    out.color = v.color;
    out.material_index = v.material_index;
    out.world_tangent = mesh_functions::mesh_tangent_local_to_world(
        world_from_local, v.tangent, v.instance_index);
    return out;
}

@fragment
fn fragment(i: VertexOutput) -> FragmentOutput {
    let slot = i.material_index;
    let p = bindings::params[slot];
    let fam = u32(p.mode.x);
    let flags = u32(p.mode.y);

    // Sampling stays in uniform control flow, as in the retail shader.
    let g = bindings::gradients(i.uv);
    let g_decal = bindings::gradients(i.color.xy);
    let g_detail = bindings::gradients_scaled(g, p.surface.z);
    let g_macro = bindings::gradients_scaled(g, p.surface.x);
    var diffuse_uv = i.uv;
    if fam == 14u {
        diffuse_uv += fract(bindings::frame_state().clock.x * p.water[1].xy * vec2<f32>(1.0, -1.0));
    }
    let a = bindings::sample_diffuse(slot, diffuse_uv, g);
    var nm = vec3<f32>(0.5, 0.5, 1.0);
    var detail = vec2<f32>(0.5);
    var overlay = vec3<f32>(0.5);
    var art = vec4<f32>(0.0);
    var masks = vec3<f32>(0.0);
    if (flags & 1u) != 0u && (fam <= 6u || fam == 13u) { nm = bindings::sample_normal_map(slot, i.uv, g).rgb; }
    if (flags & 2u) != 0u && fam != 2u {
        detail = bindings::sample_detail_map(slot, bindings::scaled_uv(i.uv, p.surface.z), g_detail).rg;
    }
    if (flags & 4u) != 0u { overlay = bindings::sample_macro_map(slot, bindings::scaled_uv(i.uv, p.surface.x), g_macro).rgb; }
    if (flags & 8u) != 0u && (fam == 3u || fam == 4u) { art = bindings::sample_decal_map(slot, i.color.xy, g_decal); }
    if (flags & 16u) != 0u { masks = bindings::sample_specular_map(slot, i.uv, g).rgb; }

    // Albedo: the retail shader squares the stored texel to linearise it.
    var albedo = a.rgb * a.rgb;
    if (fam == 3u || fam == 4u) && (flags & 8u) != 0u && (flags & 512u) == 0u {
        albedo = mix(albedo, art.rgb * art.rgb, art.a * p.decal.x);
    }
    if (flags & 4u) != 0u && fam < 13u && (flags & 256u) == 0u {
        albedo *= saturate((overlay - 0.5) * p.surface.y + 0.5);
    }

    // Shading normal: authored tangent frame where present, else derivatives.
    var wn = normalize(i.world_normal);
    let dp1 = dpdx(i.world_position.xyz);
    let dp2 = dpdy(i.world_position.xyz);
    let du1 = dpdx(i.uv * vec2<f32>(1.0, -1.0));
    let du2 = dpdy(i.uv * vec2<f32>(1.0, -1.0));
    var kt = cross(dp2, wn) * du1.x + cross(wn, dp1) * du2.x;
    var kb = cross(dp2, wn) * du1.y + cross(wn, dp1) * du2.y;
    kt *= -inverseSqrt(max(dot(kt, kt), 1e-12));
    kb *= inverseSqrt(max(dot(kb, kb), 1e-12));
    if dot(i.world_tangent.xyz, i.world_tangent.xyz) > 0.01 {
        kt = normalize(i.world_tangent.xyz);
        kb = normalize(cross(wn, kt)) * i.world_tangent.w;
    }
    if (fam <= 6u || fam == 13u) && (flags & 1u) != 0u {
        var dxy = vec2<f32>(0.5);
        if (flags & 2u) != 0u && fam != 2u { dxy = detail; }
        let raw = vec3<f32>(nm.xy * 2.0 + dxy * 2.0 - 2.0, nm.z * 2.0 - 1.0);
        wn = normalize(raw.x * kt + raw.y * kb + wn * max(raw.z, 0.05));
    }

    // Retail Blinn-Phong gloss -> GGX. Exponent n maps to alpha = sqrt(2/(n+2)).
    var perceptual_roughness = 0.9;
    var reflectance = 0.5;
    if (flags & 16u) != 0u {
        let exponent = 10.0 + 290.0 * masks.y;
        perceptual_roughness = sqrt(sqrt(2.0 / (exponent + 2.0)));
        reflectance = mix(0.3, 1.0, masks.x);
    }
    if (fam == 5u || fam == 6u || fam == 13u) && (flags & 64u) != 0u {
        perceptual_roughness = mix(perceptual_roughness, 0.1, masks.z);
    }
    let water = fam == 14u || fam >= 30u;
    if water { perceptual_roughness = 0.08; reflectance = 0.5; }

    var emissive = vec3<f32>(0.0);
    if fam == 11u || fam == 12u {
        emissive = albedo * EMISSIVE_NITS;
        if fam == 11u { emissive *= p.family.w; }
    }

    // After every derivative, as in the retail shader.
#ifdef WORLD_ALPHA_CUTOFF
    if a.a < p.mode.z { discard; }
#endif

    var out: FragmentOutput;
#ifdef NORMAL_PREPASS
    out.normal = vec4<f32>(wn * 0.5 + 0.5, 1.0);
#endif
#ifdef MOTION_VECTOR_PREPASS
    let clip_t = view.unjittered_clip_from_world * i.world_position;
    let previous_t = prepass_bindings::previous_view_uniforms.clip_from_world * i.previous_world_position;
    out.motion_vector = (clip_t.xy / clip_t.w - previous_t.xy / previous_t.w) * vec2<f32>(0.5, -0.5);
#endif
#ifdef DEFERRED_PREPASS
    // Same packing as `deferred_gbuffer_from_pbr_input`, which Solari decodes.
    out.deferred = vec4<u32>(
        pbr_deferred_types::pack_unorm4x8_(vec4<f32>(pow(saturate(albedo), vec3<f32>(1.0 / 2.2)), perceptual_roughness)),
        rgb9e5::vec3_to_rgb9e5_(emissive),
        pbr_deferred_types::pack_unorm4x8_(vec4<f32>(reflectance, 0.0, 1.0, 0.0)),
        pbr_deferred_types::pack_24bit_normal_and_flags(octahedral_encode(wn), 0u),
    );
    out.deferred_lighting_pass_id = 1u;
#endif
    return out;
}
