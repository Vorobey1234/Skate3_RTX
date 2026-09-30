//! Graphics menu "Skater light": a headlamp or glowing wheels.
//!
//! Both follow bones every board rig has (`SKATEBOARD_ROOT`, the trucks and
//! the four `*_WHEEL*` joints), so they work for the stock skater and every
//! customised board. They are emissive meshes: under RTX `rtx.rs` gives them
//! ray-tracing proxies and Solari samples them as lights. The raster renderer
//! shows the wheels' glow and gets a spot light for the headlamp.
//!
//! The light entities are top-level and posed from the bones each frame after
//! transform propagation, so they never inherit a bone's scale.
use bevy::{platform::collections::HashMap, prelude::*, transform::TransformSystems};

/// Menu order of `Menu::skater_light`.
pub(crate) const SKATER_LIGHTS: &[&str] = &["Off", "Flashlight", "Glowing wheels"];

/// Wheel glow at full speed, in cd/m^2 per unit of colour.
const WHEEL_NITS: f32 = 4000.;
const WHEEL_COLOR: Vec3 = Vec3::new(0.15, 0.75, 1.0);
/// Board speed (m/s) at which the wheels start to glow and reach full glow.
const GLOW_START: f32 = 0.5;
const GLOW_FULL: f32 = 6.0;
/// Rings just outside a retail wheel (about 52 mm across, 32 mm wide).
const RING_RADIUS: f32 = 0.029;
const RING_WIDTH: f32 = 0.036;

/// Headlamp: a small disc at head height, aimed along the board and slightly
/// down. Its luminous intensity is `HEADLAMP_CANDELA` straight ahead.
const HEADLAMP_CANDELA: f32 = 1500.;
const HEADLAMP_RADIUS: f32 = 0.06;
const HEADLAMP_HEIGHT: f32 = 1.55;
const HEADLAMP_PITCH: f32 = -0.25;
/// An emissive disc shines over a whole hemisphere. A black tube in front of
/// it cuts that to a beam of about `atan(2 * radius / length)` = 35 degrees.
const HOUSING_LENGTH: f32 = 0.17;

const WHEELS: [&str; 4] = ["LEFT_WHEELFRONT", "RIGHT_WHEELFRONT", "LEFT_WHEELBACK", "RIGHT_WHEELBACK"];

pub(crate) struct SkaterLightPlugin;

impl Plugin for SkaterLightPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Rigs>()
            .add_systems(PostUpdate, (find_rigs, update_lights).chain().after(TransformSystems::Propagate));
    }
}

/// The bones of one board rig, keyed by its `SKATEBOARD_ROOT`.
#[derive(Default)]
struct Rig {
    wheels: [Option<Entity>; 4],
    truck_front: Option<Entity>,
    truck_back: Option<Entity>,
    last_position: Option<Vec3>,
    speed: f32,
    glow: f32,
    material: Option<Handle<StandardMaterial>>,
    lights: Vec<Entity>,
    mode: usize,
}

#[derive(Resource, Default)]
struct Rigs(HashMap<Entity, Rig>);

fn find_rigs(
    named: Query<(Entity, &Name), Added<Name>>,
    parents: Query<&ChildOf>,
    names: Query<&Name>,
    mut rigs: ResMut<Rigs>,
) {
    for (entity, name) in &named {
        let name = name.as_str();
        let wheel = WHEELS.iter().position(|w| *w == name);
        if wheel.is_none() && name != "TRUCK_FRONT" && name != "TRUCK_BACK" {
            continue;
        }
        let Some(board) = parents
            .iter_ancestors(entity)
            .find(|&a| names.get(a).is_ok_and(|n| n.as_str() == "SKATEBOARD_ROOT"))
        else {
            continue;
        };
        let rig = rigs.0.entry(board).or_default();
        match (wheel, name) {
            (Some(i), _) => rig.wheels[i] = Some(entity),
            (None, "TRUCK_FRONT") => rig.truck_front = Some(entity),
            _ => rig.truck_back = Some(entity),
        }
    }
}

fn update_lights(
    mut commands: Commands,
    time: Res<Time>,
    menu: Option<Res<crate::graphics_menu::Menu>>,
    bones: Query<&GlobalTransform, Without<SkaterLight>>,
    mut lights: Query<(&mut Transform, &mut GlobalTransform), With<SkaterLight>>,
    mut rigs: ResMut<Rigs>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut shapes: Local<Option<(Handle<Mesh>, Handle<Mesh>, Handle<StandardMaterial>, Handle<Mesh>, Handle<StandardMaterial>)>>,
) {
    let mode = menu.map_or(0, |menu| menu.skater_light());
    let (ring, disc, lamp, tube, black) = shapes
        .get_or_insert_with(|| {
            let radiance = HEADLAMP_CANDELA / (std::f32::consts::PI * HEADLAMP_RADIUS * HEADLAMP_RADIUS);
            (
                meshes.add(Cylinder::new(RING_RADIUS, RING_WIDTH).mesh().resolution(24)),
                meshes.add(Circle::new(HEADLAMP_RADIUS).mesh().resolution(16)),
                materials.add(StandardMaterial {
                    base_color: Color::BLACK,
                    emissive: LinearRgba::rgb(1.0, 0.92, 0.8) * radiance,
                    ..default()
                }),
                meshes.add(Cylinder::new(HEADLAMP_RADIUS, HOUSING_LENGTH).mesh().resolution(16).without_caps()),
                materials.add(StandardMaterial { base_color: Color::BLACK, ..default() }),
            )
        })
        .clone();
    let dt = time.delta_secs().max(1e-4);
    rigs.0.retain(|_, rig| {
        let alive = rig.truck_front.is_some_and(|e| bones.get(e).is_ok());
        if !alive {
            for light in rig.lights.drain(..) {
                commands.entity(light).try_despawn();
            }
        }
        alive
    });
    for rig in rigs.0.values_mut() {
        let (Some(front), Some(back)) = (
            rig.truck_front.and_then(|e| bones.get(e).ok()),
            rig.truck_back.and_then(|e| bones.get(e).ok()),
        ) else {
            continue;
        };
        let (front, back) = (front.translation(), back.translation());
        let speed = rig.last_position.map_or(0., |last| front.distance(last) / dt);
        rig.last_position = Some(front);
        // Teleports and respawns would read as one enormous frame of speed.
        if speed < 60. {
            rig.speed += (speed - rig.speed) * (dt * 6.).min(1.);
        }

        if rig.mode != mode {
            for light in rig.lights.drain(..) {
                commands.entity(light).try_despawn();
            }
            rig.mode = mode;
            rig.glow = -1.;
            match mode {
                1 => {
                    let mut lamp_entity = commands.spawn((
                        Name::new("Skater headlamp"),
                        SkaterLight,
                        Mesh3d(disc.clone()),
                        MeshMaterial3d(lamp.clone()),
                        // Ray-traced only: the disc lights the scene but is not drawn.
                        Visibility::Hidden,
                        Transform::default(),
                    ));
                    if !crate::rtx::active() {
                        lamp_entity.with_child((
                            SpotLight {
                                intensity: HEADLAMP_CANDELA * 4. * std::f32::consts::PI,
                                range: 40.,
                                outer_angle: 0.6,
                                inner_angle: 0.35,
                                shadows_enabled: true,
                                ..default()
                            },
                            // The disc faces +Z; a spot light shines along -Z.
                            Transform::from_rotation(Quat::from_rotation_y(std::f32::consts::PI)),
                        ));
                    }
                    rig.lights.push(lamp_entity.id());
                    let housing = commands.spawn((
                        Name::new("Skater headlamp housing"),
                        SkaterLight,
                        Mesh3d(tube.clone()),
                        MeshMaterial3d(black.clone()),
                        Visibility::Hidden,
                        Transform::default(),
                    ));
                    rig.lights.push(housing.id());
                }
                2 => {
                    let material = rig
                        .material
                        .get_or_insert_with(|| materials.add(StandardMaterial { base_color: Color::srgb(0.85, 0.85, 0.8), ..default() }))
                        .clone();
                    for _ in 0..4 {
                        let id = commands
                            .spawn((
                                Name::new("Skater wheel glow"),
                                SkaterLight,
                                Mesh3d(ring.clone()),
                                MeshMaterial3d(material.clone()),
                                Transform::default(),
                            ))
                            .id();
                        rig.lights.push(id);
                    }
                }
                _ => {}
            }
        }

        let forward = (front - back).try_normalize().unwrap_or(Vec3::Z);
        let mut place = |entity: Entity, transform: Transform| {
            if let Ok((mut t, mut g)) = lights.get_mut(entity) {
                *t = transform;
                *g = GlobalTransform::from(transform);
            }
        };
        match mode {
            1 => {
                let flat = Vec3::new(forward.x, 0., forward.z).try_normalize().unwrap_or(Vec3::Z);
                let aim = Quat::from_axis_angle(flat.cross(Vec3::Y).normalize_or(Vec3::X), HEADLAMP_PITCH) * flat;
                let position = (front + back) / 2. + Vec3::Y * HEADLAMP_HEIGHT + flat * 0.15;
                let facing = Transform::from_translation(position).looking_to(aim, Vec3::Y);
                if let [lamp_entity, housing] = rig.lights[..] {
                    // Turn the disc so its emitting +Z side faces along `aim`.
                    place(lamp_entity, facing.with_rotation(facing.rotation * Quat::from_rotation_y(std::f32::consts::PI)));
                    // The tube's axis is Y; lay it along `aim`, starting at the disc.
                    let tube = Transform::from_translation(position + aim * (HOUSING_LENGTH / 2.))
                        .with_rotation(Quat::from_rotation_arc(Vec3::Y, aim));
                    place(housing, tube);
                }
            }
            2 => {
                let wheels = rig.wheels.map(|w| w.and_then(|e| bones.get(e).ok()).map(|g| g.translation()));
                let axle = match (wheels[0], wheels[1]) {
                    (Some(left), Some(right)) => (right - left).try_normalize(),
                    _ => None,
                }
                .unwrap_or(Vec3::X);
                let rotation = Quat::from_rotation_arc(Vec3::Y, axle);
                for (&light, wheel) in rig.lights.iter().zip(wheels) {
                    if let Some(wheel) = wheel {
                        place(light, Transform::from_translation(wheel).with_rotation(rotation));
                    }
                }
                let glow = ((rig.speed - GLOW_START) / (GLOW_FULL - GLOW_START)).clamp(0., 1.);
                if (glow - rig.glow).abs() > 0.02
                    && let Some(material) = rig.material.as_ref().and_then(|m| materials.get_mut(m))
                {
                    rig.glow = glow;
                    let c = WHEEL_COLOR * WHEEL_NITS * glow;
                    material.emissive = LinearRgba::rgb(c.x, c.y, c.z);
                }
            }
            _ => {}
        }
    }
}

#[derive(Component)]
struct SkaterLight;
