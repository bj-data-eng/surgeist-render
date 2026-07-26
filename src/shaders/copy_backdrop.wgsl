struct PassSpatialUniform {
    source_origin_scale: vec4<f32>,
    destination_origin_scale: vec4<f32>,
    source_destination_extents: vec4<u32>,
}

@group(0) @binding(0)
var parent_texture: texture_2d<f32>;

@group(0) @binding(1)
var parent_sampler: sampler;

@group(0) @binding(2)
var<uniform> spatial: PassSpatialUniform;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
}

@vertex
fn vertex_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    var output: VertexOutput;
    output.position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
    return output;
}

fn surface_texel_from_capture_position(capture_position: vec2<f32>) -> vec2<f32> {
    let capture_point = spatial.destination_origin_scale.xy
        + capture_position / spatial.destination_origin_scale.z;
    return (capture_point - spatial.source_origin_scale.xy)
        * spatial.source_origin_scale.z;
}

fn sample_completed_parent(surface_texel: vec2<f32>) -> vec4<f32> {
    let surface_extent = vec2<f32>(spatial.source_destination_extents.xy);
    if (any(surface_texel < vec2<f32>(0.0))
        || any(surface_texel >= surface_extent)) {
        return vec4<f32>(0.0);
    }
    return textureSampleLevel(
        parent_texture,
        parent_sampler,
        surface_texel / surface_extent,
        0.0,
    );
}

@fragment
fn fragment_main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let surface_texel = surface_texel_from_capture_position(position.xy);
    return sample_completed_parent(surface_texel);
}
