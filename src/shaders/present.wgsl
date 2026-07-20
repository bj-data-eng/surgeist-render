struct PassSpatialUniform {
    source_origin_scale: vec4<f32>,
    destination_origin_scale: vec4<f32>,
    source_destination_extents: vec4<u32>,
}

@group(0) @binding(0)
var source_texture: texture_2d<f32>;

@group(0) @binding(1)
var source_sampler: sampler;

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

fn source_uv(destination_position: vec2<f32>) -> vec2<f32> {
    let destination_point = spatial.destination_origin_scale.xy
        + destination_position / spatial.destination_origin_scale.z;
    let source_texel = (destination_point - spatial.source_origin_scale.xy)
        * spatial.source_origin_scale.z;
    return source_texel
        / vec2<f32>(spatial.source_destination_extents.xy);
}

@fragment
fn fragment_main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let premultiplied = clamp(
        textureSample(source_texture, source_sampler, source_uv(position.xy)),
        vec4<f32>(0.0),
        vec4<f32>(1.0),
    );
    let quantized_alpha = floor(premultiplied.a * 255.0 + 0.5) / 255.0;
    if (quantized_alpha == 0.0) {
        return vec4<f32>(0.0);
    }
    let straight_rgb = clamp(
        premultiplied.rgb / premultiplied.a,
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );
    return vec4<f32>(straight_rgb, quantized_alpha);
}
