///todo: migrate out
///or really, migrate everything to host side C++
float3 rot_quat(const float3 point, float4 quat)
{
    quat = fast_normalize(quat);

    float3 t = 2.f * cross(quat.xyz, point);

    return point + quat.w * t + cross(quat.xyz, t);
}

float4 aa_to_quat(float3 axis, float angle)
{
    float4 q;

    q.xyz = axis.xyz * native_sin(angle/2);
    q.w = native_cos(angle/2);

    return fast_normalize(q);
}

float4 quat_multiply(float4 q1, float4 q2)
{
    float4 ret;

    ret.x = q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y;
    ret.y = q1.w * q2.y + q1.y * q2.w + q1.z * q2.x - q1.x * q2.z;
    ret.z = q1.w * q2.z + q1.z * q2.w + q1.x * q2.y - q1.y * q2.x;
    ret.w = q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z;

    return ret;
}

float3 cartesian_to_polar(float3 in)
{
    float r = length(in);
    //float theta = atan2(native_sqrt(in.x * in.x + in.y * in.y), in.z);
    float theta = acos(in.z / r);
    float phi = atan2(in.y, in.x);

    return (float3){r, theta, phi};
}

///ok I've been procrastinating this for a while
///I want to parallel transport my camera vectors, if and only if I'm on a geodesic
///so: I need to first define local camera vectors
///then I need to get a local tetrad
///then I need to multiply my camera vectors out by the tetrad to get them in global coordinates. This means converting from a tetrad basis, to a coordinate basis
///once in global coordinates, I need to parallel transport them
///then, to render, I.. want to convert them back to a tetrad basis?
__kernel
void handle_controls_free(__global float4* camera_pos_cart, __global float4* camera_rot,
                          __global float4* e0, __global float4* e1, __global float4* e2, __global float4* e3,
                          __global float4* b0, __global float4* b1, __global float4* b2,
                          float2 mouse_delta, float4 unrotated_translation, float universe_size)
{
    ///translation is: .x is forward - back, .y = right - left, .z = down - up
    ///totally arbitrary, purely to pass to the GPU
    if(get_global_id(0) != 0)
        return;

    float4 local_camera_quat = *camera_rot;
    float4 local_camera_pos_cart = *camera_pos_cart;

    if(mouse_delta.x != 0)
    {
        float4 q = aa_to_quat((float3)(0, 0, -1), mouse_delta.x);

        local_camera_quat = quat_multiply(q, local_camera_quat);
    }

    {
        float3 right = rot_quat((float3){1, 0, 0}, local_camera_quat);

        if(mouse_delta.y != 0)
        {
            float4 q = aa_to_quat(right, mouse_delta.y);

            local_camera_quat = quat_multiply(q, local_camera_quat);
        }
    }

    float3 up = {0, 0, -1};
    float3 right = rot_quat((float3){1, 0, 0}, local_camera_quat);
    float3 forw = rot_quat((float3){0, 0, 1}, local_camera_quat);

    float3 offset = {0,0,0};

    offset += forw * unrotated_translation.x;
    offset += right * unrotated_translation.y;
    offset += up * unrotated_translation.z;

    local_camera_pos_cart.y += offset.x;
    local_camera_pos_cart.z += offset.y;
    local_camera_pos_cart.w += offset.z;

    {
        float rad = length(local_camera_pos_cart.yzw);

        if(rad > universe_size * 0.99f)
        {
            local_camera_pos_cart.yzw = normalize(local_camera_pos_cart.yzw) * universe_size * 0.99f;
        }
    }

    *camera_rot = local_camera_quat;
    *camera_pos_cart = local_camera_pos_cart;
}

__kernel
void camera_cart_to_polar(__global float4* g_camera_pos_polar_out, __global float4* g_camera_pos_cart, float flip)
{
    if(get_global_id(0) != 0)
        return;

    float3 cart = g_camera_pos_cart->yzw;

    float3 polar = cartesian_to_polar(cart);

    if(flip > 0)
        polar.x = -polar.x;

    *g_camera_pos_polar_out = (float4)(g_camera_pos_cart->x, polar.xyz);
}

/*
so
axis angle to local rotation matrix, take column vectors as basis
use a passed in tetrad to convert this to coordinate basis
convert this to a global basis
the camera kernel will project this global rotation matrix back to a local rotation matrix, by projecting the column vectors in the tetrad
*/

__kernel
void produce_global_rotation_vectors(__global float4* e0, __global float4* e1, __global float4* e2, __global float4* e3,
                                     __global float4* axis_angle, __global float4* global0, __global float4* global1, __global float4* global2)
{

}

__kernel
void advance_time(__global float4* camera, float time)
{
    if(get_global_id(0) != 0)
        return;

    camera->x += time;
}

__kernel
void set_time(__global float4* camera, float time)
{
    if(get_global_id(0) != 0)
        return;

    camera->x = time;
}
