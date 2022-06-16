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
