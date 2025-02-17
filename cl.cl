#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#include "common.cl"

#define M_PIf ((float)M_PI)
#define E4(n) n.x, n.y, n.z, n.w
#define SWAP(x, y, T) do { T SWAP = x; x = y; y = SWAP; } while (0)

#define DUMP_TETRAD(str, a, b, c, d) printf(str " p1 %f %f %f %f p2 %f %f %f %f p3 %f %f %f %f p4 %f %f %f %f", a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w, c.x, c.y, c.z, c.w, d.x, d.y, d.z, d.w)

float4 sort_vector_timelike(float4 in, int which)
{
    if(which == 0)
        return in;

    float arr[4] = {in.x, in.y, in.z, in.w};

    SWAP(arr[0], arr[which], float);

    return (float4)(arr[0], arr[1], arr[2], arr[3]);
}

float4 put_timelike_in_correct_position(float4 txyz, int which)
{
    ///because this just performs a simple swap, its always valid
    return sort_vector_timelike(txyz, which);
}

float get_vector_timelike_component(float4 v, int which)
{
    if(which == 0)
        return v.x;

    if(which == 1)
        return v.y;

    if(which == 2)
        return v.z;

    if(which == 3)
        return v.w;
}

struct triangle
{
    int parent;
    float v0x, v0y, v0z;
    float v1x, v1y, v1z;
    float v2x, v2y, v2z;
};

struct intersection
{
    int sx, sy;
    int computed_parent;
};

struct object
{
    float4 pos;
};

bool approx_equal(float v1, float v2, float tol)
{
    return fabs(v1 - v2) <= tol;
}

#define IS_DEGENERATE(x) (isnan(x) || !isfinite(x))

void sort2(float* v0, float* v1)
{
    float iv0 = *v0;
    float iv1 = *v1;

    *v0 = min(iv0, iv1);
    *v1 = max(iv0, iv1);
}

bool range_overlaps_general(float s1, float s2, float e1, float e2, float period)
{
    sort2(&s1, &s2);
    sort2(&e1, &e2);

    if(period == 0)
        return range_overlaps(s1, s2, e1, e2);
    else
        return periodic_range_overlaps(s1, s2, e1, e2, period);
}

bool range_overlaps_general4(float4 s1, float4 s2, float4 e1, float4 e2, float4 period)
{
    return range_overlaps_general(s1.x, s2.x, e1.x, e2.x, period.x) &&
           range_overlaps_general(s1.y, s2.y, e1.y, e2.y, period.y) &&
           range_overlaps_general(s1.z, s2.z, e1.z, e2.z, period.z) &&
           range_overlaps_general(s1.w, s2.w, e1.w, e2.w, period.w);
}

float smooth_fmod(float a, float b)
{
    return fmod(a, b);
}

float3 cartesian_to_polar(float3 in)
{
    float r = length(in);
    //float theta = atan2(native_sqrt(in.x * in.x + in.y * in.y), in.z);
    float theta = acos(in.z / r);
    float phi = atan2(in.y, in.x);

    return (float3){r, theta, phi};
}

float3 polar_to_cartesian(float3 in)
{
    float x = in.x * native_sin(in.y) * native_cos(in.z);
    float y = in.x * native_sin(in.y) * native_sin(in.z);
    float z = in.x * native_cos(in.y);

    return (float3){x, y, z};
}

float3 cartesian_velocity_to_polar_velocity(float3 cartesian_position, float3 cartesian_velocity)
{
    float3 p = cartesian_position;
    float3 v = cartesian_velocity;

    /*float rdot = (p.x * v.x + p.y * v.y + p.z * v.z) / length(p);
    float tdot = (v.x * p.y - p.x * v.y) / (p.x * p.x + p.y * p.y);
    float pdot = (p.z * (p.x * v.x + p.y * v.y) - (p.x * p.x + p.y * p.y) * v.z) / ((p.x * p.x + p.y * p.y + p.z * p.z) * native_sqrt(p.x * p.x + p.y * p.y));*/

    float r = length(p);

    float repeated_eq = r * native_sqrt(1 - (p.z*p.z / (r * r)));

    float rdot = (p.x * v.x + p.y * v.y + p.z * v.z) / r;
    float tdot = ((p.z * rdot) / (r*repeated_eq)) - v.z / repeated_eq;
    float pdot = (p.x * v.y - p.y * v.x) / (p.x * p.x + p.y * p.y);

    return (float3){rdot, tdot, pdot};
}

float calculate_ds(float4 velocity, float g_metric[])
{
    float v[4] = {velocity.x, velocity.y, velocity.z, velocity.w};

    float ds = 0;

    ds += g_metric[0] * v[0] * v[0];
    ds += g_metric[1] * v[1] * v[1];
    ds += g_metric[2] * v[2] * v[2];
    ds += g_metric[3] * v[3] * v[3];

    return ds;
}

//#define IS_CONSTANT_THETA

#define GENERIC_METRIC

#if (defined(GENERIC_METRIC) && defined(GENERIC_CONSTANT_THETA)) || !defined(GENERIC_METRIC)
#define IS_CONSTANT_THETA
#endif

__kernel
void clear(__write_only image2d_t out)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x >= get_image_width(out) || y >= get_image_height(out))
        return;

    write_imagef(out, (int2){x, y}, (float4){0,0,0,1});
}

float3 rot_quat(const float3 point, float4 quat)
{
    quat = fast_normalize(quat);

    float3 t = 2.f * cross(quat.xyz, point);

    return point + quat.w * t + cross(quat.xyz, t);
}

float3 rot_quat_norm(const float3 point, float4 norm_quat)
{
    float3 t = 2.f * cross(norm_quat.xyz, point);

    return point + norm_quat.w * t + cross(norm_quat.xyz, t);
}

float3 spherical_velocity_to_cartesian_velocity(float3 p, float3 dp)
{
    float r = p.x;
    float dr = dp.x;

    float x = p.y;
    float dx = dp.y;

    float y = p.z;
    float dy = dp.z;

    float v1 = - r * sin(x) * sin(y) * dy + r * cos(x) * cos(y) * dx + sin(x) * cos(y) * dr;
    float v2 = sin(x) * sin(y) * dr + r * sin(x) * cos(y) * dy + r * cos(x) * sin(y) * dx;
    float v3 = cos(x) * dr - r * sin(x) * dx;

    return (float3){v1, v2, v3};
}

///https://www.ccs.neu.edu/home/fell/CS4300/Lectures/Ray-TracingFormulas.pdf
float3 fix_ray_position_cart(float3 cartesian_pos, float3 cartesian_velocity, float sphere_radius)
{
    cartesian_velocity = fast_normalize(cartesian_velocity);

    float3 C = (float3){0,0,0};

    float a = 1;
    float b = 2 * dot(cartesian_velocity, (cartesian_pos - C));
    float c = dot(C, C) + dot(cartesian_pos, cartesian_pos) - 2 * (dot(cartesian_pos, C)) - sphere_radius * sphere_radius;

    float discrim = b*b - 4 * a * c;

    if(discrim < 0)
        return cartesian_pos;

    float t0 = (-b - native_sqrt(discrim)) / (2 * a);
    float t1 = (-b + native_sqrt(discrim)) / (2 * a);

    float my_t = 0;

    if(fabs(t0) < fabs(t1))
        my_t = t0;
    else
        my_t = t1;

    return cartesian_pos + my_t * cartesian_velocity;
}

float3 fix_ray_position(float3 polar_pos, float3 polar_velocity, float sphere_radius, bool outwards_facing)
{
    float position_sign = sign(polar_pos.x);

    float3 cpolar_pos = polar_pos;
    cpolar_pos.x = fabs(cpolar_pos.x);

    polar_velocity.x *= position_sign;

    float3 cartesian_velocity = spherical_velocity_to_cartesian_velocity(cpolar_pos, polar_velocity);

    float3 cartesian_pos = polar_to_cartesian(cpolar_pos);

    float3 new_cart = fix_ray_position_cart(cartesian_pos, cartesian_velocity, sphere_radius);

    float3 new_polar = cartesian_to_polar(new_cart);

    #ifdef IS_CONSTANT_THETA
    new_polar.y = M_PIf/2;
    #endif // IS_CONSTANT_THETA

    new_polar.x *= position_sign;

    return new_polar;
}

float3 rotate_vector(float3 bx, float3 by, float3 bz, float3 v)
{
    /*
    [nxx, nyx, nzx,   [vx]
     nxy, nyy, nzy,   [vy]
     nxz, nyz, nzz]   [vz] =

     nxx * vx + nxy * vy + nzx * vz
     nxy * vx + nyy * vy + nzy * vz
     nxz * vx + nzy * vy + nzz * vz*/

     return (float3){
        bx.x * v.x + by.x * v.y + bz.x * v.z,
        bx.y * v.x + by.y * v.y + bz.y * v.z,
        bx.z * v.x + by.z * v.y + bz.z * v.z
    };
}

float4 rotate_vector4(float4 bx, float4 by, float4 bz, float4 bw, float4 v)
{
    return (float4)
    {
         bx.x * v.x + by.x * v.y + bz.x * v.z + bw.x * v.w,
         bx.y * v.x + by.y * v.y + bz.y * v.z + bw.y * v.w,
         bx.z * v.x + by.z * v.y + bz.z * v.z + bw.z * v.w,
         bx.w * v.x + by.w * v.y + bz.w * v.z + bw.w * v.w,
    };
}

float3 unrotate_vector(float3 bx, float3 by, float3 bz, float3 v)
{
    /*
    nxx, nxy, nxz,   vx,
    nyx, nyy, nyz,   vy,
    nzx, nzy, nzz    vz*/

    return rotate_vector((float3){bx.x, by.x, bz.x}, (float3){bx.y, by.y, bz.y}, (float3){bx.z, by.z, bz.z}, v);
}

float3 rejection(float3 my_vector, float3 basis)
{
    return normalize(my_vector - dot(my_vector, basis) * basis);
}

/*float3 srgb_to_lin(float3 C_srgb)
{
    return  0.012522878f * C_srgb +
            0.682171111f * C_srgb * C_srgb +
            0.305306011f * C_srgb * C_srgb * C_srgb;
}

float3 lin_to_srgb(float3 val)
{
    float3 S1 = native_sqrt(val);
    float3 S2 = native_sqrt(S1);
    float3 S3 = native_sqrt(S2);
    float3 sRGB = 0.585122381f * S1 + 0.783140355f * S2 - 0.368262736f * S3;

    return sRGB;
}*/

float lin_to_srgb_single(float in)
{
    if(in <= 0.0031308f)
        return in * 12.92f;
    else
        return 1.055f * pow(in, 1.0f / 2.4f) - 0.055f;
}

float3 lin_to_srgb(float3 in)
{
    return (float3)(lin_to_srgb_single(in.x), lin_to_srgb_single(in.y), lin_to_srgb_single(in.z));
}

float srgb_to_lin_single(float in)
{
    if(in < 0.04045f)
        return in / 12.92f;
    else
        return pow((in + 0.055f) / 1.055f, 2.4f);
}

float3 srgb_to_lin(float3 in)
{
    return (float3)(srgb_to_lin_single(in.x), srgb_to_lin_single(in.y), srgb_to_lin_single(in.z));
}

float lambert_w0_newton(float x)
{
    if(x < -(1 / M_E))
        x = -(1 / M_E);

    float current = 0;

    for(int i=0; i < 5; i++)
    {
        float next = current - ((current * exp(current) - x) / (exp(current) + current * exp(current)));

        current = next;
    }

    return current;
}

float lambert_w0_halley(float x)
{
    if(x < -(1 / M_E))
        x = -(1 / M_E);

    float current = 0;

    for(int i=0; i < 2; i++)
    {
        float cexp = exp(current);

        float denom = cexp * (current + 1) - ((current + 2) * (current * cexp - x) / (2 * current + 2));

        float next = current - ((current * cexp - x) / denom);

        current = next;
    }

    return current;
}

float lambert_w0_highprecision(float x)
{
    if(x < -(1 / M_E))
        x = -(1 / M_E);

    float current = 0;

    for(int i=0; i < 20; i++)
    {
        float cexp = exp(current);

        float denom = cexp * (current + 1) - ((current + 2) * (current * cexp - x) / (2 * current + 2));

        float next = current - ((current * cexp - x) / denom);

        current = next;
    }

    return current;
}

float lambert_w0(float x)
{
    return lambert_w0_halley(x);
}

float4 evaluate_partial_metric(float4 vel, float g_metric[])
{
    return (float4){g_metric[0] * vel.x * vel.x,
                    g_metric[1] * vel.y * vel.y,
                    g_metric[2] * vel.z * vel.z,
                    g_metric[3] * vel.w * vel.w};
}

float4 lower_index(float4 raised, float g_metric[])
{
    float4 ret;

    /*
    for(int i=0; i < 4; i++)
    {
        float sum = 0;

        for(int j=0; j < 4; j++)
        {
            sum += g_metric_cov[i * 4 + j] * vector[j];
        }

        ret.v[i] = sum;
    }
    */

    ret.x = g_metric[0] * raised.x;
    ret.y = g_metric[1] * raised.y;
    ret.z = g_metric[2] * raised.z;
    ret.w = g_metric[3] * raised.w;

    return ret;
}

#define ARRAY4(v) {v.x, v.y, v.z, v.w}

float4 tensor_contract(float t16[16], float4 vec)
{
    float4 res;

    res.x = t16[0 * 4 + 0] * vec.x + t16[0 * 4 + 1] * vec.y + t16[0 * 4 + 2] * vec.z + t16[0 * 4 + 3] * vec.w;
    res.y = t16[1 * 4 + 0] * vec.x + t16[1 * 4 + 1] * vec.y + t16[1 * 4 + 2] * vec.z + t16[1 * 4 + 3] * vec.w;
    res.z = t16[2 * 4 + 0] * vec.x + t16[2 * 4 + 1] * vec.y + t16[2 * 4 + 2] * vec.z + t16[2 * 4 + 3] * vec.w;
    res.w = t16[3 * 4 + 0] * vec.x + t16[3 * 4 + 1] * vec.y + t16[3 * 4 + 2] * vec.z + t16[3 * 4 + 3] * vec.w;

    return res;
}

///[0, 1, 2, 3]
///[4, 5, 6, 7]
///[8, 9, 10,11]
///[12,13,14,15]

void metric_inverse(const float m[16], float invOut[16])
{
    float inv[16], det;
    int i;

    inv[0] = m[5] * m[10] * m[15] -
             m[5] * m[11] * m[11] -
             m[6] * m[6]  * m[15] +
             m[6] * m[7]  * m[11] +
             m[7] * m[6]  * m[11] -
             m[7] * m[7]  * m[10];

    inv[1] = -m[1] * m[10] * m[15] +
              m[1] * m[11] * m[11] +
              m[6] * m[2] * m[15] -
              m[6] * m[3] * m[11] -
              m[7] * m[2] * m[11] +
              m[7] * m[3] * m[10];

    inv[5] = m[0] * m[10] * m[15] -
             m[0] * m[11] * m[11] -
             m[2] * m[2] * m[15] +
             m[2] * m[3] * m[11] +
             m[3] * m[2] * m[11] -
             m[3] * m[3] * m[10];


    inv[2] = m[1] * m[6] * m[15] -
             m[1] * m[7] * m[11] -
             m[5] * m[2] * m[15] +
             m[5] * m[3] * m[11] +
             m[7] * m[2] * m[7] -
             m[7] * m[3] * m[6];

    inv[6] = -m[0] * m[6] * m[15] +
              m[0] * m[7] * m[11] +
              m[1] * m[2] * m[15] -
              m[1] * m[3] * m[11] -
              m[3] * m[2] * m[7] +
              m[3] * m[3] * m[6];

    inv[10] = m[0] * m[5] * m[15] -
              m[0] * m[7] * m[7] -
              m[1] * m[1] * m[15] +
              m[1] * m[3] * m[7] +
              m[3] * m[1] * m[7] -
              m[3] * m[3] * m[5];

    inv[3] = -m[1] * m[6] * m[11] +
              m[1] * m[7] * m[10] +
              m[5] * m[2] * m[11] -
              m[5] * m[3] * m[10] -
              m[6] * m[2] * m[7] +
              m[6] * m[3] * m[6];

    inv[7] = m[0] * m[6] * m[11] -
             m[0] * m[7] * m[10] -
             m[1] * m[2] * m[11] +
             m[1] * m[3] * m[10] +
             m[2] * m[2] * m[7] -
             m[2] * m[3] * m[6];

    inv[11] = -m[0] * m[5] * m[11] +
               m[0] * m[7] * m[6] +
               m[1] * m[1] * m[11] -
               m[1] * m[3] * m[6] -
               m[2] * m[1] * m[7] +
               m[2] * m[3] * m[5];

    inv[15] = m[0] * m[5] * m[10] -
              m[0] * m[6] * m[6] -
              m[1] * m[1] * m[10] +
              m[1] * m[2] * m[6] +
              m[2] * m[1] * m[6] -
              m[2] * m[2] * m[5];

    inv[4] = inv[1];
    inv[8] = inv[2];
    inv[12] = inv[3];
    inv[9] = inv[6];
    inv[13] = inv[7];
    inv[14] = inv[11];

    det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

    det = 1.0 / det;

    for (i = 0; i < 16; i++)
        invOut[i] = inv[i] * det;
}

void matrix_inverse(const float m[16], float invOut[16])
{
    float inv[16], det;
    int i;

    inv[0] = m[5]  * m[10] * m[15] -
             m[5]  * m[11] * m[14] -
             m[9]  * m[6]  * m[15] +
             m[9]  * m[7]  * m[14] +
             m[13] * m[6]  * m[11] -
             m[13] * m[7]  * m[10];

    inv[4] = -m[4]  * m[10] * m[15] +
              m[4]  * m[11] * m[14] +
              m[8]  * m[6]  * m[15] -
              m[8]  * m[7]  * m[14] -
              m[12] * m[6]  * m[11] +
              m[12] * m[7]  * m[10];

    inv[8] = m[4]  * m[9] * m[15] -
             m[4]  * m[11] * m[13] -
             m[8]  * m[5] * m[15] +
             m[8]  * m[7] * m[13] +
             m[12] * m[5] * m[11] -
             m[12] * m[7] * m[9];

    inv[12] = -m[4]  * m[9] * m[14] +
               m[4]  * m[10] * m[13] +
               m[8]  * m[5] * m[14] -
               m[8]  * m[6] * m[13] -
               m[12] * m[5] * m[10] +
               m[12] * m[6] * m[9];

    inv[1] = -m[1]  * m[10] * m[15] +
              m[1]  * m[11] * m[14] +
              m[9]  * m[2] * m[15] -
              m[9]  * m[3] * m[14] -
              m[13] * m[2] * m[11] +
              m[13] * m[3] * m[10];

    inv[5] = m[0]  * m[10] * m[15] -
             m[0]  * m[11] * m[14] -
             m[8]  * m[2] * m[15] +
             m[8]  * m[3] * m[14] +
             m[12] * m[2] * m[11] -
             m[12] * m[3] * m[10];

    inv[9] = -m[0]  * m[9] * m[15] +
              m[0]  * m[11] * m[13] +
              m[8]  * m[1] * m[15] -
              m[8]  * m[3] * m[13] -
              m[12] * m[1] * m[11] +
              m[12] * m[3] * m[9];

    inv[13] = m[0]  * m[9] * m[14] -
              m[0]  * m[10] * m[13] -
              m[8]  * m[1] * m[14] +
              m[8]  * m[2] * m[13] +
              m[12] * m[1] * m[10] -
              m[12] * m[2] * m[9];

    inv[2] = m[1]  * m[6] * m[15] -
             m[1]  * m[7] * m[14] -
             m[5]  * m[2] * m[15] +
             m[5]  * m[3] * m[14] +
             m[13] * m[2] * m[7] -
             m[13] * m[3] * m[6];

    inv[6] = -m[0]  * m[6] * m[15] +
              m[0]  * m[7] * m[14] +
              m[4]  * m[2] * m[15] -
              m[4]  * m[3] * m[14] -
              m[12] * m[2] * m[7] +
              m[12] * m[3] * m[6];

    inv[10] = m[0]  * m[5] * m[15] -
              m[0]  * m[7] * m[13] -
              m[4]  * m[1] * m[15] +
              m[4]  * m[3] * m[13] +
              m[12] * m[1] * m[7] -
              m[12] * m[3] * m[5];

    inv[14] = -m[0]  * m[5] * m[14] +
               m[0]  * m[6] * m[13] +
               m[4]  * m[1] * m[14] -
               m[4]  * m[2] * m[13] -
               m[12] * m[1] * m[6] +
               m[12] * m[2] * m[5];

    inv[3] = -m[1] * m[6] * m[11] +
              m[1] * m[7] * m[10] +
              m[5] * m[2] * m[11] -
              m[5] * m[3] * m[10] -
              m[9] * m[2] * m[7] +
              m[9] * m[3] * m[6];

    inv[7] = m[0] * m[6] * m[11] -
             m[0] * m[7] * m[10] -
             m[4] * m[2] * m[11] +
             m[4] * m[3] * m[10] +
             m[8] * m[2] * m[7] -
             m[8] * m[3] * m[6];

    inv[11] = -m[0] * m[5] * m[11] +
               m[0] * m[7] * m[9] +
               m[4] * m[1] * m[11] -
               m[4] * m[3] * m[9] -
               m[8] * m[1] * m[7] +
               m[8] * m[3] * m[5];

    inv[15] = m[0] * m[5] * m[10] -
              m[0] * m[6] * m[9] -
              m[4] * m[1] * m[10] +
              m[4] * m[2] * m[9] +
              m[8] * m[1] * m[6] -
              m[8] * m[2] * m[5];

    det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

    det = 1.0 / det;

    for(i = 0; i < 16; i++)
        invOut[i] = inv[i] * det;
}

void get_christoffel_generic(float g_metric_generic[], float g_partials_generic[], float christoff[64])
{
    #ifndef GENERIC_BIG_METRIC
    ///diagonal of the metric, because it only has diagonals
    float g_inv[4] = {1/g_metric_generic[0], 1/g_metric_generic[1], 1/g_metric_generic[2], 1/g_metric_generic[3]};

    {
        #pragma unroll
        for(int i=0; i < 4; i++)
        {
            float ginvii = 0.5 * g_inv[i];

            #pragma unroll
            for(int m=0; m < 4; m++)
            {
                float adding = ginvii * g_partials_generic[i * 4 + m];

                christoff[i * 16 + i * 4 + m] += adding;
                christoff[i * 16 + m * 4 + i] += adding;
                christoff[i * 16 + m * 4 + m] -= ginvii * g_partials_generic[m * 4 + i];
            }
        }
    }
    #else
    float g_inv_big[16] = {0};
    metric_inverse(g_metric_generic, g_inv_big);

    #pragma unroll
    for(int i = 0; i < 4; i++)
    {
        #pragma unroll
        for(int k = 0; k < 4; k++)
        {
            #pragma unroll
            for(int l = 0; l < 4; l++)
            {
                float sum = 0;

                #pragma unroll
                for (int m = 0; m < 4; m++)
                {
                    sum += g_inv_big[i * 4 + m] * g_partials_generic[l * 16 + m * 4 + k];
                    sum += g_inv_big[i * 4 + m] * g_partials_generic[k * 16 + m * 4 + l];
                    sum -= g_inv_big[i * 4 + m] * g_partials_generic[m * 16 + k * 4 + l];
                }

                christoff[i * 16 + k * 4 + l] = 0.5f * sum;
            }
        }
    }
    #endif
}

float4 calculate_acceleration(float4 lightray_velocity, float g_metric[4], float g_partials[16])
{
    #ifdef IS_CONSTANT_THETA
    lightray_velocity.z = 0;
    #endif // IS_CONSTANT_THETA

    float christoff[64] = {0};

    ///diagonal of the metric, because it only has diagonals
    float g_inv[4] = {1/g_metric[0], 1/g_metric[1], 1/g_metric[2], 1/g_metric[3]};

    {
        #pragma unroll
        for(int i=0; i < 4; i++)
        {
            float ginvii = 0.5 * g_inv[i];

            #pragma unroll
            for(int m=0; m < 4; m++)
            {
                float adding = ginvii * g_partials[i * 4 + m];

                christoff[i * 16 + i * 4 + m] += adding;
                christoff[i * 16 + m * 4 + i] += adding;
                christoff[i * 16 + m * 4 + m] -= ginvii * g_partials[m * 4 + i];
            }
        }
    }

    float velocity_arr[4] = {lightray_velocity.x, lightray_velocity.y, lightray_velocity.z, lightray_velocity.w};

    ///no performance benefit by unrolling u into a float4
    float christ_result[4] = {0,0,0,0};

    #pragma unroll
    for(int uu=0; uu < 4; uu++)
    {
        float sum = 0;

        #pragma unroll
        for(int aa = 0; aa < 4; aa++)
        {
            #pragma unroll
            for(int bb = 0; bb < 4; bb++)
            {
                sum += (velocity_arr[aa]) * (velocity_arr[bb]) * christoff[uu * 16 + aa*4 + bb*1];
            }
        }

        christ_result[uu] = sum;
    }

    float4 acceleration = {-christ_result[0], -christ_result[1], -christ_result[2], -christ_result[3]};

    #ifdef IS_CONSTANT_THETA
    acceleration.z = 0;
    #endif // IS_CONSTANT_THETA

    return acceleration;
}

float linear_mix(float value, float min_val, float max_val)
{
    value = clamp(value, min_val, max_val);

    return (value - min_val) / (max_val - min_val);
}

float linear_val(float value, float min_val, float max_val, float val_at_min, float val_at_max)
{
    float mixd = linear_mix(value, min_val, max_val);

    return mix(val_at_min, val_at_max, mixd);
}

struct lightray
{
    float4 position;
    float4 velocity;
    float4 initial_quat;
    float4 acceleration;
    float ku_uobsu;
    float running_dlambda_dnew;
    int terminated;
    int sx;
    int sy;
};

#ifdef GENERIC_METRIC

float4 lower_index_big(float4 vec, float g_metric_big[])
{
    float vecarray[4] = {vec.x, vec.y, vec.z, vec.w};
    float ret[4] = {0,0,0,0};

    #pragma unroll
    for(int i=0; i < 4; i++)
    {
        float sum = 0;

        #pragma unroll
        for(int j=0; j < 4; j++)
        {
            sum += g_metric_big[i * 4 + j] * vecarray[j];
        }

        ret[i] = sum;
    }

    return (float4)(ret[0], ret[1], ret[2], ret[3]);
}

float4 raise_index_big(float4 vec, float g_metric_big_inv[])
{
    float vecarray[4] = {vec.x, vec.y, vec.z, vec.w};
    float ret[4] = {0,0,0,0};

    #pragma unroll
    for(int i=0; i < 4; i++)
    {
        float sum = 0;

        #pragma unroll
        for(int j=0; j < 4; j++)
        {
            sum += g_metric_big_inv[i * 4 + j] * vecarray[j];
        }

        ret[i] = sum;
    }

    return (float4)(ret[0], ret[1], ret[2], ret[3]);
}

float4 raise_index(float4 vec, float g_metric_inv[])
{
    float4 ret;

    ret.x = vec.x * g_metric_inv[0];
    ret.y = vec.y * g_metric_inv[1];
    ret.z = vec.z * g_metric_inv[2];
    ret.w = vec.w * g_metric_inv[3];

    return ret;
}

float4 raise_index_generic(float4 vec, float g_metric_inv[])
{
    #ifdef GENERIC_BIG_METRIC
    return raise_index_big(vec, g_metric_inv);
    #else
    return raise_index(vec, g_metric_inv);
    #endif // GENERIC_BIG_METRIC
}

float4 lower_index_generic(float4 vec, float g_metric[])
{
    #ifdef GENERIC_BIG_METRIC
    return lower_index_big(vec, g_metric);
    #else
    return lower_index(vec, g_metric);
    #endif // GENERIC_BIG_METRIC
}

float dot_product_big(float4 u, float4 v, float g_metric_big[])
{
    float4 lowered = lower_index_big(u, g_metric_big);

    return dot(lowered, v);
}

float dot_product_generic(float4 u, float4 v, float g_metric[])
{
    #ifdef GENERIC_BIG_METRIC
    return dot_product_big(u, v, g_metric);
    #else
    float4 lowered = lower_index(u, g_metric);

    return dot(lowered, v);
    #endif // GENERIC_BIG_METRIC
}

void small_to_big_metric(float g_metric[], float g_metric_big[])
{
    g_metric_big[0]         = g_metric[0];
    g_metric_big[1 * 4 + 1] = g_metric[1];
    g_metric_big[2 * 4 + 2] = g_metric[2];
    g_metric_big[3 * 4 + 3] = g_metric[3];
}

void small_to_big_partials(float g_metric_partials[], float g_metric_partials_big[])
{
    ///with respect to, ie the differentiating variable
    for(int wrt = 0; wrt < 4; wrt++)
    {
        for(int variable = 0; variable < 4; variable++)
        {
            g_metric_partials_big[wrt * 16 + variable * 4 + variable] = g_metric_partials[variable * 4 + wrt];
        }
    }
}

struct dynamic_config
{
    #ifdef DYNVARS
    float DYNVARS;
    #endif // DYNVARS
};

struct dynamic_feature_config
{
    #ifdef DYNAMIC_FLOAT_FEATURES
    float DYNAMIC_FLOAT_FEATURES;
    #endif // DYNAMIC_FLOAT_FEATURES

    #ifdef DYNAMIC_BOOL_FEATURES
    int DYNAMIC_BOOL_FEATURES;
    #endif // DYNAMIC_BOOL_FEATURES
};

#ifdef KERNEL_IS_STATIC
#define GET_FEATURE(name, dfg) FEATURE_##name
#endif // KERNEL_IS_STATIC

#ifdef KERNEL_IS_DYNAMIC
#define GET_FEATURE(name, dfg) dfg->name
#endif // KERNEL_IS_DYNAMIC

#define dynamic_config_space __constant

#ifndef GENERIC_BIG_METRIC
void calculate_metric_generic(float4 spacetime_position, float g_metric_out[], dynamic_config_space const struct dynamic_config* cfg)
{
    float v1 = spacetime_position.x;
    float v2 = spacetime_position.y;
    float v3 = spacetime_position.z;
    float v4 = spacetime_position.w;

    float rs = RS_IMPL;
    float c = C_IMPL;

    float TEMPORARIES0;

    g_metric_out[0] = F1_I;
    g_metric_out[1] = F2_I;
    g_metric_out[2] = F3_I;
    g_metric_out[3] = F4_I;
}

void calculate_partial_derivatives_generic(float4 spacetime_position, float g_metric_partials[], dynamic_config_space const struct dynamic_config* cfg)
{
    float v1 = spacetime_position.x;
    float v2 = spacetime_position.y;
    float v3 = spacetime_position.z;
    float v4 = spacetime_position.w;

    float rs = RS_IMPL;
    float c = C_IMPL;

    float TEMPORARIES0;

    g_metric_partials[0] = F1_P;
    g_metric_partials[1] = F2_P;
    g_metric_partials[2] = F3_P;
    g_metric_partials[3] = F4_P;
    g_metric_partials[4] = F5_P;
    g_metric_partials[5] = F6_P;
    g_metric_partials[6] = F7_P;
    g_metric_partials[7] = F8_P;
    g_metric_partials[8] = F9_P;
    g_metric_partials[9] = F10_P;
    g_metric_partials[10] = F11_P;
    g_metric_partials[11] = F12_P;
    g_metric_partials[12] = F13_P;
    g_metric_partials[13] = F14_P;
    g_metric_partials[14] = F15_P;
    g_metric_partials[15] = F16_P;
}
#endif // GENERIC_BIG_METRIC

///[0, 1, 2, 3]
///[4, 5, 6, 7]
///[8, 9, 10,11]
///[12,13,14,15]
#ifdef GENERIC_BIG_METRIC
void calculate_metric_generic_big(float4 spacetime_position, float g_metric_out[], dynamic_config_space const struct dynamic_config* cfg)
{
    float v1 = spacetime_position.x;
    float v2 = spacetime_position.y;
    float v3 = spacetime_position.z;
    float v4 = spacetime_position.w;

    float rs = RS_IMPL;
    float c = C_IMPL;

    float TEMPORARIES0;

    g_metric_out[0] = F1_I;
    g_metric_out[1] = F2_I;
    g_metric_out[2] = F3_I;
    g_metric_out[3] = F4_I;
    g_metric_out[4] = g_metric_out[1];
    g_metric_out[5] = F6_I;
    g_metric_out[6] = F7_I;
    g_metric_out[7] = F8_I;
    g_metric_out[8] = g_metric_out[2];
    g_metric_out[9] = g_metric_out[6];
    g_metric_out[10] = F11_I;
    g_metric_out[11] = F12_I;
    g_metric_out[12] = g_metric_out[3];
    g_metric_out[13] = g_metric_out[7];
    g_metric_out[14] = g_metric_out[11];
    g_metric_out[15] = F16_I;
}

void calculate_partial_derivatives_generic_big(float4 spacetime_position, float g_metric_partials[], dynamic_config_space const struct dynamic_config* cfg)
{
    //#define NUMERICAL_DIFFERENTIATION
    #ifdef NUMERICAL_DIFFERENTIATION
    float g_metric[16] = {0};

    {
         calculate_metric_generic_big(spacetime_position, g_metric, cfg);
    }

    #pragma unroll(4)
    for(int d = 0; d < 4; d++)
    {
        float g_metric_m[16] = {0};

        float eps = 1e-3f;

        float4 evector = (float4)(0,0,0,0);

        if(d == 0)
            evector.x = eps;
        if(d == 1)
            evector.y = eps;
        if(d == 2)
            evector.z = eps;
        if(d == 3)
            evector.w = eps;

        calculate_metric_generic_big(spacetime_position + evector, g_metric_m, cfg);

        for(int k=0; k < 16; k++)
        {
            g_metric_partials[d * 16 + k] = (g_metric_m[k] - g_metric[k]) / eps;
        }
    }
    #endif

    #define ANALYTIC_DIFFERENTIATION
    #ifdef ANALYTIC_DIFFERENTIATION
    float v1 = spacetime_position.x;
    float v2 = spacetime_position.y;
    float v3 = spacetime_position.z;
    float v4 = spacetime_position.w;

    float rs = RS_IMPL;
    float c = C_IMPL;

    float TEMPORARIES0;

    g_metric_partials[0 * 16 + 0] = F1_P;
    g_metric_partials[0 * 16 + 1] = F2_P;
    g_metric_partials[0 * 16 + 2] = F3_P;
    g_metric_partials[0 * 16 + 3] = F4_P;
    //g_metric_partials[0 * 16 + 4] = F5_P;
    g_metric_partials[0 * 16 + 5] = F6_P;
    g_metric_partials[0 * 16 + 6] = F7_P;
    g_metric_partials[0 * 16 + 7] = F8_P;
    //g_metric_partials[0 * 16 + 8] = F9_P;
    //g_metric_partials[0 * 16 + 9] = F10_P;
    g_metric_partials[0 * 16 + 10] = F11_P;
    g_metric_partials[0 * 16 + 11] = F12_P;
    //g_metric_partials[0 * 16 + 12] = F13_P;
    //g_metric_partials[0 * 16 + 13] = F14_P;
    //g_metric_partials[0 * 16 + 14] = F15_P;
    g_metric_partials[0 * 16 + 15] = F16_P;

    g_metric_partials[1 * 16 + 0] = F17_P;
    g_metric_partials[1 * 16 + 1] = F18_P;
    g_metric_partials[1 * 16 + 2] = F19_P;
    g_metric_partials[1 * 16 + 3] = F20_P;
    //g_metric_partials[1 * 16 + 4] = F21_P;
    g_metric_partials[1 * 16 + 5] = F22_P;
    g_metric_partials[1 * 16 + 6] = F23_P;
    g_metric_partials[1 * 16 + 7] = F24_P;
    //g_metric_partials[1 * 16 + 8] = F25_P;
    //g_metric_partials[1 * 16 + 9] = F26_P;
    g_metric_partials[1 * 16 + 10] = F27_P;
    g_metric_partials[1 * 16 + 11] = F28_P;
    //g_metric_partials[1 * 16 + 12] = F29_P;
    //g_metric_partials[1 * 16 + 13] = F30_P;
    //g_metric_partials[1 * 16 + 14] = F31_P;
    g_metric_partials[1 * 16 + 15] = F32_P;

    g_metric_partials[2 * 16 + 0] = F33_P;
    g_metric_partials[2 * 16 + 1] = F34_P;
    g_metric_partials[2 * 16 + 2] = F35_P;
    g_metric_partials[2 * 16 + 3] = F36_P;
    //g_metric_partials[2 * 16 + 4] = F37_P;
    g_metric_partials[2 * 16 + 5] = F38_P;
    g_metric_partials[2 * 16 + 6] = F39_P;
    g_metric_partials[2 * 16 + 7] = F40_P;
    //g_metric_partials[2 * 16 + 8] = F41_P;
    //g_metric_partials[2 * 16 + 9] = F42_P;
    g_metric_partials[2 * 16 + 10] = F43_P;
    g_metric_partials[2 * 16 + 11] = F44_P;
    //g_metric_partials[2 * 16 + 12] = F45_P;
    //g_metric_partials[2 * 16 + 13] = F46_P;
    //g_metric_partials[2 * 16 + 14] = F47_P;
    g_metric_partials[2 * 16 + 15] = F48_P;

    g_metric_partials[3 * 16 + 0] = F49_P;
    g_metric_partials[3 * 16 + 1] = F50_P;
    g_metric_partials[3 * 16 + 2] = F51_P;
    g_metric_partials[3 * 16 + 3] = F52_P;
    //g_metric_partials[3 * 16 + 4] = F53_P;
    g_metric_partials[3 * 16 + 5] = F54_P;
    g_metric_partials[3 * 16 + 6] = F55_P;
    g_metric_partials[3 * 16 + 7] = F56_P;
    //g_metric_partials[3 * 16 + 8] = F57_P;
    //g_metric_partials[3 * 16 + 9] = F58_P;
    g_metric_partials[3 * 16 + 10] = F59_P;
    g_metric_partials[3 * 16 + 11] = F60_P;
    //g_metric_partials[3 * 16 + 12] = F61_P;
    //g_metric_partials[3 * 16 + 13] = F62_P;
    //g_metric_partials[3 * 16 + 14] = F63_P;
    g_metric_partials[3 * 16 + 15] = F64_P;
    #endif

    g_metric_partials[0 * 16 + 4] = g_metric_partials[0 * 16 + 1];
    g_metric_partials[0 * 16 + 8] = g_metric_partials[0 * 16 + 2];
    g_metric_partials[0 * 16 + 12] = g_metric_partials[0 * 16 + 3];
    g_metric_partials[0 * 16 + 9] = g_metric_partials[0 * 16 + 6];
    g_metric_partials[0 * 16 + 13] = g_metric_partials[0 * 16 + 7];
    g_metric_partials[0 * 16 + 14] = g_metric_partials[0 * 16 + 11];

    g_metric_partials[1 * 16 + 4] = g_metric_partials[1 * 16 + 1];
    g_metric_partials[1 * 16 + 8] = g_metric_partials[1 * 16 + 2];
    g_metric_partials[1 * 16 + 12] = g_metric_partials[1 * 16 + 3];
    g_metric_partials[1 * 16 + 9] = g_metric_partials[1 * 16 + 6];
    g_metric_partials[1 * 16 + 13] = g_metric_partials[1 * 16 + 7];
    g_metric_partials[1 * 16 + 14] = g_metric_partials[1 * 16 + 11];

    g_metric_partials[2 * 16 + 4] = g_metric_partials[2 * 16 + 1];
    g_metric_partials[2 * 16 + 8] = g_metric_partials[2 * 16 + 2];
    g_metric_partials[2 * 16 + 12] = g_metric_partials[2 * 16 + 3];
    g_metric_partials[2 * 16 + 9] = g_metric_partials[2 * 16 + 6];
    g_metric_partials[2 * 16 + 13] = g_metric_partials[2 * 16 + 7];
    g_metric_partials[2 * 16 + 14] = g_metric_partials[2 * 16 + 11];

    g_metric_partials[3 * 16 + 4] = g_metric_partials[3 * 16 + 1];
    g_metric_partials[3 * 16 + 8] = g_metric_partials[3 * 16 + 2];
    g_metric_partials[3 * 16 + 12] = g_metric_partials[3 * 16 + 3];
    g_metric_partials[3 * 16 + 9] = g_metric_partials[3 * 16 + 6];
    g_metric_partials[3 * 16 + 13] = g_metric_partials[3 * 16 + 7];
    g_metric_partials[3 * 16 + 14] = g_metric_partials[3 * 16 + 11];

}
#endif // GENERIC_BIG_METRIC

float4 generic_to_spherical(float4 in, dynamic_config_space const struct dynamic_config* cfg)
{
    float v1 = in.x;
    float v2 = in.y;
    float v3 = in.z;
    float v4 = in.w;

    float o1 = TO_COORD1;
    float o2 = TO_COORD2;
    float o3 = TO_COORD3;
    float o4 = TO_COORD4;

    return (float4)(o1, o2, o3, o4);
}

float4 generic_velocity_to_spherical_velocity(float4 in, float4 inv, dynamic_config_space const struct dynamic_config* cfg)
{
    float v1 = in.x;
    float v2 = in.y;
    float v3 = in.z;
    float v4 = in.w;

    float dv1 = inv.x;
    float dv2 = inv.y;
    float dv3 = inv.z;
    float dv4 = inv.w;

    float o1 = TO_DCOORD1;
    float o2 = TO_DCOORD2;
    float o3 = TO_DCOORD3;
    float o4 = TO_DCOORD4;

    return (float4)(o1, o2, o3, o4);
}

float4 spherical_to_generic(float4 in, dynamic_config_space const struct dynamic_config* cfg)
{
    float v1 = in.x;
    float v2 = in.y;
    float v3 = in.z;
    float v4 = in.w;

    float o1 = FROM_COORD1;
    float o2 = FROM_COORD2;
    float o3 = FROM_COORD3;
    float o4 = FROM_COORD4;

    return (float4)(o1, o2, o3, o4);
}

float4 spherical_velocity_to_generic_velocity(float4 in, float4 inv, dynamic_config_space const struct dynamic_config* cfg)
{
    float v1 = in.x;
    float v2 = in.y;
    float v3 = in.z;
    float v4 = in.w;

    float dv1 = inv.x;
    float dv2 = inv.y;
    float dv3 = inv.z;
    float dv4 = inv.w;

    float o1 = FROM_DCOORD1;
    float o2 = FROM_DCOORD2;
    float o3 = FROM_DCOORD3;
    float o4 = FROM_DCOORD4;

    return (float4)(o1, o2, o3, o4);
}

float4 generic_to_cartesian(float4 in, dynamic_config_space const struct dynamic_config* cfg)
{
    float4 spherical = generic_to_spherical(in, cfg);

    return (float4)(spherical.x, polar_to_cartesian(spherical.yzw));
}

float4 generic_velocity_to_cartesian_velocity(float4 in, float4 in_v, dynamic_config_space const struct dynamic_config* cfg)
{
    float4 spherical = generic_to_spherical(in, cfg);
    float4 spherical_v = generic_velocity_to_spherical_velocity(in, in_v, cfg);

    return (float4)(spherical_v.x, spherical_velocity_to_cartesian_velocity(spherical.yzw, spherical_v.yzw));
}

float4 cartesian_to_generic(float4 in, dynamic_config_space const struct dynamic_config* cfg)
{
    float3 polar = cartesian_to_polar(in.yzw);

    return spherical_to_generic((float4)(in.x, polar), cfg);
}

float4 cartesian_velocity_to_generic_velocity(float4 in, float4 in_v, dynamic_config_space const struct dynamic_config* cfg)
{
    float3 polar = cartesian_to_polar(in.yzw);
    float3 polar_v = cartesian_velocity_to_polar_velocity(in.yzw, in_v.yzw);

    return spherical_velocity_to_generic_velocity((float4)(in.x, polar), (float4)(in_v.x, polar_v), cfg);
}

float3 cartesian_to_spherical_g(float3 v)
{
    float v1 = 0;
    float v2 = v.x;
    float v3 = v.y;
    float v4 = v.z;

    float o1 = 0;
    float o2 = CART_TO_POL1;
    float o3 = CART_TO_POL2;
    float o4 = CART_TO_POL3;

    return (float3)(o2, o3, o4);
}

float3 cartesian_velocity_to_spherical_velocity_g(float3 v, float3 inv)
{
    float v1 = 0;
    float v2 = v.x;
    float v3 = v.y;
    float v4 = v.z;

    float dv1 = 0;
    float dv2 = inv.x;
    float dv3 = inv.y;
    float dv4 = inv.z;

    float o1 = CART_TO_POL_D0;
    float o2 = CART_TO_POL_D1;
    float o3 = CART_TO_POL_D2;
    float o4 = CART_TO_POL_D3;

    return (float3)(o2, o3, o4);
}

///This function makes no sense. Why does it take an argument? Its literally unused
float4 get_coordinate_period(dynamic_config_space const struct dynamic_config* cfg)
{
    #ifdef HAS_COORDINATE_PERIODICITY
    float v1 = 0;
    float v2 = 0;
    float v3 = 0;
    float v4 = 0;

    float o1 = COORDINATE_PERIODICITY1;
    float o2 = COORDINATE_PERIODICITY2;
    float o3 = COORDINATE_PERIODICITY3;
    float o4 = COORDINATE_PERIODICITY4;

    return (float4)(o1, o2, o3, o4);
    #else
    return (float4)(0,0,0,0);
    #endif // HAS_COORDINATE_PERIODICITY
}

float positive_fmod(float a, float b)
{
    /*float v = fmod(a, b);

    if(v < 0.f)
       v += b;

    return v;*/

    return a - floor(a/b)*b;
}

float4 positive_fmod4(float4 a, float4 b)
{
    return (float4)(positive_fmod(a.x, b.x),
                    positive_fmod(a.y, b.y),
                    positive_fmod(a.z, b.z),
                    positive_fmod(a.w, b.w));
}

float4 handle_coordinate_periodicity(float4 in, dynamic_config_space const struct dynamic_config* cfg)
{
    float4 periods = get_coordinate_period(cfg);

    if(periods.x != 0)
        in.x = positive_fmod(in.x, periods.x);
    if(periods.y != 0)
        in.y = positive_fmod(in.y, periods.y);
    if(periods.z != 0)
        in.z = positive_fmod(in.z, periods.z);
    if(periods.w != 0)
        in.w = positive_fmod(in.w, periods.w);

    return in;
}

float stable_quad(float a, float d, float k, float sign)
{
    if(k <= 4.38072748497961 * pow(10.f, 16.f))
        return -(k + copysign(native_sqrt((4 * a) * d + k * k), sign)) / (a * 2);

    return -k / a;
}

float4 fix_light_velocity_big(float4 v, float g_metric_big[])
{
    return v;

    float4 c = tensor_contract(g_metric_big, v);

    ///dot(c, v) = 0
    ///c.x * v.x + c.y * v.y + c.z * v.z + c.w * v.w = 0
    float4 r = v;

    ///c.x * v.x = -c.y * v.y + c.z * v.z + c.w * v.w
    ///so, tensor contracting the t component to get c is
    ///g_metric_big[0] * v.x + g_metric_big[1] * v.y + g_metric_big[2] * v.z + g_metric_big[3] * v.w

    //(g_metric_big[0] * v.x + g_metric_big[1] * v.y + g_metric_big[2] * v.z + g_metric_big[3] * v.w) * v.x = -(c.y * v.y + c.z * v.z + c.w * v.w)

    float k = g_metric_big[1] * v.y + g_metric_big[2] * v.z + g_metric_big[3] * v.w;
    float d = -(c.y * v.y + c.z * v.z + c.w * v.w);
    float a = g_metric_big[0];

    float nx = r.x;

    float inner = 4 * a * d + k * k;

    if(inner < 0)
        return v;

    if(fabs(a) > 0.1)
        nx = stable_quad(a, d, k, v.x);
    else
        return v;

    /*if(isnan(nx) && !(any(isnan(v))))
    {
        printf("A %f d %f k %f %f %f %f %f  %f %f %f %f", a, d, k, c.x, c.y, c.z, c.w, g_metric_big[0], g_metric_big[5], g_metric_big[10], g_metric_big[15]);
    }*/

    r.x = nx;

    return r;
}

float4 calculate_acceleration_big(float4 lightray_velocity, float g_metric_big[16], float g_partials_big[64])
{
    #ifdef IS_CONSTANT_THETA
    lightray_velocity.z = 0;
    #endif // IS_CONSTANT_THETA

    float christoff[64] = {0};

    float g_inv_big[16] = {0};

    metric_inverse(g_metric_big, g_inv_big);

    #pragma unroll
    for(int i = 0; i < 4; i++)
    {
        #pragma unroll
        for(int k = 0; k < 4; k++)
        {
            #pragma unroll
            for(int l = 0; l < 4; l++)
            {
                float sum = 0;

                #pragma unroll
                for (int m = 0; m < 4; m++)
                {
                    sum += g_inv_big[i * 4 + m] * g_partials_big[l * 16 + m * 4 + k];
                    sum += g_inv_big[i * 4 + m] * g_partials_big[k * 16 + m * 4 + l];
                    sum -= g_inv_big[i * 4 + m] * g_partials_big[m * 16 + k * 4 + l];
                }

                christoff[i * 16 + k * 4 + l] = 0.5f * sum;
            }
        }
    }

    float velocity_arr[4] = {lightray_velocity.x, lightray_velocity.y, lightray_velocity.z, lightray_velocity.w};

    ///no performance benefit by unrolling u into a float4
    float christ_result[4] = {0,0,0,0};

    #define TIME_IS_AFFINE_TIME
    #ifdef TIME_IS_AFFINE_TIME
    #pragma unroll
    for(int uu=0; uu < 4; uu++)
    {
        float sum = 0;
        #pragma unroll
        for(int aa = 0; aa < 4; aa++)
        {
            #pragma unroll
            for(int bb = 0; bb < 4; bb++)
            {
                sum += velocity_arr[aa] * velocity_arr[bb] * christoff[uu * 16 + aa*4 + bb*1];
            }
        }

        christ_result[uu] = -sum;
    }
    #endif // TIME_IS_COORDINATE_TIME

    ///this doesn't seem to work all that well, especially not for eg kerr
    ///I think more in depth changes are needed, eg B.8 https://www.researchgate.net/figure/View-of-a-static-observer-located-at-x-0-y-4-in-the-positive-y-direction-for-t_fig2_225428633
    ///that said... while the paper says that numerical integration of the alcubierre edge is difficult due to an extremely low ds, I have had no problems with ds being too small
    ///hmm, looking back that's the constraint equation
    ///So, if we set up a ray with a velocity which is dPosition / dAffine, it cannot possibly be correct to simply
    ///treat it as a velocity in coordinate time
    //#define TIME_IS_COORDINATE_TIME
    #ifdef TIME_IS_COORDINATE_TIME
    #pragma unroll
    for(int uu=0; uu < 4; uu++)
    {
        float sum = 0;
        #pragma unroll
        for(int aa = 0; aa < 4; aa++)
        {
            #pragma unroll
            for(int bb = 0; bb < 4; bb++)
            {
                sum += -velocity_arr[aa] * velocity_arr[bb] * christoff[uu * 16 + aa*4 + bb*1] + christoff[0 * 16 + aa * 4 + bb] * velocity_arr[aa] * velocity_arr[bb] * velocity_arr[uu];
            }
        }

        christ_result[uu] = sum;
    }
    #endif // TIME_IS_COORDINATE_TIME

    float4 acceleration = {christ_result[0], christ_result[1], christ_result[2], christ_result[3]};

    #ifdef IS_CONSTANT_THETA
    acceleration.z = 0;
    #endif // IS_CONSTANT_THETA

    return acceleration;
}

float3 project(float3 u, float3 v)
{
    return (dot(u, v) / dot(u, u)) * u;
}

struct ortho_result
{
    float3 v1, v2, v3;
};

struct ortho_result orthonormalise(float3 i1, float3 i2, float3 i3)
{
    float3 u1 = i1;
    float3 u2 = i2;
    float3 u3 = i3;

    u2 = u2 - project(u1, u2);

    u3 = u3 - project(u1, u3);
    u3 = u3 - project(u2, u3);

    struct ortho_result result;
    result.v1 = normalize(u1);
    result.v2 = normalize(u2);
    result.v3 = normalize(u3);

    return result;
};

struct ortho_result4
{
    float3 v1, v2, v3, v4;
};

struct ortho_result4 orthonormalise4(float3 i1, float3 i2, float3 i3, float3 i4)
{
    float3 u1 = i1;
    float3 u2 = i2;
    float3 u3 = i3;
    float3 u4 = i4;

    u2 = u2 - project(u1, u2);

    u3 = u3 - project(u1, u3);
    u3 = u3 - project(u2, u3);

    u4 = u4 - project(u1, u4);
    u4 = u4 - project(u2, u4);
    u4 = u4 - project(u3, u4);

    struct ortho_result4 result;
    result.v1 = normalize(u1);
    result.v2 = normalize(u2);
    result.v3 = normalize(u3);
    result.v4 = normalize(u4);

    return result;
};

struct frame_basis
{
    float4 v1;
    float4 v2;
    float4 v3;
    float4 v4;
    int timelike_coordinate;
};

struct orthonormal_basis
{
    float4 v1;
    float4 v2;
    float4 v3;
    float4 v4;
};

float4 gram_proj(float4 u, float4 v, float big_metric[])
{
    float top = dot_product_big(u, v, big_metric);
    float bottom = dot_product_big(u, u, big_metric);

    return (top / bottom) * u;
}

float4 normalize_big_metric(float4 in, float big_metric[])
{
    float d = dot_product_big(in, in, big_metric);

    return in / native_sqrt(fabs(d));
}

float4 gram_proj_big(float4 u, float4 v, float metric[])
{
    float top = dot_product_big(u, v, metric);
    float bottom = dot_product_big(u, u, metric);

    return (top / bottom) * u;
}

float4 normalise_big(float4 in, float big_metric[])
{
    float d = dot_product_big(in, in, big_metric);

    return in / native_sqrt(fabs(d));
}

///i1-4 are raised
///this doesn't handle diagonal matrices!!!
struct orthonormal_basis orthonormalise4_metric(float4 i1, float4 i2, float4 i3, float4 i4, float big_metric[])
{
    float4 u1 = i1;

    float4 u2 = i2;
    u2 = u2 - gram_proj_big(u1, u2, big_metric);

    float4 u3 = i3;
    u3 = u3 - gram_proj_big(u1, u3, big_metric);
    u3 = u3 - gram_proj_big(u2, u3, big_metric);

    float4 u4 = i4;
    u4 = u4 - gram_proj_big(u1, u4, big_metric);
    u4 = u4 - gram_proj_big(u2, u4, big_metric);
    u4 = u4 - gram_proj_big(u3, u4, big_metric);

    u1 = normalise_big(u1, big_metric);
    u2 = normalise_big(u2, big_metric);
    u3 = normalise_big(u3, big_metric);
    u4 = normalise_big(u4, big_metric);

    struct orthonormal_basis ret;
    ret.v1 = u1;
    ret.v2 = u2;
    ret.v3 = u3;
    ret.v4 = u4;

    return ret;
};

void print_metric_big(float g_metric_big[])
{
    printf("%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n", g_metric_big[0], g_metric_big[1], g_metric_big[2], g_metric_big[3],
           g_metric_big[4], g_metric_big[5], g_metric_big[6], g_metric_big[7],
           g_metric_big[8], g_metric_big[9], g_metric_big[10], g_metric_big[11],
           g_metric_big[12], g_metric_big[13], g_metric_big[14], g_metric_big[15]);
}

///specifically: cartesian minkowski
void get_local_minkowski(float4 e0_hi, float4 e1_hi, float4 e2_hi, float4 e3_hi, float g_metric_big[], float minkowski[])
{
    ///a * 4 + mu
    float m[16] = {e0_hi.x, e0_hi.y, e0_hi.z, e0_hi.w,
                   e1_hi.x, e1_hi.y, e1_hi.z, e1_hi.w,
                   e2_hi.x, e2_hi.y, e2_hi.z, e2_hi.w,
                   e3_hi.x, e3_hi.y, e3_hi.z, e3_hi.w};

    for(int a=0; a < 4; a++)
    {
        for(int b=0; b < 4; b++)
        {
            float sum = 0;

            for(int mu=0; mu < 4; mu++)
            {
                for(int v=0; v < 4; v++)
                {
                    sum += g_metric_big[mu * 4 + v] * m[a * 4 + mu] * m[b * 4 + v];
                }
            }

            minkowski[a * 4 + b] = sum;
        }
    }
}

int calculate_which_coordinate_is_timelike(float4 e0, float4 e1, float4 e2, float4 e3, float big_metric[16])
{
    float eps = 0.00001f;

    float minkowski[16];
    get_local_minkowski(e0, e1, e2, e3, big_metric, minkowski);

    ///is_degenerate check is because its not helpful
    ///ok no, can quite generally end up with explosions
    /*if(!IS_DEGENERATE(minkowski[0]) && !approx_equal(minkowski[0], -1, eps) && approx_equal(minkowski[0], 1, eps))
    {
        for(int i=0; i < 16; i++)
        {
            printf("Mink %f %i\n", minkowski[i], i);
        }

        printf("Warning, first column vector is not timelike. Todo for me: Fix this %f\n", minkowski[0]);
    }*/


    int lowest_index = -1;
    float lowest_index_value = 0;

    for(int i=0; i < 4; i++)
    {
        if(minkowski[i * 4 + i] < lowest_index_value)
        {
            lowest_index = i;
            lowest_index_value = minkowski[i * 4 + i];
        }
    }

    //printf("Lowest index %i\n", lowest_index);

    if(lowest_index != -1)
        return lowest_index;

    for(int i=0; i < 4; i++)
    {
        printf("Kowski %f\n", minkowski[i * 4 + i]);
    }

    printf("Warning, no index is timelike, physics is broken\n");

    return 0;
}

///todo: generic orthonormalisation
struct frame_basis calculate_frame_basis_with_swap_index(float big_metric[], int index_swap)
{
    ///this is nuts, why am I doing this?
    float4 ri1 = (float4)(1, 0, 0, 0);
    float4 ri2 = (float4)(0, 1, 0, 0);
    float4 ri3 = (float4)(0, 0, 1, 0);
    float4 ri4 = (float4)(0, 0, 0, 1);

    float4 i1 = lower_index_generic(ri1, big_metric);
    float4 i2 = lower_index_generic(ri2, big_metric);
    float4 i3 = lower_index_generic(ri3, big_metric);
    float4 i4 = lower_index_generic(ri4, big_metric);

    /*return orthonormalise4_metric(ri1, ri2, ri3, ri4, big_metric);*/

    ///all of the below is to fix misner
    float4 as_array[4] = {ri1, ri2, ri3, ri4};
    float lengths[4] = {dot(ri1, i1), dot(ri2, i2), dot(ri3, i3), dot(ri4, i4)};

    SWAP(as_array[0], as_array[index_swap], float4);
    SWAP(lengths[0], lengths[index_swap], float);

    int indices[4] = {0, 1, 2, 3};

    int first_nonzero = -1;

    float eps = 0.00001f;

    for(int i=0; i < 4; i++)
    {
        if(!approx_equal(lengths[i], 0.f, eps))
        {
            first_nonzero = i;
            break;
        }
    }

    if(first_nonzero == -1)
    {
        printf("Frame basis could not be calculated\n");
        first_nonzero = 0; ///can't exactly throw an exception now
    }

    if(first_nonzero != 0)
    {
        SWAP(as_array[0], as_array[first_nonzero], float4);
        SWAP(indices[0], indices[first_nonzero], int);
    }

    struct orthonormal_basis result = orthonormalise4_metric(as_array[0], as_array[1], as_array[2], as_array[3], big_metric);

    float4 result_as_array[4] = {result.v1, result.v2, result.v3, result.v4};

    float4 sorted_result[4] = {};

    for(int i=0; i < 4; i++)
    {
        int old_index = indices[i];

        sorted_result[old_index] = result_as_array[i];
    }

    ///is_degenerate check is because its not helpful
    ///ok no, can quite generally end up with explosions
    /*if(!IS_DEGENERATE(minkowski[0]) && !approx_equal(minkowski[0], -1, eps) && approx_equal(minkowski[0], 1, eps))
    {
        for(int i=0; i < 16; i++)
        {
            printf("Mink %f %i\n", minkowski[i], i);
        }

        printf("Warning, first column vector is not timelike. Todo for me: Fix this %f\n", minkowski[0]);
    }*/

    int which_index_is_timelike = calculate_which_coordinate_is_timelike(sorted_result[0], sorted_result[1], sorted_result[2], sorted_result[3], big_metric);

    if(which_index_is_timelike > 0)
    {
        SWAP(sorted_result[0], sorted_result[which_index_is_timelike], float4);
    }

    struct frame_basis result2;
    result2.v1 = sorted_result[0];
    result2.v2 = sorted_result[1];
    result2.v3 = sorted_result[2];
    result2.v4 = sorted_result[3];
    result2.timelike_coordinate = which_index_is_timelike == -1 ? 0 : which_index_is_timelike;

    return result2;
}

///frame basis construction is truly insane
struct frame_basis calculate_frame_basis(float big_metric[])
{
    struct frame_basis frame_1 = calculate_frame_basis_with_swap_index(big_metric, 0);

    if(frame_1.timelike_coordinate == 0)
        return frame_1;

    return calculate_frame_basis_with_swap_index(big_metric, frame_1.timelike_coordinate);
};

struct frame_basis calculate_frame_basis_at(float4 position, dynamic_config_space const struct dynamic_config* cfg)
{
    #ifndef GENERIC_BIG_METRIC
    float g_metric_local[4] = {};
    calculate_metric_generic(position, g_metric_local, cfg);

    float g_metric_big_local[16] = {0};

    g_metric_big_local[0] = g_metric_local[0];
    g_metric_big_local[1*4 + 1] = g_metric_local[1];
    g_metric_big_local[2*4 + 2] = g_metric_local[2];
    g_metric_big_local[3*4 + 3] = g_metric_local[3];
    #endif

    #ifdef GENERIC_BIG_METRIC
    float g_metric_big_local[16] = {0};
    calculate_metric_generic_big(position, g_metric_big_local, cfg);
    #endif

    return calculate_frame_basis(g_metric_big_local);
};

void print_metric_big_trace(float g_metric_big[])
{
    printf("%f %f %f %f\n", g_metric_big[0], g_metric_big[5], g_metric_big[10], g_metric_big[15]);
}


void print_metric(float g_metric[])
{
    printf("%f %f %f %f\n", g_metric[0], g_metric[1], g_metric[2], g_metric[3]);
}

void print_partials(float g_partials[])
{
    for(int i=0; i < 16; i++)
    {
        if(g_partials[i] == 0)
            continue;

        printf("%i: %f\n", i, g_partials[i]);
    }
}

void print_partials_big(float g_metric_partials_big[])
{
    for(int i=0; i < 64; i++)
    {
        if(g_metric_partials_big[i] == 0)
            continue;

        printf("%i: %f\n", i, g_metric_partials_big[i]);
    }
}

#ifdef GENERIC_BIG_METRIC
void calculate_lorentz_boost_big(float4 time_basis, float4 observer_velocity, float g_metric_big[], float coeff_out[])
{
    float delta[] = {1, 0, 0, 0,
                     0, 1, 0, 0,
                     0, 0, 1, 0,
                     0, 0, 0, 1};

    float4 lowered_time_basis = lower_index_big(time_basis, g_metric_big);
    float4 lowered_observer_velocity = lower_index_big(observer_velocity, g_metric_big);

    float T[4] = ARRAY4(time_basis);
    float lT[4] = ARRAY4(lowered_time_basis);
    float uobs[4] = ARRAY4(observer_velocity);
    float luobs[4] = ARRAY4(lowered_observer_velocity);

    float lorentz_factor = -dot(lowered_time_basis, observer_velocity);

    for(int u = 0; u < 4; u++)
    {
        for(int v=0; v < 4; v++)
        {
            coeff_out[u * 4 + v] = delta[u * 4 + v] + (1 / (1 + lorentz_factor)) * (T[u] + uobs[u]) * (lT[v] + luobs[v]) - 2 * uobs[u] * lT[v];
        }
    }
}

#else

///https://arxiv.org/pdf/2404.05744
void calculate_lorentz_boost(float4 time_basis, float4 observer_velocity, float g_metric[4], float coeff_out[])
{
    float delta[] = {1, 0, 0, 0,
                     0, 1, 0, 0,
                     0, 0, 1, 0,
                     0, 0, 0, 1};

    float4 lowered_time_basis = lower_index(time_basis, g_metric);
    float4 lowered_observer_velocity = lower_index(observer_velocity, g_metric);

    float T[4] = ARRAY4(time_basis);
    float lT[4] = ARRAY4(lowered_time_basis);
    float uobs[4] = ARRAY4(observer_velocity);
    float luobs[4] = ARRAY4(lowered_observer_velocity);

    float lorentz_factor = -dot(lowered_time_basis, observer_velocity);

    for(int u = 0; u < 4; u++)
    {
        for(int v=0; v < 4; v++)
        {
            coeff_out[u * 4 + v] = delta[u * 4 + v] + (1 / (1 + lorentz_factor)) * (T[u] + uobs[u]) * (lT[v] + luobs[v]) - 2 * uobs[u] * lT[v];
        }
    }
}

#endif // GENERIC_BIG_METRIC

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

float cos_mix(float x1, float x2, float f)
{
    float f2 = (1 - native_cos(f * M_PIf))/2.f;

    return mix(x1, x2, f2);
}

float angle_to_plane(float3 vec, float3 plane_normal)
{
    return M_PIf/2 - acos(dot(normalize(vec), normalize(plane_normal)));
}

float angle_between_vectors3(float3 v1, float3 v2)
{
    return acos(dot(v1, v2));
}

float4 get_theta_adjustment_quat(float3 pixel_direction, float4 polar_camera_in, float angle_sign, bool debug)
{
    if(fast_length(pixel_direction) < 0.00001f)
    {
        pixel_direction = (float3){0, 1, 0};
    }

    float3 apolar = polar_camera_in.yzw;
    apolar.x = fabs(apolar.x);

    float3 cartesian_camera_pos = polar_to_cartesian(apolar);

    float3 bx = normalize(pixel_direction);
    float3 by = normalize(-cartesian_camera_pos);

    bx = normalize(rejection(bx, by));

    float3 plane_n = -normalize(cross(bx, by));

    float angle_to_flat = angle_between_vectors3(plane_n, (float3)(0, 0, 1));

    if(debug)
    {
        //printf("Angle %f\n", angle_to_flat);
    }

    return aa_to_quat(normalize(cross(plane_n, (float3)(0, 0, 1))), angle_to_flat * angle_sign);
}

float3 calculate_pixel_direction(int cx, int cy, float width, float height, float4 camera_quat, dynamic_config_space const struct dynamic_feature_config* dfg)
{
    float fov = GET_FEATURE(field_of_view, dfg);

    float fov_rad = (fov / 360.f) * 2 * M_PIf;

    float nonphysical_plane_half_width = width/2;
    float nonphysical_f_stop = nonphysical_plane_half_width / tan(fov_rad/2);

    float3 pixel_direction = (float3){cx - width/2, cy - height/2, nonphysical_f_stop};

    pixel_direction = normalize(pixel_direction);
    pixel_direction = rot_quat(pixel_direction, camera_quat);

    return pixel_direction;
}

int should_early_terminate(int x, int y, int width, int height, __global const int* termination_buffer)
{
    if(x < 0 || y < 0 || x > width-1 || y > height-1)
        return false;

    x = clamp(x, 0, width-1);
    y = clamp(y, 0, height-1);

    return termination_buffer[y * width + x] == 1;
}

void get_tetrad_inverse(float4 e0_hi, float4 e1_hi, float4 e2_hi, float4 e3_hi, float4* oe0_lo, float4* oe1_lo, float4* oe2_lo, float4* oe3_lo)
{
    /*float m[16] = {e0_hi.x, e0_hi.y, e0_hi.z, e0_hi.w,
                   e1_hi.x, e1_hi.y, e1_hi.z, e1_hi.w,
                   e2_hi.x, e2_hi.y, e2_hi.z, e2_hi.w,
                   e3_hi.x, e3_hi.y, e3_hi.z, e3_hi.w};*/

    float m[16] = {e0_hi.x, e1_hi.x, e2_hi.x, e3_hi.x,
                   e0_hi.y, e1_hi.y, e2_hi.y, e3_hi.y,
                   e0_hi.z, e1_hi.z, e2_hi.z, e3_hi.z,
                   e0_hi.w, e1_hi.w, e2_hi.w, e3_hi.w};

    float inv[16] = {0};

    matrix_inverse(m, inv);

    *oe0_lo = (float4){inv[0 * 4 + 0], inv[0 * 4 + 1], inv[0 * 4 + 2], inv[0 * 4 + 3]};
    *oe1_lo = (float4){inv[1 * 4 + 0], inv[1 * 4 + 1], inv[1 * 4 + 2], inv[1 * 4 + 3]};
    *oe2_lo = (float4){inv[2 * 4 + 0], inv[2 * 4 + 1], inv[2 * 4 + 2], inv[2 * 4 + 3]};
    *oe3_lo = (float4){inv[3 * 4 + 0], inv[3 * 4 + 1], inv[3 * 4 + 2], inv[3 * 4 + 3]};
}

/// e upper i, lower mu, which must be inverse of tetrad to coordinate basis vectors
float4 coordinate_to_tetrad_basis(float4 vec_up, float4 e0_lo, float4 e1_lo, float4 e2_lo, float4 e3_lo)
{
    float4 ret;

    ret.x = dot(e0_lo, vec_up);
    ret.y = dot(e1_lo, vec_up);
    ret.z = dot(e2_lo, vec_up);
    ret.w = dot(e3_lo, vec_up);

    return ret;

    //return vec_up.x * e0_lo + vec_up.y * e1_lo + vec_up.z * e2_lo + vec_up.w * e3_lo;
}

///so. The hi tetrads are the one we get out of gram schmidt
///so this is lower i, upper mu, against a vec with upper i
float4 tetrad_to_coordinate_basis(float4 vec_up, float4 e0_hi, float4 e1_hi, float4 e2_hi, float4 e3_hi)
{
    return vec_up.x * e0_hi + vec_up.y * e1_hi + vec_up.z * e2_hi + vec_up.w * e3_hi;
}

void quat_to_matrix(float4 q, float m[9])
{
    float qx = q.x;
    float qy = q.y;
    float qz = q.z;
    float qw = q.w;

    m[0 * 3 + 0] = 1 - 2*qy*qy - 2*qz*qz;
    m[0 * 3 + 1] = 2*qx*qy - 2*qz*qw;
    m[0 * 3 + 2] = 2*qx*qz + 2*qy*qw;

    m[1 * 3 + 0] = 2*qx*qy + 2*qz*qw;
    m[1 * 3 + 1] = 1 - 2*qx*qx - 2*qz*qz;
    m[1 * 3 + 2] = 2*qy*qz - 2*qx*qw;

    m[2 * 3 + 0] = 2*qx*qz - 2*qy*qw;
    m[2 * 3 + 1] = 2*qy*qz + 2*qx*qw;
    m[2 * 3 + 2] = 1 - 2*qx*qx - 2*qy*qy;
}

float4 matrix_to_quat(float m[9])
{
    float4 l;

    float m00 = m[0 * 3 + 0];
    float m11 = m[1 * 3 + 1];
    float m22 = m[2 * 3 + 2];

    l.w = sqrt( max( 0.f, 1 + m00 + m11 + m22 ) ) / 2;
    l.x = sqrt( max( 0.f, 1 + m00 - m11 - m22 ) ) / 2;
    l.y = sqrt( max( 0.f, 1 - m00 + m11 - m22 ) ) / 2;
    l.z = sqrt( max( 0.f, 1 - m00 - m11 + m22 ) ) / 2;

    float m21 = m[2 * 3 + 1];
    float m12 = m[1 * 3 + 2];
    float m02 = m[0 * 3 + 2];
    float m20 = m[2 * 3 + 0];
    float m10 = m[1 * 3 + 0];
    float m01 = m[0 * 3 + 1];

    l.x = copysign( l.x, m21 - m12 );
    l.y = copysign( l.y, m02 - m20 );
    l.z = copysign( l.z, m10 - m01 );

    return l;
}

///https://physics.stackexchange.com/questions/524242/parallel-transport-of-a-vectors
float4 parallel_transport_get_velocity(float4 X, float4 geodesic_position, float4 geodesic_velocity, dynamic_config_space const struct dynamic_config* cfg)
{
    float X_arr[4] = {X.x, X.y, X.z, X.w};
    float Y_arr[4] = {geodesic_velocity.x, geodesic_velocity.y, geodesic_velocity.z, geodesic_velocity.w};

    float christoffel[64] = {0};

    {
        #ifndef GENERIC_BIG_METRIC
        float g_metric[4] = {0};
        float g_partials[16] = {0};

        calculate_metric_generic(geodesic_position, g_metric, cfg);
        calculate_partial_derivatives_generic(geodesic_position, g_partials, cfg);
        #else
        float g_metric[16] = {0};
        float g_partials[64] = {0};

        calculate_metric_generic_big(geodesic_position, g_metric, cfg);
        calculate_partial_derivatives_generic_big(geodesic_position, g_partials, cfg);
        #endif // GENERIC_BIG_METRIC

        get_christoffel_generic(g_metric, g_partials, christoffel);
    }

    float vel[4] = {0};

    for(int a=0; a < 4; a++)
    {
        float sum = 0;

        for(int b=0; b < 4; b++)
        {
            for(int s=0; s < 4; s++)
            {
                sum += christoffel[a * 16 + b * 4 + s] * X_arr[b] * Y_arr[s];
            }
        }

        vel[a] = -sum;
    }

    return (float4){vel[0], vel[1], vel[2], vel[3]};
}

///https://arxiv.org/pdf/0904.4184.pdf 1.4.18
float4 get_timelike_vector(float3 cartesian_basis_speed, float time_direction,
                           float4 e0, float4 e1, float4 e2, float4 e3)
{

    float v2 = dot(cartesian_basis_speed, cartesian_basis_speed);

    float Y = 1 / sqrt(1 - v2);

    float4 bT = time_direction * Y * e0;
    float4 bX = Y * cartesian_basis_speed.x * e1;
    float4 bY = Y * cartesian_basis_speed.y * e2;
    float4 bZ = Y * cartesian_basis_speed.z * e3;

    return bT + bX + bY + bZ;
}

__kernel
void calculate_timelike_coordinate(__global const float4* generic_position, dynamic_config_space const struct dynamic_config* cfg, __global int* coordinate_out)
{
    if(get_global_id(0) != 0)
        return;

    float4 at_metric = *generic_position;

    #ifndef GENERIC_BIG_METRIC
    float g_metric_local[4] = {};
    calculate_metric_generic(at_metric, g_metric_local, cfg);

    float g_metric_big_local[16] = {0};

    g_metric_big_local[0] = g_metric_local[0];
    g_metric_big_local[1*4 + 1] = g_metric_local[1];
    g_metric_big_local[2*4 + 2] = g_metric_local[2];
    g_metric_big_local[3*4 + 3] = g_metric_local[3];
    #endif

    #ifdef GENERIC_BIG_METRIC
    float g_metric_big_local[16] = {0};
    calculate_metric_generic_big(at_metric, g_metric_big_local, cfg);
    #endif

    struct frame_basis basis = calculate_frame_basis(g_metric_big_local);

    *coordinate_out = basis.timelike_coordinate;
}

__kernel
void calculate_timelike_coordinates(__global const float4* positions, int count, dynamic_config_space const struct dynamic_config* cfg, __global int* coordinate_out)
{
    int id = get_global_id(0);

    if(id >= count)
        return;

    float4 at_metric = positions[id];

    #ifndef GENERIC_BIG_METRIC
    float g_metric_local[4] = {};
    calculate_metric_generic(at_metric, g_metric_local, cfg);

    float g_metric_big_local[16] = {0};

    g_metric_big_local[0] = g_metric_local[0];
    g_metric_big_local[1*4 + 1] = g_metric_local[1];
    g_metric_big_local[2*4 + 2] = g_metric_local[2];
    g_metric_big_local[3*4 + 3] = g_metric_local[3];
    #endif

    #ifdef GENERIC_BIG_METRIC
    float g_metric_big_local[16] = {0};
    calculate_metric_generic_big(at_metric, g_metric_big_local, cfg);
    #endif

    struct frame_basis basis = calculate_frame_basis(g_metric_big_local);

    coordinate_out[id] = basis.timelike_coordinate;
}

void calculate_tetrads(float4 at_metric, float3 cartesian_basis_speed,
                       float4* e0_out, float4* e1_out, float4* e2_out, float4* e3_out,
                       dynamic_config_space const struct dynamic_config* cfg, int should_orient)
{
    float4 polar_camera = generic_to_spherical(at_metric, cfg);

    if(IS_DEGENERATE(at_metric.x) || IS_DEGENERATE(at_metric.y) || IS_DEGENERATE(at_metric.z) || IS_DEGENERATE(at_metric.w))
    {
        *e0_out = (float4)(1, 0, 0, 0);
        *e1_out = (float4)(0, 1, 0, 0);
        *e2_out = (float4)(0, 0, 1, 0);
        *e3_out = (float4)(0, 0, 0, 1);

        return;
    }

    #ifndef GENERIC_BIG_METRIC
    float g_metric_local[4] = {};
    calculate_metric_generic(at_metric, g_metric_local, cfg);

    float g_metric_big_local[16] = {0};

    g_metric_big_local[0] = g_metric_local[0];
    g_metric_big_local[1*4 + 1] = g_metric_local[1];
    g_metric_big_local[2*4 + 2] = g_metric_local[2];
    g_metric_big_local[3*4 + 3] = g_metric_local[3];
    #endif

    #ifdef GENERIC_BIG_METRIC
    float g_metric_big_local[16] = {0};
    calculate_metric_generic_big(at_metric, g_metric_big_local, cfg);
    #endif

    struct frame_basis basis = calculate_frame_basis(g_metric_big_local);

    ///contravariant
    float4 e0 = basis.v1;
    float4 e1 = basis.v2;
    float4 e2 = basis.v3;
    float4 e3 = basis.v4;

    if(should_orient)
    {
        /*printf("Basis bT %f %f %f %f\n", e0.x, e0.y, e0.z, e0.w);
        printf("Basis sVx %f %f %f %f\n", e1.x, e1.y, e1.z, e1.w);
        printf("Basis sVy %f %f %f %f\n", e2.x, e2.y, e2.z, e2.w);
        printf("Basis sVz %f %f %f %f\n", e3.x, e3.y, e3.z, e3.w);*/

        ///void get_tetrad_inverse(float4 e0_hi, float4 e1_hi, float4 e2_hi, float4 e3_hi, float4* oe0_lo, float4* oe1_lo, float4* oe2_lo, float4* oe3_lo)

        float4 le0;
        float4 le1;
        float4 le2;
        float4 le3;

        {
            float3 apolar = polar_camera.yzw;
            apolar.x = fabs(apolar.x);

            float3 cart_camera = polar_to_cartesian(apolar);

            float4 e_lo[4];
            get_tetrad_inverse(e0, e1, e2, e3, &e_lo[0], &e_lo[1], &e_lo[2], &e_lo[3]);

            float3 cx = (float3)(1, 0, 0);
            float3 cy = (float3)(0, 1, 0);
            float3 cz = (float3)(0, 0, 1);

            float3 sx = cartesian_velocity_to_polar_velocity(cart_camera, cx);
            float3 sy = cartesian_velocity_to_polar_velocity(cart_camera, cy);
            float3 sz = cartesian_velocity_to_polar_velocity(cart_camera, cz);

            if(polar_camera.y < 0)
            {
                sx.x = -sx.x;
                sy.x = -sy.x;
                sz.x = -sz.x;
            }

            float4 gx = spherical_velocity_to_generic_velocity(polar_camera, (float4)(0, sx), cfg);
            float4 gy = spherical_velocity_to_generic_velocity(polar_camera, (float4)(0, sy), cfg);
            float4 gz = spherical_velocity_to_generic_velocity(polar_camera, (float4)(0, sz), cfg);

            ///normalise with y first, so that the camera controls always work intuitively - as they are inherently a 'global' concept
            ///ok so this is in global coordinate
            float4 approximate_basis[3] = {gy, gx, gz};

            ///push it into the tetrad
            float4 tE1 = coordinate_to_tetrad_basis(approximate_basis[0], e_lo[0], e_lo[1], e_lo[2], e_lo[3]);
            float4 tE2 = coordinate_to_tetrad_basis(approximate_basis[1], e_lo[0], e_lo[1], e_lo[2], e_lo[3]);
            float4 tE3 = coordinate_to_tetrad_basis(approximate_basis[2], e_lo[0], e_lo[1], e_lo[2], e_lo[3]);

            ///orthonormalise the spatial parts of the projected vectors
            struct ortho_result result = orthonormalise(tE1.yzw, tE2.yzw, tE3.yzw);

            ///discard the timelike component
            float4 basis1 = (float4)(0, result.v1);
            float4 basis2 = (float4)(0, result.v2);
            float4 basis3 = (float4)(0, result.v3);

            float4 x_basis = basis2;
            float4 y_basis = basis1;
            float4 z_basis = basis3;

            ///use the original tetrads, because we know where the timelike component lives
            float4 x_out = tetrad_to_coordinate_basis(x_basis, e0, e1, e2, e3);
            float4 y_out = tetrad_to_coordinate_basis(y_basis, e0, e1, e2, e3);
            float4 z_out = tetrad_to_coordinate_basis(z_basis, e0, e1, e2, e3);

            le0 = e0;
            le1 = x_out;
            le2 = y_out;
            le3 = z_out;
        }

        /*printf("Out Basis bT %f %f %f %f\n", le0.x, le0.y, le0.z, le0.w);
        printf("Out Basis sVx %f %f %f %f\n", le1.x, le1.y, le1.z, le1.w);
        printf("Out Basis sVy %f %f %f %f\n", le2.x, le2.y, le2.z, le2.w);
        printf("Out Basis sVz %f %f %f %f\n", le3.x, le3.y, le3.z, le3.w);*/

        e0 = le0;
        e1 = le1;
        e2 = le2;
        e3 = le3;
    }

    {
        float4 observer_velocity = get_timelike_vector(cartesian_basis_speed, 1, e0, e1, e2, e3);

        float lorentz[16] = {};

        #ifndef GENERIC_BIG_METRIC
        float g_metric[4] = {};
        calculate_metric_generic(at_metric, g_metric, cfg);
        calculate_lorentz_boost(e0, observer_velocity, g_metric, lorentz);
        #else
        float g_metric_big[16] = {0};
        calculate_metric_generic_big(at_metric, g_metric_big, cfg);
        calculate_lorentz_boost_big(e0, observer_velocity, g_metric_big, lorentz);
        #endif // GENERIC_METRIC

        e0 = observer_velocity;
        e1 = tensor_contract(lorentz, e1);
        e2 = tensor_contract(lorentz, e2);
        e3 = tensor_contract(lorentz, e3);
    }

    *e0_out = e0;
    *e1_out = e1;
    *e2_out = e2;
    *e3_out = e3;
}

__kernel
void boost_tetrad(__global float4* generic_in, int count, __global float3* geodesic_basis_speed,
                  __global float4* e0_io, __global float4* e1_io, __global float4* e2_io, __global float4* e3_io,
                  dynamic_config_space const struct dynamic_config* cfg)
{
    int id = get_global_id(0);

    if(id >= count)
        return;

    float4 at_metric = generic_in[id];

    float4 e0 = e0_io[id];
    float4 e1 = e1_io[id];
    float4 e2 = e2_io[id];
    float4 e3 = e3_io[id];

    float4 observer_velocity = get_timelike_vector(geodesic_basis_speed[id], 1, e0, e1, e2, e3);

    float lorentz[16] = {};

    #ifndef GENERIC_BIG_METRIC
    float g_metric[4] = {};
    calculate_metric_generic(at_metric, g_metric, cfg);
    calculate_lorentz_boost(e0, observer_velocity, g_metric, lorentz);
    #else
    float g_metric_big[16] = {};
    calculate_metric_generic_big(at_metric, g_metric_big, cfg);
    calculate_lorentz_boost_big(e0, observer_velocity, g_metric_big, lorentz);
    #endif // GENERIC_METRIC

    e0 = observer_velocity;
    e1 = tensor_contract(lorentz, e1);
    e2 = tensor_contract(lorentz, e2);
    e3 = tensor_contract(lorentz, e3);

    e0_io[id] = e0;
    e1_io[id] = e1;
    e2_io[id] = e2;
    e3_io[id] = e3;
}

__kernel
void init_basis_vectors(__global const float4* generic_in, int count,
                        float3 cartesian_basis_speed,
                        __global float4* e0_out, __global float4* e1_out, __global float4* e2_out, __global float4* e3_out,
                        dynamic_config_space const struct dynamic_config* cfg)
{
    int id = get_global_id(0);

    if(id >= count)
        return;

    float4 e0;
    float4 e1;
    float4 e2;
    float4 e3;

    calculate_tetrads(generic_in[id], cartesian_basis_speed, &e0, &e1, &e2, &e3, cfg, 1);

    e0_out[id] = e0;
    e1_out[id] = e1;
    e2_out[id] = e2;
    e3_out[id] = e3;

    //DUMP_TETRAD("Basis Vector", e0, e1, e2, e3);
}

float4 mix_spherical(float4 in1, float4 in2, float a)
{
    float4 ain1 = in1;
    float4 ain2 = in2;

    ain1.y = fabs(ain1.y);
    ain2.y = fabs(ain2.y);

    float3 cart1 = polar_to_cartesian(ain1.yzw);
    float3 cart2 = polar_to_cartesian(ain2.yzw);

    float r1 = in1.y;
    float r2 = in2.y;

    float3 mixed = mix(cart1, cart2, a);

    float3 as_polar = cartesian_to_polar(mixed);

    as_polar.x = mix(r1, r2, a);

    float t = mix(in1.x, in2.x, a);

    return (float4)(t, as_polar);
}

__kernel
void calculate_tetrad_inverse(__global int* global_count, int count,
                              __global float4* t_e0, __global float4* t_e1, __global float4* t_e2, __global float4* t_e3,
                              __global float4* ie0, __global float4* ie1, __global float4* ie2, __global float4* ie3,
                              __global float4* positions,
                              dynamic_config_space const struct dynamic_config* cfg)
{
    int id = get_global_id(0);

    if(id >= count)
        return;

    int cnt = global_count[id];

    int stride = count;

    for(int kk=0; kk < cnt; kk++)
    {
        int current_idx = kk * count + id;

        float4 e0 = t_e0[current_idx];
        float4 e1 = t_e1[current_idx];
        float4 e2 = t_e2[current_idx];
        float4 e3 = t_e3[current_idx];

        float4 e_lo[4];
        get_tetrad_inverse(e0, e1, e2, e3, &e_lo[0], &e_lo[1], &e_lo[2], &e_lo[3]);

        ie0[current_idx] = e_lo[0];
        ie1[current_idx] = e_lo[1];
        ie2[current_idx] = e_lo[2];
        ie3[current_idx] = e_lo[3];
    }
}

__kernel
void parallel_transport_quantity(__global float4* geodesic_path, __global float4* geodesic_velocity, __global float* ds_in, __global float4* quantity, __global int* count_in, int count, __global float4* quantity_out, dynamic_config_space const struct dynamic_config* cfg)
{
    int id = get_global_id(0);

    if(id >= count)
        return;

    int cnt = count_in[id];

    if(cnt == 0)
        return;

    int stride = count;

    ///i * stride + id
    float4 current_quantity = quantity[0 * stride + id];

    quantity_out[0 * stride + id] = current_quantity;

    if(cnt == 1)
        return;

    for(int kk=0; kk < cnt - 1; kk++)
    {
        int current_idx = kk * stride + id;
        int next_idx = (kk + 1) * stride + id;

        float ds = ds_in[current_idx];

        float4 current_position = geodesic_path[current_idx];
        float4 next_position = geodesic_path[next_idx];

        float4 current_velocity = geodesic_velocity[current_idx];
        float4 next_velocity = geodesic_velocity[next_idx];

        ///this isn't verlet, its generic 2nd order integration
        float4 f_x = parallel_transport_get_velocity(current_quantity, current_position, current_velocity, cfg);

        float4 intermediate_next = current_quantity + f_x * ds;

        float4 next = current_quantity + 0.5f * ds * (f_x + parallel_transport_get_velocity(intermediate_next, next_position, next_velocity, cfg));

        ///so. quantity_out[0] ends up being initial, quantity_out[1] = after one transport
        quantity_out[current_idx] = current_quantity;

        current_quantity = next;
    }

    ///need to write final one
    quantity_out[(cnt - 1) * stride + id] = current_quantity;
}

float4 transport(float4 current_quantity,
                 float4 current_position, float4 current_velocity,
                 float4 next_position, float4 next_velocity,
                 float ds,
                 dynamic_config_space const struct dynamic_config* cfg)
{

    ///this isn't verlet, its generic 2nd order integration
    float4 f_x = parallel_transport_get_velocity(current_quantity, current_position, current_velocity, cfg);

    float4 intermediate_next = current_quantity + f_x * ds;

    float4 next = current_quantity + 0.5f * ds * (f_x + parallel_transport_get_velocity(intermediate_next, next_position, next_velocity, cfg));

    return next;
}

__kernel
void parallel_transport_tetrads(__global float4* geodesic_path, __global float4* geodesic_velocity, __global float* ds_in,
                                __global float4* in_e0, __global float4* in_e1, __global float4* in_e2, __global float4* in_e3,
                                __global int* count_in, int count,
                                __global float4* out_e0, __global float4* out_e1, __global float4* out_e2, __global float4* out_e3,
                                dynamic_config_space const struct dynamic_config* cfg)
{
    int id = get_global_id(0);

    if(id >= count)
        return;

    int cnt = count_in[id];

    if(cnt == 0)
        return;

    int stride = count;

    ///i * stride + id
    float4 e0 = in_e0[0 * stride + id];
    float4 e1 = in_e1[0 * stride + id];
    float4 e2 = in_e2[0 * stride + id];
    float4 e3 = in_e3[0 * stride + id];

    out_e0[0 * stride + id] = e0;
    out_e1[0 * stride + id] = e1;
    out_e2[0 * stride + id] = e2;
    out_e3[0 * stride + id] = e3;

    if(cnt == 1)
        return;

    for(int kk=0; kk < cnt - 1; kk++)
    {
        int current_idx = kk * stride + id;
        int next_idx = (kk + 1) * stride + id;

        float ds = ds_in[current_idx];

        float4 current_position = geodesic_path[current_idx];
        float4 next_position = geodesic_path[next_idx];

        float4 current_velocity = geodesic_velocity[current_idx];
        float4 next_velocity = geodesic_velocity[next_idx];

        float4 ne0 = transport(e0, current_position, current_velocity, next_position, next_velocity, ds, cfg);
        float4 ne1 = transport(e1, current_position, current_velocity, next_position, next_velocity, ds, cfg);
        float4 ne2 = transport(e2, current_position, current_velocity, next_position, next_velocity, ds, cfg);
        float4 ne3 = transport(e3, current_position, current_velocity, next_position, next_velocity, ds, cfg);

        #ifndef GENERIC_BIG_METRIC
        float g_metric_local[4] = {};
        calculate_metric_generic(current_position, g_metric_local, cfg);

        float g_metric_big_local[16] = {0};

        g_metric_big_local[0] = g_metric_local[0];
        g_metric_big_local[1*4 + 1] = g_metric_local[1];
        g_metric_big_local[2*4 + 2] = g_metric_local[2];
        g_metric_big_local[3*4 + 3] = g_metric_local[3];
        #endif

        #ifdef GENERIC_BIG_METRIC
        float g_metric_big_local[16] = {0};
        calculate_metric_generic_big(current_position, g_metric_big_local, cfg);
        #endif

        struct orthonormal_basis ortho = orthonormalise4_metric(ne0, ne1, ne2, ne3, g_metric_big_local);

        ///so. quantity_out[0] ends up being initial, quantity_out[1] = after one transport
        out_e0[current_idx] = ortho.v1;
        out_e1[current_idx] = ortho.v2;
        out_e2[current_idx] = ortho.v3;
        out_e3[current_idx] = ortho.v4;

        e0 = ortho.v1;
        e1 = ortho.v2;
        e2 = ortho.v3;
        e3 = ortho.v4;

        /*out_e0[current_idx] = ne0;
        out_e1[current_idx] = ne1;
        out_e2[current_idx] = ne2;
        out_e3[current_idx] = ne3;

        e0 = ne0;
        e1 = ne1;
        e2 = ne2;
        e3 = ne3;*/
    }

    ///need to write final one
    out_e0[(cnt - 1) * stride + id] = e0;
    out_e1[(cnt - 1) * stride + id] = e1;
    out_e2[(cnt - 1) * stride + id] = e2;
    out_e3[(cnt - 1) * stride + id] = e3;
}

__kernel
void handle_interpolating_geodesic(__global const float4* geodesic_path, __global const float4* geodesic_velocity, __global const float* ds_in,
                                   __global float4* g_camera_generic_out,
                                   __global const float4* t_e0_in, __global const float4* t_e1_in, __global const float4* t_e2_in, __global const float4* t_e3_in,
                                   __global float4* e0_out, __global float4* e1_out, __global float4* e2_out, __global float4* e3_out,
                                   float target_time,
                                   __global const int* count_in,
                                   int parallel_transport_observer,
                                   __global const float3* geodesic_basis_speed,
                                   __global float4* interpolated_geodesic_velocity,
                                   dynamic_config_space const struct dynamic_config* cfg)
{
    if(get_global_id(0) != 0)
        return;

    if(*count_in == 0)
        return;

    float4 start_generic = geodesic_path[0];

    if(!parallel_transport_observer)
    {
        float4 e0, e1, e2, e3;
        calculate_tetrads(start_generic, *geodesic_basis_speed, &e0, &e1, &e2, &e3, cfg, 1);

        *e0_out = e0;
        *e1_out = e1;
        *e2_out = e2;
        *e3_out = e3;
    }
    else
    {
        *e0_out = t_e0_in[0];
        *e1_out = t_e1_in[0];
        *e2_out = t_e2_in[0];
        *e3_out = t_e3_in[0];
    }

    int cnt = *count_in;

    //float current_time = geodesic_path[0].x;
    float current_proper_time = 0;

    *g_camera_generic_out = start_generic;
    *interpolated_geodesic_velocity = geodesic_velocity[0];

    if(cnt == 1)
        return;

    for(int i=0; i < cnt - 1; i++)
    {
        float4 current_pos = geodesic_path[i];
        float4 next_pos = geodesic_path[i + 1];

        float next_proper_time = current_proper_time + ds_in[i];

        #ifdef INTERPOLATE_USE_COORDINATE_TIME
        if(target_time >= current_pos.x && target_time < next_pos.x || target_time < current_pos.x)
        #else
        if(target_time >= current_proper_time && target_time < next_proper_time || target_time < current_proper_time)
        #endif
        {
            #ifdef INTERPOLATE_USE_COORDINATE_TIME
            float dx = (target_time - current_pos.x) / (next_pos.x - current_pos.x);
            #else
            float dx = (target_time - current_proper_time) / (next_proper_time - current_proper_time);
            #endif

            #ifdef INTERPOLATE_USE_COORDINATE_TIME
            if(target_time < current_pos.x)
                dx = 0;
            #else
            if(target_time < current_proper_time)
                dx = 0;
            #endif

            float4 fin_generic = mix(current_pos, next_pos, dx);

            *g_camera_generic_out = fin_generic;

            float4 e0 = t_e0_in[i];
            float4 e1 = t_e1_in[i];
            float4 e2 = t_e2_in[i];
            float4 e3 = t_e3_in[i];

            float4 ne0 = t_e0_in[i + 1];
            float4 ne1 = t_e1_in[i + 1];
            float4 ne2 = t_e2_in[i + 1];
            float4 ne3 = t_e3_in[i + 1];

            float4 oe0 = mix(e0, ne0, dx);
            float4 oe1 = mix(e1, ne1, dx);
            float4 oe2 = mix(e2, ne2, dx);
            float4 oe3 = mix(e3, ne3, dx);

            if(!parallel_transport_observer)
            {
                calculate_tetrads(fin_generic, *geodesic_basis_speed, &oe0, &oe1, &oe2, &oe3, cfg, 1);
            }

            *interpolated_geodesic_velocity = mix(geodesic_velocity[i], geodesic_velocity[i + 1], dx);

            *e0_out = oe0;
            *e1_out = oe1;
            *e2_out = oe2;
            *e3_out = oe3;

            return;
        }

        //current_time += dt;
        current_proper_time = next_proper_time;
    }

    *g_camera_generic_out = geodesic_path[cnt - 1];
    *interpolated_geodesic_velocity = geodesic_velocity[cnt - 1];

    if(!parallel_transport_observer)
    {
        float4 e0, e1, e2, e3;
        calculate_tetrads(geodesic_path[cnt - 1], *geodesic_basis_speed, &e0, &e1, &e2, &e3, cfg, 1);

        *e0_out = e0;
        *e1_out = e1;
        *e2_out = e2;
        *e3_out = e3;
    }
    else
    {
        *e0_out = t_e0_in[cnt - 1];
        *e1_out = t_e1_in[cnt - 1];
        *e2_out = t_e2_in[cnt - 1];
        *e3_out = t_e3_in[cnt - 1];
    }
}

__kernel void pull_object_positions(__global struct object* current_pos, __global float4* geodesic_out, int object_count)
{
    int id = get_global_id(0);

    if(id >= object_count)
        return;

    geodesic_out[id] = current_pos[id].pos;
}

#if 0
__kernel void push_object_positions(__global float4* geodesics_in, __global int* counts_in, __global struct object* pos_out, float target_time, int object_count, dynamic_config_space const struct dynamic_config* cfg)
{
    int id = get_global_id(0);

    if(id >= object_count)
        return;

    int path_length = counts_in[id];

    if(path_length == 0)
        return;

    __global struct object* my_obj = &pos_out[id];

    my_obj->pos = generic_to_cartesian(geodesics_in[id], cfg);

    if(path_length == 1)
        return;

    for(int i=0; i < path_length - 1; i++)
    {
        float4 current_pos = geodesics_in[i * object_count + id];
        float4 next_pos = geodesics_in[(i + 1) * object_count + id];

        if(next_pos.x < current_pos.x)
        {
            float4 im = current_pos;
            current_pos = next_pos;
            next_pos = im;
        }

        if(target_time >= current_pos.x && target_time < next_pos.x || target_time < current_pos.x)
        {
            float dx = (target_time - current_pos.x) / (next_pos.x - current_pos.x);

            if(target_time < current_pos.x)
                dx = 0;

            ///this is overly complicated and can be directly interpolated
            float4 spherical1 = generic_to_spherical(current_pos, cfg);
            float4 spherical2 = generic_to_spherical(next_pos, cfg);

            float4 fin_polar = mix_spherical(spherical1, spherical2, dx);

            float3 fin_cart = polar_to_cartesian(fin_polar.yzw);

            float4 out_pos = (float4)(fin_polar.x, fin_cart.xyz);

            my_obj->pos = out_pos;
            return;
        }
    }

    my_obj->pos = generic_to_cartesian(geodesics_in[(path_length - 1) * object_count + id], cfg);
}
#endif

struct corrected_lightray
{
    float4 position;
    float4 velocity;
    float4 inverse_quat;
};

struct corrected_lightray correct_lightray(float4 position, float4 velocity, dynamic_config_space const struct dynamic_config* cfg)
{
    float4 extra_quat = (float4)(0, 0, 0, 1);

    #if defined(GENERIC_CONSTANT_THETA) || defined(DEBUG_CONSTANT_THETA)
    {
        float4 polar_pos = generic_to_spherical(position, cfg);

        float4 pos_spherical = generic_to_spherical(position, cfg);
        float4 vel_spherical = generic_velocity_to_spherical_velocity(position, velocity, cfg);

        float fsign = sign(pos_spherical.y);
        pos_spherical.y = fabs(pos_spherical.y);

        float3 pos_cart = polar_to_cartesian(pos_spherical.yzw);
        float3 vel_cart = spherical_velocity_to_cartesian_velocity(pos_spherical.yzw, vel_spherical.yzw);

        float4 quat = get_theta_adjustment_quat(vel_cart, polar_pos, 1, false);
        extra_quat  = get_theta_adjustment_quat(vel_cart, polar_pos,-1, false);

        pos_cart = rot_quat(pos_cart, quat);
        vel_cart = rot_quat(vel_cart, quat);

        float3 next_pos_spherical = cartesian_to_polar(pos_cart);
        float3 next_vel_spherical = cartesian_velocity_to_polar_velocity(pos_cart, vel_cart);

        if(fsign < 0)
        {
            next_pos_spherical.x = -next_pos_spherical.x;
        }

        float4 next_pos_generic = spherical_to_generic((float4)(pos_spherical.x, next_pos_spherical), cfg);
        float4 next_vel_generic = spherical_velocity_to_generic_velocity((float4)(pos_spherical.x, next_pos_spherical), (float4)(vel_spherical.x, next_vel_spherical), cfg);

        position = next_pos_generic;
        velocity = next_vel_generic;
    }
    #endif // GENERIC_CONSTANT_THETA

    //position = handle_coordinate_periodicity(position, cfg);

    struct corrected_lightray ret;

    ret.position = position;
    ret.velocity = velocity;
    ret.inverse_quat = extra_quat;

    return ret;
};

///fully set up except for ray.early_terminate
struct lightray geodesic_to_render_ray(int cx, int cy, float4 position, float4 velocity, float4 observer_velocity, dynamic_config_space const struct dynamic_config* cfg)
{
    struct corrected_lightray corrected = correct_lightray(position, velocity, cfg);

    position = corrected.position;
    velocity = corrected.velocity;

    float4 lightray_acceleration = (float4)(0,0,0,0);

    #ifdef IS_CONSTANT_THETA
    position.z = M_PIf/2;
    velocity.z = 0;
    #endif // IS_CONSTANT_THETA

    #ifndef GENERIC_BIG_METRIC
    float g_metric[4] = {0};
    #else
    float g_metric_big[16] = {0};
    #endif // GENERIC_BIG_METRIC

    {
        #ifndef GENERIC_BIG_METRIC
        float g_partials[16] = {0};

        calculate_metric_generic(position, g_metric, cfg);
        calculate_partial_derivatives_generic(position, g_partials, cfg);

        lightray_acceleration = calculate_acceleration(velocity, g_metric, g_partials);
        #else
        float g_partials_big[64] = {0};

        calculate_metric_generic_big(position, g_metric_big, cfg);
        calculate_partial_derivatives_generic_big(position, g_partials_big, cfg);

        velocity = fix_light_velocity_big(velocity, g_metric_big);
        lightray_acceleration = calculate_acceleration_big(velocity, g_metric_big, g_partials_big);
        #endif // GENERIC_BIG_METRIC
    }

    struct lightray ray;
    ray.position = position;
    ray.velocity = velocity;
    ray.acceleration = lightray_acceleration;
    ray.initial_quat = corrected.inverse_quat;
    ray.running_dlambda_dnew = 1;
    ray.terminated = 0;

    {
        float4 uobsu_upper = observer_velocity;

        #ifdef GENERIC_BIG_METRIC
        float4 uobsu_lower = lower_index_big(uobsu_upper, g_metric_big);
        #else
        float4 uobsu_lower = lower_index(uobsu_upper, g_metric);
        #endif // GENERIC_BIG_METRIC

        float final_val = dot(velocity, uobsu_lower);

        ray.ku_uobsu = final_val;
    }

    ray.sx = cx;
    ray.sy = cy;

    return ray;
};

struct lightray geodesic_to_trace_ray(float4 position, float4 velocity, dynamic_config_space const struct dynamic_config* cfg)
{
    struct corrected_lightray corrected = correct_lightray(position, velocity, cfg);

    position = corrected.position;
    velocity = corrected.velocity;

    float4 lightray_acceleration = (float4)(0,0,0,0);

    #ifdef IS_CONSTANT_THETA
    position.z = M_PIf/2;
    velocity.z = 0;
    #endif // IS_CONSTANT_THETA

    #ifndef GENERIC_BIG_METRIC
    float g_metric[4] = {0};
    #else
    float g_metric_big[16] = {0};
    #endif // GENERIC_BIG_METRIC

    {
        #ifndef GENERIC_BIG_METRIC
        float g_partials[16] = {0};

        calculate_metric_generic(position, g_metric, cfg);
        calculate_partial_derivatives_generic(position, g_partials, cfg);

        lightray_acceleration = calculate_acceleration(velocity, g_metric, g_partials);
        #else
        float g_partials_big[64] = {0};

        calculate_metric_generic_big(position, g_metric_big, cfg);
        calculate_partial_derivatives_generic_big(position, g_partials_big, cfg);

        lightray_acceleration = calculate_acceleration_big(velocity, g_metric_big, g_partials_big);
        #endif // GENERIC_BIG_METRIC
    }

    struct lightray ray;
    ray.position = position;
    ray.velocity = velocity;
    ray.acceleration = lightray_acceleration;
    ray.initial_quat = corrected.inverse_quat;
    ray.ku_uobsu = 1;
    ray.running_dlambda_dnew = 1;
    ray.terminated = 0;

    return ray;
};

__kernel
void init_inertial_ray(__global float4* g_generic_position_in, int ray_count,
                       __global struct lightray* metric_rays, __global int* metric_ray_count,
                       __global float4* e0, __global float4* e1, __global float4* e2, __global float4* e3,
                       __global float3* geodesic_basis_speed,
                       dynamic_config_space const struct dynamic_config* cfg)
{
    int id = get_global_id(0);

    if(id >= ray_count)
        return;

    float4 velocity = get_timelike_vector(geodesic_basis_speed[id], 1, e0[id], e1[id], e2[id], e3[id]);

    float4 position = g_generic_position_in[id];

    struct lightray trace = geodesic_to_trace_ray(position, velocity, cfg);
    trace.sx = 0;
    trace.sy = 0;

    metric_rays[id] = trace;

    if(id == 0)
        *metric_ray_count = ray_count;
}

__kernel
void init_rays_generic(__global const float4* g_generic_camera_in, __global const float4* g_camera_quat,
                       __global struct lightray* metric_rays, __global int* metric_ray_count,
                       int width, int height,
                       __global const int* termination_buffer,
                       int prepass_width, int prepass_height,
                       int flip_geodesic_direction,
                       __global const float4* e0, __global const float4* e1, __global const float4* e2, __global const float4* e3,
                       dynamic_config_space const struct dynamic_config* cfg, dynamic_config_space const struct dynamic_feature_config* dfg,
                       int i_am_prepass)
{
    int id = get_global_id(0);

    if(id >= width * height)
        return;

    const int cx = id % width;
    const int cy = id / width;

    float3 pixel_direction = calculate_pixel_direction(cx, cy, width, height, *g_camera_quat, dfg);

    float4 at_metric = *g_generic_camera_in;

    /*{
        float g_metric[4] = {0};
        calculate_metric_generic(at_metric, g_metric, cfg);

        float len = dot_product_generic(timelike, timelike, g_metric);

        if(cx == width/2 && cy == height/2)
        {
            printf("Len %f\n", len);
        }
    }*/

    float4 bT = *e0;
    float4 observer_velocity = *e0;

    float4 le1 = *e1;
    float4 le2 = *e2;
    float4 le3 = *e3;

    pixel_direction = normalize(pixel_direction);

    float4 pixel_x = pixel_direction.x * le1;
    float4 pixel_y = pixel_direction.y * le2;
    float4 pixel_z = pixel_direction.z * le3;

    ///when people say backwards in time, what they mean is backwards in affine time, not coordinate time
    ///going backwards in coordinate time however should be identical

    ///so, the -bT path traces geodesics backwards in time, aka where did this light ray originate from?
    ///the forward geodesic path says: I'm at this point, if I were to travel at the speed of light in the direction of a pixel
    ///where would I end up?
    #ifndef FORWARD_GEODESIC_PATH
    float4 pixel_t = -bT;
    #else
    float4 pixel_t = bT;
    #endif // FORWARD_GEODESIC_PATH

    if(flip_geodesic_direction)
    {
        pixel_t = -pixel_t;
    }

    float4 lightray_velocity = pixel_x + pixel_y + pixel_z + pixel_t;
    float4 lightray_spacetime_position = at_metric;

    struct lightray ray = geodesic_to_render_ray(cx, cy, lightray_spacetime_position, lightray_velocity, observer_velocity, cfg);

    #define USE_PREPASS
    #ifdef USE_PREPASS
    if(prepass_width != width && prepass_height != height)
    {
        float fx = (float)cx / width;
        float fy = (float)cy / height;

        int lx = round(fx * prepass_width);
        int ly = round(fy * prepass_height);

        if(should_early_terminate(lx-1, ly, prepass_width, prepass_height, termination_buffer) &&
           should_early_terminate(lx, ly, prepass_width, prepass_height, termination_buffer) &&
           should_early_terminate(lx+1, ly, prepass_width, prepass_height, termination_buffer) &&
           should_early_terminate(lx, ly-1, prepass_width, prepass_height, termination_buffer) &&
           should_early_terminate(lx, ly+1, prepass_width, prepass_height, termination_buffer))
        {
            ray.terminated = 2;
        }
    }
    #endif // USE_PREPASS

    if(i_am_prepass || !GET_FEATURE(adaptive_sampling, dfg) || GET_FEATURE(use_triangle_rendering, dfg))
    {
        if(id == 0)
            *metric_ray_count = height * width;

        metric_rays[id] = ray;
    }
    else
    {
        if(id == 0)
            *metric_ray_count = (height * width)/4;

        if((cx % 2) != 0 || (cy % 2) != 0)
            return;

        metric_rays[(cy/2) * (width/2) + cx/2] = ray;
    }
}

float4 fix_light_velocity(float4 position, float4 velocity, bool always_lightlike, dynamic_config_space const struct dynamic_config* cfg)
{
    float v1 = position.x;
    float v2 = position.y;
    float v3 = position.z;
    float v4 = position.w;

    float iv1 = velocity.x;
    float iv2 = velocity.y;
    float iv3 = velocity.z;
    float iv4 = velocity.w;

    return (float4){FIX_LIGHT0, FIX_LIGHT1, FIX_LIGHT2, FIX_LIGHT3};
}

///https://www.math.kit.edu/ianm3/lehre/geonumint2009s/media/gni_by_stoermer-verlet.pdf
///todo:
///it would be useful to be able to combine data from multiple ticks which are separated by some delta, but where I don't have control over that delta
///I wonder if a taylor series expansion of F(y + dt) might be helpful
///this is actually regular velocity verlet with no modifications https://en.wikipedia.org/wiki/Verlet_integration#Velocity_Verlet
void step_verlet(float4 position, float4 velocity, float4 acceleration, bool always_lightlike, float ds, float4* __restrict__ position_out, float4* __restrict__ velocity_out, float4* __restrict__ acceleration_out, float* __restrict__ dLambda_dNew, dynamic_config_space const struct dynamic_config* cfg, dynamic_config_space const struct dynamic_feature_config* dfg)
{
    float4 next_position = position + velocity * ds + 0.5f * acceleration * ds * ds;
    float4 intermediate_next_velocity = velocity + acceleration * ds;

    float4 next_acceleration;

    ///handles always_lightlike on the cpu side
    {
        float v1 = next_position.x;
        float v2 = next_position.y;
        float v3 = next_position.z;
        float v4 = next_position.w;

        float iv1 = intermediate_next_velocity.x;
        float iv2 = intermediate_next_velocity.y;
        float iv3 = intermediate_next_velocity.z;
        float iv4 = intermediate_next_velocity.w;

        float TEMPORARIES0;

        #ifdef GENERIC_CONSTANT_THETA
        v3 = M_PI/2;
        iv3 = 0;
        #endif // GENERIC_CONSTANT_THETA

        next_acceleration.x = GEO_ACCEL0;
        next_acceleration.y = GEO_ACCEL1;

        #ifndef GENERIC_CONSTANT_THETA
        next_acceleration.z = GEO_ACCEL2;
        #else
        next_acceleration.z = 0;
        #endif // GENERIC_CONSTANT_THETA

        next_acceleration.w = GEO_ACCEL3;
    }

    float4 next_velocity = velocity + 0.5f * (acceleration + next_acceleration) * ds;

    //next_position = handle_coordinate_periodicity(next_position, cfg);

    //float4 final_velocity = fix_light_velocity(next_position, next_velocity, always_lightlike, cfg);

    float max_divisor = max(max(fabs(next_velocity.x), fabs(next_velocity.y)), max(fabs(next_velocity.z), fabs(next_velocity.w)));
    //float max_divisor = fabs(next_velocity.x);
    float K = 1/max_divisor;

    ///so. In the x position we have dt/dlambda
    ///and our whole ray is parameterised as dX/dlambda
    ///we have a new divisor for a new parameter, here multiplication by a constant K
    ///we want to relate the old parameterisatin to the new parameterisation
    ///dX/dNew = dlambda/dNew * dX/dlambda
    ///dX/dNew = (dX/dLambda) * K
    ///1/dNew = K/dLambda
    ///dLambda/dNew = K, obviously ok i'm bad at rearranging simple equations today
    ///but ok, so, if we want to recover the original, we have dX/dNew
    ///so (dX/dNew) / (dLambda/dNew) = dX/dLambda
    ///= (dX/dNew) / K
    ///all of this is fairly obvious but its worth spelling out

    if(!GET_FEATURE(reparameterisation, dfg))
        K = 1;

    if(dLambda_dNew)
        *dLambda_dNew = K;

    *position_out = next_position;

    *velocity_out = next_velocity * K;
    ///ok, this is a second order coordinate change
    ///at each step we're making a linear coordinate transform, which means that d^2u/dx^2 is 0, and I think this is justifiable
    *acceleration_out = next_acceleration * K * K;
}

void step_euler(float4 position, float4 velocity, float ds, float4* position_out, float4* velocity_out, dynamic_config_space const struct dynamic_config* cfg)
{
    #ifndef GENERIC_BIG_METRIC
    float g_metric[4] = {};
    float g_partials[16] = {};
    #else
    float g_metric_big[16] = {};
    float g_partials_big[64] = {};
    #endif // GENERIC_BIG_METRIC

    #ifndef GENERIC_BIG_METRIC
    calculate_metric_generic(position, g_metric, cfg);
    calculate_partial_derivatives_generic(position, g_partials, cfg);

    float4 lacceleration = calculate_acceleration(velocity, g_metric, g_partials);
    #else
    calculate_metric_generic_big(position, g_metric_big, cfg);
    calculate_partial_derivatives_generic_big(position, g_partials_big, cfg);

    float4 lacceleration = calculate_acceleration_big(velocity, g_metric_big, g_partials_big);
    #endif // GENERIC_BIG_METRIC

    velocity += lacceleration * ds;
    position += velocity * ds;

    *position_out = position;
    *velocity_out = velocity;
}

float get_distance_to_object(float4 polar, dynamic_config_space const struct dynamic_config* cfg)
{
    float v1 = polar.x;
    float v2 = polar.y;
    float v3 = polar.z;
    float v4 = polar.w;

    float result = DISTANCE_FUNC;

    return result;
}

enum ds_result
{
    DS_NONE,
    DS_SKIP,
    DS_RETURN,
};

#ifdef ADAPTIVE_PRECISION

#define I_HATE_COMPUTERS (256*256)

float acceleration_to_precision(float4 acceleration, float max_acceleration, float* next_ds_out)
{
    float uniform_coordinate_precision_divisor = max(max(W_V1, W_V2), max(W_V3, W_V4));

    float current_acceleration_err = fast_length(acceleration * (float4)(W_V1, W_V2, W_V3, W_V4)) * 0.01f;
    current_acceleration_err /= uniform_coordinate_precision_divisor;

    float experienced_acceleration_change = current_acceleration_err;

    float err = max_acceleration;

    //#define MIN_STEP 0.00001f
    //#define MIN_STEP 0.000001f

    float max_timestep = 100000;

    float diff = experienced_acceleration_change * I_HATE_COMPUTERS;

    if(diff < err * I_HATE_COMPUTERS / pow(max_timestep, 2))
        diff = err * I_HATE_COMPUTERS / pow(max_timestep, 2);

    ///of course, as is tradition, whatever works for kerr does not work for alcubierre
    ///the sqrt error calculation is significantly better for alcubierre, largely in terms of having no visual artifacts at all
    ///whereas the pow version is nearly 2x faster for kerr
    float next_ds = native_sqrt(((err * I_HATE_COMPUTERS) / diff));

    *next_ds_out = next_ds;

    return diff;
}

int calculate_ds_error(float current_ds, float4 next_acceleration, float4 acceleration, float max_acceleration, float* next_ds_out, dynamic_config_space const struct dynamic_feature_config* dfg)
{
    float next_ds = 0;
    float diff = acceleration_to_precision(next_acceleration, max_acceleration, &next_ds);

    ///produces strictly worse results for kerr
    next_ds = 0.99f * current_ds * clamp(next_ds / current_ds, 0.3f, 2.f);

    float min_step = GET_FEATURE(min_step, dfg);

    next_ds = max(next_ds, min_step);

    *next_ds_out = next_ds;

    float err = max_acceleration;

    #ifdef SINGULARITY_DETECTION
    if(next_ds == min_step && (diff/I_HATE_COMPUTERS) > err * 10000)
        return DS_RETURN;
    #endif // SINGULARITY_DETECTION

    if(next_ds < current_ds/1.95f)
        return DS_SKIP;

    return DS_NONE;
}
#endif // ADAPTIVE_PRECISION

bool ray_plane_intersection(float3 plane_origin, float3 plane_normal, float3 ray_origin, float3 ray_direction, float* t)
{
    float denom = dot(ray_direction, plane_normal);

    if(fabs(denom) < 0.000001f)
    {
        *t = 0;
        return false;
    }

    *t = dot(plane_origin - ray_origin, plane_normal) / denom;
    return true;
}

bool ray_intersects_triangle(float3 origin, float3 direction, float3 v0, float3 v1, float3 v2, float* t_out, float* u_out, float* v_out)
{
    float eps = 0.0000001;
    float3 edge1, edge2, h, s, q;
    float a,f,u,v;
    edge1 = v1 - v0;
    edge2 = v2 - v0;

    h = cross(direction, edge2);
    a = dot(edge1, h);

    if (a > -eps && a < eps)
        return false;    // This ray is parallel to this triangle.

    f = 1.0/a;
    s = origin - v0;

    u = f * dot(s, h);

    if (u < 0.0 || u > 1.0)
        return false;

    q = cross(s, edge1);

    v = f * dot(direction, q);

    if (v < 0.0 || u + v > 1.0)
        return false;

    float t = f * dot(edge2, q);

    if(t_out)
        *t_out = t;

    if(u_out)
        *u_out = u;

    if(v_out)
        *v_out = v;

    return true;
}

int mod(int a, int b)
{
    int r = a % b;
    return r < 0 ? r + b : r;
}

__kernel void pull_to_geodesics(__global struct object* current_pos, __global float4* geodesic_out, int max_path_length, int object_count)
{
    int id = get_global_id(0);

    if(id >= object_count)
        return;

    geodesic_out[0 * max_path_length + id] = current_pos[id].pos;
}

float fast_fmod(float a, float b)
{
    return a - b * trunc(a/b);
}

float sign_of_sin(float x)
{
    x = positive_fmod(x, 2 * M_PI);

    if(x == 0 || x == M_PI)
        return 0;

    if(x > 0 && x < M_PI)
        return 1;

    return -1;
}

float sign_of_cos(float x)
{
    x = positive_fmod(x, 2 * M_PI);

    if(x == M_PI/2 || x == 3 * M_PI/2)
        return 0;

    if(x < M_PI/2 || x >= 3 * M_PI/2)
        return 1;

    return -1;
}

float pseudo_atan_tan(float in)
{
    return positive_fmod(in + M_PI, 2 * M_PI) - M_PI;
}

float fast_pseudo_atan2(float arg)
{
    ///top = sin(x)
    ///bottom = cos(x)

    ///s_y = sin(x)
    ///s_x = cos(x)
    ///atan(s_y/s_x) == atan(sin/cos) == atan(tan) == x

    float s_x = sign_of_cos(arg);
    float s_y = sign_of_sin(arg);

    if(s_x >= 0)
        return pseudo_atan_tan(arg);

    if(s_x < 0 && s_y >= 0)
        return pseudo_atan_tan(arg) + M_PI;

    if(s_x < 0 && s_y < 0)
        return pseudo_atan_tan(arg) - M_PI;

    if(s_x == 0 && s_y > 0)
        return M_PI/2;

    if(s_x == 0 && s_y < 0)
        return -M_PI/2;

    return pseudo_atan_tan(arg);
}

float circular_diff(float f1, float f2, float period)
{
    f1 = f1 * (2 * M_PI/period);
    f2 = f2 * (2 * M_PI/period);

    //return period * fast_pseudo_atan2(f2 - f1) / (2 * M_PI);
    return period * atan2(native_sin(f2 - f1), native_cos(f2 - f1)) / (2 * M_PI);
}

float2 circular_diff2(float2 f1, float2 f2)
{
    return (float2)(circular_diff(f1.x, f2.x, 1.f), circular_diff(f1.y, f2.y, 1.f));
}

float circular_diff_p(float f1, float f2, float period)
{
    return circular_diff(f1, f2, period);
}


float4 periodic_diff(float4 in1, float4 in2, float4 periods)
{
    float4 ret = in1 - in2;

    #define CHECK_PERIOD(v) if(periods.v != 0)\
                            {ret.v = circular_diff_p(in2.v, in1.v, periods.v);}
    CHECK_PERIOD(x)
    CHECK_PERIOD(y)
    CHECK_PERIOD(z)
    CHECK_PERIOD(w)

    return ret;
}

#define TRI_GEODESIC_SKIP 1
#define TRI_RAY_SKIP 1

#define FAST_TRI
#ifdef FAST_TRI
#define TRI_GEODESIC_SKIP 8
#define TRI_RAY_SKIP 8
#endif

///float4 tE1 = coordinate_to_tetrad_basis(approximate_basis[0], e_lo[0], e_lo[1], e_lo[2], e_lo[3]);

__kernel
void subsample_tri_quantity(int count, __global const int* geodesic_counts, __global const float4* geodesic_path, __global const float4* geodesic_velocities, __global const float* geodesic_ds,
                            __global const float4* t_e0, __global const float4* t_e1, __global const float4* t_e2, __global const float4* t_e3,
                            dynamic_config_space const struct dynamic_config* cfg,
                            int element_size, __global const char* data_in, __global char* data_out, __global int* out_counts)
{
    int id = get_global_id(0);

    if(id >= count)
        return;

    int cnt = geodesic_counts[id];

    if(cnt == 0)
    {
        out_counts[id] = 0;
        return;
    }

    float4 periods = get_coordinate_period(cfg);

    //#define FIXED_SKIPPING
    #ifdef FIXED_SKIPPING
    int skip = TRI_GEODESIC_SKIP;

    int next_count = 0;

    for(int kk=0; kk < cnt - skip; kk += skip)
    {
        int current_idx = kk * count + id;
        int out_idx = next_count * count + id;

        for(int i=0; i < element_size; i++)
        {
            data_out[out_idx * element_size + i] = data_in[current_idx * element_size + i];
        }

        next_count++;
    }

    printf("Count %i real %i\n", next_count, cnt);

    out_counts[id] = next_count;
    #endif // FIXED_SKIPPING

    //#define CAPPED_SKIPPING
    #ifdef CAPPED_SKIPPING
    int next_count = 0;

    int max_steps = 32;
    float fskip = cnt / (float)max_steps;

    if(fskip < 1)
        fskip = 1;

    for(float fidx = 0; fidx < cnt; fidx += fskip)
    {
        int kk = fidx;

        int current_idx = kk * count + id;
        int out_idx = next_count * count + id;

        for(int i=0; i < element_size; i++)
        {
            data_out[out_idx * element_size + i] = data_in[current_idx * element_size + i];
        }

        next_count++;
    }

    //printf("Count %i real %i\n", next_count, cnt);

    out_counts[id] = next_count;
    #endif

    ///need to do this by distance, not time
    //#define DS_SKIPPING
    #ifdef DS_SKIPPING
    float max_ds = 0.5;
    float current_ds_budget = 0;

    int next_count = 0;

    {
        int current_idx = 0 * count + id;

        current_ds_budget = geodesic_ds[current_idx];

        for(int i=0; i < element_size; i++)
            data_out[id * element_size + i] = data_in[current_idx * element_size + i];

        next_count = 1;
    }

    for(int kk=1; kk < cnt; kk++)
    {
        int current_idx = kk * count + id;

        current_ds_budget += geodesic_ds[current_idx];

        if(current_ds_budget < max_ds)
            continue;

        while(current_ds_budget >= max_ds)
            current_ds_budget -= max_ds;

        int out_idx = next_count * count + id;

        for(int i=0; i < element_size; i++)
            data_out[out_idx * element_size + i] = data_in[current_idx * element_size + i];

        next_count++;
    }

    printf("Count %i real %i\n", next_count, cnt);

    out_counts[id] = next_count;
    #endif

    #define DISTANCE_SKIPPING
    #ifdef DISTANCE_SKIPPING
    float max_dist = 0.005;
    float current_budget = 0;
    float4 last_position;

    int next_count = 0;

    {
        int current_idx = 0 * count + id;

        last_position = geodesic_path[current_idx];

        for(int i=0; i < element_size; i++)
            data_out[id * element_size + i] = data_in[current_idx * element_size + i];

        next_count = 1;
    }

    for(int kk=1; kk < cnt - 1; kk++)
    {
        int current_idx = kk * count + id;

        float4 e0 = t_e0[current_idx];
        float4 e1 = t_e1[current_idx];
        float4 e2 = t_e2[current_idx];
        float4 e3 = t_e3[current_idx];

        float4 e_lo[4];
        get_tetrad_inverse(e0, e1, e2, e3, &e_lo[0], &e_lo[1], &e_lo[2], &e_lo[3]);

        float4 current_position = geodesic_path[current_idx];

        //printf("hi %f %f %f %f which %i\n", current_position.x, current_position.y, current_position.z, current_position.w, basis.timelike_coordinate);

        float4 to_next = periodic_diff(current_position, last_position, periods);

        float4 in_tetrad = coordinate_to_tetrad_basis(to_next, e_lo[0], e_lo[1], e_lo[2], e_lo[3]);

        last_position = current_position;

        float dist = length(in_tetrad.yzw);

        current_budget += dist;

        if(current_budget < max_dist)
            continue;

        while(current_budget >= max_dist)
            current_budget -= max_dist;

        int out_idx = next_count * count + id;

        for(int i=0; i < element_size; i++)
            data_out[out_idx * element_size + i] = data_in[current_idx * element_size + i];

        next_count++;
    }

    {
        int current_idx = (cnt - 1) * count + id;
        int out_idx = next_count * count + id;

        for(int i=0; i < element_size; i++)
            data_out[out_idx * element_size + i] = data_in[current_idx * element_size + i];

        next_count++;
    }

    //printf("Count %i real %i\n", next_count, cnt);

    out_counts[id] = next_count;
    #endif
}

///todo: winding order
float3 triangle_normal(float3 v0, float3 v1, float3 v2)
{
    float3 U = v1 - v0;
    float3 V = v2 - v0;

    return normalize(cross(U, V));
}

bool ray_intersects_toblerone2(float4 global_pos, float4 next_global_pos, float3 v0, float3 v1, float3 v2, float4 object_geodesic_origin, float4 next_object_geodesic_origin,
                               float4 i_re0, float4 i_re1, float4 i_re2, float4 i_re3, ///inverse current geodesic segment tetrad
                               float4 i_ne0, float4 i_ne1, float4 i_ne2, float4 i_ne3, ///inverse next geodesic segment tetrad
                               float4 periods, float* t_out, bool debug)
{
    float3 plane_normal = normalize(cross(v1 - v0, v2 - v0));

    float4 object_pos_1 = object_geodesic_origin;
    //float4 object_pos_2 = periodic_diff(next_object_geodesic_origin, object_geodesic_origin, periods) + object_geodesic_origin;
    float4 object_pos_2 = next_object_geodesic_origin;

    float4 ray_origin = global_pos;
    float4 ray_vel = next_global_pos - global_pos;

    //float4 initial_diff = ray_origin - object_pos_1;
    ///still don't think periodic diff is 100% correct
    float4 initial_diff = periodic_diff(ray_origin, object_pos_1, periods);
    float4 initial_origin = object_pos_1;

    float4 last_pos;
    float4 last_dir;

    float next_frac = 0;

    float last_object_start_t = 0;
    float last_object_end_t = 0;

    #pragma unroll
    for(int i=0; i < 8; i++)
    {
        float frac = clamp(next_frac, 0.f, 1.f);

        float last_dt = 0;

        float4 i_e0 = mix(i_re0, i_ne0, frac);
        float4 i_e1 = mix(i_re1, i_ne1, frac);
        float4 i_e2 = mix(i_re2, i_ne2, frac);
        float4 i_e3 = mix(i_re3, i_ne3, frac);
        float4 object_position = mix(object_pos_1, object_pos_2, frac);

        {

            float4 diff = initial_diff + initial_origin - object_position;

            float4 pos = coordinate_to_tetrad_basis(diff, i_e0, i_e1, i_e2, i_e3);
            float4 dir = coordinate_to_tetrad_basis(ray_vel, i_e0, i_e1, i_e2, i_e3);

            float found_t = 0;

            if(!ray_plane_intersection(v0, plane_normal, pos.yzw, dir.yzw, &found_t))
                return false;

            last_dt = found_t;
            last_pos = pos;
            last_dir = dir;
        }

        float4 object_start_in_tetrad = coordinate_to_tetrad_basis(object_pos_1 - object_position, i_e0, i_e1, i_e2, i_e3);
        float4 object_end_in_tetrad = coordinate_to_tetrad_basis(object_pos_2 - object_position, i_e0, i_e1, i_e2, i_e3);

        float object_end_t = object_end_in_tetrad.x;
        float object_start_t = object_start_in_tetrad.x;

        last_object_end_t = object_end_t;
        last_object_start_t = object_start_t;

        ///so. here we express our tri bounds in the local tetrad, which is situated between the two
        ///we want to calculate the interpolating fraction of our next guess, which is with respect to our object bounds
        ///we have a new coordinate time position, which is last_pos.x + last_dir.x * last_dt
        ///so our new fraction is (new_x - object_start_t) / (object_end_t - object_start_t)
        float4 intersection_point = last_pos + last_dir * last_dt;

        next_frac = (intersection_point.x - object_start_t) / (object_end_t - object_start_t);
    }

    float ray_t = 0;
    bool intersected = ray_intersects_triangle(last_pos.yzw, last_dir.yzw, v0, v1, v2, &ray_t, 0, 0);

    float end_t = last_pos.x + last_dir.x * ray_t;

    if(end_t < last_object_start_t || end_t > last_object_end_t)
        return false;

    if(ray_t < 0 || ray_t > 1)
        return false;

    ///the issue is the ray time is just slightly outside of the tri time
    /*if(debug)
    {
        printf("Intersected %i time %f lower ray %f upper ray %f lower tri %f upper tri %f\n", intersected, new_x, ray_lower_t, ray_upper_t, tri_lower_t, tri_upper_t);
    }*/

    /*if(debug)
    {
        printf("%i timelike pos %f %f\n", which_coordinate_timelike, last_gintersection_point.x, last_gintersection_point.y);
    }*/

    *t_out = ray_t;

    /*if(debug)
    {
        printf("Hello %f pos1 %f %f %f %f pos2 %f %f %f %f geo1 %f %f %f %f geo2 %f %f %f %f\n", ray_t, E4(global_pos), E4(next_global_pos), E4(object_geodesic_origin), E4(next_object_geodesic_origin));
    }*/

    return intersected;

}

__kernel
void do_generic_rays (__global struct lightray* restrict generic_rays_in,
                      __global const int* restrict generic_count_in,
                      __global int* restrict ray_time_min, __global int* restrict ray_time_max,
                      dynamic_config_space const struct dynamic_config* restrict cfg,
                      dynamic_config_space const struct dynamic_feature_config* restrict dfg,
                      int width, int height,
                      int mouse_x, int mouse_y,
                      __global float4* ray_write,
                      __global int* ray_write_counts,
                      int max_write)
{
    int id = get_global_id(0);

    if(id >= *generic_count_in)
        return;

    ray_write_counts[id] = 0;

     __global struct lightray* ray = &generic_rays_in[id];

    if(ray->terminated == 2)
        return;

    float4 position = ray->position;
    float4 velocity = ray->velocity;
    float4 acceleration = ray->acceleration;

    #if 1
    if(GET_FEATURE(use_triangle_rendering, dfg))
    {
        atomic_min(ray_time_min, (int)floor(position.x));
        atomic_max(ray_time_max, (int)ceil(position.x));
    }
    #endif

    float f_in_x = fabs(velocity.x);

    #ifdef IS_CONSTANT_THETA
    position.z = M_PIf/2;
    velocity.z = 0;
    acceleration.z = 0;
    #endif // IS_CONSTANT_THETA

    float next_ds = 0.00001;

    #ifdef ADAPTIVE_PRECISION
    (void)acceleration_to_precision(acceleration, GET_FEATURE(max_acceleration_change, dfg), &next_ds);
    #endif // ADAPTIVE_PRECISION

    ///results:
    ///subambient_precision can't go above 0.5 much while in verlet mode without the size of the event horizon changing
    ///in euler mode this is actually already too low

    ///ambient precision however looks way too low at 0.01, testing up to 0.3 showed no noticable difference, needs more precise tests though
    ///only in the case without kruskals and event horizon crossings however, any precision > 0.01 is insufficient in that case
    ///this super affects being able to render alcubierre at thin shells
    float subambient_precision = 0.5;
    float ambient_precision = 0.2;

    float uniform_coordinate_precision_divisor = max(max(W_V1, W_V2), max(W_V3, W_V4));

    int loop_limit = 4096 * 4;

    #ifdef DEVICE_SIDE_ENQUEUE
    loop_limit /= 125;
    #endif // DEVICE_SIDE_ENQUEUE

    float4 last_real_pos = (float4)(0,0,0,0);
    float4 last_real_velocity = (float4)(0,0,0,0);

    float my_min = position.x;
    float my_max = position.x;

    float4 ray_quat = ray->initial_quat;

    int any_visible_tris = 0;

    #if 0
    if(GET_FEATURE(use_triangle_rendering, dfg))
        any_visible_tris = *any_visible;
    #endif

    float4 periods = get_coordinate_period(cfg);

    float running_dlambda_dnew = 1;

    int which_ray_write = 0;

    int ray_skipping = GET_FEATURE(ray_skip, dfg);

    //#pragma unroll
    for(int i=0; i < loop_limit; i++)
    {
        {
            my_min = min(my_min, position.x);
            my_max = max(my_max, position.x);
        }

        #ifdef IS_CONSTANT_THETA
        position.z = M_PIf/2;
        velocity.z = 0;
        acceleration.z = 0;
        #endif // IS_CONSTANT_THETA

        float new_max = GET_FEATURE(max_precision_radius, dfg);
        float new_min = 3;

        float4 polar_position = generic_to_spherical(position, cfg);

        #ifdef IS_CONSTANT_THETA
        polar_position.z = M_PIf/2;
        #endif // IS_CONSTANT_THETA

        float r_value = get_distance_to_object(polar_position, cfg);

        float ds = linear_val(fabs(r_value), new_min, new_max, ambient_precision, subambient_precision);

        #ifndef RK4_GENERIC
        #ifdef ADAPTIVE_PRECISION
        ds = next_ds;
        #endif // ADAPTIVE_PRECISION
        #endif // RK4_GENERIC

        if(fabs(r_value) < new_max)
        {
            ds = min(ds, ambient_precision);
        }
        else
        {
            ds = 0.1f * pow((fabs(r_value) - new_max), 1) + ambient_precision;
            //ds = (0.1 * pow((fabs(r_value) - new_max), 2) / (uniform_coordinate_precision_divisor * uniform_coordinate_precision_divisor)) + subambient_precision;
        }

        bool should_terminate = fabs(polar_position.y) >= GET_FEATURE(universe_size, dfg);

        #ifdef SINGULAR
        should_terminate |= fabs(polar_position.y) < SINGULAR_TERMINATOR;
        #endif // SINGULAR

        #ifdef HAS_CYLINDRICAL_SINGULARITY
        if(position.y < CYLINDRICAL_TERMINATOR)
            return;
        #endif // CYLINDRICAL_SINGULARITY

        #ifndef UNCONDITIONALLY_NONSINGULAR
        if(fabs(velocity.x / running_dlambda_dnew) > 1000 + f_in_x && fabs(acceleration.x / running_dlambda_dnew) > 100)
        {
            #if 1
            if(GET_FEATURE(use_triangle_rendering, dfg))
            {
                atomic_min(ray_time_min, (int)floor(my_min));
                atomic_max(ray_time_max, (int)ceil(my_max));
            }
            #endif

            return;
        }
        #endif // UNCONDITIONALLY_NONSINGULAR

        if(should_terminate)
        {
            #if 1
            if(GET_FEATURE(use_triangle_rendering, dfg))
            {
                atomic_min(ray_time_min, (int)floor(my_min));
                atomic_max(ray_time_max, (int)ceil(my_max));
            }
            #endif

            generic_rays_in[id].position = position;
            generic_rays_in[id].velocity = velocity;
            generic_rays_in[id].running_dlambda_dnew = running_dlambda_dnew;
            generic_rays_in[id].terminated = 1;

            return;
        }

        #ifdef EULER_INTEGRATION_GENERIC

        float4 next_position;
        float4 next_velocity;

        step_euler(position, velocity, ds, &next_position, &next_velocity, cfg);

        position = next_position;
        velocity = next_velocity;

        #endif // EULER_INTEGRATsION

        #ifdef VERLET_INTEGRATION_GENERIC

        float dLambda_dNew = 1;

        float4 next_position, next_velocity, next_acceleration;

        step_verlet(position, velocity, acceleration, true, ds, &next_position, &next_velocity, &next_acceleration, &dLambda_dNew, cfg, dfg);

        running_dlambda_dnew *= dLambda_dNew;

        #ifdef ADAPTIVE_PRECISION

        if(fabs(r_value) < new_max)
        {
            int res = calculate_ds_error(ds, next_acceleration, acceleration, GET_FEATURE(max_acceleration_change, dfg), &next_ds, dfg);

            if(res == DS_RETURN)
                return;

            if(res == DS_SKIP)
            {
                i--;
                continue;
            }
        }

        #endif // ADAPTIVE_PRECISION

        position = next_position;
        velocity = next_velocity;
        acceleration = next_acceleration;
        #endif // VERLET_INTEGRATION

        #ifdef RK4_GENERIC
        rk4_generic_big(&position, &velocity, &ds);
        #endif // RK4_GENERIC

        if(GET_FEATURE(use_triangle_rendering, dfg) && ((i % ray_skipping) == 0))
        {
            float4 native_position = position;

            #if (defined(GENERIC_METRIC) && defined(GENERIC_CONSTANT_THETA)) || !defined(GENERIC_METRIC) || defined(DEBUG_CONSTANT_THETA)
            {
                float4 pos_spherical = generic_to_spherical(position, cfg);

                float fsign = sign(pos_spherical.y);
                pos_spherical.y = fabs(pos_spherical.y);

                float3 pos_cart = polar_to_cartesian(pos_spherical.yzw);

                float4 quat = ray_quat;

                pos_cart = rot_quat_norm(pos_cart, quat);

                float3 next_pos_spherical = cartesian_to_polar(pos_cart);

                if(fsign < 0)
                {
                    next_pos_spherical.x = -next_pos_spherical.x;
                }

                float4 next_pos_generic = spherical_to_generic((float4)(pos_spherical.x, next_pos_spherical), cfg);

                native_position = next_pos_generic;
            }
            #endif

            if(i == 0)
            {
                last_real_pos = native_position;
            }

            float4 real_pos = native_position;

            float4 pdiff = periodic_diff(native_position, last_real_pos, periods);

            ///I think this periodic diff is only necessary in constant theta metrics?
            float4 next_real_pos = pdiff + last_real_pos;

            if(which_ray_write < max_write)
            {
                ray_write[which_ray_write * width * height + id] = next_real_pos;

                which_ray_write++;
                ray_write_counts[ray->sy * width + ray->sx] = which_ray_write;
            }

            last_real_pos = next_real_pos;
        }

        if(any(IS_DEGENERATE(position)) || any(IS_DEGENERATE(velocity)) || any(IS_DEGENERATE(acceleration)))
        {
            return;
        }
    }

    #if 1
    if(GET_FEATURE(use_triangle_rendering, dfg))
    {
        atomic_min(ray_time_min, (int)floor(my_min));
        atomic_max(ray_time_max, (int)ceil(my_max));
    }
    #endif
}

int2 get_bounds(int count, int offset, int segments)
{
    int div = count / segments;

    int2 val;
    val.x = div * offset;
    val.y = min(div * (offset + 1) + 1, count);

    if(offset == segments-1)
        val.y = count;

    return val;
}

#define SEGMENTS 4

__kernel
void generate_clip_regions(const global float4* restrict ray_write,
                           const global int* restrict ray_write_counts,
                           int max_write,
                           global float4* restrict mins_out,
                           global float4* restrict maxs_out,
                           int width, int height,
                           global float4* restrict chunked_mins,
                           global float4* restrict chunked_maxs,
                           int offset,
                           local float4* restrict lmins,
                           local float4* restrict lmaxs,
                           local char* restrict exists
                           )
{
    size_t x = get_global_id(0);
    size_t y = get_global_id(1);

    size_t lid = get_local_id(1) * get_local_size(0) + get_local_id(0);

    int count = 0;
    size_t id = y * width + x;

    if(x < width && y < height)
    {
        count = ray_write_counts[id];
    }

    float4 current_min = (float4)(0,0,0,0);
    float4 current_max = (float4)(0,0,0,0);

    if(count > 0)
    {
        int2 bounds = get_bounds(count, offset, SEGMENTS);

        int has_any = 0;

        for(int i=bounds.x; i < bounds.y; i++)
        {
            float4 val = ray_write[i * width * height + id];

            if(has_any == 0)
            {
                current_min = val;
                current_max = val;
                has_any = 1;
            }
            else
            {
                current_min = min(current_min, val);
                current_max = max(current_max, val);
            }
        }

        lmins[lid] = current_min;
        lmaxs[lid] = current_max;
    }

    if(x < width && y < height)
    {
        mins_out[id] = current_min;
        maxs_out[id] = current_max;
    }

    exists[lid] = count > 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    if(get_local_id(0) == 0 && get_local_id(1) == 0)
    {
        float4 clip_min = (float4)(0,0,0,0);
        float4 clip_max = (float4)(0,0,0,0);
        int any_clip = 0;

        size_t size_x = get_local_size(0);
        size_t size_y = get_local_size(1);

        for(size_t ly=0; ly < size_y; ly++)
        {
            for(size_t lx=0; lx < size_x; lx++)
            {
                size_t lid = ly * size_x + lx;

                if(exists[lid] == 0)
                    continue;

                if(any_clip == 0)
                {
                    clip_min = lmins[lid];
                    clip_max = lmaxs[lid];
                    any_clip = 1;
                }
                else
                {
                    clip_min = min(clip_min, lmins[lid]);
                    clip_max = max(clip_max, lmaxs[lid]);
                }
            }
        }
        size_t block_x = get_group_id(0);
        size_t block_y = get_group_id(1);

        size_t block_width = get_num_groups(0);

        chunked_mins[block_y * block_width + block_x] = clip_min;
        chunked_maxs[block_y * block_width + block_x] = clip_max;
    }
}

struct computed
{
    ///in coordinate space, in global coordinates, not tied to a geodesic
    float4 min_extents;
    float4 max_extents;

    int root_tri_id;
    int geodesic_segment;
};

#define COMPUTED_SKIP 1

__kernel
void generate_computed_tris(global struct triangle* tris, int tri_count,
                            int object_count,
                            global float4* object_geodesics, global int* object_geodesic_counts,
                            global float4* p_e0, global float4* p_e1, global float4* p_e2, global float4* p_e3,
                            global struct computed* ctris, global int* ctri_count,
                            global int* restrict ray_time_min, global int* restrict ray_time_max,
                            dynamic_config_space const struct dynamic_config* cfg)
{
    int tri_id = get_global_id(0);

    if(tri_id >= tri_count)
        return;

    struct triangle tri = tris[tri_id];

    int count = object_geodesic_counts[tri.parent];

    int stride = object_count;

    int skip = COMPUTED_SKIP;

    float4 coordinate_period = get_coordinate_period(cfg);

    for(int cc=0; cc < count - skip; cc+=skip)
    {
        float4 native_current = object_geodesics[cc * object_count + tri.parent];
        float4 native_next = object_geodesics[(cc + skip) * object_count + tri.parent];

        if(!range_overlaps_general(native_current.x, native_next.x, *ray_time_min, *ray_time_max, coordinate_period.x))
            continue;

        ///todo: precalculate me
        float4 min_extents;
        float4 max_extents;

        {
            ///current tetrads
            float4 s_e0 = p_e0[cc * stride + tri.parent];
            float4 s_e1 = p_e1[cc * stride + tri.parent];
            float4 s_e2 = p_e2[cc * stride + tri.parent];
            float4 s_e3 = p_e3[cc * stride + tri.parent];

            ///next tetrads
            float4 n_e0 = p_e0[(cc + skip) * stride + tri.parent];
            float4 n_e1 = p_e1[(cc + skip) * stride + tri.parent];
            float4 n_e2 = p_e2[(cc + skip) * stride + tri.parent];
            float4 n_e3 = p_e3[(cc + skip) * stride + tri.parent];

            ///triangle coordinates in local space
            float4 vert_0 = (float4)(0, tri.v0x, tri.v0y, tri.v0z);
            float4 vert_1 = (float4)(0, tri.v1x, tri.v1y, tri.v1z);
            float4 vert_2 = (float4)(0, tri.v2x, tri.v2y, tri.v2z);

            ///start triangle coordinates (as a vector) in tangent space
            float4 s_coordinate_v0 = tetrad_to_coordinate_basis(vert_0, s_e0, s_e1, s_e2, s_e3);
            float4 s_coordinate_v1 = tetrad_to_coordinate_basis(vert_1, s_e0, s_e1, s_e2, s_e3);
            float4 s_coordinate_v2 = tetrad_to_coordinate_basis(vert_2, s_e0, s_e1, s_e2, s_e3);

            ///end triangle coordinates (as a vector) in tangent space
            float4 e_coordinate_v0 = tetrad_to_coordinate_basis(vert_0, n_e0, n_e1, n_e2, n_e3);
            float4 e_coordinate_v1 = tetrad_to_coordinate_basis(vert_1, n_e0, n_e1, n_e2, n_e3);
            float4 e_coordinate_v2 = tetrad_to_coordinate_basis(vert_2, n_e0, n_e1, n_e2, n_e3);

            //printf("t1 %f %f %f t2 %f %f %f\n", s_coordinate_v0.x, s_coordinate_v1.x, s_coordinate_v2.x,
            //                                    e_coordinate_v0.x, e_coordinate_v1.x, e_coordinate_v2.x
            //       );


            ///Approximate triangle coordinates in coordinate space
            float4 sgv0 = s_coordinate_v0 + native_current;
            float4 sgv1 = s_coordinate_v1 + native_current;
            float4 sgv2 = s_coordinate_v2 + native_current;

            ///Approximate triangle coordinates in coordinate space
            float4 egv0 = e_coordinate_v0 + native_next;
            float4 egv1 = e_coordinate_v1 + native_next;
            float4 egv2 = e_coordinate_v2 + native_next;

            float4 min1 = min(sgv0, min(sgv1, sgv2));
            float4 min2 = min(egv0, min(egv1, egv2));

            float4 max1 = max(sgv0, max(sgv1, sgv2));
            float4 max2 = max(egv0, max(egv1, egv2));

            min_extents = min(min1, min2);
            max_extents = max(max1, max2);
        }

        struct computed ctri;
        ctri.root_tri_id = tri_id;
        ctri.geodesic_segment = cc * object_count + tri.parent;
        ctri.min_extents = min_extents;
        ctri.max_extents = max_extents;

        int id = atomic_inc(ctri_count);

        ctris[id] = ctri;
    }
}

///if i sorted computed tris, generate_tri_lists2 could binary search based on its clipping region.x
///clipping regions

__kernel
void generate_tri_lists2(global struct computed* ctri,
                        global int* ctri_count,
                        global int* chunked_tri_list_out,
                        global int* chunked_tri_list_count,
                        global int* chunked_global_count,
                        global int* chunked_offsets,
                        int max_tris,
                        global float4* chunked_mins,
                        global float4* chunked_maxs,
                        int chunk_x, int chunk_y,
                        int width, int height,
                        dynamic_config_space const struct dynamic_config* cfg)
{
    size_t idx = get_global_id(0);
    size_t idy = get_global_id(1);

    int chunk_dim_x = get_chunk_size(width, chunk_x);
    int chunk_dim_y = get_chunk_size(height, chunk_y);

    if(idx >= chunk_dim_x || idy >= chunk_dim_y)
        return;

    float4 coordinate_period = get_coordinate_period(cfg);

    size_t cid = idy * chunk_dim_x + idx;

    float4 chunk_clip_min = chunked_mins[cid];
    float4 chunk_clip_max = chunked_maxs[cid];

    if(all(chunk_clip_min == chunk_clip_max))
    {
        chunked_tri_list_count[cid] = 0;
        return;
    }

    int tris_num = *ctri_count;

    int chunked_count = 0;

    for(int i=0; i < tris_num; i++)
    {
        struct computed my_tri = ctri[i];

        ///could improve memory layout
        if(!range_overlaps_general4(chunk_clip_min, chunk_clip_max, my_tri.min_extents, my_tri.max_extents, coordinate_period))
            continue;

        chunked_count++;
    }

    int root_offset = 0;

    if(chunked_count > 0)
        root_offset = atomic_add(chunked_global_count, chunked_count);

    if(root_offset + chunked_count > max_tris || chunked_count == 0)
    {
        chunked_tri_list_count[cid] = 0;
        chunked_offsets[cid] = 0;
        return;
    }

    chunked_tri_list_count[cid] = chunked_count;

    int chunked_id = 0;

    for(int i=0; i < tris_num; i++)
    {
        struct computed my_tri = ctri[i];

        ///could improve memory layout
        if(!range_overlaps_general4(chunk_clip_min, chunk_clip_max, my_tri.min_extents, my_tri.max_extents, coordinate_period))
            continue;

        int my_id = chunked_id++;

        chunked_tri_list_out[root_offset + my_id] = i;
    }

    chunked_offsets[cid] = root_offset;
}

__kernel
void render_chunked_tris(global const struct triangle* const tris,
                         global struct computed* ctri, global int* ctri_count,
                         int object_count,
                         __write_only image2d_t screen,
                         global int* chunked_tri_list,
                         global int* chunked_tri_list_count,
                         global int* chunked_offsets,
                         int width,
                         int height,
                         int chunk_x,
                         int chunk_y,
                         global float4* ray_segments,
                         global int* ray_segments_count,
                         global float4* object_geodesics, global int* object_geodesic_counts,
                         global float4* p_e0, global float4* p_e1, global float4* p_e2, global float4* p_e3,
                         global const float4* restrict inverse_e0s, __global const float4* restrict inverse_e1s, __global const float4* restrict inverse_e2s, __global const float4* restrict inverse_e3s,
                         global float4* fine_clip_min, global float4* fine_clip_max,
                         global int* already_rendered,
                         dynamic_config_space const struct dynamic_config* cfg,
                         dynamic_config_space const struct dynamic_feature_config* dfg,
                         float mouse_x,
                         float mouse_y,
                         int offset
                         )
{
    int ray_x = get_global_id(0);
    int ray_y = get_global_id(1);

    if(ray_x >= width || ray_y >= height)
        return;

    int ray_id = ray_y * width + ray_x;

    if(already_rendered[ray_id])
        return;

    int chunk_idx = get_group_id(0);
    int chunk_idy = get_group_id(1);

    int chunk_dim_x = get_chunk_size(width, chunk_x);
    int chunk_id = chunk_idy * chunk_dim_x + chunk_idx;

    int found_tris = chunked_tri_list_count[chunk_id];

    if(found_tris == 0)
        return;

    int my_ray_segment_count = ray_segments_count[ray_id];

    int2 bounds = get_bounds(my_ray_segment_count, offset, SEGMENTS);

    /*if(found_tris > 0)
    {
        write_imagef(screen, (int2)(ray_x, ray_y), (float4)((float)found_tris / 100.f, 0, 0, 1));
        return;
    }*/

    int root_offset = chunked_offsets[chunk_id];

    //float4 ray_clip_min = fine_clip_min[ray_id];
    //float4 ray_clip_max = fine_clip_max[ray_id];

    float4 periods = get_coordinate_period(cfg);

    float last_ray_t = FLT_MAX;
    float last_ray_frac = 0;

    int last_tri_id = -1;

    for(int rs = bounds.x; rs < bounds.y - 1; rs++)
    {
        float4 current_pos = ray_segments[rs * width * height + ray_id];
        float4 next_pos = ray_segments[(rs+1) * width * height + ray_id];

        ///...could i stuff you in local memory? or even an array?
        for(int t=0; t < found_tris; t++)
        {
            int tri_id = chunked_tri_list[root_offset + t];

            struct computed tri = ctri[tri_id];
            struct triangle ttri = tris[tri.root_tri_id];

            int stride = object_count;

            float4 min_extents = tri.min_extents;
            float4 max_extents = tri.max_extents;

            ///could improve the memroy layout to have min_extents and max_extents accessible
            //if(!range_overlaps_general4(ray_clip_min, ray_clip_max, min_extents, max_extents, periods))
            //    continue;

            if(!range_overlaps_general4(current_pos, next_pos, min_extents, max_extents, periods))
                continue;

            ///current position of triangle in coordinate space
            float4 native_current = object_geodesics[tri.geodesic_segment];
            float4 native_next = object_geodesics[tri.geodesic_segment + stride * COMPUTED_SKIP];

            ///current inverse tetrads
            float4 s_ie0 = inverse_e0s[tri.geodesic_segment];
            float4 s_ie1 = inverse_e1s[tri.geodesic_segment];
            float4 s_ie2 = inverse_e2s[tri.geodesic_segment];
            float4 s_ie3 = inverse_e3s[tri.geodesic_segment];

            ///next inverse tetrads
            float4 n_ie0 = inverse_e0s[tri.geodesic_segment + stride * COMPUTED_SKIP];
            float4 n_ie1 = inverse_e1s[tri.geodesic_segment + stride * COMPUTED_SKIP];
            float4 n_ie2 = inverse_e2s[tri.geodesic_segment + stride * COMPUTED_SKIP];
            float4 n_ie3 = inverse_e3s[tri.geodesic_segment + stride * COMPUTED_SKIP];

            float3 v0 = (float3)(ttri.v0x, ttri.v0y, ttri.v0z);
            float3 v1 = (float3)(ttri.v1x, ttri.v1y, ttri.v1z);
            float3 v2 = (float3)(ttri.v2x, ttri.v2y, ttri.v2z);

            float ray_t = FLT_MAX;

            ///todo: sort by minimum ray intersection length? whats going on here
            ///wait. do i want the last hit??
            ///no its because i might hit multiple in this segment, rip
            ///i'm doing this raywise, but traversing the entire ray, so early terminate is borked
            ///this is very inefficient, go along the way and then check the tris
            ///or maybe not actually, there's some good short circuiting i can do, and the tris need more memory fetche
            if(ray_intersects_toblerone2(current_pos, next_pos, v0, v1, v2, native_current, native_next,
                                         s_ie0, s_ie1, s_ie2, s_ie3, n_ie0, n_ie1, n_ie2, n_ie3, periods, &ray_t, ray_x == mouse_x && ray_y == mouse_y))
            {
                if(last_ray_t != FLT_MAX && ray_t >= last_ray_t)
                    continue;

                last_ray_t = ray_t;

                last_tri_id = tri_id;
            }
        }

        if(last_ray_t != FLT_MAX)
            break;
    }

    ///1030, 280
    if(last_tri_id >= 0)
    {
        struct computed found_ctri = ctri[last_tri_id];
        struct triangle tri = tris[found_ctri.root_tri_id];

        float3 v0 = (float3)(tri.v0x, tri.v0y, tri.v0z);
        float3 v1 = (float3)(tri.v1x, tri.v1y, tri.v1z);
        float3 v2 = (float3)(tri.v2x, tri.v2y, tri.v2z);

        float3 ncol = fabs(triangle_normal(v0, v1, v2));

        if(GET_FEATURE(redshift, dfg))
        {

        }

        write_imagef(screen, (int2)(ray_x, ray_y), (float4)(ncol.x, ncol.y, ncol.z, 1));

        already_rendered[ray_id] = 1;
    }
}

__kernel
void get_geodesic_path(__global struct lightray* generic_rays_in,
                       __global float4* positions_out,
                       __global float4* velocities_out,
                       __global float* ds_out,
                       __global int* generic_count_in,
                       int max_path_length,
                       dynamic_config_space const struct dynamic_config* cfg,
                       dynamic_config_space const struct dynamic_feature_config* dfg,
                       __global int* count_out)
{
    int id = get_global_id(0);

    if(id >= *generic_count_in)
        return;

    __global struct lightray* ray = &generic_rays_in[id];

    float4 position = ray->position;
    float4 velocity = ray->velocity;
    float4 acceleration = ray->acceleration;

    float f_in_x = fabs(velocity.x);

    //printf("Pos %f %f %f %f\n", position.x,position.y,position.z,position.w);

    #ifdef IS_CONSTANT_THETA
    position.z = M_PIf/2;
    velocity.z = 0;
    acceleration.z = 0;
    #endif // IS_CONSTANT_THETA

    #ifdef ADAPTIVE_PRECISION
    float max_accel = min(0.00001000f, GET_FEATURE(max_acceleration_change, dfg));
    #endif // ADAPTIVE_PRECISION

    float next_ds = 0.00001;

    #ifdef ADAPTIVE_PRECISION
    (void)acceleration_to_precision(acceleration, max_accel, &next_ds);
    #endif // ADAPTIVE_PRECISION

    float subambient_precision = 0.5;
    float ambient_precision = 0.2;

    int stride_out = *generic_count_in;
    int bufc = 0;

    float4 periods = get_coordinate_period(cfg);
    float4 last_pos_generic;

    float running_dlambda_dnew = 1;

    //#pragma unroll
    for(int i=0; i < max_path_length; i++)
    {
        #ifdef IS_CONSTANT_THETA
        position.z = M_PIf/2;
        velocity.z = 0;
        acceleration.z = 0;
        #endif // IS_CONSTANT_THETA

        float new_max = GET_FEATURE(max_precision_radius, dfg);
        float new_min = 3;

        float4 polar_position = generic_to_spherical(position, cfg);

        #ifdef IS_CONSTANT_THETA
        polar_position.z = M_PIf/2;
        #endif // IS_CONSTANT_THETA

        float r_value = get_distance_to_object(polar_position, cfg);

        float ds = linear_val(fabs(r_value), new_min, new_max, ambient_precision, subambient_precision);

        #ifndef RK4_GENERIC
        #ifdef ADAPTIVE_PRECISION
        ds = next_ds;
        #endif // ADAPTIVE_PRECISION
        #endif // RK4_GENERIC

        if(fabs(r_value) < new_max)
        {
            ds = min(ds, ambient_precision);
        }
        else
        {
            ds = 0.1 * pow((fabs(r_value) - new_max), 1) + ambient_precision;
        }

        bool should_break = false;

        #ifndef SINGULAR
        if(fabs(polar_position.y) >= GET_FEATURE(universe_size, dfg))
        #else
        if(fabs(polar_position.y) < SINGULAR_TERMINATOR || fabs(polar_position.y) >= GET_FEATURE(universe_size, dfg))
        #endif // SINGULAR
        {
            //printf("Escaped\n");

            should_break = true;
        }

        float dLambda_dNew = 1;

        float4 next_position, next_velocity, next_acceleration;

        step_verlet(position, velocity, acceleration, false, ds, &next_position, &next_velocity, &next_acceleration, &dLambda_dNew, cfg, dfg);

        float old_dlambda = running_dlambda_dnew;

        running_dlambda_dnew *= dLambda_dNew;

        #ifdef ADAPTIVE_PRECISION
        if(fabs(r_value) < new_max)
        {
            int res = calculate_ds_error(ds, next_acceleration, acceleration, max_accel, &next_ds, dfg);

            if(res == DS_RETURN)
            {
                should_break = true;
            }

            if(res == DS_SKIP)
                continue;
        }
        #endif // ADAPTIVE_PRECISION

        #ifndef UNCONDITIONALLY_NONSINGULAR
        if(fabs(velocity.x / running_dlambda_dnew) > 1000 + f_in_x && fabs(acceleration.x / running_dlambda_dnew) > 100)
        {
            should_break = true;
        }
        #endif // UNCONDITIONALLY_NONSINGULAR

        float4 generic_position_out = position;
        float4 generic_velocity_out = velocity / old_dlambda;

        #if (defined(GENERIC_METRIC) && defined(GENERIC_CONSTANT_THETA)) || !defined(GENERIC_METRIC) || defined(DEBUG_CONSTANT_THETA)
        {
            float4 pos_spherical = generic_to_spherical(position, cfg);
            float4 vel_spherical = generic_velocity_to_spherical_velocity(position, velocity / old_dlambda, cfg);

            float fsign = sign(pos_spherical.y);
            pos_spherical.y = fabs(pos_spherical.y);

            float3 pos_cart = polar_to_cartesian(pos_spherical.yzw);
            float3 vel_cart = spherical_velocity_to_cartesian_velocity(pos_spherical.yzw, vel_spherical.yzw);

            float4 quat = ray->initial_quat;

            pos_cart = rot_quat(pos_cart, quat);
            vel_cart = rot_quat(vel_cart, quat);

            float3 next_pos_spherical = cartesian_to_polar(pos_cart);
            float3 next_vel_spherical = cartesian_velocity_to_polar_velocity(pos_cart, vel_cart);

            if(fsign < 0)
            {
                next_pos_spherical.x = -next_pos_spherical.x;
            }

            float4 next_pos_generic = spherical_to_generic((float4)(pos_spherical.x, next_pos_spherical), cfg);
            float4 next_vel_generic = spherical_velocity_to_generic_velocity((float4)(pos_spherical.x, next_pos_spherical), (float4)(vel_spherical.x, next_vel_spherical), cfg);

            if(i != 0)
                next_pos_generic = periodic_diff(next_pos_generic, last_pos_generic, periods) + last_pos_generic;

            last_pos_generic = next_pos_generic;

            generic_position_out = next_pos_generic;
            generic_velocity_out = next_vel_generic;

            //printf("In pos %f %f %f %f out %f %f %f %f\n", position.x, position.y, position.z, position.w, next_pos_generic.x, next_pos_generic.y, next_pos_generic.z, next_pos_generic.w);
        }
        #endif

        ///in the event that velocity and acceleration is 0, it'd be ideal to have a fast path
        if(any(IS_DEGENERATE(next_position)) || any(IS_DEGENERATE(next_velocity)) || any(IS_DEGENERATE(next_acceleration)))
        {
            //printf("Degenerate");
            break;
        }

        position = next_position;
        velocity = next_velocity;
        acceleration = next_acceleration;

        positions_out[bufc * stride_out + id] = generic_position_out;

        ///do I want to reparameterise?
        ///yes, because we interpolate velocities
        if(velocities_out)
            velocities_out[bufc * stride_out + id] = generic_velocity_out;

        if(ds_out)
            ds_out[bufc * stride_out + id] = ds * old_dlambda;

        bufc++;

        if(should_break)
            break;
    }

    count_out[id] = bufc;
}

#ifdef DEVICE_SIDE_ENQUEUE
__kernel
void relauncher_generic(__global struct lightray* generic_rays_in, __global struct lightray* generic_rays_out,
                        __global struct lightray* finished_rays,
                        __global int* restrict generic_count_in, __global int* restrict generic_count_out,
                        __global int* finished_count_out,
                        int fallback,
                        dynamic_config_space const struct dynamic_config* cfg)
{
    ///failed to converge
    if(fallback > 125)
        return;

    if((*generic_count_in) == 0)
        return;

    if(fallback == 0)
        *finished_count_out = 0;

    int generic_count = *generic_count_in;

    int offset = 0;
    int loffset = 256;

    int one = 1;
    int oneoffset = 1;

    *generic_count_out = 0;

    clk_event_t f3;

    enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_NO_WAIT,
                   ndrange_1D(offset, generic_count, loffset),
                   0, NULL, &f3,
                   ^{
                        do_generic_rays (generic_rays_in, generic_rays_out,
                                         finished_rays,
                                         generic_count_in, generic_count_out,
                                         finished_count_out, cfg);
                   });

    enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_WAIT_KERNEL,
                   ndrange_1D(offset, one, oneoffset),
                   1, &f3, NULL,
                   ^{
                        relauncher_generic(generic_rays_out, generic_rays_in,
                                           finished_rays,
                                           generic_count_out, generic_count_in,
                                           finished_count_out, fallback + 1, cfg);
                   });

    release_event(f3);
}
#endif // DEVICE_SIDE_ENQUEUE

__kernel
void clear_termination_buffer(__global int* termination_buffer, int width, int height)
{
    int id = get_global_id(0);

    if(id >= width * height)
        return;

    termination_buffer[id] = 1;
}

__kernel
void calculate_singularities(__global const struct lightray* finished_rays, __global const int* finished_count, __global int* termination_buffer, int width, int height)
{
    int id = get_global_id(0);

    if(id >= *finished_count)
        return;

    int sx = id % width;
    int sy = id / width;

    termination_buffer[sy * width + sx] = !finished_rays[id].terminated;
}

#endif // GENERIC_METRIC

float4 get_intersection_position(struct lightray ray, dynamic_config_space const struct dynamic_config* cfg, dynamic_config_space const struct dynamic_feature_config* dfg)
{
    float4 position = generic_to_spherical(ray.position, cfg);
    float4 velocity = generic_velocity_to_spherical_velocity(ray.position, ray.velocity, cfg);

    #ifdef IS_CONSTANT_THETA
    position.z = M_PIf/2;
    velocity.z = 0;
    #endif // IS_CONSTANT_THETA

    {
        if(fabs(position.y) >= GET_FEATURE(universe_size, dfg))
        {
            position.yzw = fix_ray_position(position.yzw, velocity.yzw, GET_FEATURE(universe_size, dfg), true);
        }

        ///I'm not 100% sure this is working as well as it could be
        #if defined(SINGULAR) && defined(TRAVERSABLE_EVENT_HORIZON)
        if(fabs(position.y) < SINGULAR_TERMINATOR)
        {
            position.yzw = fix_ray_position(position.yzw, velocity.yzw, SINGULAR_TERMINATOR, true);
        }
        #endif
    }

	float3 npolar = position.yzw;

    #if (defined(GENERIC_METRIC) && defined(GENERIC_CONSTANT_THETA)) || !defined(GENERIC_METRIC) || defined(DEBUG_CONSTANT_THETA)
	{
        float4 quat = ray.initial_quat;

	    float3 cart_pos = polar_to_cartesian(position.yzw);

	    cart_pos = rot_quat(cart_pos, quat);

	    npolar = cartesian_to_polar(cart_pos);
	}
    #endif // GENERIC_CONSTANT_THETA

    return (float4)(position.x, npolar.xyz);
}

struct render_data
{
    float2 tex_coord;
    float z_shift;
    int sx;
    int sy;
    int terminated;
    int side;
};

float3 angle_to_vec(float2 a)
{
    return polar_to_cartesian((float3)(1.f, a));
}

float2 angle_to_tex(float2 angle)
{
    float thetaf = fmod(angle.x, 2 * M_PIf);
    float phif = angle.y;

    if(thetaf >= M_PIf)
    {
        phif += M_PIf;
        thetaf -= M_PIf;
    }

    phif = fmod(phif, 2 * M_PIf);

    float sxf = (phif) / (2 * M_PIf);
    float syf = thetaf / M_PIf;

    sxf += 0.5f;

    return (float2)(sxf, syf);
}

float2 tex_to_angle(float2 tex)
{
    tex.x -= 0.5f;
    tex.x *= 2 * M_PIf;
    tex.y *= M_PIf;

    return tex;
}

struct render_data interpolate_render_data(struct render_data ray1, struct render_data ray2)
{
    float2 a1 = tex_to_angle(ray1.tex_coord);
    float2 a2 = tex_to_angle(ray2.tex_coord);

    float3 v1 = angle_to_vec(a1.yx);
    float3 v2 = angle_to_vec(a2.yx);

    float3 vc = (v1 + v2)/2.f;

    float3 fangle = cartesian_to_polar(vc);

    float2 tc = angle_to_tex(fangle.yz);

    struct render_data out;
    out.tex_coord = tc;
    out.z_shift = (ray1.z_shift + ray2.z_shift)/2.f;
    out.terminated = ray1.terminated;
    out.sx = (ray1.sx + ray2.sx)/2;
    out.sy = (ray1.sy + ray2.sy)/2;
    out.side = (ray1.side + ray2.side)/2;
    return out;
};

__kernel
void calculate_render_data(global const struct lightray* rays_in, global const int* rays_in_count,
                           global struct render_data* rdata, global int* rdata_count,
                           int width, int height,
                           dynamic_config_space const struct dynamic_config* cfg, dynamic_config_space const struct dynamic_feature_config* dfg)
{
    int gid = get_global_id(0);

    if(gid >= *rays_in_count)
        return;

    struct render_data dat;

    __global const struct lightray* ray = &rays_in[gid];

    dat.terminated = ray->terminated;
    dat.sx = ray->sx;
    dat.sy = ray->sy;
    dat.z_shift = 0;
    dat.tex_coord = (float2){0,0};
    dat.side = 1;

    if(gid == 0)
        *rdata_count = width * height;

    //int id = atomic_inc(rdata_count);

    int id = dat.sy * width + dat.sx;

    if(ray->terminated != 1)
    {
        rdata[id] = dat;
        return;
    }

    int sx = ray->sx;
    int sy = ray->sy;

    float4 position = get_intersection_position(*ray, cfg, dfg);
    float4 generic_velocity = ray->velocity / ray->running_dlambda_dnew;

    dat.side = generic_to_spherical(ray->position, cfg).y < 0 ? 0 : 1;

    float r_value = position.y;

    #if !defined(TRAVERSABLE_EVENT_HORIZON) || (defined(NO_EVENT_HORIZON_CROSSING) && !defined(GENERIC_METRIC))
    if(fabs(r_value) <= 1)
    {
        rdata[id] = dat;
        return;
    }
    #endif

    float4 generic_position = ray->position;

    float4 fe0, fe1, fe2, fe3;
    calculate_tetrads(generic_position, (float3)(0,0,0), &fe0, &fe1, &fe2, &fe3, cfg, 0);

    #ifndef GENERIC_BIG_METRIC
    float g_metric[4] = {0};
    calculate_metric_generic(generic_position, g_metric, cfg);
    #else
    float g_metric[16] = {0};
    calculate_metric_generic_big(generic_position, g_metric, cfg);
    #endif

    float4 obvs_low = lower_index_generic(fe0, g_metric);

    ///[-1, +infinity]
    float z_shift = (dot(generic_velocity, obvs_low) / ray->ku_uobsu) - 1;

    z_shift = max(z_shift, -0.999f);

    dat.z_shift = z_shift;

    dat.tex_coord = angle_to_tex(position.zw);

    rdata[id] = dat;
}

float2 angle_between_angles2(float2 a1, float2 a2)
{
    float3 v1 = angle_to_vec(a1);
    float3 v2 = angle_to_vec(a2);

    return acos(clamp(dot(v1, v2), -1.f, 1.f));
}

__kernel
void handle_adaptive_sampling(global const struct lightray* rays_in, global const int* rays_in_count,
                              global struct render_data* rdat, global int* rdata_count,
                              global struct lightray* unprocessed_rays_out, global int* unprocessed_rays_out_count,
                              global float4* g_generic_camera_in, global float4* g_camera_quat,
                              __global const float4* e0, __global const float4* e1, __global const float4* e2, __global const float4* e3,
                              int width, int height,
                              dynamic_config_space const struct dynamic_config* cfg, dynamic_config_space const struct dynamic_feature_config* dfg)
{
    int sx = get_global_id(0);
    int sy = get_global_id(1);

    if(sx >= width/2 || sy >= height/2)
        return;

    struct lightray my_ray = rays_in[sy * (width/2) + sx];

    bool should_sample = true;

    if(sx != 0 && sx != (width/2)-1 && sy != 0 && sy != (height/2)-1)
    {
        int lx = sx-1;
        int rx = sx+1;

        int ly = sy-1;
        int ry = sy+1;

        #define GET_RAY(x, y) rays_in[y * (width/2) + x];

        struct lightray centre = GET_RAY(sx, sy);
        struct lightray left = GET_RAY(lx, sy);
        struct lightray right = GET_RAY(rx, sy);
        struct lightray up = GET_RAY(sx, ly);
        struct lightray down = GET_RAY(sx, ry);
        struct lightray down_right = GET_RAY(rx, ry);

        float4 lpos = get_intersection_position(left, cfg, dfg);
        float4 rpos = get_intersection_position(right, cfg, dfg);
        float4 upos = get_intersection_position(up, cfg, dfg);
        float4 dpos = get_intersection_position(down, cfg, dfg);

        ///circular diff2 isn't right here
        float2 x_error = fabs(angle_between_angles2(lpos.zw, rpos.zw));
        float2 y_error = fabs(angle_between_angles2(dpos.zw, upos.zw));

        ///hmm. Perhaps a better error measure would be to work out how divergent our view would be if a regular ray hit the wall?
        ///the thing is, if we have a constant shear applied to the camera view, i don't want to render the whole scene in great detail, that's unnecessary
        ///what we're looking for is when the approximation that (lpos + rpos)/2 == centre is no longer true basically
        ///I guess the error is quantified by the differences in derivatives, of our expected derivative, vs our actual derivative
        float relative_angular_error = ((x_error.x + x_error.y + y_error.x + y_error.y)/4.f) / 2 * M_PI;

        float fov = GET_FEATURE(field_of_view, dfg);

        float fov_angle_pi = fov * 2 * M_PI/360.f;

        float rough_angular_change_per_pixel = fov_angle_pi / width;

        should_sample = relative_angular_error >= rough_angular_change_per_pixel * GET_FEATURE(adaptive_sampling_threshold, dfg);

        if(centre.terminated != left.terminated || centre.terminated != right.terminated || centre.terminated != up.terminated || centre.terminated != down.terminated || centre.terminated != down_right.terminated)
            should_sample = true;
    }

    if(should_sample)
    {
        ///output the other 3 rays
        int base_sx = sx * 2;
        int base_sy = sy * 2;

        int2 ray_pos[3] = {{base_sx+1, base_sy}, {base_sx, base_sy+1}, {base_sx+1, base_sy+1}};

        int root_id = atomic_add(unprocessed_rays_out_count, 3);

        for(int i=0; i < 3; i++)
        {
            int cx = ray_pos[i].x;
            int cy = ray_pos[i].y;

            float3 pixel_direction = calculate_pixel_direction(cx, cy, width, height, *g_camera_quat, dfg);

            float4 at_metric = *g_generic_camera_in;

            float4 bT = *e0;
            float4 observer_velocity = *e0;

            float4 le1 = *e1;
            float4 le2 = *e2;
            float4 le3 = *e3;

            pixel_direction = normalize(pixel_direction);

            float4 pixel_x = pixel_direction.x * le1;
            float4 pixel_y = pixel_direction.y * le2;
            float4 pixel_z = pixel_direction.z * le3;

            float4 pixel_t = -bT;

            float4 lightray_velocity = pixel_x + pixel_y + pixel_z + pixel_t;
            float4 lightray_spacetime_position = at_metric;

            struct lightray ray = geodesic_to_render_ray(cx, cy, lightray_spacetime_position, lightray_velocity, observer_velocity, cfg);

            int out_id = root_id + i;
            unprocessed_rays_out[out_id] = ray;
        }
    }
    else
    {
        int lsx = my_ray.sx;
        int lsy = my_ray.sy;

        struct render_data cdata = rdat[lsy * width + lsx];
        struct render_data rdata = rdat[lsy * width + lsx + 2];
        struct render_data ldata = rdat[lsy * width + lsx - 2];
        struct render_data udata = rdat[(lsy - 2) * width + lsx];
        struct render_data ddata = rdat[(lsy + 2) * width + lsx];
        struct render_data drdata = rdat[(lsy + 2) * width + lsx + 2];

        rdat[lsy * width + lsx + 1] = interpolate_render_data(cdata, rdata);
        rdat[(lsy + 1) * width + lsx] = interpolate_render_data(cdata, ddata);
        rdat[(lsy + 1) * width + lsx + 1] = interpolate_render_data(cdata, drdata);
    }
}

float3 linear_rgb_to_XYZ(float3 in)
{
    float X = 0.4124564f * in.x + 0.3575761f * in.y + 0.1804375f * in.z;
    float Y = 0.2126729f * in.x + 0.7151522f * in.y + 0.0721750f * in.z;
    float Z = 0.0193339f * in.x + 0.1191920f * in.y + 0.9503041f * in.z;

    return (float3){X, Y, Z};
}

bool vector_lies_between(float2 v1, float2 v2, float2 c)
{
    return (v1.y * v2.x - v1.x * v2.y) * (v1.y * c.x - v1.x * c.y) < 0;
}

float angle_between_vectors(float2 v1, float2 v2)
{
    return acos(clamp(dot(fast_normalize(v1), fast_normalize(v2)), -1.f, 1.f));
}

float energy_of(float3 v)
{
    return v.x*0.2125f + v.y*0.7154f + v.z*0.0721f;
}

float3 redshift(float3 v, float z, dynamic_config_space const struct dynamic_feature_config* dfg)
{
    ///1 + z = gtt(recv) / gtt(src)
    ///1 + z = lnow / lthen
    ///1 + z = wsrc / wobs

    float radiant_energy = energy_of(v);

    float3 red = (float3){1/0.2125f, 0.f, 0.f};
    float3 green = (float3){0, 1/0.7154, 0.f};
    float3 blue = (float3){0.f, 0.f, 1/0.0721};

    float3 result;

    if(z > 0)
    {
        result = mix(v, radiant_energy * red, tanh(z));
    }
    else
    {
        float iv1pz = (1/(1 + z)) - 1;

        float interpolating_fraction = tanh(iv1pz);

        float3 col = mix(v, radiant_energy * blue, interpolating_fraction);

        if(!GET_FEATURE(use_old_redshift, dfg))
        {
            float final_energy = energy_of(clamp(col, 0.f, 1.f));
            float real_energy = energy_of(col);

            float remaining_energy = real_energy - final_energy;

            col.xy += remaining_energy * (red + green).xy;
        }

        result = col;
    }

    result = clamp(result, 0.f, 1.f);

    return result;
}

///this function is drastically complicated by nvidias terrible opencl support
///can't believe i fully have to implement trilinear filtering, mipmapping, and anisotropic texture filtering in raw opencl
///such is the terrible state of support across amd and nvidia
///amd doesn't correctly support shared opengl textures with mipmaps, and there's no anisotropic filtering i can see
///and nvidia don't support mipmapped textxures at all, or clCreateSampler
///what a mess!
float4 read_mipmap(image2d_array_t mipmap1, image2d_array_t mipmap2, int side, float2 pos, float lod)
{
    lod = max(lod, 0.f);

    sampler_t sam = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_REPEAT | CLK_FILTER_LINEAR;

    pos = fmod(pos, 1.f);

    //lod = 0;

    float mip_lower = floor(lod);
    float mip_upper = ceil(lod);

    float lower_divisor = pow(2.f, mip_lower);
    float upper_divisor = pow(2.f, mip_upper);

    float2 lower_coord = pos / lower_divisor;
    float2 upper_coord = pos / upper_divisor;

    float4 full_lower_coord = (float4)(lower_coord.xy, mip_lower, 0.f);
    float4 full_upper_coord = (float4)(upper_coord.xy, mip_upper, 0.f);

    float lower_weight = (lod - mip_lower);

    float4 v1 = mix(read_imagef(mipmap1, sam, full_lower_coord), read_imagef(mipmap1, sam, full_upper_coord), lower_weight);
    float4 v2 = mix(read_imagef(mipmap2, sam, full_lower_coord), read_imagef(mipmap2, sam, full_upper_coord), lower_weight);

    return side >= 1 ? v1 : v2;
}

#define MIPMAP_CONDITIONAL(x) ((side > 0) ? x(mip_background) : x(mip_background2))

__kernel
void render(__global const struct render_data* rdata, __global const int* rdata_count, __write_only image2d_t out,
            __read_only image2d_array_t mip_background, __read_only image2d_array_t mip_background2,
            int width, int height, int maxProbes,
            dynamic_config_space const struct dynamic_config* cfg, dynamic_config_space const struct dynamic_feature_config* dfg)
{
    int id = get_global_id(0);

    if(id >= *rdata_count)
        return;

    struct render_data rdat = rdata[id];

    int sx = rdat.sx;
    int sy = rdat.sy;

    int side = rdat.side;

    if(rdat.terminated != 1)
    {
        write_imagef(out, (int2)(sx, sy), (float4)(0,0,0,1));
        return;
    }

    float sxf = rdat.tex_coord.x;
    float syf = rdat.tex_coord.y;

    #if 0
    ///we actually do have an event horizon
    if(fabs(r_value) <= 1)
    //if(fabs(r_value) <= rs || r_value < 0)
    {
        float4 val = (float4)(0,0,0,1);

        int x_half = fabs(fmod((sxf + 1) * 10.f, 1.f)) > 0.5 ? 1 : 0;
        int y_half = fabs(fmod((syf + 1) * 10.f, 1.f)) > 0.5 ? 1 : 0;

        //val.x = (x_half + y_half) % 2;

        val.x = x_half;
        val.y = y_half;

        if(syf < 0.1 || syf >= 0.9)
        {
            val.x = 0;
            val.y = 0;
            val.z = 1;
        }

        write_imagef(out, (int2){sx, sy}, val);
        return;
    }
    #endif

    #define MIPMAPPING
    #ifdef MIPMAPPING
    int dx = 1;
    int dy = 1;

    if(sx == width-1)
        dx = -1;

    if(sy == height-1)
        dy = -1;

    float2 tl = rdata[sy * width + sx].tex_coord;
    float2 tr = rdata[sy * width + sx + dx].tex_coord;
    float2 bl = rdata[(sy + dy) * width + sx].tex_coord;

    ///higher = sharper
    float bias_frac = 1.3;

    //TL x 0.435143 TR 0.434950 TD -0.000149, aka (tr.x - tl.x) / 1.3
    float2 dx_vtc = circular_diff2(tl, tr) / bias_frac;
    float2 dy_vtc = circular_diff2(tl, bl) / bias_frac;

    if(dx == -1)
    {
        dx_vtc = -dx_vtc;
    }

    if(dy == -1)
    {
        dy_vtc = -dy_vtc;
    }

    //#define TRILINEAR
    #ifdef TRILINEAR
    dx_vtc.x *= MIPMAP_CONDITIONAL(get_image_width);
    dy_vtc.x *= MIPMAP_CONDITIONAL(get_image_width);

    dx_vtc.y *= MIPMAP_CONDITIONAL(get_image_height);
    dy_vtc.y *= MIPMAP_CONDITIONAL(get_image_height);

    //dx_vtc.x /= 10.f;
    //dy_vtc.x /= 10.f;

    dx_vtc /= 2.f;
    dy_vtc /= 2.f;

    float delta_max_sqr = max(dot(dx_vtc, dx_vtc), dot(dy_vtc, dy_vtc));

    float mip_level = 0.5 * log2(delta_max_sqr);

    //mip_level -= 0.5;

    float mip_clamped = clamp(mip_level, 0.f, 5.f);

    float4 end_result = MIPMAP_CONDITIONAL_READ(read_imagef, sam, ((float2){sxf, syf}), mip_clamped);
    #else

    dx_vtc.x *= MIPMAP_CONDITIONAL(get_image_width);
    dy_vtc.x *= MIPMAP_CONDITIONAL(get_image_width);

    dx_vtc.y *= MIPMAP_CONDITIONAL(get_image_height);
    dy_vtc.y *= MIPMAP_CONDITIONAL(get_image_height);

    ///http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.1002.1336&rep=rep1&type=pdf
    float dv_dx = dx_vtc.y;
    float dv_dy = dy_vtc.y;

    float du_dx = dx_vtc.x;
    float du_dy = dy_vtc.x;

    float Ann = dv_dx * dv_dx + dv_dy * dv_dy;
    float Bnn = -2 * (du_dx * dv_dx + du_dy * dv_dy);
    float Cnn = du_dx * du_dx + du_dy * du_dy; ///only tells lies

    ///hecc
    #define HECKBERT
    #ifdef HECKBERT
    Ann = dv_dx * dv_dx + dv_dy * dv_dy + 1;
    Cnn = du_dx * du_dx + du_dy * du_dy + 1;
    #endif // HECKBERT

    float F = Ann * Cnn - Bnn * Bnn / 4;
    float A = Ann / F;
    float B = Bnn / F;
    float C = Cnn / F;

    float root = sqrt((A - C) * (A - C) + B*B);
    float a_prime = (A + C - root) / 2;
    float c_prime = (A + C + root) / 2;

    float majorRadius = native_rsqrt(a_prime);
    float minorRadius = native_rsqrt(c_prime);

    float theta = atan2(B, (A - C)/2);

    majorRadius = max(majorRadius, 1.f);
    minorRadius = max(minorRadius, 1.f);

    majorRadius = max(majorRadius, minorRadius);

    float fProbes = 2 * (majorRadius / minorRadius) - 1;
    int iProbes = floor(fProbes + 0.5f);

    iProbes = min(iProbes, maxProbes);

    if(iProbes < fProbes)
        minorRadius = 2 * majorRadius / (iProbes + 1);

    float levelofdetail = log2(minorRadius);

    int maxLod = MIPMAP_CONDITIONAL(get_image_array_size) - 1;

    if(levelofdetail > maxLod)
    {
        levelofdetail = maxLod;
        iProbes = 1;
    }

    float4 end_result = 0;

    if(iProbes == 1 || iProbes <= 1)
    {
        if(iProbes < 1)
            levelofdetail = maxLod;

        end_result = read_mipmap(mip_background, mip_background2, side, (float2){sxf, syf}, levelofdetail);
    }
    else
    {
        float lineLength = 2 * (majorRadius - minorRadius);
        float du = cos(theta) * lineLength / (iProbes - 1);
        float dv = sin(theta) * lineLength / (iProbes - 1);

        float4 totalWeight = 0;
        float accumulatedProbes = 0;

        int startN = 0;

        ///odd probes
        if((iProbes % 2) == 1)
        {
            int probeArm = (iProbes - 1) / 2;

            startN = -2 * probeArm;
        }
        else
        {
            int probeArm = (iProbes / 2);

            startN = -2 * probeArm - 1;
        }

        int currentN = startN;
        float alpha = 2;

        float sU = du / MIPMAP_CONDITIONAL(get_image_width);
        float sV = dv / MIPMAP_CONDITIONAL(get_image_height);

        for(int cnt = 0; cnt < iProbes; cnt++)
        {
            float d_2 = (currentN * currentN / 4.f) * (du * du + dv * dv) / (majorRadius * majorRadius);

            ///not a performance issue
            float relativeWeight = native_exp(-alpha * d_2);

            float centreu = sxf;
            float centrev = syf;

            float cu = centreu + (currentN / 2.f) * sU;
            float cv = centrev + (currentN / 2.f) * sV;

            float4 fval = read_mipmap(mip_background, mip_background2, side, (float2){cu, cv}, levelofdetail);

            totalWeight += relativeWeight * fval;
            accumulatedProbes += relativeWeight;

            currentN += 2;
        }

        end_result = totalWeight / accumulatedProbes;
    }

    #endif // TRILINEAR

    //float4 end_result = read_imagef(*background, sam, (float2){sxf, syf}, dx_vtc, dy_vtc);

    #else
    float4 end_result = read_mipmap(mip_background, mip_background2, side, sam, (float2){sxf, syf}, 0);
    #endif // MIPMAPPING

    if(GET_FEATURE(redshift, dfg))
    {
        float z_shift = rdat.z_shift;

        /*if(sx == width/2 && sy == height/2)
        {
            printf("Vx %f ray %f\n", velocity.x, -ray->ku_uobsu);
        }*/

        ///linf / le = z + 1
        ///le =  linf / (z + 1)

        ///So, this is an incredibly, incredibly gross approximation
        ///there are several problems here
        ///1. Fundamentally I do not have a spectrographic map of the surrounding universe, which means any data is very approximate
        ///EG blueshifting of infrared into visible light is therefore impossible
        ///2. Converting sRGB information into wavelengths is possible, but also unphysical
        ///This might be a worthwhile approximation as it might correctly bunch frequencies together
        ///3. Its not possible to correctly render red/blueshifting, so it maps the range [-1, +inf] to [red, blue], mixing the colours with parameter [x <= 0 -> abs(x), x > 0 -> tanh(x)]]
        ///this means that even if I did all the above correctly, its still a mess

        ///This estimates luminance from the rgb value, which should be pretty ok at least!

        float3 lin_result = srgb_to_lin(end_result.xyz);

        float real_sol = 299792458;

        //#define DOMINANT_COLOUR
        #ifndef DOMINANT_COLOUR
        ///Pick an arbitrary wavelength, the peak of human vision
        float test_wavelength = 555 / real_sol;
        #else

        float r_wavelength = 612;
        float g_wavelength = 549;
        float b_wavelength = 464;

        float r_angle = -2.161580;
        float g_angle = 1.695013;
        float b_angle = -0.010759;

        float3 as_xyz = linear_rgb_to_XYZ(lin_result);

        float sum = as_xyz.x + as_xyz.y + as_xyz.z;

        if(sum < 0.00001f)
            sum = 0.00001f;

        float2 as_xy = (float2)(as_xyz.x / sum, as_xyz.y / sum);

        float2 as_vec = as_xy - (float2)(0.3333f, 0.3333f);

        float angle = atan2(as_xy.y, as_xy.x);

        float2 v_r = {cos(r_angle), sin(r_angle)};
        float2 v_g = {cos(g_angle), sin(g_angle)};
        float2 v_b = {cos(b_angle), sin(b_angle)};

        float2 p1;
        float2 p2;
        float w1, w2;

        if(vector_lies_between(v_r, v_g, as_vec))
        {
            p1 = v_r;
            p2 = v_g;
            w1 = r_wavelength;
            w2 = g_wavelength;
        }

        else if(vector_lies_between(v_g, v_b, as_vec))
        {
            p1 = v_g;
            p2 = v_b;
            w1 = g_wavelength;
            w2 = b_wavelength;
        }

        else
        {
            p1 = v_r;
            p2 = v_b;
            w1 = r_wavelength;
            w2 = b_wavelength;
        }

        float fraction = angle_between_vectors(p1, as_vec) / angle_between_vectors(p1, p2);

        float test_wavelength = mix(w1, w2, fraction) / real_sol;

        /*if(sx == 700 && sy == 400)
        {
            printf("wave %f\n", test_wavelength * real_sol);
        }*/

        #endif // DOMINANT_COLOUR

        float local_wavelength = test_wavelength / (z_shift + 1);

        ///this is relative luminance instead of absolute specific intensity, but relative_luminance / wavelength^3 should still be lorenz invariant (?)
        float relative_luminance = 0.2126f * lin_result.x + 0.7152f * lin_result.y + 0.0722f * lin_result.z;

        ///Iv = I1 / v1^3, where Iv is lorenz invariant
        ///Iv = I2 / v2^3 in our new frame of reference
        ///therefore we can calculate the new intensity in our new frame of reference as...
        ///I1/v1^3 = I2 / v2^3
        ///I2 = v2^3 * I1/v1^3

        float new_relative_luminance = pow(local_wavelength, 3) * relative_luminance / pow(test_wavelength, 3);

        new_relative_luminance = clamp(new_relative_luminance, 0.f, 1.f);

        if(relative_luminance > 0.00001)
        {
            lin_result = (new_relative_luminance / relative_luminance) * lin_result;

            lin_result = clamp(lin_result, 0.f, 1.f);
        }

        /*if(sx == 700 && sy == 400)
        {
            printf("Shift %f vx %f obvsu %f\n", z_shift, velocity.x, -ray->ku_uobsu);
        }*/

        //if(fabs(r_value) > 2)
        {
            /*if(sx == width/2 && sy == height/2)
            {
                printf("ZShift %f\n", z_shift);
            }*/

            lin_result = redshift(lin_result, z_shift, dfg);

            lin_result = clamp(lin_result, 0.f, 1.f);
        }

        #ifndef LINEAR_FRAMEBUFFER
        end_result.xyz = lin_to_srgb(lin_result);
        #else
        end_result.xyz = lin_result;
        #endif // LINEAR_FRAMEBUFFER
    }

    #ifdef LINEAR_FRAMEBUFFER
    if(!GET_FEATURE(redshift, dfg)) //redshift already handles this for roundtrip accuracy reasons
        end_result.xyz = srgb_to_lin(end_result.xyz);
    #endif // LINEAR_FRAMEBUFFER

    write_imagef(out, (int2){sx, sy}, end_result);
}

__kernel
void render_tris(__global struct triangle* tris, int tri_count,
                 __global struct lightray* finished_rays, __global int* finished_count_in,
                 __global float4* traced_positions, __global int* traced_positions_count,
                 int width, int height,
                 __write_only image2d_t screen,
                 dynamic_config_space const struct dynamic_config* cfg)
{
    /*int id = get_global_id(0);

    if(id >= *finished_count_in)
        return;

    int sx = finished_rays[id].sx;
    int sy = finished_rays[id].sy;

    int big_id = sy * width + sx;*/

    int sx = get_global_id(0);
    int sy = get_global_id(1);

    if(sx >= width || sy >= height)
        return;

    int big_id = sy * width + sx;

    int cnt = traced_positions_count[big_id];

    if(sx == width/2 && sy == height/2)
    {
        //printf("ello %i\n", tri_count);
    }

    for(int kk=0; kk < tri_count; kk++)
    {
        __global struct triangle* ctri = &tris[kk];

        //float tri_time = ctri->time;

        float3 v0_pos = {ctri->v0x, ctri->v0y, ctri->v0z};
        float3 v1_pos = {ctri->v1x, ctri->v1y, ctri->v1z};
        float3 v2_pos = {ctri->v2x, ctri->v2y, ctri->v2z};

        for(int i=0; i < cnt - 1; i++)
        {
            float4 pos = generic_to_cartesian(traced_positions[i * width * height + big_id], cfg);
            float4 next_pos = generic_to_cartesian(traced_positions[(i + 1) * width * height + big_id], cfg);

            float time = pos.x;
            float next_time = next_pos.x;

            //if(time >= tri_time && time < next_time)
            {
                //float dx = (tri_time - time) / (next_time - time);

                float dx = 0;

                float3 ray_pos = mix(next_pos.yzw, pos.yzw, dx);
                float3 ray_dir = next_pos.yzw - pos.yzw;

                if(ray_intersects_triangle(ray_pos, ray_dir, v0_pos, v1_pos, v2_pos, 0, 0, 0))
                {
                    write_imagef(screen, (int2){sx, sy}, (float4)(1, 0, 0, 1));
                    return;
                }
            }
        }
    }
}

#if 0
__kernel
void render_potential_intersections(__global struct potential_intersection* in, __global int* cnt,
                                    __global int* counts, __global int* offsets, __global int* linear_mem, float accel_width, int accel_width_num,
                                    __global struct triangle* tris, __write_only image2d_t screen)
{
    int id = get_global_id(0);

    if(id >= *cnt)
        return;

    struct potential_intersection mine = in[id];
    int cx = mine.cx;
    int cy = mine.cy;

    float4 rt_pos = {mine.st, mine.sx, mine.sy, mine.sz};
    float4 next_rt_pos = {mine.et, mine.ex, mine.ey, mine.ez};

    float3 current_pos = world_to_voxel(rt_pos.yzw, accel_width, accel_width_num);
    float3 next_pos = world_to_voxel(next_rt_pos.yzw, accel_width, accel_width_num);

    current_pos = round(current_pos);
    next_pos = round(next_pos);

    float3 diff2 = next_pos - current_pos;
    float3 adiff2 = fabs(diff2);

    float max_len2 = max(max(adiff2.x, adiff2.y), adiff2.z);

    float3 step = diff2 / max_len2;

    for(int i=0; i <= max_len2; i++)
    {
        float3 floordf = floor(current_pos);

        int3 ifloor = (int3)(floordf.x, floordf.y, floordf.z);

        ifloor = loop_voxel(ifloor, accel_width_num);

        int voxel_id = ifloor.z * accel_width_num * accel_width_num + ifloor.y * accel_width_num + ifloor.x;

        int cnt = counts[voxel_id];

        if(cnt > 0)
        {
            int tri_count = counts[voxel_id];
            int base_offset = offsets[voxel_id];

            __global int* tri_indices = &linear_mem[base_offset];

            for(int i=0; i < tri_count; i++)
            {
                int idx = tri_indices[i];

                __global struct triangle* ctri = &tris[idx];

                float3 v0_pos = {ctri->v0x, ctri->v0y, ctri->v0z};
                float3 v1_pos = {ctri->v1x, ctri->v1y, ctri->v1z};
                float3 v2_pos = {ctri->v2x, ctri->v2y, ctri->v2z};

                float dx = 0;
                //float3 ray_pos = mix(next_rt_pos.yzw, rt_pos.yzw, dx);

                float3 ray_pos = rt_pos.yzw;
                float3 ray_dir = next_rt_pos.yzw - rt_pos.yzw;

                ///ehhhh need to take closest
                if(ray_intersects_triangle(ray_pos, ray_dir, v0_pos, v1_pos, v2_pos, 0))
                {
                    write_imagef(screen, (int2){cx, cy}, (float4)(1, 0, 0, 1));
                    return;
                }
            }
        }

        current_pos += step;
    }
}
#endif // 0

///todo: make flip global
__kernel
void cart_to_polar_kernel(__global const float4* position_cart_in, __global float4* position_polar_out, int count, float flip)
{
    int id = get_global_id(0);

    if(id >= count)
        return;

    float3 cart = position_cart_in[id].yzw;

    float3 polar = cartesian_to_polar(cart);

    if(flip > 0)
        polar.x = -polar.x;

    position_polar_out[id] = (float4)(position_cart_in[id].x, polar.xyz);
}

///todo: make flip global
__kernel
void cart_to_generic_kernel(__global const float4* position_cart_in, __global float4* position_generic_out, int count, float flip, dynamic_config_space const struct dynamic_config* cfg)
{
    int id = get_global_id(0);

    if(id >= count)
        return;

    float3 cart = position_cart_in[id].yzw;

    float3 polar = cartesian_to_polar(cart);

    if(flip > 0)
        polar.x = -polar.x;

    position_generic_out[id] = spherical_to_generic((float4)(position_cart_in[id].x, polar.xyz), cfg);
}

__kernel
void camera_polar_to_generic(__global const float4* g_camera_pos_polar, __global float4* g_camera_pos_generic, dynamic_config_space const struct dynamic_config* cfg)
{
    if(get_global_id(0) != 0)
        return;

    *g_camera_pos_generic = spherical_to_generic(*g_camera_pos_polar, cfg);
}

