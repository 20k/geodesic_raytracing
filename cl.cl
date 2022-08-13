#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define M_PIf ((float)M_PI)

struct triangle
{
    int parent;
    float v0x, v0y, v0z;
    float v1x, v1y, v1z;
    float v2x, v2y, v2z;
};

#define TRI_PRECESSION
#ifdef TRI_PRECESSION
struct computed_triangle
{
    float v0x, v0y, v0z;
    float v1x, v1y, v1z;
    float v2x, v2y, v2z;

    float e0x, e0y, e0z;
    float e1x, e1y, e1z;
    float e2x, e2y, e2z;
};
#else
struct computed_triangle
{
    //float time;
    float v0x, v0y, v0z;
    half dv1x, dv1y, dv1z;
    half dv2x, dv2y, dv2z;

    //float dvt;
    half dvx;
    half dvy;
    half dvz;
};

#endif // TRI_PRECESSION

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

///ds2 = guv dx^u dx^v
float4 fix_light_velocity2(float4 v, float g_metric[])
{
    ///g_metric[1] * v[1]^2 + g_metric[2] * v[2]^2 + g_metric[3] * v[3]^2 = -g_metric[0] * v[0]^2

    float3 vmetric = {g_metric[1], g_metric[2], g_metric[3]};

    #ifdef IS_CONSTANT_THETA
    v.z = 0;
    #endif // IS_CONSTANT_THETA

    float tvl_2 = dot(vmetric, v.yzw * v.yzw) / -g_metric[0];

    v.x = copysign(native_sqrt(tvl_2), v.x);

    return v;
}

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
    int sx, sy;
    float ku_uobsu;
    int early_terminate;
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

#define dynamic_config_space __constant

#ifndef GENERIC_BIG_METRIC
void calculate_metric_generic(float4 spacetime_position, float g_metric_out[], dynamic_config_space struct dynamic_config* cfg)
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

void calculate_partial_derivatives_generic(float4 spacetime_position, float g_metric_partials[], dynamic_config_space struct dynamic_config* cfg)
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
void calculate_metric_generic_big(float4 spacetime_position, float g_metric_out[], dynamic_config_space struct dynamic_config* cfg)
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

void calculate_partial_derivatives_generic_big(float4 spacetime_position, float g_metric_partials[], dynamic_config_space struct dynamic_config* cfg)
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

float4 generic_to_spherical(float4 in, dynamic_config_space struct dynamic_config* cfg)
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

float4 generic_velocity_to_spherical_velocity(float4 in, float4 inv, dynamic_config_space struct dynamic_config* cfg)
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

float4 spherical_to_generic(float4 in, dynamic_config_space struct dynamic_config* cfg)
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

float4 spherical_velocity_to_generic_velocity(float4 in, float4 inv, dynamic_config_space struct dynamic_config* cfg)
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

float4 generic_to_cartesian(float4 in, dynamic_config_space struct dynamic_config* cfg)
{
    float4 spherical = generic_to_spherical(in, cfg);

    return (float4)(spherical.x, polar_to_cartesian(spherical.yzw));
}

float4 generic_velocity_to_cartesian_velocity(float4 in, float4 in_v, dynamic_config_space struct dynamic_config* cfg)
{
    float4 spherical = generic_to_spherical(in, cfg);
    float4 spherical_v = generic_velocity_to_spherical_velocity(in, in_v, cfg);

    return (float4)(spherical_v.x, spherical_velocity_to_cartesian_velocity(spherical.yzw, spherical_v.yzw));
}

float4 cartesian_to_generic(float4 in, dynamic_config_space struct dynamic_config* cfg)
{
    float3 polar = cartesian_to_polar(in.yzw);

    return spherical_to_generic((float4)(in.x, polar), cfg);
}

float4 cartesian_velocity_to_generic_velocity(float4 in, float4 in_v, dynamic_config_space struct dynamic_config* cfg)
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

float4 handle_coordinate_periodicity(float4 in, dynamic_config_space struct dynamic_config* cfg)
{
    #ifdef HAS_COORDINATE_PERIODICITY
    float v1 = in.x;
    float v2 = in.y;
    float v3 = in.z;
    float v4 = in.w;

    float o1 = COORDINATE_PERIODICITY1;
    float o2 = COORDINATE_PERIODICITY2;
    float o3 = COORDINATE_PERIODICITY3;
    float o4 = COORDINATE_PERIODICITY4;

    return (float4)(o1, o2, o3, o4);
    #else
    return in;
    #endif // HAS_COORDINATE_PERIODICITY
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

float4 gram_proj_generic(float4 u, float4 v, float metric[])
{
    float top = dot_product_generic(u, v, metric);
    float bottom = dot_product_generic(u, u, metric);

    return (top / bottom) * u;
}

float4 normalise_generic(float4 in, float metric[])
{
    float d = dot_product_generic(in, in, metric);

    return in / native_sqrt(fabs(d));
}

#define SWAP(x, y, T) do { T SWAP = x; x = y; y = SWAP; } while (0)

///i1-4 are raised
struct frame_basis orthonormalise4_metric(float4 i1, float4 i2, float4 i3, float4 i4, float metric[])
{
    float4 u1 = i1;

    float4 u2 = i2;
    u2 = u2 - gram_proj_generic(u1, u2, metric);

    float4 u3 = i3;
    u3 = u3 - gram_proj_generic(u1, u3, metric);
    u3 = u3 - gram_proj_generic(u2, u3, metric);

    float4 u4 = i4;
    u4 = u4 - gram_proj_generic(u1, u4, metric);
    u4 = u4 - gram_proj_generic(u2, u4, metric);
    u4 = u4 - gram_proj_generic(u3, u4, metric);

    u1 = normalise_generic(u1, metric);
    u2 = normalise_generic(u2, metric);
    u3 = normalise_generic(u3, metric);
    u4 = normalise_generic(u4, metric);

    struct frame_basis ret;
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

///todo: generic orthonormalisation
struct frame_basis calculate_frame_basis(float big_metric[])
{
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

    struct frame_basis result = orthonormalise4_metric(as_array[0], as_array[1], as_array[2], as_array[3], big_metric);

    float4 result_as_array[4] = {result.v1, result.v2, result.v3, result.v4};

    float4 sorted_result[4] = {};

    for(int i=0; i < 4; i++)
    {
        int old_index = indices[i];

        sorted_result[old_index] = result_as_array[i];
    }

    float minkowski[16];

    get_local_minkowski(sorted_result[0], sorted_result[1], sorted_result[2], sorted_result[3], big_metric, minkowski);

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

    ///do I need to reshuffle spatial indices?
    if(!approx_equal(minkowski[0], -1, eps))
    {
        int which_index_is_timelike = -1;

        for(int i=0; i < 4; i++)
        {
            if(approx_equal(minkowski[i * 4 + i], -1, eps))
            {
                which_index_is_timelike = i;
                break;
            }
        }

        ///catastrophic failure
        if(which_index_is_timelike == -1)
        {

        }
        else
        {
            SWAP(sorted_result[0], sorted_result[which_index_is_timelike], float4);
        }
    }

    struct frame_basis result2;
    result2.v1 = sorted_result[0];
    result2.v2 = sorted_result[1];
    result2.v3 = sorted_result[2];
    result2.v4 = sorted_result[3];

    return result2;
}

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
            coeff_out[u * 4 + v] = delta[u * 4 + v] - (1 / (1 + lorentz_factor)) * (T[u] + uobs[u]) * (-lT[v] - luobs[v]) - 2 * uobs[u] * lT[v];
        }
    }
}

#else

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
            coeff_out[u * 4 + v] = delta[u * 4 + v] - (1 / (1 + lorentz_factor)) * (T[u] + uobs[u]) * (-lT[v] - luobs[v]) - 2 * uobs[u] * lT[v];
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

float3 get_theta_axis(float3 pixel_direction, float4 polar_camera_in)
{
    float3 apolar = polar_camera_in.yzw;
    apolar.x = fabs(apolar.x);

    float3 cartesian_camera_pos = polar_to_cartesian(apolar);

    float3 by = normalize(-cartesian_camera_pos);

    return normalize(-cross((float3)(0, 0, 1), by));
}

float3 get_phi_axis(float3 pixel_direction, float4 polar_camera_in)
{
    float3 apolar = polar_camera_in.yzw;
    apolar.x = fabs(apolar.x);

    float3 cartesian_camera_pos = polar_to_cartesian(apolar);

    float3 by = normalize(-cartesian_camera_pos);

    return -cross(get_theta_axis(pixel_direction, polar_camera_in), by);
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

float3 calculate_pixel_direction(int cx, int cy, float width, float height, float4 camera_quat)
{
    #define FOV 90

    float fov_rad = (FOV / 360.f) * 2 * M_PIf;

    float nonphysical_plane_half_width = width/2;
    float nonphysical_f_stop = nonphysical_plane_half_width / tan(fov_rad/2);

    float3 pixel_direction = (float3){cx - width/2, cy - height/2, nonphysical_f_stop};

    pixel_direction = normalize(pixel_direction);
    pixel_direction = rot_quat(pixel_direction, camera_quat);

    return pixel_direction;
}

int should_early_terminate(int x, int y, int width, int height, __global int* termination_buffer)
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

float4 parallel_transport_get_acceleration(float4 X, float4 geodesic_position, float4 geodesic_velocity, dynamic_config_space struct dynamic_config* cfg)
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

    float accel[4] = {0};

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

        accel[a] = -sum;
    }

    return (float4){accel[0], accel[1], accel[2], accel[3]};
}

__kernel
void init_basis_speed(__global float4* camera_rot, float speed, __global float4* basis_speed_out)
{
    if(get_global_id(0) != 0)
        return;

    float3 base = {0, 0, 1};

    float3 rotated = rot_quat(base, *camera_rot);

    *basis_speed_out = (float4)(normalize(rotated) * speed, 0);
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
                          float2 mouse_delta, float4 unrotated_translation, float universe_size,
                          dynamic_config_space struct dynamic_config* cfg)
{
    ///translation is: .x is forward - back, .y = right - left, .z = down - up
    ///totally arbitrary, purely to pass to the GPU
    if(get_global_id(0) != 0)
        return;

    float4 local_camera_quat = *camera_rot;

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

    float4 local_camera_pos_cart = *camera_pos_cart;

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

    *camera_pos_cart = local_camera_pos_cart;
    *camera_rot = local_camera_quat;
}

/*__kernel
void quat_to_basis(__global float4* camera_quat,
                   __global float4* e0, __global float4* e1, __global float4* e2, __global float4* e3,
                   __global float4* b0, __global float4* b1, __global float4* b2)
{
    if(get_global_id(0) != 0)

}*/

///https://arxiv.org/pdf/0904.4184.pdf 1.4.18
float4 get_timelike_vector(float3 cartesian_basis_speed, float time_direction,
                           float4 e0, float4 e1, float4 e2, float4 e3)
{
    float v = length(cartesian_basis_speed);
    float Y = 1 / sqrt(1 - v*v);

    float B = v;

    float psi = B * Y;

    float4 bT = time_direction * Y * e0;

    float3 dir = normalize(cartesian_basis_speed);

    if(v == 0)
        dir = (float3)(0, 0, 1);

    float4 bX = psi * dir.x * e1;
    float4 bY = psi * dir.y * e2;
    float4 bZ = psi * dir.z * e3;

    return bT + bX + bY + bZ;
}

void calculate_tetrads(float4 polar_camera, float3 cartesian_basis_speed,
                       float4* e0_out, float4* e1_out, float4* e2_out, float4* e3_out,
                       dynamic_config_space struct dynamic_config* cfg, int should_orient)
{
    float4 at_metric = spherical_to_generic(polar_camera, cfg);

    if(IS_DEGENERATE(at_metric.x) || IS_DEGENERATE(at_metric.y) || IS_DEGENERATE(at_metric.z) || IS_DEGENERATE(at_metric.w))
    {
        *e0_out = (float4)(1, 0, 0, 0);
        *e1_out = (float4)(0, 1, 0, 0);
        *e2_out = (float4)(0, 0, 1, 0);
        *e3_out = (float4)(0, 0, 0, 1);

        return;
    }

    #ifndef GENERIC_BIG_METRIC
    float g_metric[4] = {};
    calculate_metric_generic(at_metric, g_metric, cfg);

    float4 co_basis = (float4){native_sqrt(-g_metric[0]), native_sqrt(g_metric[1]), native_sqrt(g_metric[2]), native_sqrt(g_metric[3])};

    float4 e0 = (float4)(1/co_basis.x, 0, 0, 0); ///or bt
    float4 e1 = (float4)(0, 1/co_basis.y, 0, 0); ///or br
    float4 e2 = (float4)(0, 0, 1/co_basis.z, 0);
    float4 e3 = (float4)(0, 0, 0, 1/co_basis.w);
    #else
    float g_metric_big[16] = {0};
    calculate_metric_generic_big(at_metric, g_metric_big, cfg);

    ///contravariant
    float4 e0;
    float4 e1;
    float4 e2;
    float4 e3;

    {
        struct frame_basis basis = calculate_frame_basis(g_metric_big);

        /*if(cx == 500 && cy == 400)
        {
            float d1 = dot_product_big(basis.v1, basis.v2, g_metric_big);
            float d2 = dot_product_big(basis.v1, basis.v3, g_metric_big);
            float d3 = dot_product_big(basis.v1, basis.v4, g_metric_big);
            float d4 = dot_product_big(basis.v2, basis.v3, g_metric_big);
            float d5 = dot_product_big(basis.v3, basis.v4, g_metric_big);

            printf("ORTHONORMAL? %f %f %f %f %f\n", d1, d2, d3, d4, d5);
        }*/

        e0 = basis.v1;
        e1 = basis.v2;
        e2 = basis.v3;
        e3 = basis.v4;
    }
    #endif // GENERIC_BIG_METRIC

    /*
    ///???
    float4 observer_velocity = bT;

    //float4 observer_velocity = {1, 0.5, 0, 0};
    float lorentz[16] = {};

    #ifndef GENERIC_BIG_METRIC
    calculate_lorentz_boost(bT, observer_velocity, g_metric, lorentz);
    #else
    calculate_lorentz_boost_big(bT, observer_velocity, g_metric_big, lorentz);
    #endif // GENERIC_BIG_METRIC

    bT = observer_velocity;

    float4 sVx = tensor_contract(lorentz, btheta);
    float4 sVy = tensor_contract(lorentz, bphi);
    float4 sVz = tensor_contract(lorentz, bX);
    */

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

            /*float3 right = rot_quat((float3){1, 0, 0}, local_camera_quat);
            float3 forw = rot_quat((float3){0, 0, 1}, local_camera_quat);*/

            ///normalise with y first, so that the camera controls always work intuitively - as they are inherently a 'global' concept
            float4 approximate_basis[3] = {gy, gx, gz};

            float4 tE1 = coordinate_to_tetrad_basis(approximate_basis[0], e_lo[0], e_lo[1], e_lo[2], e_lo[3]);
            float4 tE2 = coordinate_to_tetrad_basis(approximate_basis[1], e_lo[0], e_lo[1], e_lo[2], e_lo[3]);
            float4 tE3 = coordinate_to_tetrad_basis(approximate_basis[2], e_lo[0], e_lo[1], e_lo[2], e_lo[3]);

            struct ortho_result result = orthonormalise(tE1.yzw, tE2.yzw, tE3.yzw);

            float4 basis1 = (float4)(0, result.v1);
            float4 basis2 = (float4)(0, result.v2);
            float4 basis3 = (float4)(0, result.v3);

            float4 x_basis = basis2;
            float4 y_basis = basis1;
            float4 z_basis = basis3;

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
void boost_tetrad(__global float4* polar_in, int count, __global float3* geodesic_basis_speed,
                  __global float4* e0_io, __global float4* e1_io, __global float4* e2_io, __global float4* e3_io,
                  dynamic_config_space struct dynamic_config* cfg)
{
    int id = get_global_id(0);

    if(id >= count)
        return;

    float4 at_metric = spherical_to_generic(polar_in[id], cfg);

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
void init_basis_vectors(__global float4* polar_in, int count,
                        float3 cartesian_basis_speed,
                        __global float4* e0_out, __global float4* e1_out, __global float4* e2_out, __global float4* e3_out,
                        dynamic_config_space struct dynamic_config* cfg)
{
    int id = get_global_id(0);

    if(id >= count)
        return;

    float4 polar = polar_in[id];

    float4 e0;
    float4 e1;
    float4 e2;
    float4 e3;

    calculate_tetrads(polar, cartesian_basis_speed, &e0, &e1, &e2, &e3, cfg, 1);

    e0_out[id] = e0;
    e1_out[id] = e1;
    e2_out[id] = e2;
    e3_out[id] = e3;
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
void parallel_transport_quantity(__global float4* geodesic_path, __global float4* geodesic_velocity, __global float* ds_in, __global float4* quantity, __global int* count_in, int count, __global float4* quantity_out, dynamic_config_space struct dynamic_config* cfg)
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

        float4 current_pos = geodesic_path[current_idx];
        float4 next_pos = geodesic_path[next_idx];

        float4 f_x = parallel_transport_get_acceleration(current_quantity, geodesic_path[current_idx], geodesic_velocity[current_idx], cfg);

        float4 intermediate_next = current_quantity + f_x * ds;

        float4 next = current_quantity + 0.5f * ds * (f_x + parallel_transport_get_acceleration(intermediate_next, geodesic_path[next_idx], geodesic_velocity[next_idx], cfg));

        ///so. quantity_out[0] ends up being initial, quantity_out[1] = after one transport
        quantity_out[current_idx] = current_quantity;

        current_quantity = next;
    }

    ///need to write final one
    quantity_out[(cnt - 1) * stride + id] = current_quantity;
}

__kernel
void handle_interpolating_geodesic(__global float4* geodesic_path, __global float4* geodesic_velocity, __global float* dT_ds, __global float* ds_in,
                                   __global float4* g_camera_polar_out,
                                   __global float4* t_e0_in, __global float4* t_e1_in, __global float4* t_e2_in, __global float4* t_e3_in,
                                   __global float4* e0_out, __global float4* e1_out, __global float4* e2_out, __global float4* e3_out,
                                   float target_time,
                                   __global int* count_in,
                                   int parallel_transport_observer,
                                   __global float3* geodesic_basis_speed,
                                   __global float4* interpolated_geodesic_velocity,
                                   dynamic_config_space struct dynamic_config* cfg)
{
    if(get_global_id(0) != 0)
        return;

    if(*count_in == 0)
        return;

    float4 start_generic = geodesic_path[0];

    if(!parallel_transport_observer)
    {
        float4 e0, e1, e2, e3;
        calculate_tetrads(generic_to_spherical(start_generic, cfg), *geodesic_basis_speed, &e0, &e1, &e2, &e3, cfg, 1);

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

    *g_camera_polar_out = generic_to_spherical(start_generic, cfg);
    *interpolated_geodesic_velocity = geodesic_velocity[0];

    if(cnt == 1)
        return;

    for(int i=0; i < cnt - 1; i++)
    {
        float4 current_pos = geodesic_path[i];
        float4 next_pos = geodesic_path[i + 1];

        /*{
            #ifndef GENERIC_BIG_METRIC
            float g_metric[4] = {0};
            calculate_metric_generic(next_pos, g_metric, cfg);
            #else
            float g_metric[16] = {0};
            calculate_metric_generic_big(next_pos, g_metric, cfg);
            #endif // GENERIC_BIG_METRIC

            struct frame_basis basis = orthonormalise4_metric(ne0, ne1, ne2, ne3, g_metric);

            ne0 = basis.v1;
            ne1 = basis.v2;
            ne2 = basis.v3;
            ne3 = basis.v4;
        }*/

        if(next_pos.x < current_pos.x)
        {
            float4 im = current_pos;
            current_pos = next_pos;
            next_pos = im;
        }

        //float dt = next_pos.x - current_pos.x;
        //float next_proper_time = current_proper_time + dT_dt[i] * dt;

        float next_proper_time = current_proper_time + dT_ds[i] * ds_in[i];

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

            #ifndef HAS_COORDINATE_PERIODICITY
            float4 spherical1 = generic_to_spherical(current_pos, cfg);
            float4 spherical2 = generic_to_spherical(next_pos, cfg);

            float4 fin_polar = mix_spherical(spherical1, spherical2, dx);
            *g_camera_polar_out = fin_polar;

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
                calculate_tetrads(fin_polar, *geodesic_basis_speed, &oe0, &oe1, &oe2, &oe3, cfg, 1);
            }

            *interpolated_geodesic_velocity = mix(geodesic_velocity[i], geodesic_velocity[i + 1], dx);

            *e0_out = oe0;
            *e1_out = oe1;
            *e2_out = oe2;
            *e3_out = oe3;
            #else
            float4 fin_polar = generic_to_spherical(current_pos, cfg);

            *g_camera_polar_out = fin_polar;

            float4 oe0 = t_e0_in[i];
            float4 oe1 = t_e1_in[i];
            float4 oe2 = t_e2_in[i];
            float4 oe3 = t_e3_in[i];

            if(!parallel_transport_observer)
            {
                calculate_tetrads(fin_polar, *geodesic_basis_speed, &oe0, &oe1, &oe2, &oe3, cfg, 1);
            }

            *interpolated_geodesic_velocity = geodesic_velocity[i];

            *e0_out = oe0;
            *e1_out = oe1;
            *e2_out = oe2;
            *e3_out = oe3;

            #endif // HAS_COORDINATE_PERIODICITY

            return;
        }

        //current_time += dt;
        current_proper_time = next_proper_time;
    }

    *g_camera_polar_out = generic_to_spherical(geodesic_path[cnt - 1], cfg);
    *interpolated_geodesic_velocity = geodesic_velocity[cnt - 1];

    if(!parallel_transport_observer)
    {
        float4 e0, e1, e2, e3;
        calculate_tetrads(generic_to_spherical(geodesic_path[cnt - 1], cfg), *geodesic_basis_speed, &e0, &e1, &e2, &e3, cfg, 1);

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

__kernel void push_object_positions(__global float4* geodesics_in, __global int* counts_in, __global struct object* pos_out, float target_time, int object_count, dynamic_config_space struct dynamic_config* cfg)
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

struct corrected_lightray
{
    float4 position;
    float4 velocity;
    float4 inverse_quat;
};

struct corrected_lightray correct_lightray(float4 position, float4 velocity, dynamic_config_space struct dynamic_config* cfg)
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

    position = handle_coordinate_periodicity(position, cfg);

    struct corrected_lightray ret;

    ret.position = position;
    ret.velocity = velocity;
    ret.inverse_quat = extra_quat;

    return ret;
};

///fully set up except for ray.early_terminate
struct lightray geodesic_to_render_ray(int cx, int cy, float4 position, float4 velocity, float4 observer_velocity, dynamic_config_space struct dynamic_config* cfg)
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

        velocity = fix_light_velocity2(velocity, g_metric);
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
    ray.sx = cx;
    ray.sy = cy;
    ray.position = position;
    ray.velocity = velocity;
    ray.acceleration = lightray_acceleration;
    ray.initial_quat = corrected.inverse_quat;
    ray.early_terminate = 0;

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

    return ray;
};

struct lightray geodesic_to_trace_ray(float4 position, float4 velocity, dynamic_config_space struct dynamic_config* cfg)
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
    ray.sx = 0;
    ray.sy = 0;
    ray.position = position;
    ray.velocity = velocity;
    ray.acceleration = lightray_acceleration;
    ray.initial_quat = corrected.inverse_quat;
    ray.early_terminate = 0;
    ray.ku_uobsu = 1;

    return ray;
};

__kernel
void init_inertial_ray(__global float4* g_polar_position_in, int ray_count,
                       __global struct lightray* metric_rays, __global int* metric_ray_count,
                       __global float4* e0, __global float4* e1, __global float4* e2, __global float4* e3,
                       __global float3* geodesic_basis_speed,
                       dynamic_config_space struct dynamic_config* cfg)
{
    int id = get_global_id(0);

    if(id >= ray_count)
        return;

    float4 velocity = get_timelike_vector(geodesic_basis_speed[id], 1, e0[id], e1[id], e2[id], e3[id]);

    float4 position = spherical_to_generic(g_polar_position_in[id], cfg);

    struct lightray trace = geodesic_to_trace_ray(position, velocity, cfg);

    metric_rays[id] = trace;

    atomic_inc(metric_ray_count);
}

__kernel
void init_rays_generic(__global float4* g_polar_camera_in, __global float4* g_camera_quat,
                       __global struct lightray* metric_rays, __global int* metric_ray_count,
                       int width, int height,
                       __global int* termination_buffer,
                       int prepass_width, int prepass_height,
                       int flip_geodesic_direction,
                       __global float4* e0, __global float4* e1, __global float4* e2, __global float4* e3,
                       dynamic_config_space struct dynamic_config* cfg)
{
    int id = get_global_id(0);

    if(id >= width * height)
        return;

    const int cx = id % width;
    const int cy = id / width;

    float3 pixel_direction = calculate_pixel_direction(cx, cy, width, height, *g_camera_quat);

    float4 polar_camera = *g_polar_camera_in;

    float4 at_metric = spherical_to_generic(polar_camera, cfg);

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
            ray.early_terminate = 1;
        }
    }
    #endif // USE_PREPASS

    if(id == 0)
        *metric_ray_count = height * width;

    metric_rays[id] = ray;
}

/*void rk4_evaluate_velocity_at(float4 position, float4 velocity, float4* out_k_velocity, float dt, float dt_factor, float4 k)
{
    ///dt f(tn + ds, xn + k)

    float4 position_pds = position + velocity * dt * dt_factor;

    calculate_metric_generic_big(position_pds, g_metric_big);
    calculate_partial_derivatives_generic_big(position_pds, g_partials_big);

    float4 acceleration = calculate_acceleration_big(velocity + k, g_metric_big, g_partials_big);

    *out_k_velocity = acceleration * dt;
}

void rk4_evaluate_position_at(float4 position, float4 velocity, float* out_k_position, float dt, float dt_factor, float4 k)
{
    *out_k_position = dt * (velocity ;
}*/

///http://homepages.cae.wisc.edu/~blanchar/eps/ivp/ivp

///ok so
///we have x = defined by dx/ds is our position
///dx/ds = d2x/ds2 = is our velocity
///d2x/ds2 = f(s, x, dx/ds)

///so define dx/ds = z = g(s, x, z);

///dx/ds = z aka g(t, x, z). Velocity
///dz/dt = f(t, x, dx/dt) aka f(t, x, z) aka d2x/ds2. Acceleration

///so x(t) is exact position, ideally analytic but here produced by our approximation
///and z(t) is exact velocity

#ifdef GENERIC_BIG_METRIC
///velocity
#if 0
float4 rk4_g(float t, float4 position, float4 velocity)
{
    /*float4 estimated_pos = position + velocity * t;

    float g_metric_big[16] = {0};
    float g_partials_big[64] = {0};

    calculate_metric_generic_big(position_pds, g_metric_big);
    calculate_partial_derivatives_generic_big(position_pds, g_partials_big);

    float4 acceleration = calculate_acceleration_big(velocity, g_metric_big, g_partials_big);

    return velocity + acceleration * t;*/

    return velocity;
}

///acceleration
float4 rk4_f(float t, float4 position, float4 velocity)
{
    float g_metric_big[16] = {0};
    float g_partials_big[64] = {0};

    calculate_metric_generic_big(position, g_metric_big);
    calculate_partial_derivatives_generic_big(position, g_partials_big);

    return calculate_acceleration_big(velocity, g_metric_big, g_partials_big);
}
#endif // 0

/*void rk4_generic_big(float4* position, float4* velocity, float* step)
{
    float g_metric[16] = {0};

    float ds = *step;

    ///its important to remember here that dt is nothing to do with coordinate time
    ///its the affine parameter ds

    ///so
    //x(tn + 1) approx x(tn) + dt * dx(tn)
    //y(tn + 1) approx y(tn) + dt * dy(tn)

    //vx(tn + 1) approx vx(tn) + dt * dvx(tn)
    //vy(tn + 1) approx vy(tn) + dt * dvy(tn)

    float4 x = *position;
    float4 z = *velocity;

    float4 k1 = ds * rk4_g(0, x, z);
    float4 l1 = ds * rk4_f(0, x, z);

    float4 k2 = ds * rk4_g(ds/2, x + k1/2, z + l1/2);
    float4 l2 = ds * rk4_f(ds/2, x + k1/2, z + l1/2);

    float4 k3 = ds * rk4_g(ds/2, x + k2/2, z + l2/2);
    float4 l3 = ds * rk4_f(ds/2, x + k2/2, z + l2/2);

    float4 k4 = ds * rk4_g(ds/2, x + k3/2, z + l3/2);
    float4 l4 = ds * rk4_f(ds/2, x + k3/2, z + l3/2);

    float4 xt_t = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6;
    float4 zt_t = z + (l1 + 2 * l2 + 2 * l3 + l4) / 6;

    *position = xt_t;
    *velocity = zt_t;
}*/


/*void rk4_generic_big(float4* position, float4* velocity, float* step)
{
    float g_metric[16] = {0};

    float ds = *step;

    ///its important to remember here that dt is nothing to do with coordinate time
    ///its the affine parameter ds

    float a21 = 1/5.f;
    float a31 = 3/40.f;
    float a32 = 9/40.f;
    float a41 = 3/10.f;
    float a42 = -9/10.f;
    float a43 = 6/5.f;
    float a51 = -11/54.f;
    float a52 = 5/2.f;
    float a53 = -70/27.f;
    float a54 = 35/27.f;
    float a61 = 1631/55296.f;
    float a62 = 175/512.f;
    float a63 = 575/13824.f;
    float a64 = 44275/110592.f;
    float a65 = 253/4096.f;

    ///so
    //x(tn + 1) approx x(tn) + dt * dx(tn)
    //y(tn + 1) approx y(tn) + dt * dy(tn)

    //vx(tn + 1) approx vx(tn) + dt * dvx(tn)
    //vy(tn + 1) approx vy(tn) + dt * dvy(tn)

    float4 x = *position;
    float4 z = *velocity;

    float4 k1 = ds * rk4_g(0, x, z);
    float4 l1 = ds * rk4_f(0, x, z);

    float4 k2 = ds * rk4_g(ds/2, x + k1 * a21, z + l1 * a21);
    float4 l2 = ds * rk4_f(ds/2, x + k1 * a21, z + l1 * a21);

    float4 k3 = ds * rk4_g(ds/2, x + k1 * a31 + k2 * a32, z + k1 * a31 + k2 * a32);
    float4 l3 = ds * rk4_f(ds/2, x + k1 * a31 + k2 * a32, z + k1 * a31 + k2 * a32);

    float4 k4 = ds * rk4_g(ds/2, x + k3/2, z + l3/2);
    float4 l4 = ds * rk4_f(ds/2, x + k3/2, z + l3/2);

    float4 xt_t = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6;
    float4 zt_t = z + (l1 + 2 * l2 + 2 * l3 + l4) / 6;

    *position = xt_t;
    *velocity = zt_t;
}*/
#endif // GENERIC_BIG_METRIC

///https://www.math.kit.edu/ianm3/lehre/geonumint2009s/media/gni_by_stoermer-verlet.pdf
///todo:
///it would be useful to be able to combine data from multiple ticks which are separated by some delta, but where I don't have control over that delta
///I wonder if a taylor series expansion of F(y + dt) might be helpful
///this is actually regular velocity verlet with no modifications https://en.wikipedia.org/wiki/Verlet_integration#Velocity_Verlet
void step_verlet(float4 position, float4 velocity, float4 acceleration, bool always_lightlike, float ds, float4* position_out, float4* velocity_out, float4* acceleration_out, float* dT_ds, dynamic_config_space struct dynamic_config* cfg)
{
    #ifdef OLD_METRIC_STEP
    #ifndef GENERIC_BIG_METRIC
    float g_metric[4] = {};
    float g_partials[16] = {};
    #else
    float g_metric_big[16] = {};
    float g_partials_big[64] = {};
    #endif // GENERIC_BIG_METRIC

    float4 next_position = position + velocity * ds + 0.5f * acceleration * ds * ds;

    float4 intermediate_next_velocity = velocity + acceleration * ds;

    #ifndef GENERIC_BIG_METRIC
    calculate_metric_generic(next_position, g_metric, cfg);
    calculate_partial_derivatives_generic(next_position, g_partials, cfg);

    if(g_00_out)
    {
        *g_00_out = g_metric[0];
    }

    ///1ms
    if(always_lightlike)
        intermediate_next_velocity = fix_light_velocity2(intermediate_next_velocity, g_metric);

    float4 next_acceleration = calculate_acceleration(intermediate_next_velocity, g_metric, g_partials);
    #else
    calculate_metric_generic_big(next_position, g_metric_big, cfg);
    calculate_partial_derivatives_generic_big(next_position, g_partials_big, cfg);

    if(g_00_out)
    {
        *g_00_out = g_metric_big[0];
    }

    //intermediate_next_velocity = fix_light_velocity_big(intermediate_next_velocity, g_metric_big);
    float4 next_acceleration = calculate_acceleration_big(intermediate_next_velocity, g_metric_big, g_partials_big);
    #endif // GENERIC_BIG_METRIC
    #endif // 0

    if(dT_ds)
    {
        #ifndef GENERIC_BIG_METRIC
        float g_metric[4] = {};
        calculate_metric_generic(position, g_metric, cfg);
        #else
        float g_metric[16] = {};
        calculate_metric_generic_big(position, g_metric, cfg);
        #endif // GENERIC_BIG_METRIC

        *dT_ds = native_sqrt(fabs(dot_product_generic(velocity, velocity, g_metric)));
    }

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

    next_position = handle_coordinate_periodicity(next_position, cfg);

    *position_out = next_position;
    *velocity_out = next_velocity;
    *acceleration_out = next_acceleration;
}

void step_euler(float4 position, float4 velocity, float ds, float4* position_out, float4* velocity_out, dynamic_config_space struct dynamic_config* cfg)
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

    //velocity = fix_light_velocity2(velocity, g_metric);

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

float get_distance_to_object(float4 polar, dynamic_config_space struct dynamic_config* cfg)
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
    #define MIN_STEP 0.000001f

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

int calculate_ds_error(float current_ds, float4 next_acceleration, float4 acceleration, float max_acceleration, float* next_ds_out)
{
    float next_ds = 0;
    float diff = acceleration_to_precision(next_acceleration, max_acceleration, &next_ds);

    ///produces strictly worse results for kerr
    next_ds = 0.99f * current_ds * clamp(next_ds / current_ds, 0.3f, 2.f);

    next_ds = max(next_ds, MIN_STEP);

    *next_ds_out = next_ds;

    float err = max_acceleration;

    #ifdef SINGULARITY_DETECTION
    if(next_ds == MIN_STEP && (diff/I_HATE_COMPUTERS) > err * 10000)
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

struct sub_point
{
    float x, y, z;
    int parent;
    int object_parent;
};

#if 0
__kernel
void generate_acceleration_structure(__global struct triangle* tris, int tri_count, __global int* offset_map, __global int* offset_counts, __global int* pool, float width, int width_num)
{
    int id = get_global_id(0);

    if(id >= tri_count)
        return;

    __global struct triangle* tri = &tris[id];

    #define MIN3(x, y, z) min(min(x, y), z)
    #define MAX3(x, y, z) max(max(x, y), z)

    float3 minimum = (float3)(MIN3(tri->v0x, tri->v1x, tri->v2x), MIN3(tri->v0y, tri->v1y, tri->v2y), MIN3(tri->v0z, tri->v1z, tri->v2z));
    float3 maximum = (float3)(MAX3(tri->v0x, tri->v1x, tri->v2x), MAX3(tri->v0y, tri->v1y, tri->v2y), MAX3(tri->v0z, tri->v1z, tri->v2z));

    float voxel_cube_size = width / width_num;
}
#endif // 0

float3 world_to_voxel(float3 world, float width, int width_num)
{
    float scale = width / width_num;

    return world / scale;
}

float4 world_to_voxel4(float4 world, float width, float time_width, int width_num)
{
    float scale = width / width_num;
    float time_scale = time_width / width_num;

    float4 vscale = (float4)(time_scale, scale, scale, scale);

    return world / vscale;
}

int mod(int a, int b)
{
    int r = a % b;
    return r < 0 ? r + b : r;
}

int3 loop_voxel(int3 in, int width_num)
{
    in.x = mod(in.x, width_num);
    in.y = mod(in.y, width_num);
    in.z = mod(in.z, width_num);

    return in;
}

int4 loop_voxel4(int4 in, int width_num)
{
    in.x = mod(in.x, width_num);
    in.y = mod(in.y, width_num);
    in.z = mod(in.z, width_num);
    in.w = mod(in.w, width_num);

    return in;
}

__kernel void pull_to_geodesics(__global struct object* current_pos, __global float4* geodesic_out, int max_path_length, int object_count)
{
    int id = get_global_id(0);

    if(id >= object_count)
        return;

    geodesic_out[0 * max_path_length + id] = current_pos[id].pos;
}

__kernel
void clear_accel_counts(__global int* offset_counts, int size)
{
    int x = get_global_id(0);

    if(x >= size)
        return;

    offset_counts[x] = 0;
}

__kernel
void generate_acceleration_counts(__global struct sub_point* sp, int sp_count, __global struct object* objs, __global int* offset_counts, float width, int width_num)
{
    int id = get_global_id(0);

    if(id >= sp_count)
        return;

    float voxel_cube_size = width / width_num;

    struct sub_point mine = sp[id];

    float3 pos = (float3)(mine.x, mine.y, mine.z) + objs[mine.object_parent].pos.yzw;
    int w_id = mine.parent;

    float3 grid_pos = floor(pos / voxel_cube_size);

    int3 int_grid_pos = (int3)(grid_pos.x, grid_pos.y, grid_pos.z);

    for(int z=-1; z <= 1; z++)
    {
        for(int y=-1; y <= 1; y++)
        {
            for(int x=-1; x <= 1; x++)
            {
                int3 off = (int3)(x, y, z);
                int3 fin = int_grid_pos + off;

                fin.x = mod(fin.x, width_num);
                fin.y = mod(fin.y, width_num);
                fin.z = mod(fin.z, width_num);

                int oid = fin.z * width_num * width_num + fin.y * width_num + fin.x;

                atomic_inc(&offset_counts[oid]);
            }
        }
    }
}

float positive_fmod(float a, float b)
{
    float v = fmod(a, b);

    if(v < 0.f)
       v += b;

    return v;
}

float4 positive_fmod4(float4 a, float4 b)
{
    return (float4)(positive_fmod(a.x, b.x),
                    positive_fmod(a.y, b.y),
                    positive_fmod(a.z, b.z),
                    positive_fmod(a.w, b.w));
}

///https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.42.3443&rep=rep1&type=pdf
struct step_setup
{
    int4 end_grid_pos;
    char4 sign_step; ///ie stepX, stepY, stepZ

    float4 tMax;
    float4 tDelta;

    int4 current;
    int idx;
    bool should_end;
};

struct step_setup setup_step(float4 grid1, float4 grid2)
{
    float4 ray_dir = grid2 - grid1;

    struct step_setup ret;

    float4 floor_grid1 = floor(grid1);
    float4 floor_grid2 = floor(grid2);

    ret.end_grid_pos = (int4)(floor_grid2.x, floor_grid2.y, floor_grid2.z, floor_grid2.w);
    ret.current = (int4)(floor_grid1.x, floor_grid1.y, floor_grid1.z, floor_grid1.w);

    ret.sign_step = convert_char4(sign(ray_dir));

    ///so, we're a ray, going at a speed of ray_dir.x
    ///our position is grid1.x
    ///here we're working in voxel coordinates, which means that there are
    ///voxel boundaries at 0, 1, 2, 3, 4, 5
    ///so, if I'm at 0.3, moving at a speed of 0.5 in the +x direction, we'll hit 1 in
    ///(1 - 0.3) / 0.5f units of time == 1.4
    ///so 0.3 + 0.5 * 1.4 == 1
    ///if I'm a ray at 0.3, moving at a speed of -0.5 in the +x direction, we're looking to hit 0
    ///in which case the calcultion becomes (0 - 0.3) / -0.5 == 0.6
    ///which gives 0.3 + -0.5 * 0.6 == 0
    ///therefore, if signstep < 0 we're looking to intersect with 0, and if signstep > 0 we're looking to intersect with 1
    ///where I take the fractional part of the coordinate, ie fmod(pos, 1)
    ///though careful!! we CAN have negative coordinates here, the looping is done elsewhere
    ///this means need to positive_fmod
    float4 position_fraction = grid1 - floor(grid1);

    float4 target = (float4)(1,1,1,1);

    if(ret.sign_step.x < 0)
        target.x = 0;

    if(ret.sign_step.y < 0)
        target.y = 0;

    if(ret.sign_step.z < 0)
        target.z = 0;

    if(ret.sign_step.w < 0)
        target.w = 0;

    if(ray_dir.x == 0)
        ray_dir.x = 1;

    if(ray_dir.y == 0)
        ray_dir.y = 1;

    if(ray_dir.z == 0)
        ray_dir.z = 1;

    if(ray_dir.w == 0)
        ray_dir.w = 1;

    float4 tMax = (target - position_fraction) / ray_dir;

    ///if we move at a speed of 0.7, the time it takes in t to traverse a voxel of size 1 is 1/0.7 == 1.42
    float4 tDelta = 1.f / fabs(ray_dir);

    ret.tMax = tMax;
    ret.tDelta = tDelta;
    ret.idx = 0;
    ret.should_end = false;

    return ret;
};

void do_step(struct step_setup* step)
{
    if(all(step->current == step->end_grid_pos))
    {
        step->idx++;
        step->should_end = true;
        return;
    }

    float tMaxArray[4] = {step->tMax.x, step->tMax.y, step->tMax.z, step->tMax.w};
    char tStep[4] = {step->sign_step.x, step->sign_step.y, step->sign_step.z, step->sign_step.w};

    int current[4] = {step->current.x, step->current.y, step->current.z, step->current.w};
    int end_pos[4] = {step->end_grid_pos.x, step->end_grid_pos.y, step->end_grid_pos.z, step->end_grid_pos.w};

    int which_min = -1;
    float my_min = FLT_MAX;

    for(int i=0; i < 4; i++)
    {
        if(tMaxArray[i] < my_min && tStep[i] != 0 && current[i] != end_pos[i])
        {
            which_min = i;
            my_min = tMaxArray[i];
        }
    }

    if(which_min == -1)
        step->should_end = true;

    ///i sure love programming
    if(which_min == 0)
    {
        step->tMax.x += step->tDelta.x;
        step->current.x += step->sign_step.x;
    }

    if(which_min == 1)
    {
        step->tMax.y += step->tDelta.y;
        step->current.y += step->sign_step.y;
    }

    if(which_min == 2)
    {
        step->tMax.z += step->tDelta.z;
        step->current.z += step->sign_step.z;
    }

    if(which_min == 3)
    {
        step->tMax.w += step->tDelta.w;
        step->current.w += step->sign_step.w;
    }

    //step->current += step->step;
    step->idx++;
}

bool is_step_finished(struct step_setup* step)
{
    ///do I sometimes want to include the end point in smearing?
    return step->should_end || step->idx > 600;
}

unsigned int index_acceleration(struct step_setup* setup, int width_num)
{
    int4 ifloor = setup->current;

    ifloor = loop_voxel4(ifloor, width_num);

    return ifloor.x * width_num * width_num * width_num + ifloor.w * width_num * width_num + ifloor.z * width_num + ifloor.y;

    //return ifloor.w * width_num * width_num * width_num + ifloor.z * width_num * width_num + ifloor.y * width_num + ifloor.x;
}

void sort2(float* v0, float* v1)
{
    float iv0 = *v0;
    float iv1 = *v1;

    *v0 = min(iv0, iv1);
    *v1 = max(iv0, iv1);
}

bool range_overlaps(float s0, float s1, float e0, float e1)
{
    sort2(&s0, &s1);
    sort2(&e0, &e1);

    return s0 <= e1 && e0 <= s1;
}

__kernel
void generate_smeared_acceleration(__global struct sub_point* sp, int sp_count,
                                          int obj_count,
                                          __global struct triangle* reference_tris,
                                          __global int* offset_map, __global int* offset_counts,
                                          __global struct computed_triangle* mem_buffer, __global float* start_times_memory, __global float* delta_times_memory,
                                          __global int* unculled_counts,
                                          __global int* any_visible,
                                          __global float4* object_geodesics, __global int* object_geodesic_counts,
                                          __global float* object_geodesic_ds,
                                          __global float4* p_e0, __global float4* p_e1, __global float4* p_e2, __global float4* p_e3,
                                          float start_ds, float end_ds,
                                          float width, float time_width, int width_num,
                                          __global int* ray_time_min, __global int* ray_time_max,
                                          __global int* old_cell_time_min, __global int* old_cell_time_max,
                                          int should_store,
                                          int generate_unculled_counts,
                                          dynamic_config_space struct dynamic_config* cfg)
{
    int id = get_global_id(0);

    if(id >= sp_count)
        return;

    struct sub_point mine = sp[id];

    ///number of geodesics == objs.size()
    int stride = obj_count;

    int count = object_geodesic_counts[mine.object_parent];

    float last_output_ds = -10;
    float ds_accum = 0;
    //float current_dt = object_geodesics[0 * stride + mine.object_parent];

    ///this is super suboptimal in regions of high curvature
    ///to future james: you cannot parameterise these curves by coordinate time, especially due to CTCs
    float max_ds_step = 0.25f;
    float ds_error = 0;

    float voxel_cube_size = width / width_num;

    struct triangle my_tri = reference_tris[mine.parent];

    float lowest_time = *ray_time_min - 1;
    float maximum_time = *ray_time_max + 1;

    int skip = 8;

    ///if I'm doing bresenhams, then ds_stepping makes no sense and I am insane
    for(int cc=0; cc < count - skip; cc += skip)
    {
        if(cc > 2048)
            return;

        int next_cc = cc + skip;

        float4 native_current = object_geodesics[cc * stride + mine.object_parent];
        float4 native_next = object_geodesics[next_cc * stride + mine.object_parent];

        if(!range_overlaps(lowest_time, maximum_time, native_current.x, native_next.x))
            continue;

        float4 current = generic_to_cartesian(native_current, cfg);
        float4 next = generic_to_cartesian(native_next, cfg);

        #ifndef TRI_PRECESSION
        //if(should_store)
        //printf("Current %f %f %f %f\n", current.x, current.y, current.z, current.w);

        struct computed_triangle local_tri;

        float delta_time = (next - current).x;
        float output_time = current.x;

        local_tri.v0x = my_tri.v0x + current.y;
        local_tri.v0y = my_tri.v0y + current.z;
        local_tri.v0z = my_tri.v0z + current.w;

        ///this is a constant. These are actually just all offsets. Could massively reduce memory storage
        ///by storing location, and tri index
        local_tri.dv1x = my_tri.v1x - my_tri.v0x;
        local_tri.dv1y = my_tri.v1y - my_tri.v0y;
        local_tri.dv1z = my_tri.v1z - my_tri.v0z;

        local_tri.dv2x = my_tri.v2x - my_tri.v0x;
        local_tri.dv2y = my_tri.v2y - my_tri.v0y;
        local_tri.dv2z = my_tri.v2z - my_tri.v0z;

        local_tri.dvx = (next - current).y;
        local_tri.dvy = (next - current).z;
        local_tri.dvz = (next - current).w;

        #ifdef TIME_CULL
        ///my later pos is actually earlier than the last pos. Swap start and end
        if(next.x < current.x)
        {
            delta_time = (current - next).x;
            output_time = next.x;

            local_tri.v0x = my_tri.v0x + next.y;
            local_tri.v0y = my_tri.v0y + next.z;
            local_tri.v0z = my_tri.v0z + next.w;

            local_tri.dvx = (current - next).y;
            local_tri.dvy = (current - next).z;
            local_tri.dvz = (current - next).w;
        }
        #endif // TIME_CULL

        float4 cartesian_mine_start = (float4)(0.f, mine.x, mine.y, mine.z);
        float4 cartesian_mine_next = (float4)(0.f, mine.x, mine.y, mine.z);
        #else
        float delta_time = (next - current).x;
        float output_time = current.x;

        float4 s_e0 = p_e0[cc * stride + mine.object_parent];
        float4 s_e1 = p_e1[cc * stride + mine.object_parent];
        float4 s_e2 = p_e2[cc * stride + mine.object_parent];
        float4 s_e3 = p_e3[cc * stride + mine.object_parent];

        float4 e_e0 = p_e0[next_cc * stride + mine.object_parent];
        float4 e_e1 = p_e1[next_cc * stride + mine.object_parent];
        float4 e_e2 = p_e2[next_cc * stride + mine.object_parent];
        float4 e_e3 = p_e3[next_cc * stride + mine.object_parent];

        float4 vert_0 = (float4)(0, my_tri.v0x, my_tri.v0y, my_tri.v0z);
        float4 vert_1 = (float4)(0, my_tri.v1x, my_tri.v1y, my_tri.v1z);
        float4 vert_2 = (float4)(0, my_tri.v2x, my_tri.v2y, my_tri.v2z);

        float4 s_coordinate_v0 = tetrad_to_coordinate_basis(vert_0, s_e0, s_e1, s_e2, s_e3);
        float4 s_coordinate_v1 = tetrad_to_coordinate_basis(vert_1, s_e0, s_e1, s_e2, s_e3);
        float4 s_coordinate_v2 = tetrad_to_coordinate_basis(vert_2, s_e0, s_e1, s_e2, s_e3);

        float4 e_coordinate_v0 = tetrad_to_coordinate_basis(vert_0, e_e0, e_e1, e_e2, e_e3);
        float4 e_coordinate_v1 = tetrad_to_coordinate_basis(vert_1, e_e0, e_e1, e_e2, e_e3);
        float4 e_coordinate_v2 = tetrad_to_coordinate_basis(vert_2, e_e0, e_e1, e_e2, e_e3);

        float4 cart_s_coordinate_v0 = generic_velocity_to_cartesian_velocity(native_current, s_coordinate_v0, cfg) + current;
        float4 cart_s_coordinate_v1 = generic_velocity_to_cartesian_velocity(native_current, s_coordinate_v1, cfg) + current;
        float4 cart_s_coordinate_v2 = generic_velocity_to_cartesian_velocity(native_current, s_coordinate_v2, cfg) + current;

        float4 cart_e_coordinate_v0 = generic_velocity_to_cartesian_velocity(native_next, e_coordinate_v0, cfg) + next;
        float4 cart_e_coordinate_v1 = generic_velocity_to_cartesian_velocity(native_next, e_coordinate_v1, cfg) + next;
        float4 cart_e_coordinate_v2 = generic_velocity_to_cartesian_velocity(native_next, e_coordinate_v2, cfg) + next;

        struct computed_triangle local_tri;

        local_tri.v0x = cart_s_coordinate_v0.y;
        local_tri.v0y = cart_s_coordinate_v0.z;
        local_tri.v0z = cart_s_coordinate_v0.w;

        local_tri.v1x = cart_s_coordinate_v1.y;
        local_tri.v1y = cart_s_coordinate_v1.z;
        local_tri.v1z = cart_s_coordinate_v1.w;

        local_tri.v2x = cart_s_coordinate_v2.y;
        local_tri.v2y = cart_s_coordinate_v2.z;
        local_tri.v2z = cart_s_coordinate_v2.w;

        local_tri.e0x = cart_e_coordinate_v0.y;
        local_tri.e0y = cart_e_coordinate_v0.z;
        local_tri.e0z = cart_e_coordinate_v0.w;

        local_tri.e1x = cart_e_coordinate_v1.y;
        local_tri.e1y = cart_e_coordinate_v1.z;
        local_tri.e1z = cart_e_coordinate_v1.w;

        local_tri.e2x = cart_e_coordinate_v2.y;
        local_tri.e2y = cart_e_coordinate_v2.z;
        local_tri.e2z = cart_e_coordinate_v2.w;

        float4 coordinate_mine_start = tetrad_to_coordinate_basis((float4)(0.f, mine.x, mine.y, mine.z), s_e0, s_e1, s_e2, s_e3);
        float4 cartesian_mine_start = generic_velocity_to_cartesian_velocity(native_current, coordinate_mine_start, cfg);

        float4 coordinate_mine_next = tetrad_to_coordinate_basis((float4)(0.f, mine.x, mine.y, mine.z), e_e0, e_e1, e_e2, e_e3);
        float4 cartesian_mine_next = generic_velocity_to_cartesian_velocity(native_next, coordinate_mine_next, cfg);
        #endif // TRI_PRECESSION

        float4 pos = current + cartesian_mine_start;
        float4 next_pos = next + cartesian_mine_next;

        float4 grid_pos = world_to_voxel4(pos, width, time_width, width_num);
        float4 next_grid_pos = world_to_voxel4(next_pos, width, time_width, width_num);

        //grid_pos.x = 0;
        //last_grid_pos.x = 0;

        //if(all(grid_pos == next_grid_pos))
        //    continue;

        struct step_setup steps = setup_step(grid_pos, next_grid_pos);

        //if(should_store)
        //printf("Fgp %f %f %f %f ngp %f %f %f %f\n", grid_pos.x, grid_pos.y, grid_pos.z, grid_pos.w, next_grid_pos.x, next_grid_pos.y, next_grid_pos.z, next_grid_pos.w);

        while(!is_step_finished(&steps))
        {

            //if(should_store)
            //printf("Gp %i %i %i %i\n", steps.current.x, steps.current.y, steps.current.z, steps.current.w);

            unsigned int oid = index_acceleration(&steps, width_num);

            //#define USE_FEEDBACK_CULLING
            #ifdef USE_FEEDBACK_CULLING
            if(generate_unculled_counts)
            {
                unculled_counts[oid] = 1;
            }
            #endif // USE_FEEDBACK_CULLING

            #ifdef USE_FEEDBACK_CULLING
            bool any_valid = false;
            #else
            bool any_valid = true;
            #endif // USE_FEEDBACK_CULLING

            #ifdef USE_FEEDBACK_CULLING
            int test_time_min = old_cell_time_min[oid];
            int test_time_max = old_cell_time_max[oid];

            bool checked = true;

            if(test_time_min == 2147483647 || test_time_max == (-2147483647 - 1))
                checked = false;

            if(!range_overlaps(current.x, next.x, test_time_min, test_time_max))
                checked = false;

            if(checked)
                any_valid = true;
            #endif // USE_FEEDBACK_CULLING

            if(any_valid)
            {
                int lid = atomic_inc(&offset_counts[oid]);

                if(should_store)
                {
                    int mem_start = offset_map[oid];

                    mem_buffer[mem_start + lid] = local_tri;
                    start_times_memory[mem_start + lid] = output_time;
                    delta_times_memory[mem_start + lid] = delta_time;

                    *any_visible = 1;
                }
            }

            do_step(&steps);
        }
    }
}

///so
///need a generate_smeared_acceleration_counts
///will take in a start coordinate time, and an end coordinate time. End coordinate time can be camera coordinate time, as rays
///go strictly backwards in time (?) (????)
///then, step along the ray for a particular tri, calculating its interpolated position at constant steps along the coordinate time
///at every step, dump it into the acceleration structure, which *could* be 3d but more likely will want to be 4d for perf
///then in step_verlet, look up in the correct chunk by time, skip any tris with an incorrect time coordinate (ie loop), then
///do a constant time approximation. This latter part may be very problematic, so could assume a constant velocity and interpolate in time

///resets offset_counts
__kernel
void alloc_acceleration(__global int* offset_map, __global int* offset_counts, int max_count, __global int* mem_count, int max_memory_size)
{
    int idx = get_global_id(0);

    if(idx >= max_count)
        return;

    int my_count = offset_counts[idx];

    int offset = 0;

    if(my_count > 0)
        offset = atomic_add(mem_count, my_count);

    if(offset + my_count > (max_memory_size / sizeof(struct computed_triangle)))
    {
        offset = (max_memory_size / sizeof(struct computed_triangle)) - my_count;

        printf("Error in alloc acceleration\n");
    }

    offset_map[idx] = offset;
}

__kernel
void generate_acceleration_data(__global struct sub_point* sp, int sp_count, __global struct object* objs, __global struct triangle* reference_tris, __global int* offset_map, __global int* offset_counts, __global struct triangle* mem_buffer, float width, int width_num)
{
    int id = get_global_id(0);

    if(id >= sp_count)
        return;

    float voxel_cube_size = width / width_num;

    struct sub_point mine = sp[id];

    float3 pos = (float3)(mine.x, mine.y, mine.z) + objs[mine.object_parent].pos.yzw;
    int w_id = mine.parent;

    struct triangle my_tri = reference_tris[w_id];

    struct object my_object = objs[my_tri.parent];

    my_tri.v0x += my_object.pos.y;
    my_tri.v0y += my_object.pos.z;
    my_tri.v0z += my_object.pos.w;

    my_tri.v1x += my_object.pos.y;
    my_tri.v1y += my_object.pos.z;
    my_tri.v1z += my_object.pos.w;

    my_tri.v2x += my_object.pos.y;
    my_tri.v2y += my_object.pos.z;
    my_tri.v2z += my_object.pos.w;

    float3 grid_pos = floor(pos / voxel_cube_size);

    int3 int_grid_pos = (int3)(grid_pos.x, grid_pos.y, grid_pos.z);

    for(int z=-1; z <= 1; z++)
    {
        for(int y=-1; y <= 1; y++)
        {
            for(int x=-1; x <= 1; x++)
            {
                int3 off = (int3)(x, y, z);
                int3 fin = int_grid_pos + off;

                fin.x = mod(fin.x, width_num);
                fin.y = mod(fin.y, width_num);
                fin.z = mod(fin.z, width_num);

                int oid = fin.z * width_num * width_num + fin.y * width_num + fin.x;

                int lid = atomic_inc(&offset_counts[oid]);

                int mem_start = offset_map[oid];

                mem_buffer[mem_start + lid] = my_tri;
            }
        }
    }
}

void swap3f(float3* i1, float3* i2)
{
    float3 v = *i1;

    *i1 = *i2;
    *i2 = v;
}

struct potential_intersection
{
    int cx, cy;
    float st, sx, sy, sz;
    float et, ex, ey, ez;
};

///todo: winding order
float3 triangle_normal(float3 v0, float3 v1, float3 v2)
{
    float3 U = v1 - v0;
    float3 V = v2 - v0;

    return normalize(cross(U, V));
}

float3 triangle_normal_U(float3 v0, float3 v1, float3 v2)
{
    float3 U = v1 - v0;
    float3 V = v2 - v0;

    return cross(U, V);
}

float3 triangle_normal_D(float3 v1_m_v0, float3 v2_m_v0)
{
    return cross(v1_m_v0, v2_m_v0);
}

bool ray_toblerone_could_intersect(float4 pos, float4 dir, float tri_start_t, float tri_end_t)
{
    float ray_start_t = pos.x;
    float ray_end_t = pos.x + dir.x;

    return range_overlaps(ray_start_t, ray_end_t, tri_start_t, tri_end_t);
}

///returns from point to line
float3 point_to_line(float3 line_pos, float3 line_dir, float3 point)
{
    line_dir = normalize(line_dir);

    float3 on_line = line_pos + dot(point - line_pos, line_dir) * line_dir;

    return on_line - point;
}

///dir is not normalised, should really use a pos2
bool ray_intersects_toblerone(float4 pos, float4 dir, __global struct computed_triangle* ctri, float tri_start_time, float tri_end_time, float* t_out)
{
    #ifndef TRI_PRECESSION
    float3 to_v1 = (float3)(ctri->dv1x, ctri->dv1y, ctri->dv1z);
    float3 to_v2 = (float3)(ctri->dv2x, ctri->dv2y, ctri->dv2z);

    float3 v0 = (float3)(ctri->v0x, ctri->v0y, ctri->v0z);
    float3 v1 = v0 + to_v1;
    float3 v2 = v0 + to_v2;

    float3 t_diff = (float3)(ctri->dvx, ctri->dvy, ctri->dvz);

    bool is_static = all(t_diff == 0.f);

    float3 e0 = v0 + t_diff;
    float3 e1 = v1 + t_diff;
    float3 e2 = v2 + t_diff;
    #else
    #define TRI_COLLIDE
    float3 v0 = (float3)(ctri->v0x, ctri->v0y, ctri->v0z);
    float3 v1 = (float3)(ctri->v1x, ctri->v1y, ctri->v1z);
    float3 v2 = (float3)(ctri->v2x, ctri->v2y, ctri->v2z);

    float3 e0 = (float3)(ctri->e0x, ctri->e0y, ctri->e0z);
    float3 e1 = (float3)(ctri->e1x, ctri->e1y, ctri->e1z);
    float3 e2 = (float3)(ctri->e2x, ctri->e2y, ctri->e2z);

    bool is_static = all(v0 == e0) && all(v1 == e1) && all(v2 == e2);
    #endif // TRI_PRECESSION

    /*if(tri_start_time == tri_end_time)
    {
        return ray_intersects_triangle(pos.yzw, dir.yzw, v0, v1, v2, 0);
    }*/

    if(is_static)
    {
        if(!range_overlaps(pos.x, pos.x + dir.x, tri_start_time, tri_end_time))
            return false;

        float ray_t = 0;

        bool success = ray_intersects_triangle(pos.yzw, dir.yzw, v0, v1, v2, &ray_t, 0, 0);

        if(success && t_out)
        {
            *t_out = ray_t;
        }

        return success;
    }


    ///this is marginally faster if there's lots of false positives
    /*{
        float3 max_v = max(max(v0, v1), v2);
        float3 max_e = max(max(e0, e1), e2);

        float3 min_v = min(min(v0, v1), v2);
        float3 min_e = min(min(e0, e1), e2);

        float3 max_all = max(max_v, max_e);
        float3 min_all = min(min_v, min_e);

        if(!range_overlaps(min_all.x, max_all.x, pos.y, pos.y + dir.y))
            return false;

        if(!range_overlaps(min_all.y, max_all.y, pos.z, pos.z + dir.z))
            return false;

        if(!range_overlaps(min_all.z, max_all.z, pos.w, pos.w + dir.w))
            return false;
    }*/

    //#define TRI_COLLIDE
    #ifdef TRI_COLLIDE

    bool any_t = false;
    float min_t = 0;
    float max_t = 0;

    ///time of this triangle
    float interpolated_min_time = 0;
    float interpolated_max_time = 0;

    float3 intermediate_planes[3 * 4] =
    {
        v1, v0, e0, e1,
        v0, v2, e2, e0,
        v2, v1, e1, e2,
    };

    float3 base_tris[3 * 2] = {
        v0, v1, v2,
        e0, e1, e2,
    };

    for(int i=0; i < 3; i++)
    {
        float3 base_0 = intermediate_planes[i * 4 + 0];
        float3 base_1 = intermediate_planes[i * 4 + 1];
        float3 end_0 = intermediate_planes[i * 4 + 2];
        float3 end_1 = intermediate_planes[i * 4 + 3];

        ///triangles are!
        ///base_0, base1, end0
        ///base_0, end0, end1

        float3 tri_0_0 = base_0;
        float3 tri_0_1 = base_1;
        float3 tri_0_2 = end_0;

        float3 tri_1_0 = base_0;
        float3 tri_1_1 = end_0;
        float3 tri_1_2 = end_1;

        float t_0 = 0;
        float t_1 = 0;

        bool hit_1 = ray_intersects_triangle(pos.yzw, dir.yzw, tri_0_0, tri_0_1, tri_0_2, &t_0, 0, 0);
        bool hit_2 = ray_intersects_triangle(pos.yzw, dir.yzw, tri_1_0, tri_1_1, tri_1_2, &t_1, 0, 0);

        if(!hit_1 && !hit_2)
            continue;

        float which_t = 0;

        if(hit_1)
            which_t = t_0;

        if(hit_2)
            which_t = t_1;

        float3 hit_pos = pos.yzw + dir.yzw * which_t;

        #if 0
        float3 to_upper_edge = point_to_line(end_0, end_1 - end_0, hit_pos);
        float3 to_lower_edge = point_to_line(base_0, base_1 - base_0, hit_pos);

        float full_length = length(to_upper_edge) + length(to_lower_edge);

        float value_at_upper = tri_end_time;
        float value_at_lower = tri_start_time;

        ///if 0, we're situated at the upper line
        ///if 1, we're situated at the lower line
        //float fraction_from_upper = length(to_upper_edge) / full_length;

        float fraction_from_upper = 0.f;

        if(full_length > 0.00001f)
        {
            fraction_from_upper = length(to_upper_edge) / full_length;
        }

        fraction_from_upper = clamp(fraction_from_upper, 0.f, 1.f);

        float my_time = value_at_upper * (1 - fraction_from_upper) + value_at_lower * fraction_from_upper;
        #endif // 0


        #if 0
        {
            float3 v00 = base_0;
            float3 v10 = base_1;
            float3 v11 = end_0;
            float3 v01 = end_1;

            float3 X = hit_pos;

            float3 A = v00 - X;
            float3 B = v10 - v00;
            float3 C = v01 - v00;
            float3 D = v00 - v10 - v01 + v11;

            ///https://www.reedbeta.com/blog/quadrilateral-interpolation-part-2/
        }
        #endif // 0

        float my_time = 0;

        ///https://www.wolframalpha.com/input?i=%28%28c+%2B+d+*+y+-+a+-+b+*+y%29+%2F+%28%28c+%2B+d+*+y+-+a+-+b+*+y%29%5E2%29%29+*+%28%28k+-+a+-+b+*+y%29+%2F+%28k+-+a+-+b+*+y%29%5E2%29+%3D+1+solve+for+y
        {
            float3 e = end_0;
            float3 v = base_0;

            float3 en = normalize(end_1 - end_0);
            float3 vn = normalize(base_1 - base_0);

            float3 ps = hit_pos;

            float3 a = v;
            float3 b = vn;
            float3 c = e;
            float3 d = en;
            float3 k = ps;

            float start = 1 / (2 * dot(b, b - d));

            float chunk_1 = -2 * dot(b, d) * (dot(a, c) - dot(a, k) - dot(c, k) + dot(k, k) + 2);
            float chunk_2 = dot(d, d) * dot(a - k, a - k);
            float chunk_3 = dot(b, b) * (dot(c, c) - 2 * dot(c, k) + dot(k,k) + 4);
            float chunk_4 = dot(a, d - 2 * b);
            float chunk_5 = dot(b, c) + dot(b, k) - dot(d, k);

            float rhs_2 = -sqrt(chunk_1 + chunk_2 + chunk_3) + chunk_4 + chunk_5;

            float y = start * rhs_2;

            float t = y;

            ///t = y

            float3 upper = e + en * t;
            float3 lower = v * vn * t;

            float ul_len = length(upper - lower);
            float my_len = length(ps - lower);

            ///when frac = 0, ps == lower. When frac = 1, ps == upper
            float frac = my_len / ul_len;

            my_time = frac * tri_end_time + (1 - frac) * tri_start_time;
        }

        if(!any_t)
        {
            min_t = which_t;
            max_t = which_t;

            interpolated_min_time = my_time;
            interpolated_max_time = my_time;
        }
        else
        {
            if(which_t < min_t)
            {
                min_t = which_t;
                interpolated_min_time = my_time;
            }

            if(which_t > max_t)
            {
                max_t = which_t;
                interpolated_max_time = my_time;
            }
        }

        any_t = true;
    }

    for(int i=0; i < 2; i++)
    {
        float3 tri_0 = base_tris[i * 3 + 0];
        float3 tri_1 = base_tris[i * 3 + 1];
        float3 tri_2 = base_tris[i * 3 + 2];

        float my_time = 0;

        if(i == 0)
            my_time = tri_start_time;

        if(i == 1)
            my_time = tri_end_time;

        float which_t = 0;

        if(ray_intersects_triangle(pos.yzw, dir.yzw, tri_0, tri_1, tri_2, &which_t, 0, 0))
        {
            if(!any_t)
            {
                min_t = which_t;
                max_t = which_t;

                interpolated_min_time = my_time;
                interpolated_max_time = my_time;
            }
            else
            {
                if(which_t < min_t)
                {
                    min_t = which_t;
                    interpolated_min_time = my_time;
                }

                if(which_t > max_t)
                {
                    max_t = which_t;
                    interpolated_max_time = my_time;
                }
            }

            any_t = true;
        }
    }

    if(!any_t)
        return false;

    #if 0
    float3 vertices[8 * 3] = {
        v0, v1, v2,
        v0, e0, v1,
        v0, e0, v2,
        e0, e1, e2,
        e0, e1, v1,
        e0, e2, v2,
        e1, v2, v1,
        e1, e2, v2
    };

    float vert_times[8 * 3] = {
        tri_start_time, tri_start_time, tri_start_time,
        tri_start_time, tri_end_time, tri_start_time,
        tri_start_time, tri_end_time, tri_start_time,
        tri_end_time, tri_end_time, tri_end_time,
        tri_end_time, tri_end_time, tri_start_time,
        tri_end_time, tri_end_time, tri_start_time,
        tri_end_time, tri_start_time, tri_start_time,
        tri_end_time, tri_end_time, tri_start_time
    };

    ///tri * 3 + vert

    ///bool ray_intersects_triangle(float3 origin, float3 direction, float3 vertex0, float3 vertex1, float3 vertex2, float* t_out)

    bool any_t = false;
    float min_t = 0;
    float max_t = 0;

    ///time of this triangle
    float interpolated_min_time = 0;
    float interpolated_max_time = 0;

    for(int i=0; i < 8; i++)
    {
        float3 l_v0 = vertices[i * 3 + 0];
        float3 l_v1 = vertices[i * 3 + 1];
        float3 l_v2 = vertices[i * 3 + 2];

        float my_t = 0;

        float u = 0;
        float v = 0;

        if(ray_intersects_triangle(pos.yzw, dir.yzw, l_v0, l_v1, l_v2, &my_t, &u, &v))
        {
            float w = 1 - u - v;

            float tv0 = vert_times[i * 3 + 0];
            float tv1 = vert_times[i * 3 + 1];
            float tv2 = vert_times[i * 3 + 2];

            ///ok barycentric coordinates is wrong
            ///because we only want to interpolate vertically
            ///but its close to a good idea!
            //float i_t = tv0 * u + tv1 * v + tv2 * (1 - u - v);

            float i_t = 0;

            /*if(tv0 == tv1 && tv1 == tv2)
            {
                i_t = tv0;
            }
            else
            {
                ///tv0 != tv2
                if(tv0 == tv1)
                {
                    float n_u = u / (u + w);
                    float n_w = w / (u + w);

                    i_t = n_u * tv0 + n_w * tv2;
                }

                ///tv0 != tv1
                else if(tv0 == tv2)
                {
                    float n_u = u / (u + v);
                    float n_v = v / (u + v);

                    i_t = n_u * tv0 + n_v * tv1;
                }

                ///tv0 != tv1
                else //if(tv1 == tv2)
                {
                    float n_u = u / (u + v);
                    float n_v = v / (u + v);

                    i_t = n_u * tv0 + n_v * tv1;
                }
            }*/

            if(tv0 == tv1 && tv1 == tv2)
            {
                i_t = tv0;
            }
            else
            {
                ///tv0 != tv2
                if(tv0 == tv1)
                {
                    float a_1 = (u + v)/2.f;

                    float n_u = a_1 / (a_1 + w);
                    float n_w = w / (a_1 + w);

                    i_t = n_u * tv0 + n_w * tv2;
                }

                ///tv0 != tv1
                else if(tv0 == tv2)
                {
                    float a_1 = (u + w)/2.f;

                    float n_u = a_1 / (a_1 + v);
                    float n_v = v / (a_1 + v);

                    i_t = n_u * tv0 + n_v * tv1;
                }

                ///tv0 != tv1
                else //if(tv1 == tv2)
                {
                    float a_1 = (v + w)/2.f;

                    float n_v = a_1 / (a_1 + u);
                    float n_u = u / (a_1 + u);

                    i_t = n_u * tv0 + n_v * tv1;
                }
            }

            if(any_t)
            {
                if(my_t < min_t)
                {
                    min_t = my_t;
                    interpolated_min_time = i_t;
                }

                if(my_t > max_t)
                {
                    max_t = my_t;
                    interpolated_max_time = i_t;
                }
            }
            else
            {
                min_t = my_t;
                max_t = my_t;

                interpolated_min_time = i_t;
                interpolated_max_time = i_t;

                any_t = true;
            }
        }
    }

    ///min_t and max_t correspond to pos.yzw + dir.yzw * t positions, ie toblerone entry and exit
    ///min_t and max_t must be inherently in range [0, 1]

    if(!any_t)
        return false;
    #endif // 0
    #else
    /*float3 planes[5 * 3] = {
        v0, v1, v2,
        e0, e2, e1,
        v0, e0, v1,
        v0, v2, e0,
        v1, e1, v2,
    };*/

    float3 base_normal = triangle_normal_D(to_v1, to_v2);
    float base_sign = dot(base_normal, t_diff);

    float3 origins[5] = {
        v0,
        e0,
        v0,
        v0,
        v1
    };

    float3 normals[5] = {
        base_normal * base_sign,
        -base_normal * base_sign,
        triangle_normal_D(t_diff, to_v1) * base_sign,
        triangle_normal_D(to_v2, t_diff) * base_sign,
        triangle_normal_D(t_diff, to_v2 - to_v1) * base_sign
    };

    float denoms[5] = {
        dot(dir.yzw, normals[0]),
        -dot(dir.yzw, normals[0]),
        dot(dir.yzw, normals[2]),
        dot(dir.yzw, normals[3]),
        dot(dir.yzw, normals[4]),
    };

    float start_t = -FLT_MAX;
    float end_t = FLT_MAX;

    for(int i=0; i < 5; i++)
    {
        float3 N = normals[i];
        float denom = denoms[i];

        float t = dot(origins[i] - pos.yzw, N) / denom;

        if(denom > 0)
        {
            start_t = max(start_t, t - 1e-7);
        }
        else
        {
            end_t = min(end_t, t + 1e-7);
        }

        //if(start_t > end_t)
        //    return false;
    }

    if(start_t > end_t)
        return false;

    float min_t = start_t;
    float max_t = end_t;
    #endif // 0

    //float3 toblerone_origin = (v0 + v1 + v2)/3.f;
    //float3 toblerone_end = (e0 + e1 + e2)/3.f;

    #ifndef TRI_COLLIDE
    float3 toblerone_origin = v0;
    float3 toblerone_end = e0;
    float3 toblerone_normal = toblerone_end - toblerone_origin;

    float toblerone_height = length(toblerone_normal);

    float3 entry_position = pos.yzw + dir.yzw * min_t;
    float3 exit_position = pos.yzw + dir.yzw * max_t;

    float3 norm_toblerone_normal = normalize(toblerone_normal);

    float3 tri_normal = triangle_normal(v0, v1, v2);

    float entry_height = 0;
    float exit_height = 0;

    {
        float3 new_origin = entry_position;
        float3 new_dir = -norm_toblerone_normal;

        float ray_plane_t = 0;

        if(!ray_plane_intersection(toblerone_origin, tri_normal, new_origin, new_dir, &ray_plane_t))
            return false;

        //float3 plane_intersection_location = new_origin + new_dir * ray_plane_t;
        //entry_height = length(new_origin - plane_intersection_location);

        entry_height = fabs(ray_plane_t);
    }

    {
        float3 new_origin = exit_position;
        float3 new_dir = -norm_toblerone_normal;

        float ray_plane_t = 0;

        if(!ray_plane_intersection(toblerone_origin, tri_normal, new_origin, new_dir, &ray_plane_t))
            return false;

        //float3 plane_intersection_location = new_origin + new_dir * ray_plane_t;
        //exit_height = length(new_origin - plane_intersection_location);

        exit_height = fabs(ray_plane_t);
    }

    float entry_time = (entry_height / toblerone_height) * (tri_end_time - tri_start_time) + tri_start_time;
    float exit_time = (exit_height / toblerone_height) * (tri_end_time - tri_start_time) + tri_start_time;
    #else
    float entry_time = interpolated_min_time;
    float exit_time = interpolated_max_time;
    #endif // TRI_COLLIDE
    float ray_entry_time = pos.x + dir.x * min_t;
    float ray_exit_time = pos.x + dir.x * max_t;

    /*sort2(&entry_time, &exit_time);
    sort2(&ray_entry_time, &ray_exit_time);

    entry_time -= 0.0001f;
    exit_time += 0.0001f;

    return ray_entry_time <= exit_time && entry_time <= ray_exit_time;*/

    /**
    ///so time line1 = entry + (exit - entry) * t0
    ///time line 2 = ray_entry + (ray_exit - ray_entry) * t1;
    ///I thiiink t1 = t2? and I want to solve for t

    ///so y = entry + (exit - entry) * t
    ///y = ray_entry + (ray_exit - ray_entry) * t
    ///entry + (exit - entry) * t = ray_entry + (ray_exit - ray_entry) * t
    ///entry - ray_entry + (exit - entry) * t - (ray_exit - ray_entry) * t = 0
    ///entry - ray_entry + t * ((exit - entry) - (ray_exit - ray_entry)) = 0
    ///t * ((exit - entry) - (ray_exit - ray_entry)) = -(entry - ray_entry)
    ///t = -(entry - ray_entry) / ((exit - entry) - (ray_exit - ray_entry))


    ///so. t = -(e - re) / ((x - e) - (rx - re)). Constrain to [0, 1]
    ///t = -(e - re) / div -> [eps, 1 - eps]
    ///t * div = -(e - re) -> [eps * div, (1 - eps) * div]
    */

    float eps = 0.00001f;

    float tdiv = -(entry_time - ray_entry_time);

    float div = ((exit_time - entry_time) - (ray_exit_time - ray_entry_time));

    float intersection_time = tdiv / div;

    bool success = intersection_time >= -eps && intersection_time <= 1 + eps;

    if(success && t_out)
    {
        ///so. t = tdiv / div. entry + (exit - entry) * t = time, and we have t, exit, and entry so neat
        ///now we want pos.x + dir.x * p = time
        ///((entry + (exit - entry) * t) - pos.x) / dir.x = p
        float fin_time = ray_entry_time + (ray_exit_time - ray_entry_time) * (tdiv / div);

        *t_out = (fin_time - pos.x) / dir.x;
    }

    #if 0
    eps *= div;

    float lower = -eps;
    float upper = div + eps;

    sort2(&lower, &upper);

    bool success = tdiv >= lower && tdiv <= upper;

    if(t_out)
    {
        ///so. t = tdiv / div. entry + (exit - entry) * t = time, and we have t, exit, and entry so neat
        ///now we want pos.x + dir.x * p = time
        ///((entry + (exit - entry) * t) - pos.x) / dir.x = p
        float fin_time = ray_entry_time + (ray_exit_time - ray_entry_time) * (tdiv / div);

        *t_out = (fin_time - pos.x) / dir.x;
    }
    #endif // 0

    return success;

    /*float intersection_time = -(entry_time - ray_entry_time) / ((exit_time - entry_time) - (ray_exit_time - ray_entry_time));

    float eps = 0.0001f;

    return intersection_time >= -eps && intersection_time < 1 + eps;*/
}

__kernel
void do_generic_rays (__global struct lightray* restrict generic_rays_in, __global struct lightray* restrict generic_rays_out,
                      __global struct lightray* restrict finished_rays,
                      __global int* restrict generic_count_in, __global int* restrict generic_count_out,
                      __global int* restrict finished_count_out,
                      //__global float4* path_out, __global int* counts_out,
                      __global struct triangle* tris, int tri_count, __global struct intersection* intersections_out, __global int* intersection_count,
                      __global struct potential_intersection* intersections_p, __global int* intersection_count_p,
                      __global int* counts, __global int* offsets, __global struct computed_triangle* linear_mem, __global float* linear_start_times, __global float* linear_delta_times,
                      __global int* unculled_counts,
                      __constant int* any_visible,
                      float accel_width, float accel_time_width, int accel_width_num,
                      __global int* cell_time_min, __global int* cell_time_max,
                      __global struct object* objs,
                      __global int* ray_time_min, __global int* ray_time_max,
                      dynamic_config_space struct dynamic_config* cfg)
{
    int id = get_global_id(0);

    if(id >= *generic_count_in)
        return;

    int ray_num = *generic_count_in;

    __global struct lightray* ray = &generic_rays_in[id];

    //counts_out[id] = 0;

    if(ray->early_terminate)
        return;

    float4 position = ray->position;
    float4 velocity = ray->velocity;
    float4 acceleration = ray->acceleration;

    {
        atomic_min(ray_time_min, (int)floor(position.x));
        atomic_max(ray_time_max, (int)ceil(position.x));
    }

    float f_in_x = fabs(velocity.x);

    int sx = ray->sx;
    int sy = ray->sy;

    #ifndef GENERIC_BIG_METRIC
    {
        float g_metric[4] = {0};
        calculate_metric_generic(position, g_metric, cfg);

        velocity = fix_light_velocity2(velocity, g_metric);
    }
    #endif // GENERIC_BIG_METRIC

    #ifdef IS_CONSTANT_THETA
    position.z = M_PIf/2;
    velocity.z = 0;
    acceleration.z = 0;
    #endif // IS_CONSTANT_THETA

    float next_ds = 0.00001;

    #ifdef ADAPTIVE_PRECISION
    (void)acceleration_to_precision(acceleration, MAX_ACCELERATION_CHANGE, &next_ds);
    #endif // ADAPTIVE_PRECISION

    ///results:
    ///subambient_precision can't go above 0.5 much while in verlet mode without the size of the event horizon changing
    ///in euler mode this is actually already too low

    ///ambient precision however looks way too low at 0.01, testing up to 0.3 showed no noticable difference, needs more precise tests though
    ///only in the case without kruskals and event horizon crossings however, any precision > 0.01 is insufficient in that case
    ///this super affects being able to render alcubierre at thin shells
    float subambient_precision = 0.5;
    float ambient_precision = 0.2;

    float rs = 1;

    float uniform_coordinate_precision_divisor = max(max(W_V1, W_V2), max(W_V3, W_V4));

    int loop_limit = 4096  * 2;

    #ifdef DEVICE_SIDE_ENQUEUE
    loop_limit /= 125;
    #endif // DEVICE_SIDE_ENQUEUE

    float4 last_pos = (float4)(0,0,0,0);

    float my_min = position.x;
    float my_max = position.x;

    float4 ray_quat = ray->initial_quat;

    int any_visible_tris = *any_visible;

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

        float new_max = MAX_PRECISION_RADIUS * rs;
        float new_min = 3 * rs;

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

        bool should_terminate = fabs(polar_position.y) >= UNIVERSE_SIZE;

        #ifdef SINGULAR
        should_terminate |= fabs(polar_position.y) < rs*SINGULAR_TERMINATOR;
        #endif // SINGULAR

        #ifdef HAS_CYLINDRICAL_SINGULARITY
        if(position.y < CYLINDRICAL_TERMINATOR)
            return;
        #endif // CYLINDRICAL_SINGULARITY

        #ifndef UNCONDITIONALLY_NONSINGULAR
        if(fabs(velocity.x) > 1000 + f_in_x && fabs(acceleration.x) > 100)
        {
            atomic_min(ray_time_min, (int)floor(my_min));
            atomic_max(ray_time_max, (int)ceil(my_max));
            return;
        }
        #endif // UNCONDITIONALLY_NONSINGULAR

        if((i % 8) == 0 && any_visible_tris > 0)
        {
            float4 out_position = generic_to_cartesian(position, cfg);

            #if (defined(GENERIC_METRIC) && defined(GENERIC_CONSTANT_THETA)) || !defined(GENERIC_METRIC) || defined(DEBUG_CONSTANT_THETA)
            {
                float4 pos_spherical = generic_to_spherical(position, cfg);

                float fsign = sign(pos_spherical.y);
                pos_spherical.y = fabs(pos_spherical.y);

                float3 pos_cart = polar_to_cartesian(pos_spherical.yzw);

                pos_cart = rot_quat_norm(pos_cart, ray_quat);

                out_position = (float4)(position.x, pos_cart);
            }
            #endif

            if(i == 0)
                last_pos = out_position;

            bool should_check = true;

            ///we could evaluate some of this with the metric tensor, ie angles
            float4 rt_pos = last_pos;
            float4 next_rt_pos = out_position;

            float4 rt_diff = next_rt_pos - rt_pos;

            //if(dot(rt_diff, rt_diff) > 1)
            //    should_check = true;

            if((should_check || should_terminate) && !any(IS_DEGENERATE(out_position)))
            {
                last_pos = next_rt_pos;

                #if 1

                {
                    struct step_setup setup;

                    {
                        float4 current_pos = world_to_voxel4(rt_pos, accel_width, accel_time_width, accel_width_num);
                        float4 next_pos = world_to_voxel4(next_rt_pos, accel_width, accel_time_width, accel_width_num);

                        setup = setup_step(current_pos, next_pos);
                    }

                    float best_intersection = FLT_MAX;

                    ///saves about 2ms, but I'm concerned about validity
                    #ifdef FIRST_BEST_INTERSECTION
                    int hit_id = -1;
                    #endif

                    struct intersection out;
                    out.sx = sx;
                    out.sy = sy;

                    while(!is_step_finished(&setup))
                    {
                        unsigned int voxel_id = index_acceleration(&setup, accel_width_num);

                        #ifdef USE_FEEDBACK_CULLING
                        float rmin = min(rt_pos.x, next_rt_pos.x);
                        float rmax = max(rt_pos.x, next_rt_pos.x);

                        int unculled = unculled_counts[voxel_id];

                        if(unculled > 0)
                        {
                            atomic_min(&cell_time_min[voxel_id], (int)floor(rmin));
                            atomic_max(&cell_time_max[voxel_id], (int)ceil(rmax));
                        }
                        #endif // USE_FEEDBACK_CULLING

                        do_step(&setup);

                        int tri_count = counts[voxel_id];
                        int base_offset = offsets[voxel_id];

                        #if 1
                        for(int t_off=0; t_off < tri_count; t_off++)
                        {
                            #ifdef FIRST_BEST_INTERSECTION
                            if(hit_id == t_off + base_offset)
                                continue;
                            #endif

                            ///so, I now know for a fact that start time < end_time
                            ///could sort this instead by end times, and then use end time as the quick check
                            float start_time = linear_start_times[base_offset + t_off];

                            #ifdef TIME_CULL
                            if(rt_pos.x < start_time - 0.0001f && (next_rt_pos - rt_pos).x < 0)
                                continue;
                            #endif // TIME_CULL

                            float end_time = start_time + linear_delta_times[base_offset + t_off];

                            #ifdef TIME_CULL
                            if(!ray_toblerone_could_intersect(rt_pos, next_rt_pos - rt_pos, start_time, end_time))
                                continue;
                            #endif // TIME_CULL

                            ///use dot current_pos tri centre pos (?) as distance metric
                            /*float3 pdiff = parent_pos.yzw - rt_pos.yzw;

                            if(dot(pdiff, pdiff) > 5)
                                continue;*/

                            __global struct computed_triangle* ctri = &linear_mem[base_offset + t_off];

                            float ray_t = 0;

                            if(ray_intersects_toblerone(rt_pos, next_rt_pos - rt_pos, ctri, start_time, end_time, &ray_t))
                            {
                                if(ray_t < best_intersection)
                                {
                                    out.computed_parent = base_offset + t_off;
                                    best_intersection = ray_t;

                                    #ifdef FIRST_BEST_INTERSECTION
                                    hit_id = base_offset + t_off;
                                    #endif
                                }
                            }

                            #if 0
                            #define OBSERVER_DEPENDENCE
                            #ifdef OBSERVER_DEPENDENCE
                            float tri_time = ctri->time;
                            float next_tri_time = ctri->end_time;

                            float interpolate_frac = (float)kk / max_len2;

                            float ray_time = mix(rt_pos.x, next_rt_pos.x, interpolate_frac);

                            //if(tri_time < ray_time + 0.5f && tri_time >= ray_time - 0.5f)

                            if(ray_time >= tri_time && ray_time < next_tri_time)
                            #endif // OBSERVER_DEPENDENCE
                            {
                                float time_elapsed = (ray_time - tri_time);

                                float3 v0_pos = {ctri->v0x, ctri->v0y, ctri->v0z};
                                float3 v1_pos = {ctri->v1x, ctri->v1y, ctri->v1z};
                                float3 v2_pos = {ctri->v2x, ctri->v2y, ctri->v2z};

                                //printf("Elapsed %f velocity %f %f %f\n", time_elapsed, ctri->velocity.y, ctri->velocity.z, ctri->velocity.w);

                                /*v0_pos += ctri->velocity.yzw * time_elapsed;
                                v1_pos += ctri->velocity.yzw * time_elapsed;
                                v2_pos += ctri->velocity.yzw * time_elapsed;*/

                                //v0_pos += parent_pos.yzw;
                                //v1_pos += parent_pos.yzw;
                                //v2_pos += parent_pos.yzw;

                                float3 ray_pos = rt_pos.yzw;
                                float3 ray_dir = next_rt_pos.yzw - rt_pos.yzw;

                                ///ehhhh need to take closest
                                if(ray_intersects_triangle(ray_pos, ray_dir, v0_pos, v1_pos, v2_pos, 0, 0, 0))
                                {
                                    struct intersection out;
                                    out.sx = sx;
                                    out.sy = sy;

                                    int isect = atomic_inc(intersection_count);

                                    intersections_out[isect] = out;
                                    return;
                                }
                            }
                            #endif // 0
                        }


                        #endif // 0

                        /*struct intersection out;
                        out.sx = sx;
                        out.sy = sy;

                        int isect = atomic_inc(intersection_count);

                        intersections_out[isect] = out;
                        return;*/
                    }

                    if(best_intersection != FLT_MAX)
                    {
                        atomic_min(ray_time_min, (int)floor(my_min));
                        atomic_max(ray_time_max, (int)ceil(my_max));

                        int isect = atomic_inc(intersection_count);

                        intersections_out[isect] = out;

                        return;
                    }
                }
                #endif // 0

                #ifdef VOX_SAMP
                float3 floord = floor(world_to_voxel(rt_pos.yzw, accel_width, accel_width_num));

                int3 ipos = (int3)(floord.x, floord.y, floord.z);

                ipos = loop_voxel(ipos, accel_width_num);

                int bidx = ipos.z * accel_width_num * accel_width_num + ipos.y * accel_width_num + ipos.x;

                int cnt = counts[bidx];

                if(cnt > 0)
                {
                    int isect = atomic_inc(intersection_count);

                    struct intersection out;
                    out.sx = sx;
                    out.sy = sy;

                    intersections_out[isect] = out;
                    return;
                }
                #endif // VOX_SAMP
            }
        }

        if(should_terminate)
        {
            atomic_min(ray_time_min, (int)floor(my_min));
            atomic_max(ray_time_max, (int)ceil(my_max));

            int out_id = atomic_inc(finished_count_out);

            float4 polar_velocity = generic_velocity_to_spherical_velocity(position, velocity, cfg);

            struct lightray out_ray;
            out_ray.sx = sx;
            out_ray.sy = sy;
            out_ray.position = polar_position;
            out_ray.velocity = polar_velocity;
            out_ray.initial_quat = ray->initial_quat;
            out_ray.acceleration = 0;
            out_ray.ku_uobsu = ray->ku_uobsu;
            out_ray.early_terminate = 0;

            finished_rays[out_id] = out_ray;
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

        float4 next_position, next_velocity, next_acceleration;

        step_verlet(position, velocity, acceleration, true, ds, &next_position, &next_velocity, &next_acceleration, 0, cfg);

        #ifdef ADAPTIVE_PRECISION

        if(fabs(r_value) < new_max)
        {
            int res = calculate_ds_error(ds, next_acceleration, acceleration, MAX_ACCELERATION_CHANGE, &next_ds);

            if(res == DS_RETURN)
                return;

            if(res == DS_SKIP)
                continue;
        }

        #endif // ADAPTIVE_PRECISION

        position = next_position;
        //velocity = fix_light_velocity2(next_velocity, g_metric);
        velocity = next_velocity;
        acceleration = next_acceleration;
        #endif // VERLET_INTEGRATION

        #ifdef RK4_GENERIC
        rk4_generic_big(&position, &velocity, &ds);
        #endif // RK4_GENERIC

        /*if(sx == 1920/2 && sy == 1080/2)
        {
            printf("Pos %f %f %f %f vel %f %f %f %f\n", position.x, position.y, position.z, position.w, velocity.x, velocity.y, velocity.z, velocity.w);
        }*/

        if(any(IS_DEGENERATE(position)) || any(IS_DEGENERATE(velocity)) || any(IS_DEGENERATE(acceleration)))
        {
            return;
        }
    }

    atomic_min(ray_time_min, (int)floor(my_min));
    atomic_max(ray_time_max, (int)ceil(my_max));

    int out_id = atomic_inc(generic_count_out);

    struct lightray out_ray;
    out_ray.sx = sx;
    out_ray.sy = sy;
    out_ray.position = position;
    out_ray.velocity = velocity;
    out_ray.acceleration = acceleration;
    out_ray.initial_quat = ray->initial_quat;
    out_ray.ku_uobsu = ray->ku_uobsu;
    out_ray.early_terminate = 0;

    generic_rays_out[out_id] = out_ray;
}

__kernel
void get_geodesic_path(__global struct lightray* generic_rays_in,
                       __global float4* positions_out,
                       __global float4* velocities_out,
                       __global float* dT_ds_out,
                       __global float* ds_out,
                       __global int* generic_count_in,
                       int max_path_length,
                       dynamic_config_space struct dynamic_config* cfg, __global int* count_out)
{
    int id = get_global_id(0);

    if(id >= *generic_count_in)
        return;

    __global struct lightray* ray = &generic_rays_in[id];

    float4 position = ray->position;
    float4 velocity = ray->velocity;
    float4 acceleration = ray->acceleration;

    //printf("Pos %f %f %f %f\n", position.x,position.y,position.z,position.w);

    /*#ifndef GENERIC_BIG_METRIC
    {
        float g_metric[4] = {0};
        calculate_metric_generic(position, g_metric, cfg);

        velocity = fix_light_velocity2(velocity, g_metric);
    }
    #endif // GENERIC_BIG_METRIC*/

    #ifdef IS_CONSTANT_THETA
    position.z = M_PIf/2;
    velocity.z = 0;
    acceleration.z = 0;
    #endif // IS_CONSTANT_THETA

    #ifdef ADAPTIVE_PRECISION
    float max_accel = min(0.00001000f, MAX_ACCELERATION_CHANGE);
    #endif // ADAPTIVE_PRECISION

    float next_ds = 0.00001;

    #ifdef ADAPTIVE_PRECISION
    (void)acceleration_to_precision(acceleration, max_accel, &next_ds);
    #endif // ADAPTIVE_PRECISION

    float subambient_precision = 0.5;
    float ambient_precision = 0.2;

    float rs = 1;

    int stride_out = *generic_count_in;
    int bufc = 0;

    //#pragma unroll
    for(int i=0; i < max_path_length; i++)
    {
        #ifdef IS_CONSTANT_THETA
        position.z = M_PIf/2;
        velocity.z = 0;
        acceleration.z = 0;
        #endif // IS_CONSTANT_THETA

        float new_max = MAX_PRECISION_RADIUS * rs;
        float new_min = 3 * rs;

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
        if(fabs(polar_position.y) >= UNIVERSE_SIZE)
        #else
        if(fabs(polar_position.y) < rs*SINGULAR_TERMINATOR || fabs(polar_position.y) >= UNIVERSE_SIZE)
        #endif // SINGULAR
        {
            //printf("Escaped\n");

            should_break = true;
        }

        float4 next_position, next_velocity, next_acceleration;
        float dT_ds = 0;

        step_verlet(position, velocity, acceleration, false, ds, &next_position, &next_velocity, &next_acceleration, &dT_ds, cfg);

        #ifdef ADAPTIVE_PRECISION
        if(fabs(r_value) < new_max)
        {
            int res = calculate_ds_error(ds, next_acceleration, acceleration, max_accel, &next_ds);

            if(res == DS_RETURN)
            {
                should_break = true;
            }

            if(res == DS_SKIP)
                continue;
        }
        #endif // ADAPTIVE_PRECISION

        float4 generic_position_out = position;
        float4 generic_velocity_out = velocity;

        #if (defined(GENERIC_METRIC) && defined(GENERIC_CONSTANT_THETA)) || !defined(GENERIC_METRIC) || defined(DEBUG_CONSTANT_THETA)
        {
            float4 pos_spherical = generic_to_spherical(position, cfg);
            float4 vel_spherical = generic_velocity_to_spherical_velocity(position, velocity, cfg);

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

            generic_position_out = next_pos_generic;
            generic_velocity_out = next_vel_generic;
        }
        #endif

        ///in the event that velocity and acceleration is 0, it'd be ideal to have a fast path
        if(any(IS_DEGENERATE(next_position)) || any(IS_DEGENERATE(next_velocity)) || any(IS_DEGENERATE(next_acceleration)))
        {
            //printf("Degenerate");
            break;
        }

        //printf("Pos %f %f %f %f\n", next_position.x,next_position.y,next_position.z,next_position.w);

        position = next_position;
        velocity = next_velocity;
        acceleration = next_acceleration;

        positions_out[bufc * stride_out + id] = generic_position_out;

        if(velocities_out)
            velocities_out[bufc * stride_out + id] = generic_velocity_out;

        if(dT_ds_out)
            dT_ds_out[bufc * stride_out + id] = dT_ds;

        if(ds_out)
            ds_out[bufc * stride_out + id] = ds;

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
                        dynamic_config_space struct dynamic_config* cfg)
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
void calculate_singularities(__global struct lightray* finished_rays, __global int* finished_count, __global int* termination_buffer, int width, int height)
{
    int id = get_global_id(0);

    if(id >= *finished_count)
        return;

    int sx = finished_rays[id].sx;
    int sy = finished_rays[id].sy;

    termination_buffer[sy * width + sx] = 0;
}

#endif // GENERIC_METRIC

__kernel
void calculate_texture_coordinates(__global struct lightray* finished_rays, __global int* finished_count_in, __global float2* texture_coordinates, int width, int height)
{
    int id = get_global_id(0);

    if(id >= *finished_count_in)
        return;

    __global struct lightray* ray = &finished_rays[id];

    int pos = ray->sy * width + ray->sx;
    int sx = ray->sx;
    int sy = ray->sy;

    float4 position = ray->position;
    float4 velocity = ray->velocity;

    #ifdef IS_CONSTANT_THETA
    position.z = M_PIf/2;
    velocity.z = 0;
    #endif // IS_CONSTANT_THETA

    #if defined(UNIVERSE_SIZE)
    {
        if(fabs(position.y) >= UNIVERSE_SIZE)
        {
            position.yzw = fix_ray_position(position.yzw, velocity.yzw, UNIVERSE_SIZE, true);
        }

        ///I'm not 100% sure this is working as well as it could be
        #if defined(SINGULAR) && defined(TRAVERSABLE_EVENT_HORIZON)
        if(fabs(position.y) < SINGULAR_TERMINATOR)
        {
            position.yzw = fix_ray_position(position.yzw, velocity.yzw, SINGULAR_TERMINATOR, true);
        }
        #endif
    }
    #endif

    float rs = 1;
    float r_value = position.y;

    #if !defined(TRAVERSABLE_EVENT_HORIZON) || (defined(NO_EVENT_HORIZON_CROSSING) && !defined(GENERIC_METRIC))
    if(fabs(r_value) <= rs)
    {
        return;
    }
    #endif

	float3 npolar = position.yzw;

    #if (defined(GENERIC_METRIC) && defined(GENERIC_CONSTANT_THETA)) || !defined(GENERIC_METRIC) || defined(DEBUG_CONSTANT_THETA)
	{
        float4 quat = ray->initial_quat;

	    float3 cart_pos = polar_to_cartesian(position.yzw);

	    cart_pos = rot_quat(cart_pos, quat);

	    npolar = cartesian_to_polar(cart_pos);
	}
    #endif // GENERIC_CONSTANT_THETA

    float thetaf = fmod(npolar.y, 2 * M_PIf);
    float phif = npolar.z;

    if(thetaf >= M_PIf)
    {
        phif += M_PIf;
        thetaf -= M_PIf;
    }

    phif = fmod(phif, 2 * M_PIf);

    float sxf = (phif) / (2 * M_PIf);
    float syf = thetaf / M_PIf;

    sxf += 0.5f;

    /*if(sx == width/2 && sy == height/2)
    {
        printf("COORD %f %f\n", sxf, syf);
    }*/

    texture_coordinates[pos] = (float2)(sxf, syf);
}

float smallest(float f1, float f2)
{
    if(fabs(f1) < fabs(f2))
        return f1;

    return f2;
}

float circular_diff(float f1, float f2)
{
    float a1 = f1 * M_PIf * 2;
    float a2 = f2 * M_PIf * 2;

    float2 v1 = {cos(a1), sin(a1)};
    float2 v2 = {cos(a2), sin(a2)};

    return atan2(v1.x * v2.y - v1.y * v2.x, v1.x * v2.x + v1.y * v2.y) / (2 * M_PIf);
}

float2 circular_diff2(float2 f1, float2 f2)
{
    return (float2)(circular_diff(f1.x, f2.x), circular_diff(f1.y, f2.y));
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

float3 redshift(float3 v, float z)
{
    ///1 + z = gtt(recv) / gtt(src)
    ///1 + z = lnow / lthen
    ///1 + z = wsrc / wobs

    float radiant_energy = v.x*0.2125f + v.y*0.7154f + v.z*0.0721f;

    float3 red = (float3){1/0.2125f, 0.f, 0.f};
    float3 blue = (float3){0.f, 0.f, 1/0.0721};

    float3 result;

    if(z > 0)
    {
        result = mix(v, radiant_energy * red, tanh(z));
    }
    else
    {
        float iv1pz = (1/(1 + z)) - 1;

        //result = mix(v, radiant_energy * blue, fabs(z));
        result = mix(v, radiant_energy * blue, tanh(iv1pz));
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
float4 read_mipmap(image2d_array_t mipmap1, image2d_array_t mipmap2, float position_y, float2 pos, float lod)
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

    return position_y >= 0 ? v1 : v2;
}

#define MIPMAP_CONDITIONAL(x) ((position.y >= 0) ? x(mip_background) : x(mip_background2))

__kernel
void render(__global struct lightray* finished_rays, __global int* finished_count_in, __write_only image2d_t out,
            __read_only image2d_array_t mip_background, __read_only image2d_array_t mip_background2,
            int width, int height, __global float2* texture_coordinates, int maxProbes)
{
    int id = get_global_id(0);

    if(id >= *finished_count_in)
        return;

    __global struct lightray* ray = &finished_rays[id];

    int sx = ray->sx;
    int sy = ray->sy;

    if(sx >= width || sy >= height)
        return;

    if(sx < 0 || sy < 0)
        return;

    float4 position = ray->position;
    float4 velocity = ray->velocity;

    #ifdef IS_CONSTANT_THETA
    position.z = M_PIf/2;
    velocity.z = 0;
    #endif // IS_CONSTANT_THETA

    #if defined(UNIVERSE_SIZE)
    {
        if(fabs(position.y) >= UNIVERSE_SIZE)
        {
            position.yzw = fix_ray_position(position.yzw, velocity.yzw, UNIVERSE_SIZE, true);
        }

        #if defined(SINGULAR) && defined(TRAVERSABLE_EVENT_HORIZON)
        if(fabs(position.y) < SINGULAR_TERMINATOR)
        {
            position.yzw = fix_ray_position(position.yzw, velocity.yzw, SINGULAR_TERMINATOR, true);
        }
        #endif
    }
    #endif

    float rs = 1;
    float r_value = position.y;

    #if !defined(TRAVERSABLE_EVENT_HORIZON) || (defined(NO_EVENT_HORIZON_CROSSING) && !defined(GENERIC_METRIC))
    if(fabs(r_value) <= rs * 2)
    {
        write_imagef(out, (int2){sx, sy}, (float4)(0, 0, 0, 1));
        return;
    }
    #endif

    float sxf = texture_coordinates[sy * width + sx].x;
    float syf = texture_coordinates[sy * width + sx].y;

    ///we actually do have an event horizon
    if(fabs(r_value) <= rs)
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

    #define MIPMAPPING
    #ifdef MIPMAPPING
    int dx = 1;
    int dy = 1;

    if(sx == width-1)
        dx = -1;

    if(sy == height-1)
        dy = -1;

    float2 tl = texture_coordinates[sy * width + sx];
    float2 tr = texture_coordinates[sy * width + sx + dx];
    float2 bl = texture_coordinates[(sy + dy) * width + sx];

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

        end_result = read_mipmap(mip_background, mip_background2, position.y, (float2){sxf, syf}, levelofdetail);
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

            float4 fval = read_mipmap(mip_background, mip_background2, position.y, (float2){cu, cv}, levelofdetail);

            totalWeight += relativeWeight * fval;
            accumulatedProbes += relativeWeight;

            currentN += 2;
        }

        end_result = totalWeight / accumulatedProbes;
    }

    #endif // TRILINEAR

    //float4 end_result = read_imagef(*background, sam, (float2){sxf, syf}, dx_vtc, dy_vtc);

    #else
    float4 end_result = read_mipmap(mip_background, mip_background2, position.y, sam, (float2){sxf, syf}, 0);
    #endif // MIPMAPPING

    #ifdef REDSHIFT
    ///[-1, +infinity]
    float z_shift = (velocity.x / -ray->ku_uobsu) - 1;

    z_shift = max(z_shift, -0.999f);

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

    if(fabs(r_value) > rs * 2)
    {
        /*if(sx == width/2 && sy == height/2)
        {
            printf("ZShift %f\n", z_shift);
        }*/

        lin_result = redshift(lin_result, z_shift);

        lin_result = clamp(lin_result, 0.f, 1.f);
    }

    #ifndef LINEAR_FRAMEBUFFER
    end_result.xyz = lin_to_srgb(lin_result);
    #else
    end_result.xyz = lin_result;
    #endif // LINEAR_FRAMEBUFFER

    #endif // REDSHIFT

    #ifdef LINEAR_FRAMEBUFFER
    #ifndef REDSHIFT //redshift already handles this for roundtrip accuracy reasons
    end_result.xyz = srgb_to_lin(end_result.xyz);
    #endif // REDSHIFT
    #endif // LINEAR_FRAMEBUFFER

    write_imagef(out, (int2){sx, sy}, end_result);
}

__kernel
void render_tris(__global struct triangle* tris, int tri_count,
                 __global struct lightray* finished_rays, __global int* finished_count_in,
                 __global float4* traced_positions, __global int* traced_positions_count,
                 int width, int height,
                 __write_only image2d_t screen,
                 dynamic_config_space struct dynamic_config* cfg)
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

__kernel
void render_intersections(__global struct intersection* in, __global int* cnt, __global struct computed_triangle* linear_mem, __write_only image2d_t screen)
{
    int id = get_global_id(0);

    if(id >= *cnt)
        return;

    __global struct intersection* mine = &in[id];
    __global struct computed_triangle* ctri = &linear_mem[mine->computed_parent];

    #ifndef TRI_PRECESSION
    float3 to_v1 = (float3)(ctri->dv1x, ctri->dv1y, ctri->dv1z);
    float3 to_v2 = (float3)(ctri->dv2x, ctri->dv2y, ctri->dv2z);

    float3 v0 = (float3)(ctri->v0x, ctri->v0y, ctri->v0z);
    float3 v1 = v0 + to_v1;
    float3 v2 = v0 + to_v2;

    float3 t_diff = (float3)(ctri->dvx, ctri->dvy, ctri->dvz);

    float3 e0 = v0 + t_diff;
    float3 e1 = v1 + t_diff;
    float3 e2 = v2 + t_diff;
    #else
    float3 v0 = (float3)(ctri->v0x, ctri->v0y, ctri->v0z);
    float3 v1 = (float3)(ctri->v1x, ctri->v1y, ctri->v1z);
    float3 v2 = (float3)(ctri->v2x, ctri->v2y, ctri->v2z);
    #endif

    //float3 N = triangle_normal(iv0, iv1, iv2);

    float3 N = triangle_normal(v0, v1, v2);

    int2 pos = (int2)(mine->sx, mine->sy);

    float3 N_as_col = fabs(N);

    write_imagef(screen, pos, (float4)(N_as_col.xyz, 1));
}

///todo: make flip global
__kernel
void cart_to_polar_kernel(__global float4* position_cart_in, __global float4* position_polar_out, int count, float flip)
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

__kernel
void camera_polar_to_generic(__global float4* g_camera_pos_polar, __global float4* g_camera_pos_generic, dynamic_config_space struct dynamic_config* cfg)
{
    if(get_global_id(0) != 0)
        return;

    *g_camera_pos_generic = spherical_to_generic(*g_camera_pos_polar, cfg);
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
