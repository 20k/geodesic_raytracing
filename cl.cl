#define M_PIf ((float)M_PI)

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
    float x = in.x * sin(in.y) * cos(in.z);
    float y = in.x * sin(in.y) * sin(in.z);
    float z = in.x * cos(in.y);

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

float stable_quad(float a, float d, float k, float sign)
{
    if(k <= 4.38072748497961 * pow(10.f, 16.f))
        return -(k + copysign(native_sqrt((4 * a) * d + k * k), sign)) / (a * 2);

    return -k / a;
}

float4 fix_light_velocity_big(float4 v, float g_metric_big[])
{
    //return v;

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
    float dot = dot_product_big(in, in, big_metric);

    return in / native_sqrt(fabs(dot));
}

///todo: generic orthonormalisation
struct frame_basis calculate_frame_basis(float big_metric[])
{
    ///while really i think it should be columns, the metric tensor is always symmetric
    ///this seems better for memory ordering
    float4 i1 = (float4)(big_metric[0], big_metric[1], big_metric[2], big_metric[3]);
    float4 i2 = (float4)(big_metric[1], big_metric[5], big_metric[6], big_metric[7]);
    float4 i3 = (float4)(big_metric[2], big_metric[6], big_metric[10], big_metric[11]);
    float4 i4 = (float4)(big_metric[3], big_metric[7], big_metric[11], big_metric[15]);

    float g_big_metric_inverse[16] = {};
    metric_inverse(big_metric, g_big_metric_inverse);

    i1 = raise_index_big(i1, g_big_metric_inverse);
    i2 = raise_index_big(i2, g_big_metric_inverse);
    i3 = raise_index_big(i3, g_big_metric_inverse);
    i4 = raise_index_big(i4, g_big_metric_inverse);

    float4 u1 = i1;

    float4 u2 = i2;
    u2 = u2 - gram_proj(u1, u2, big_metric);

    float4 u3 = i3;
    u3 = u3 - gram_proj(u1, u3, big_metric);
    u3 = u3 - gram_proj(u2, u3, big_metric);

    float4 u4 = i4;
    u4 = u4 - gram_proj(u1, u4, big_metric);
    u4 = u4 - gram_proj(u2, u4, big_metric);
    u4 = u4 - gram_proj(u3, u4, big_metric);

    u1 = normalize(u1);
    u2 = normalize(u2);
    u3 = normalize(u3);
    u4 = normalize(u4);

    u1 = normalize_big_metric(u1, big_metric);
    u2 = normalize_big_metric(u2, big_metric);
    u3 = normalize_big_metric(u3, big_metric);
    u4 = normalize_big_metric(u4, big_metric);

    struct frame_basis ret;
    ret.v1 = u1;
    ret.v2 = u2;
    ret.v3 = u3;
    ret.v4 = u4;

    return ret;
}

void print_metric_big_trace(float g_metric_big[])
{
    printf("%f %f %f %f\n", g_metric_big[0], g_metric_big[5], g_metric_big[10], g_metric_big[15]);
}

void print_metric_big(float g_metric_big[])
{
    printf("%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n", g_metric_big[0], g_metric_big[1], g_metric_big[2], g_metric_big[3],
           g_metric_big[4], g_metric_big[5], g_metric_big[6], g_metric_big[7],
           g_metric_big[8], g_metric_big[9], g_metric_big[10], g_metric_big[11],
           g_metric_big[12], g_metric_big[13], g_metric_big[14], g_metric_big[15]);
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

void adjust_pixel_direction_and_camera_theta(float3 pixel_direction, float4 polar_camera_in, float3* pixel_direction_out, float4* polar_camera_out, bool debug)
{
    float4 theta_quat = get_theta_adjustment_quat(pixel_direction, polar_camera_in, 1, debug);

    float3 apolar = polar_camera_in.yzw;
    apolar.x = fabs(apolar.x);

    float3 cartesian_camera =  polar_to_cartesian(apolar);

    pixel_direction = rot_quat(pixel_direction, theta_quat);
    cartesian_camera = rot_quat(cartesian_camera, theta_quat);

    float3 polar_camera = cartesian_to_polar(cartesian_camera);

    if(polar_camera_in.y < 0)
    {
        polar_camera.x = -polar_camera.x;
    }

    *pixel_direction_out = pixel_direction;
    *polar_camera_out = (float4)(polar_camera_in.x, polar_camera);
}

float3 get_texture_constant_theta_rotation(float3 pixel_direction, float4 polar_camera_in, float4 final_position)
{
    float4 theta_quat = get_theta_adjustment_quat(pixel_direction, polar_camera_in, -1, false);

    float3 afinal_position = final_position.yzw;
    afinal_position.x = fabs(afinal_position.x);

    float3 cartesian_position = polar_to_cartesian(afinal_position);

    cartesian_position = rot_quat(cartesian_position, theta_quat);

    float3 polar_position = cartesian_to_polar(cartesian_position);

    if(final_position.y < 0)
    {
        polar_position.x = -polar_position.x;
    }

	return polar_position;
}

float3 calculate_pixel_direction(int cx, int cy, float width, float height, float4 polar_camera, float4 camera_quat, float2 base_angle)
{
    #define FOV 90

    float fov_rad = (FOV / 360.f) * 2 * M_PIf;

    float nonphysical_plane_half_width = width/2;
    float nonphysical_f_stop = nonphysical_plane_half_width / tan(fov_rad/2);

    float3 pixel_direction = (float3){cx - width/2, cy - height/2, nonphysical_f_stop};

    pixel_direction = normalize(pixel_direction);
    pixel_direction = rot_quat(pixel_direction, camera_quat);

    //pixel_direction = rotate_vector(b0, b1, b2, pixel_direction);

    float3 up = {0, 0, 1};

    if(base_angle.y != 0)
    {
        float4 goff2 = aa_to_quat(up, base_angle.y);

        if(polar_camera.y < 0)
        {
            goff2 = aa_to_quat(up, -base_angle.y);
        }

        pixel_direction = rot_quat(pixel_direction, goff2);
    }

    if(base_angle.x != M_PIf/2)
    {
        ///gets the rotation associated with the theta intersection of r=0
        float base_theta_angle = cos_mix(M_PIf/2, base_angle.x, clamp(1 - fabs(polar_camera.y), 0.f, 1.f));

        float4 theta_goff = aa_to_quat(get_theta_axis(pixel_direction, polar_camera), -(-base_theta_angle + M_PIf/2));

        if(polar_camera.y < 0)
        {
            float3 theta_axis = get_theta_axis(pixel_direction, polar_camera);

            float4 new_thetaquat = aa_to_quat(theta_axis, -(-polar_camera.z + M_PIf/2));

            pixel_direction = rot_quat(pixel_direction, new_thetaquat);

            float3 phi_axis = get_phi_axis(pixel_direction, polar_camera);

            ///the phi axis... is a basis up axis?
            float4 new_quat = aa_to_quat(phi_axis, -(polar_camera.w + M_PIf/2));

            pixel_direction = rot_quat(pixel_direction, new_quat);

            float4 new_quat2 = aa_to_quat(phi_axis, -(polar_camera.w - M_PIf/2));

            pixel_direction = rot_quat(pixel_direction, new_quat2);

            float4 new_thetaquat2 = aa_to_quat(theta_axis, -(-polar_camera.z + M_PIf/2));

            pixel_direction = rot_quat(pixel_direction, new_thetaquat2);

            theta_goff = aa_to_quat(get_theta_axis(pixel_direction, polar_camera), (-base_theta_angle + M_PIf/2));
        }

        pixel_direction = rot_quat(pixel_direction, theta_goff);
    }

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

void calculate_tetrads(float4 polar_camera,
                       float4* e0_out, float4* e1_out, float4* e2_out, float4* e3_out,
                       dynamic_config_space struct dynamic_config* cfg, int should_orient)
{
    float4 at_metric = spherical_to_generic(polar_camera, cfg);

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
            float4 cart_camera = (float4)(polar_camera.x, polar_to_cartesian(polar_camera.yzw));

            float4 e_lo[4];
            get_tetrad_inverse(e0, e1, e2, e3, &e_lo[0], &e_lo[1], &e_lo[2], &e_lo[3]);

            float4 cx = (float4)(0, 1, 0, 0);
            float4 cy = (float4)(0, 0, 1, 0);
            float4 cz = (float4)(0, 0, 0, 1);

            cx = cartesian_velocity_to_generic_velocity(cart_camera, cx, cfg);
            cy = cartesian_velocity_to_generic_velocity(cart_camera, cy, cfg);
            cz = cartesian_velocity_to_generic_velocity(cart_camera, cz, cfg);

            /*float3 right = rot_quat((float3){1, 0, 0}, local_camera_quat);
            float3 forw = rot_quat((float3){0, 0, 1}, local_camera_quat);*/

            ///normalise with y first, so that the camera controls always work intuitively - as they are inherently a 'global' concept
            float4 approximate_basis[3] = {cy, cx, cz};

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

    *e0_out = e0;
    *e1_out = e1;
    *e2_out = e2;
    *e3_out = e3;
}

__kernel
void init_basis_vectors(__global float4* g_polar_camera_in, __global float4* g_camera_quat,
                        __global float4* e0_out, __global float4* e1_out, __global float4* e2_out, __global float4* e3_out,
                        dynamic_config_space struct dynamic_config* cfg)
{
    if(get_global_id(0) != 0)
        return;

    float4 polar_camera_in = *g_polar_camera_in;

    float4 e0;
    float4 e1;
    float4 e2;
    float4 e3;

    calculate_tetrads(polar_camera_in, &e0, &e1, &e2, &e3, cfg, 1);

    *e0_out = e0;
    *e1_out = e1;
    *e2_out = e2;
    *e3_out = e3;
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

float4 mix_spherical_velocity(float4 p1, float4 p2, float4 in1, float4 in2, float a)
{
    float4 ap1 = p1;
    float4 ap2 = p2;

    ap1.y = fabs(ap1.y);
    ap2.y = fabs(ap2.y);

    float3 cart1 = spherical_velocity_to_cartesian_velocity(ap1.yzw, in1.yzw);
    float3 cart2 = spherical_velocity_to_cartesian_velocity(ap2.yzw, in2.yzw);

    float3 cp_1 = polar_to_cartesian(ap1.yzw);
    float3 cp_2 = polar_to_cartesian(ap2.yzw);

    //float r1 = in1.y;
    //float r2 = in2.y;

    float3 mixed = mix(cart1, cart2, a);
    float3 mixed_cart_pos = mix(cp_1, cp_2, a);

    float3 as_polar = cartesian_velocity_to_polar_velocity(mixed_cart_pos, mixed);

    //as_polar.x = mix(r1, r2, a);

    float t = mix(in1.x, in2.x, a);

    return (float4)(t, as_polar);
}

__kernel
void handle_interpolating_geodesic(__global float4* geodesic_path, __global float4* geodesic_velocity, __global float* dT_dt, __global float* ds_in,
                                   __global float4* g_camera_quat,
                                   __global float4* g_camera_polar_out,
                                   __global float4* e0_out, __global float4* e1_out, __global float4* e2_out, __global float4* e3_out,
                                   float target_time,
                                   __global int* count_in,
                                   dynamic_config_space struct dynamic_config* cfg)
{
    if(get_global_id(0) != 0)
        return;

    float4 start_generic = geodesic_path[0];

    float4 e0, e1, e2, e3;
    calculate_tetrads(generic_to_spherical(start_generic, cfg), &e0, &e1, &e2, &e3, cfg, 1);

    int cnt = *count_in;

    //printf("Count %i\n", cnt);

    float current_time = geodesic_path[0].x;

    *g_camera_polar_out = generic_to_spherical(start_generic, cfg);

    *e0_out = e0;
    *e1_out = e1;
    *e2_out = e2;
    *e3_out = e3;

    for(int i=0; i < cnt - 1; i++)
    {
        float ds = ds_in[i];

        float4 current_pos = geodesic_path[i];
        float4 next_pos = geodesic_path[i + 1];

        float4 velocity = geodesic_velocity[i];

        float4 ne0 = e0 + parallel_transport_get_acceleration(e0, geodesic_path[i], velocity, cfg) * ds;
        float4 ne1 = e1 + parallel_transport_get_acceleration(e1, geodesic_path[i], velocity, cfg) * ds;
        float4 ne2 = e2 + parallel_transport_get_acceleration(e2, geodesic_path[i], velocity, cfg) * ds;
        float4 ne3 = e3 + parallel_transport_get_acceleration(e3, geodesic_path[i], velocity, cfg) * ds;

        if(next_pos.x < current_pos.x)
        {
            float4 im = current_pos;
            current_pos = next_pos;
            next_pos = im;
        }

        float dt = next_pos.x - current_pos.x;

        if(target_time >= current_pos.x && target_time < next_pos.x)
        {
            float dx = (target_time - current_pos.x) / (next_pos.x - current_pos.x);

            ///NEED TO BACKWARDS ROTATE POS FOR CONSTANT THETA
            float4 spherical1 = generic_to_spherical(current_pos, cfg);
            float4 spherical2 = generic_to_spherical(next_pos, cfg);

            float4 fin_polar = mix_spherical(spherical1, spherical2, dx);
            *g_camera_polar_out = fin_polar;

            float4 oe0 = mix(e0, ne0, dx);
            float4 oe1 = mix(e1, ne1, dx);
            float4 oe2 = mix(e2, ne2, dx);
            float4 oe3 = mix(e3, ne3, dx);

            *e0_out = oe0;
            *e1_out = oe1;
            *e2_out = oe2;
            *e3_out = oe3;

            ///so. now we have the basis. Need to apply camera rotation to it
            ///or... could just parallel transport the whole rotation initially?

            return;
        }

        current_time += dt;

        e0 = ne0;
        e1 = ne1;
        e2 = ne2;
        e3 = ne3;
    }
}

__kernel
void init_rays_generic(__global float4* g_polar_camera_in, __global float4* g_camera_quat,
                       __global struct lightray* metric_rays, __global int* metric_ray_count,
                       int width, int height,
                       __global int* termination_buffer,
                       int prepass_width, int prepass_height,
                       int flip_geodesic_direction, float2 base_angle,
                       __global float4* e0, __global float4* e1, __global float4* e2, __global float4* e3,
                       float on_geodesic,
                       dynamic_config_space struct dynamic_config* cfg)
{
    int id = get_global_id(0);

    if(id >= width * height)
        return;

    float4 polar_camera_in = *g_polar_camera_in;

    const int cx = id % width;
    const int cy = id / width;

    float3 pixel_direction = calculate_pixel_direction(cx, cy, width, height, polar_camera_in, *g_camera_quat, base_angle);

    float4 polar_camera = polar_camera_in;

    float4 at_metric = spherical_to_generic(polar_camera, cfg);

    float4 bT = *e0;
    float4 observer_velocity = bT;

    float4 sVx = *e1;
    float4 sVy = *e2;
    float4 sVz = *e3;

    float4 polar_x = generic_velocity_to_spherical_velocity(at_metric, sVx, cfg);
    float4 polar_y = generic_velocity_to_spherical_velocity(at_metric, sVy, cfg);
    float4 polar_z = generic_velocity_to_spherical_velocity(at_metric, sVz, cfg);

    float3 apolar = polar_camera.yzw;
    apolar.x = fabs(apolar.x);

    float3 cartesian_cx = normalize(spherical_velocity_to_cartesian_velocity(apolar, polar_x.yzw));
    float3 cartesian_cy = normalize(spherical_velocity_to_cartesian_velocity(apolar, polar_y.yzw));
    float3 cartesian_cz = normalize(spherical_velocity_to_cartesian_velocity(apolar, polar_z.yzw));

    //if(on_geodesic == 0)
    //    pixel_direction = unrotate_vector(normalize(cartesian_cx), normalize(cartesian_cy), normalize(cartesian_cz), pixel_direction);

    pixel_direction = normalize(pixel_direction);

    float4 pixel_x = pixel_direction.x * sVx;
    float4 pixel_y = pixel_direction.y * sVy;
    float4 pixel_z = pixel_direction.z * sVz;

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

    #if defined(GENERIC_CONSTANT_THETA) || defined(DEBUG_CONSTANT_THETA)
    {
        float4 pos_spherical = generic_to_spherical(lightray_spacetime_position, cfg);
        float4 vel_spherical = generic_velocity_to_spherical_velocity(lightray_spacetime_position, lightray_velocity, cfg);

        float fsign = sign(pos_spherical.y);
        pos_spherical.y = fabs(pos_spherical.y);

        float3 pos_cart = polar_to_cartesian(pos_spherical.yzw);
        float3 vel_cart = spherical_velocity_to_cartesian_velocity(pos_spherical.yzw, vel_spherical.yzw);

        float4 quat = get_theta_adjustment_quat(vel_cart, polar_camera, 1, false);

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

        lightray_spacetime_position = next_pos_generic;
        lightray_velocity = next_vel_generic;
    }
    #endif // GENERIC_CONSTANT_THETA

    float4 lightray_acceleration = (float4)(0,0,0,0);

    #ifdef IS_CONSTANT_THETA
    lightray_spacetime_position.z = M_PIf/2;
    lightray_velocity.z = 0;
    lightray_acceleration.z = 0;
    #endif // IS_CONSTANT_THETA

    #ifndef GENERIC_BIG_METRIC
    float g_metric[4] = {0};
    #else
    float g_metric_big[16] = {0};
    #endif // GENERIC_BIG_METRIC

    #if 1
    {
        #ifndef GENERIC_BIG_METRIC
        float g_partials[16] = {0};

        calculate_metric_generic(lightray_spacetime_position, g_metric, cfg);
        calculate_partial_derivatives_generic(lightray_spacetime_position, g_partials, cfg);

        lightray_velocity = fix_light_velocity2(lightray_velocity, g_metric);
        lightray_acceleration = calculate_acceleration(lightray_velocity, g_metric, g_partials);
        #else
        float g_partials_big[64] = {0};

        calculate_metric_generic_big(lightray_spacetime_position, g_metric_big, cfg);
        calculate_partial_derivatives_generic_big(lightray_spacetime_position, g_partials_big, cfg);

        //float4 prefix = lightray_velocity;

        lightray_velocity = fix_light_velocity_big(lightray_velocity, g_metric_big);

        /*if(cx == 500 && cy == 400)
        {
            printf("pre %f %f %f %f post %f %f %f %f", prefix.x, prefix.y, prefix.z, prefix.w,
                                                         lightray_velocity.x, lightray_velocity.y, lightray_velocity.z, lightray_velocity.w);
        }*/

        lightray_acceleration = calculate_acceleration_big(lightray_velocity, g_metric_big, g_partials_big);
        #endif // GENERIC_BIG_METRIC
    }
    #endif // 0

    //if(cx == 500 && cy == 400)
    //    printf("Posr %f %f %f\n", polar_camera.y, polar_camera.z, polar_camera.w);
    //    printf("DS %f\n", dot_product_big(lightray_velocity, lightray_velocity, g_metric_big));

    /*lightray_spacetime_position.z = M_PIf/2;
    lightray_velocity.z = 0;
    lightray_acceleration.z = 0;*/

    struct lightray ray;
    ray.sx = cx;
    ray.sy = cy;
    ray.position = lightray_spacetime_position;
    ray.velocity = lightray_velocity;
    ray.acceleration = lightray_acceleration;
    ray.early_terminate = 0;

    {
        float4 uobsu_upper = observer_velocity;

        #ifdef GENERIC_BIG_METRIC
        float4 uobsu_lower = lower_index_big(uobsu_upper, g_metric_big);
        #else
        float4 uobsu_lower = lower_index(uobsu_upper, g_metric);
        #endif // GENERIC_BIG_METRIC

        float final_val = dot(lightray_velocity, uobsu_lower);

        ray.ku_uobsu = final_val;
    }

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
void step_verlet(float4 position, float4 velocity, float4 acceleration, float ds, float4* position_out, float4* velocity_out, float4* acceleration_out, float* g_00_out, dynamic_config_space struct dynamic_config* cfg)
{
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

    float4 next_velocity = velocity + 0.5f * (acceleration + next_acceleration) * ds;

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

    velocity = fix_light_velocity2(velocity, g_metric);

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

float acceleration_to_precision(float4 acceleration, float* next_ds_out)
{
    float uniform_coordinate_precision_divisor = max(max(W_V1, W_V2), max(W_V3, W_V4));

    float current_acceleration_err = fast_length(acceleration * (float4)(W_V1, W_V2, W_V3, W_V4)) * 0.01f;
    current_acceleration_err /= uniform_coordinate_precision_divisor;

    float experienced_acceleration_change = current_acceleration_err;

    float err = MAX_ACCELERATION_CHANGE;

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

int calculate_ds_error(float current_ds, float4 next_acceleration, float4 acceleration, float* next_ds_out)
{
    float next_ds = 0;
    float diff = acceleration_to_precision(next_acceleration, &next_ds);

    ///produces strictly worse results for kerr
    next_ds = 0.99f * current_ds * clamp(next_ds / current_ds, 0.3f, 2.f);

    next_ds = max(next_ds, MIN_STEP);

    *next_ds_out = next_ds;

    float err = MAX_ACCELERATION_CHANGE;

    #ifdef SINGULARITY_DETECTION
    if(next_ds == MIN_STEP && (diff/I_HATE_COMPUTERS) > err * 10000)
        return DS_RETURN;
    #endif // SINGULARITY_DETECTION

    if(next_ds < current_ds/1.95f)
        return DS_SKIP;

    return DS_NONE;
}
#endif // ADAPTIVE_PRECISION

__kernel
void do_generic_rays (__global struct lightray* restrict generic_rays_in, __global struct lightray* restrict generic_rays_out,
                      __global struct lightray* restrict finished_rays,
                      __global int* restrict generic_count_in, __global int* restrict generic_count_out,
                      __global int* restrict finished_count_out, dynamic_config_space struct dynamic_config* cfg)
{
    int id = get_global_id(0);

    if(id >= *generic_count_in)
        return;

    __global struct lightray* ray = &generic_rays_in[id];

    if(ray->early_terminate)
        return;

    float4 position = ray->position;
    float4 velocity = ray->velocity;
    float4 acceleration = ray->acceleration;

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
    (void)acceleration_to_precision(acceleration, &next_ds);
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

    int loop_limit = 64000;

    #ifdef DEVICE_SIDE_ENQUEUE
    loop_limit /= 125;
    #endif // DEVICE_SIDE_ENQUEUE

    //#pragma unroll
    for(int i=0; i < loop_limit; i++)
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

        if(should_terminate)
        {
            int out_id = atomic_inc(finished_count_out);

            float4 polar_velocity = generic_velocity_to_spherical_velocity(position, velocity, cfg);

            struct lightray out_ray;
            out_ray.sx = sx;
            out_ray.sy = sy;
            out_ray.position = polar_position;
            out_ray.velocity = polar_velocity;
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

        step_verlet(position, velocity, acceleration, ds, &next_position, &next_velocity, &next_acceleration, 0, cfg);

        #ifdef ADAPTIVE_PRECISION

        if(fabs(r_value) < new_max)
        {
            int res = calculate_ds_error(ds, next_acceleration, acceleration, &next_ds);

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

        if(any(isnan(position)) || any(isnan(velocity)) || any(isnan(acceleration)))
        {
            return;
        }
    }

    int out_id = atomic_inc(generic_count_out);

    struct lightray out_ray;
    out_ray.sx = sx;
    out_ray.sy = sy;
    out_ray.position = position;
    out_ray.velocity = velocity;
    out_ray.acceleration = acceleration;
    out_ray.ku_uobsu = ray->ku_uobsu;
    out_ray.early_terminate = 0;

    generic_rays_out[out_id] = out_ray;
}

__kernel
void get_geodesic_path(__global struct lightray* generic_rays_in,
                       __global float4* positions_out,
                       __global float4* velocities_out,
                       __global float* dT_dt_out,
                       __global float* ds_out,
                       __global int* generic_count_in, int geodesic_start, int width, int height,
                       __global float4* g_polar_camera_pos, __global float4* g_camera_quat,
                       __global float4* e0, __global float4* e1, __global float4* e2, __global float4* e3,
                       float2 base_angle, dynamic_config_space struct dynamic_config* cfg, __global int* count_out)
{
    int id = geodesic_start;

    if(id >= *generic_count_in)
        return;

    if(get_global_id(0) > 1)
        return;

    __global struct lightray* ray = &generic_rays_in[id];

    float4 polar_camera_pos = *g_polar_camera_pos;
    float4 camera_quat = *g_camera_quat;

    float4 position = ray->position;
    float4 velocity = ray->velocity;
    float4 acceleration = ray->acceleration;

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
    (void)acceleration_to_precision(acceleration, &next_ds);
    #endif // ADAPTIVE_PRECISION

    float subambient_precision = 0.5;
    float ambient_precision = 0.2;

    float rs = 1;

    int bufc = 0;

    //#pragma unroll
    for(int i=0; i < 64000; i++)
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

        #ifndef SINGULAR
        if(fabs(polar_position.y) >= UNIVERSE_SIZE)
        #else
        if(fabs(polar_position.y) < rs*SINGULAR_TERMINATOR || fabs(polar_position.y) >= UNIVERSE_SIZE)
        #endif // SINGULAR
        {
            *count_out = bufc;
            return;
        }

        float4 next_position, next_velocity, next_acceleration;
        float g00 = 0;

        step_verlet(position, velocity, acceleration, ds, &next_position, &next_velocity, &next_acceleration, &g00, cfg);

        ///https://en.wikipedia.org/wiki/Coordinate_time#Mathematics
        float dT_dt = native_sqrt(fabs(g00));

        #ifdef ADAPTIVE_PRECISION

        if(fabs(r_value) < new_max)
        {
            int res = calculate_ds_error(ds, next_acceleration, acceleration, &next_ds);

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

        /*
        float4 polar_out = generic_to_spherical(position, cfg);

        #if (defined(GENERIC_METRIC) && defined(GENERIC_CONSTANT_THETA)) || !defined(GENERIC_METRIC) || defined(DEBUG_CONSTANT_THETA)
        polar_out.yzw = get_texture_constant_theta_rotation(pixel_direction, polar_camera_pos, polar_out);
        #endif
        */

        positions_out[bufc] = position;
        velocities_out[bufc] = velocity;
        dT_dt_out[bufc] = dT_dt;
        ds_out[bufc] = ds;
        bufc++;

        if(any(isnan(position)) || any(isnan(velocity)) || any(isnan(acceleration)))
        {
            *count_out = bufc;
            return;
        }
    }

    *count_out = bufc;
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
void calculate_texture_coordinates(__global struct lightray* finished_rays, __global int* finished_count_in, __global float2* texture_coordinates, int width, int height, __global float4* g_polar_camera_pos, __global float4* g_camera_quat,
                                   __global float4* e0, __global float4* e1, __global float4* e2, __global float4* e3,
                                   float2 base_angle)
{
    int id = get_global_id(0);

    if(id >= *finished_count_in)
        return;

    float4 polar_camera_pos = *g_polar_camera_pos;
    float4 camera_quat = *g_camera_quat;

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

    float3 pixel_direction = calculate_pixel_direction(sx, sy, width, height, polar_camera_pos, *g_camera_quat, base_angle);

	float3 npolar = position.yzw;

    #if (defined(GENERIC_METRIC) && defined(GENERIC_CONSTANT_THETA)) || !defined(GENERIC_METRIC) || defined(DEBUG_CONSTANT_THETA)
	npolar = get_texture_constant_theta_rotation(pixel_direction, polar_camera_pos, position);
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
        lin_result = redshift(lin_result, z_shift);
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
