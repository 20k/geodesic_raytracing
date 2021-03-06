/*__kernel
void generate_mips(__global uchar4* in, int page_width, int page_height, int width, int height, int lower_level)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x >= width || y >= height)
        return;

    uint4 avg = 0;

    avg += in[lower_level * ]
}*/

#define M_PIf ((float)M_PI)

float3 cartesian_to_polar(float3 in)
{
    float r = length(in);
    //float theta = atan2(native_sqrt(in.x * in.x + in.y * in.y), in.z);
    float theta = acos(in.z / r);
    float phi = atan2(in.y, in.x);

    return (float3){r, theta, phi};
}

float3 cartesian_to_polar_signed(float3 in, bool positive)
{
    float r = length(in);

    if(!positive)
        r = -r;

    //float theta = atan2(native_sqrt(in.x * in.x + in.y * in.y), in.z);
    float theta = acos(in.z / r);
    float phi = atan2(in.y, in.x);

    if(!positive)
        phi += M_PIf;

    if(phi >= M_PIf)
        phi -= 2 * M_PIf;

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

void calculate_metric(float4 spacetime_position, float g_metric_out[])
{
    float rs = 1;
    float c = 1;

    float r = spacetime_position.y;

    #ifndef IS_CONSTANT_THETA
    float theta = spacetime_position.z;
    #else
    float theta = M_PIf/2;
    #endif // IS_CONSTANT_THETA

    g_metric_out[0] = -c * c * (1 - rs / r);
    g_metric_out[1] = 1/(1 - rs / r);
    g_metric_out[2] = r * r;
    g_metric_out[3] = r * r * sin(theta) * sin(theta);
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

float3 fix_ray_position(float3 polar_pos, float3 polar_velocity, float sphere_radius, bool outwards_facing)
{
    float position_sign = sign(polar_pos.x);

    float3 cpolar_pos = polar_pos;
    cpolar_pos.x = fabs(cpolar_pos.x);

    polar_velocity.x *= position_sign;

    float3 cartesian_velocity = spherical_velocity_to_cartesian_velocity(cpolar_pos, polar_velocity);

    float3 cartesian_pos = polar_to_cartesian(cpolar_pos);

    float3 C = (float3){0,0,0};

    float3 L = C - cartesian_pos;
    float tca = dot(L, cartesian_velocity);

    float d2 = dot(L, L) - tca * tca;

    if(d2 > sphere_radius * sphere_radius)
        return polar_pos;

    float thc = native_sqrt(sphere_radius * sphere_radius - d2);

    float t0 = tca - thc;
    float t1 = tca + thc;

    float my_t = 0;

    if(t0 > 0 && t1 > 0)
        return polar_pos;

    if(t0 < 0 && t1 < 0)
        my_t = max(t0, t1);

    if(t0 < 0 && t1 > 0)
        my_t = t0;

    if(t0 > 0 && t1 < 0)
        my_t = t1;

    float3 new_cart = cartesian_pos + my_t * cartesian_velocity;

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

/*float4 unrotate_vector4(float4 bx, float4 by, float4 bz, float4 bw, float4 v)
{

}*/

///normalize
float3 rejection(float3 my_vector, float3 basis)
{
    return normalize(my_vector - dot(my_vector, basis) * basis);
}

float3 srgb_to_lin(float3 C_srgb)
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

void calculate_metric_krus(float4 spacetime_position, float g_metric_out[])
{
    float rs = 1;
    float k = rs;

    #ifndef IS_CONSTANT_THETA
    float theta = spacetime_position.z;
    #else
    float theta = M_PIf/2;
    #endif // IS_CONSTANT_THETA

    float T = spacetime_position.x;
    float X = spacetime_position.y;

    float lambert_interior = lambert_w0((X*X - T*T) / M_E);

    float fXT = k * (1 + lambert_interior);

    g_metric_out[0] = - (4 * k * k * k / fXT) * exp(-fXT / k);
    g_metric_out[1] = (4 * k * k * k / fXT) * exp(-fXT / k);
    g_metric_out[2] = fXT * fXT;
    g_metric_out[3] = fXT * fXT * sin(theta) * sin(theta);
}

void calculate_metric_krus_with_r(float4 spacetime_position, float r, float g_metric_out[])
{
    float rs = 1;
    float k = rs;

    #ifndef IS_CONSTANT_THETA
    float theta = spacetime_position.z;
    #else
    float theta = M_PIf/2;
    #endif // IS_CONSTANT_THETA

    float T = spacetime_position.x;
    float X = spacetime_position.y;

    float fXT = r;

    g_metric_out[0] = - (4 * k * k * k / fXT) * exp(-fXT / k);
    g_metric_out[1] = (4 * k * k * k / fXT) * exp(-fXT / k);
    g_metric_out[2] = fXT * fXT;
    g_metric_out[3] = fXT * fXT * sin(theta) * sin(theta);
}

void calculate_partial_derivatives_krus(float4 spacetime_position, float g_metric_partials[16])
{
    /*TTTT,
      XXXX,
      oooo,
      pppp,*/

    float rs = 1;
    float k = rs;

    #ifndef IS_CONSTANT_THETA
    float theta = spacetime_position.z;
    #else
    float theta = M_PIf/2;
    #endif // IS_CONSTANT_THETA

    float T = spacetime_position.x;
    float X = spacetime_position.y;

    float lambert_interior = lambert_w0((X*X - T*T) / M_E);

    float fXT = k * (1 + lambert_interior);

    float f10 = 0;

    if(fabs(lambert_interior) > 0.00001)
        f10 = (2 * k * X * lambert_interior) / ((X*X - T*T) * (lambert_interior + 1));

    float f01 = 0;

    if(fabs(lambert_interior) > 0.00001)
        f01 = (2 * k * T * lambert_interior) / ((T*T - X*X) * (lambert_interior + 1));

    float back_component = exp(-fXT/k) * ((4 * k * k * k / (fXT * fXT)) + 4 * k * k / fXT);

    //dT by dT
    g_metric_partials[0 * 4 + 0] = f01 * back_component;
    //dT by dX
    g_metric_partials[0 * 4 + 1] = f10 * back_component;

    //dX by dT
    g_metric_partials[1 * 4 + 0] = -f01 * back_component;
    //dX by dX
    g_metric_partials[1 * 4 + 1] = -f10 * back_component;

    //dtheta by dT
    g_metric_partials[2 * 4 + 0] = 2 * fXT * f01;
    //dtheta by dX
    g_metric_partials[2 * 4 + 1] = 2 * fXT * f10;

    //dphi by dT
    g_metric_partials[3 * 4 + 0] = 2 * sin(theta) * sin(theta) * fXT * f01;
    //dphi by dX
    g_metric_partials[3 * 4 + 1] = 2 * sin(theta) * sin(theta) * fXT * f10;
    //dphi by dtheta
    g_metric_partials[3 * 4 + 2] = 2 * sin(theta) * cos(theta) * fXT * fXT;
}

void calculate_partial_derivatives(float4 spacetime_position, float g_metric_partials[])
{
    /*dt
    dr
    dtheta
    dphi*/

    float r = spacetime_position.y;

    float rs = 1;
    float c = 1;

    #ifndef IS_CONSTANT_THETA
    float theta = spacetime_position.z;
    #else
    float theta = M_PIf/2;
    #endif // IS_CONSTANT_THETA

    //dt dr
    g_metric_partials[0 * 4 + 1] = -c*c*rs/(r*r);
    //dr dr
    g_metric_partials[1 * 4 + 1] = -rs / ((rs - r) * (rs - r));
    //dtheta by dr
    g_metric_partials[2 * 4 + 1] = 2 * r; //this is irrelevant for constant theta, but i think the compiler figures it out now
    //dphi by dr
    g_metric_partials[3 * 4 + 1] = 2 * r * sin(theta) * sin(theta);
    //dphi by dtheta
    g_metric_partials[3 * 4 + 2] = 2 * r * r * sin(theta) * cos(theta);
}

float rt_to_T_krus(float r, float t)
{
    float rs = 1;
    float k = rs;

    if(r > rs)
        return native_sqrt(r/k - 1) * exp(0.5 * r/k) * sinh(0.5 * t/k);
    else
        return native_sqrt(1 - r/k) * exp(0.5 * r/k) * cosh(0.5 * t/k);
}

float rt_to_X_krus(float r, float t)
{
    float rs = 1;
    float k = rs;

    if(r > rs)
        return native_sqrt(r/k - 1) * exp(0.5 * r/k) * cosh(0.5 * t/k);
    else
        return native_sqrt(1 - r/k) * exp(0.5 * r/k) * sinh(0.5 * t/k);
}

float TX_to_r_krus(float T, float X)
{
    float rs = 1;
    float k = rs;

    return k * (1 + lambert_w0((X * X - T * T) / M_E));
}

float TX_to_r_krus_highprecision(float T, float X)
{
    float rs = 1;
    float k = rs;

    return k * (1 + lambert_w0_highprecision((X * X - T * T) / M_E));
}

float trdtdr_to_dX(float t, float r, float dt, float dr)
{
    float rs = 1;
    float k = rs;

    if(r > rs)
    {
        ///https://www.wolframalpha.com/input/?i=D%5B%28r%2Fk+-+1%29%5E0.5+*+%28e%5E%280.5+*+r%2Fk%29%29+*+cosh%280.5+*+t%2Fk%29%2C+r%5D+*+r0+%2B+D%5B%28r%2Fk+-+1%29%5E0.5+*+%28e%5E%280.5+*+r%2Fk%29%29+*+cosh%280.5+*+t%2Fk%29%2C+t%5D+*+t0
        return exp((0.5 * r)/k) * (r * (2 * dr * cosh((0.5 * t)/k) + dt * (exp((0.5 * t/k)) - exp(-(0.5 * t)/k))) - 2 * k * dt * sinh((0.5 * t)/k)) / (4 * k * k * native_sqrt((r - k)/k));
    }
    else
    {
        ///https://www.wolframalpha.com/input/?i=D%5B%281+-+r%2Fk%29%5E0.5+*+%28e%5E%280.5+*+r%2Fk%29%29+*+sinh%280.5+*+t%2Fk%29%2C+r%5D+*+r0+%2B+D%5B%281+-+r%2Fk%29%5E0.5+*+%28e%5E%280.5+*+r%2Fk%29%29+*+sinh%280.5+*+t%2Fk%29%2C+t%5D+*+t0
        return exp((0.5 * r)/k) * (dt * (0.5 * k - 0.5 * r) * cosh((0.5 * t) / k) - 0.5 * r * dr * sinh((0.5 * t/k))) / (k * k * native_sqrt(1-r/k));
    }

    /*if(r > rs)
        return exp(0.5 * r/k) * (dt * (0.5 * r - 0.5 * k) * sinh((0.5 * t) / k) + 0.5 * r * dr * cosh((0.5 * t) / k)) / (k * k * native_sqrt(r/k - 1));
    else
        return exp(0.5 * r/k) * (dt * (0.5 * k - 0.5 * r) * cosh((0.5 * t) / k) - 0.5 * r * dr * sinh((0.5 * t) / k)) / (k * k * native_sqrt(1 - k/r));*/
}

float trdtdr_to_dT(float t, float r, float dt, float dr)
{
    float rs = 1;
    float k = rs;

    if(r > rs)
    {
        ///https://www.wolframalpha.com/input/?i=D%5B%28r%2Fk+-+1%29%5E0.5+*+%28e%5E%280.5+*+r%2Fk%29%29+*+sinh%280.5+*+t%2Fk%29%2C+r%5D+*+r0+%2B+D%5B%28r%2Fk+-+1%29%5E0.5+*+%28e%5E%280.5+*+r%2Fk%29%29+*+sinh%280.5+*+t%2Fk%29%2C+t%5D+*+t0
        return exp((0.5 * r)/k) * (0.5 * r * dr * sinh((0.5 * t)/k) + dt * (0.5 * r - 0.5 * k) * cosh((0.5 * t/k))) / (k * k * native_sqrt(r/k - 1));
    }
    else
    {
        ///https://www.wolframalpha.com/input/?i=D%5B%281+-+r%2Fk%29%5E0.5+*+%28e%5E%280.5+*+r%2Fk%29%29+*+cosh%280.5+*+t%2Fk%29%2C+r%5D+*+r0+%2B+D%5B%281+-+r%2Fk%29%5E0.5+*+%28e%5E%280.5+*+r%2Fk%29%29+*+cosh%280.5+*+t%2Fk%29%2C+t%5D+*+t0
        return -exp((0.5 * r)/k) * (r * (2 * dr * cosh((0.5 * t)/k) + dt * (exp((0.5 * t)/k) - exp(-(0.5 * t)/k))) - 2 * k * dt * sinh((0.5 * t)/k)) / (4 * k * k * native_sqrt(-(r-k)/k));
    }

    /*if(r > rs)
        return exp(0.5 * r/k) * (dt * (0.5 * r - 0.5 * k) * cosh((0.5 * t) / k) + 0.5 * r * dr * sinh((0.5 * t) / k)) / (k * k * native_sqrt(r/k - 1));
    else
        return exp(0.5 * r/k) * (dt * (0.5 * k - 0.5 * r) * sinh((0.5 * t) / k) - 0.5 * r * dr * cosh((0.5 * t) / k)) / (k * k * native_sqrt(1 - k/r));*/
}

float TX_to_t(float T, float X)
{
    float rs = 1;

    if((T * T - X * X) < 0)
        return 2 * rs * atanh(T / X);
    else
        return 2 * rs * atanh(X / T);
}

float TXdTdX_to_dt(float T, float X, float dT, float dX)
{
    float rs = 1;

    /*if(T * T - X * X < 0)
    {
        return 2 * rs * (T * dX - dT * X) / (T * T - X * X);
    }
    else
    {
        return 2 * rs * (T * dX - dT * X) / (T * T - X * X);
    }*/

    return 2 * rs * (T * dX - dT * X) / (T * T - X * X);
}

///so the problem with this function, and the kruskal partial derivative function
///is that X*X - T * T = 0 at the horizon, so divide by 0
///however, the equation is basically 2 * k * lambert * (X * dX - T * dT) / ((X * X - T * T) * (lambert + 1))
///need to figure out the limit of this equation

///So
///

///https://www.wolframalpha.com/input/?i=D%5Bk+*+%281+%2B+W%28%28X+*+X+-+T+*+T%29+%2F+e%29%29%2C+T%5D+*+t0+%2B+D%5Bk+*+%281+%2B+W%28%28X+*+X+-+T+*+T%29+%2F+e%29%29%2C+X%5D+*+x0+
float TXdTdX_to_dr(float T, float X, float dT, float dX)
{
    float rs = 1;
    float k = rs;

    float lambert = lambert_w0((X * X - T * T) / M_E);

    if(fabs(T * T - X * X) < 0.0001)
    {
        float dU = dT - dX;
        float dV = dT + dX;

        float left = 0;
        float right = 0;

        float U = (T - X);
        float V = (T + X);

        if(fabs(U) > 0.0001)
        {
            left = k * dU * lambert / (U * (lambert + 1));
        }

        if(fabs(V) > 0.0001)
        {
            right = k * dV * lambert / (V * (lambert + 1));
        }

        return left + right;
    }

    float denom = (X * X - T * T) * (lambert + 1);

    float num = 2 * k * X * lambert * dX - 2 * k * T * lambert * dT;

    return num / denom;
}

float TXdTdX_to_dr_with_r(float T, float X, float dT, float dX, float r)
{
    float rs = 1;
    float k = rs;

    float lambert = (r / rs) - 1;

    if(fabs(T * T - X * X) < 0.0001)
    {
        float dU = dT - dX;
        float dV = dT + dX;

        float left = 0;
        float right = 0;

        float U = (T - X);
        float V = (T + X);

        if(fabs(U) > 0.0001)
        {
            left = k * dU * lambert / (U * (lambert + 1));
        }

        if(fabs(V) > 0.0001)
        {
            right = k * dV * lambert / (V * (lambert + 1));
        }

        return left + right;
    }

    float denom = (X * X - T * T) * (lambert + 1);
    float num = 2 * k * X * lambert * dX - 2 * k * T * lambert * dT;

    return num / denom;
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

float4 kruskal_position_to_schwarzs_position(float4 krus)
{
    float T = krus.x;
    float X = krus.y;

    float rs = 1;
    float r = TX_to_r_krus(T, X);

    float t = TX_to_t(T, X);

    return (float4)(t, r, krus.zw);
}

float4 kruskal_velocity_to_schwarzs_velocity(float4 krus, float4 dkrus)
{
    float dt = TXdTdX_to_dt(krus.x, krus.y, dkrus.x, dkrus.y);
    float dr = TXdTdX_to_dr(krus.x, krus.y, dkrus.x, dkrus.y);

    return (float4)(dt, dr, dkrus.zw);
}

float4 kruskal_position_to_schwarzs_position_with_r(float4 krus, float r)
{
    float T = krus.x;
    float X = krus.y;

    float rs = 1;

    float t = TX_to_t(T, X);

    return (float4)(t, r, krus.zw);
}

float4 kruskal_velocity_to_schwarzs_velocity_with_r(float4 krus, float4 dkrus, float r)
{
    float dt = TXdTdX_to_dt(krus.x, krus.y, dkrus.x, dkrus.y);
    float dr = TXdTdX_to_dr_with_r(krus.x, krus.y, dkrus.x, dkrus.y, r);

    return (float4)(dt, dr, dkrus.zw);
}

float4 schwarzs_position_to_kruskal_position(float4 pos)
{
    float T = rt_to_T_krus(pos.y, pos.x);
    float X = rt_to_X_krus(pos.y, pos.x);

    return (float4)(T, X, pos.zw);
}

float4 schwarzs_velocity_to_kruskal_velocity(float4 pos, float4 dpos)
{
    float dX = trdtdr_to_dX(pos.x, pos.y, dpos.x, dpos.y);
    float dT = trdtdr_to_dT(pos.x, pos.y, dpos.x, dpos.y);

    return (float4)(dT, dX, dpos.zw);
}

float r_to_T2_m_X2(float radius)
{
    float rs = 1;

    return (1 - radius / rs) * exp(radius / rs);
}

bool is_radius_leq_than(float4 spacetime_position, bool is_kruskal, float radius)
{
    if(!is_kruskal)
    {
        return spacetime_position.y < radius;
    }
    else
    {
        float cr = r_to_T2_m_X2(radius);

        float T = spacetime_position.x;
        float X = spacetime_position.y;

        return T*T - X*X >= cr;
    }
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

float dot_product_big(float4 u, float4 v, float g_metric_big[])
{
    float4 lowered = lower_index_big(u, g_metric_big);

    return dot(lowered, v);
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

#ifndef GENERIC_BIG_METRIC
void calculate_metric_generic(float4 spacetime_position, float g_metric_out[])
{
    float v1 = spacetime_position.x;
    float v2 = spacetime_position.y;
    float v3 = spacetime_position.z;
    float v4 = spacetime_position.w;

    float rs = RS_IMPL;
    float c = C_IMPL;

    g_metric_out[0] = F1_I;
    g_metric_out[1] = F2_I;
    g_metric_out[2] = F3_I;
    g_metric_out[3] = F4_I;
}

void calculate_partial_derivatives_generic(float4 spacetime_position, float g_metric_partials[])
{
    float v1 = spacetime_position.x;
    float v2 = spacetime_position.y;
    float v3 = spacetime_position.z;
    float v4 = spacetime_position.w;

    float rs = RS_IMPL;
    float c = C_IMPL;

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
void calculate_metric_generic_big(float4 spacetime_position, float g_metric_out[])
{
    float v1 = spacetime_position.x;
    float v2 = spacetime_position.y;
    float v3 = spacetime_position.z;
    float v4 = spacetime_position.w;

    float rs = RS_IMPL;
    float c = C_IMPL;

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

void calculate_partial_derivatives_generic_big(float4 spacetime_position, float g_metric_partials[])
{
    float v1 = spacetime_position.x;
    float v2 = spacetime_position.y;
    float v3 = spacetime_position.z;
    float v4 = spacetime_position.w;

    float rs = RS_IMPL;
    float c = C_IMPL;

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

    g_metric_partials[0 * 16 + 4] = g_metric_partials[0 * 16 + 1];
    g_metric_partials[0 * 16 + 8] = g_metric_partials[0 * 16 + 2];
    g_metric_partials[0 * 16 + 12] = g_metric_partials[0 * 16 + 3];
    g_metric_partials[0 * 16 + 9] = g_metric_partials[0 * 16 + 6];
    g_metric_partials[0 * 16 + 13] = g_metric_partials[0 * 16 + 7];
    g_metric_partials[0 * 16 + 14] = g_metric_partials[0 * 16 + 11];

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

    g_metric_partials[1 * 16 + 4] = g_metric_partials[1 * 16 + 1];
    g_metric_partials[1 * 16 + 8] = g_metric_partials[1 * 16 + 2];
    g_metric_partials[1 * 16 + 12] = g_metric_partials[1 * 16 + 3];
    g_metric_partials[1 * 16 + 9] = g_metric_partials[1 * 16 + 6];
    g_metric_partials[1 * 16 + 13] = g_metric_partials[1 * 16 + 7];
    g_metric_partials[1 * 16 + 14] = g_metric_partials[1 * 16 + 11];

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

    g_metric_partials[2 * 16 + 4] = g_metric_partials[2 * 16 + 1];
    g_metric_partials[2 * 16 + 8] = g_metric_partials[2 * 16 + 2];
    g_metric_partials[2 * 16 + 12] = g_metric_partials[2 * 16 + 3];
    g_metric_partials[2 * 16 + 9] = g_metric_partials[2 * 16 + 6];
    g_metric_partials[2 * 16 + 13] = g_metric_partials[2 * 16 + 7];
    g_metric_partials[2 * 16 + 14] = g_metric_partials[2 * 16 + 11];

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

    g_metric_partials[3 * 16 + 4] = g_metric_partials[3 * 16 + 1];
    g_metric_partials[3 * 16 + 8] = g_metric_partials[3 * 16 + 2];
    g_metric_partials[3 * 16 + 12] = g_metric_partials[3 * 16 + 3];
    g_metric_partials[3 * 16 + 9] = g_metric_partials[3 * 16 + 6];
    g_metric_partials[3 * 16 + 13] = g_metric_partials[3 * 16 + 7];
    g_metric_partials[3 * 16 + 14] = g_metric_partials[3 * 16 + 11];

}
#endif // GENERIC_BIG_METRIC

float4 generic_to_spherical(float4 in)
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

float4 generic_velocity_to_spherical_velocity(float4 in, float4 inv)
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

float4 spherical_to_generic(float4 in)
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

float4 spherical_velocity_to_generic_velocity(float4 in, float4 inv)
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
    x = clamp(x, 0, width-1);
    y = clamp(y, 0, height-1);

    return termination_buffer[y * width + x] == 1;
}

__kernel
void init_rays_generic(float4 polar_camera_in, float4 camera_quat,
                       __global struct lightray* metric_rays, __global int* metric_ray_count,
                       int width, int height,
                       __global int* termination_buffer,
                       int prepass_width, int prepass_height,
                       int flip_geodesic_direction, float2 base_angle)
{
    int id = get_global_id(0);

    if(id >= width * height)
        return;

    const int cx = id % width;
    const int cy = id / width;

    float3 pixel_direction = calculate_pixel_direction(cx, cy, width, height, polar_camera_in, camera_quat, base_angle);

    float4 polar_camera = polar_camera_in;

    #if defined(GENERIC_CONSTANT_THETA) || defined(DEBUG_CONSTANT_THETA)
    {
        adjust_pixel_direction_and_camera_theta(pixel_direction, polar_camera, &pixel_direction, &polar_camera, cx==500&&cy==400);
    }
    #endif // GENERIC_CONSTANT_THETA

    if(cx == 500 && cy == 400)
    {
        //printf("Pixel direction %f %f %f cam %f %f %f\n", pixel_direction.x, pixel_direction.y, pixel_direction.z, polar_camera.y, polar_camera.z, polar_camera.w);
    }

    float4 at_metric = spherical_to_generic(polar_camera);

    /*if(cx == 500 && cy == 400)
    {
        printf("At %f %f %f %f\n", at_metric.x, at_metric.y, at_metric.z, at_metric.w);
        printf("was %f %f %f %f\n", cartesian_camera_pos.x, cartesian_camera_pos.y, cartesian_camera_pos.z, cartesian_camera_pos.w);
    }*/

    #ifndef GENERIC_BIG_METRIC
    float g_metric[4] = {};
    calculate_metric_generic(at_metric, g_metric);

    float4 co_basis = (float4){native_sqrt(-g_metric[0]), native_sqrt(g_metric[1]), native_sqrt(g_metric[2]), native_sqrt(g_metric[3])};

    float4 bT = (float4)(1/co_basis.x, 0, 0, 0); ///or bt
    float4 bX = (float4)(0, 1/co_basis.y, 0, 0); ///or br
    float4 btheta = (float4)(0, 0, 1/co_basis.z, 0);
    float4 bphi = (float4)(0, 0, 0, 1/co_basis.w);
    #else
    float g_metric_big[16] = {0};
    calculate_metric_generic_big(at_metric, g_metric_big);

    ///contravariant
    float4 bT;
    float4 bX;
    float4 btheta;
    float4 bphi;

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

        bT = basis.v1;
        bX = basis.v2;
        btheta = basis.v3;
        bphi = basis.v4;
    }
    #endif // GENERIC_BIG_METRIC

    #if 0
    if(cx == 500 && cy == 400)
    {
        /*printf("BT %f %f %f %f\n", bT.x, bT.y, bT.z, bT.w);
        printf("bX %f %f %f %f\n", bX.x, bX.y, bX.z, bX.w);
        printf("btheta %f %f %f %f\n", btheta.x, btheta.y, btheta.z, btheta.w);
        printf("bphi %f %f %f %f\n", bphi.x, bphi.y, bphi.z, bphi.w);*/

        /*printf("oBT %f %f %f %f\n", obT.x, obT.y, obT.z, obT.w);
        printf("obX %f %f %f %f\n", obX.x, obX.y, obX.z, obX.w);
        printf("obtheta %f %f %f %f\n", obtheta.x, obtheta.y, obtheta.z, obtheta.w);
        printf("obphi %f %f %f %f\n", obphi.x, obphi.y, obphi.z, obphi.w);*/
    }
    #endif // 0

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

    float4 polar_x = generic_velocity_to_spherical_velocity(at_metric, sVx);
    float4 polar_y = generic_velocity_to_spherical_velocity(at_metric, sVy);
    float4 polar_z = generic_velocity_to_spherical_velocity(at_metric, sVz);

    float3 apolar = polar_camera.yzw;
    apolar.x = fabs(apolar.x);

    float3 cartesian_cx = normalize(spherical_velocity_to_cartesian_velocity(apolar, polar_x.yzw));
    float3 cartesian_cy = normalize(spherical_velocity_to_cartesian_velocity(apolar, polar_y.yzw));
    float3 cartesian_cz = normalize(spherical_velocity_to_cartesian_velocity(apolar, polar_z.yzw));

    /*if(cx == 500 && cy == 400)
    {
        printf("ANGLES %f %f\n", base_angle.y, base_angle.x);
    }*/

    pixel_direction = unrotate_vector(normalize(cartesian_cx), normalize(cartesian_cy), normalize(cartesian_cz), pixel_direction);

    pixel_direction = normalize(pixel_direction);
    float4 pixel_x = pixel_direction.x * polar_x;
    float4 pixel_y = pixel_direction.y * polar_y;
    float4 pixel_z = pixel_direction.z * polar_z;

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

    pixel_x = spherical_velocity_to_generic_velocity(polar_camera, pixel_x);
    pixel_y = spherical_velocity_to_generic_velocity(polar_camera, pixel_y);
    pixel_z = spherical_velocity_to_generic_velocity(polar_camera, pixel_z);

    float4 lightray_velocity = pixel_x + pixel_y + pixel_z + pixel_t;
    float4 lightray_spacetime_position = at_metric;

    float4 lightray_acceleration = (float4)(0,0,0,0);

    #ifdef IS_CONSTANT_THETA
    lightray_spacetime_position.z = M_PIf/2;
    lightray_velocity.z = 0;
    lightray_acceleration.z = 0;
    #endif // IS_CONSTANT_THETA

    #if 1
    {
        #ifndef GENERIC_BIG_METRIC
        float g_partials[16] = {0};

        calculate_metric_generic(lightray_spacetime_position, g_metric);
        calculate_partial_derivatives_generic(lightray_spacetime_position, g_partials);

        lightray_velocity = fix_light_velocity2(lightray_velocity, g_metric);
        lightray_acceleration = calculate_acceleration(lightray_velocity, g_metric, g_partials);
        #else
        float g_partials_big[64] = {0};

        calculate_metric_generic_big(lightray_spacetime_position, g_metric_big);
        calculate_partial_derivatives_generic_big(lightray_spacetime_position, g_partials_big);

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
        *metric_ray_count = (height - 1) * width + width - 1;

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
void step_verlet(float4 position, float4 velocity, float4 acceleration, float ds, float4* position_out, float4* velocity_out, float4* acceleration_out)
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
    calculate_metric_generic(next_position, g_metric);
    calculate_partial_derivatives_generic(next_position, g_partials);

    ///1ms
    intermediate_next_velocity = fix_light_velocity2(intermediate_next_velocity, g_metric);

    float4 next_acceleration = calculate_acceleration(intermediate_next_velocity, g_metric, g_partials);
    #else
    calculate_metric_generic_big(next_position, g_metric_big);
    calculate_partial_derivatives_generic_big(next_position, g_partials_big);

    //intermediate_next_velocity = fix_light_velocity_big(intermediate_next_velocity, g_metric_big);
    float4 next_acceleration = calculate_acceleration_big(intermediate_next_velocity, g_metric_big, g_partials_big);
    #endif // GENERIC_BIG_METRIC

    float4 next_velocity = velocity + 0.5f * (acceleration + next_acceleration) * ds;

    *position_out = next_position;
    *velocity_out = next_velocity;
    *acceleration_out = next_acceleration;
}

void step_euler(float4 position, float4 velocity, float ds, float4* position_out, float4* velocity_out)
{
    #ifndef GENERIC_BIG_METRIC
    float g_metric[4] = {};
    float g_partials[16] = {};
    #else
    float g_metric_big[16] = {};
    float g_partials_big[64] = {};
    #endif // GENERIC_BIG_METRIC

    #ifndef GENERIC_BIG_METRIC
    calculate_metric_generic(position, g_metric);
    calculate_partial_derivatives_generic(position, g_partials);

    velocity = fix_light_velocity2(velocity, g_metric);

    float4 lacceleration = calculate_acceleration(velocity, g_metric, g_partials);
    #else
    calculate_metric_generic_big(position, g_metric_big);
    calculate_partial_derivatives_generic_big(position, g_partials_big);

    float4 lacceleration = calculate_acceleration_big(velocity, g_metric_big, g_partials_big);
    #endif // GENERIC_BIG_METRIC

    velocity += lacceleration * ds;
    position += velocity * ds;

    *position_out = position;
    *velocity_out = velocity;
}

float get_distance_to_object(float4 polar)
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
int calculate_ds_error(float current_ds, float4 next_acceleration, float4 acceleration, float* next_ds_out)
{
    float uniform_coordinate_precision_divisor = max(max(W_V1, W_V2), max(W_V3, W_V4));

    float current_acceleration_err = fast_length(next_acceleration * (float4)(W_V1, W_V2, W_V3, W_V4)) * 0.01f;
    current_acceleration_err /= uniform_coordinate_precision_divisor;

    float experienced_acceleration_change = current_acceleration_err;

    float err = MAX_ACCELERATION_CHANGE;
    float i_hate_computers = 256*256;

    //#define MIN_STEP 0.00001f
    #define MIN_STEP 0.000001f

    float max_timestep = 100000;

    float diff = experienced_acceleration_change * i_hate_computers;

    if(diff < err * i_hate_computers / pow(max_timestep, 2))
        diff = err * i_hate_computers / pow(max_timestep, 2);

    ///of course, as is tradition, whatever works for kerr does not work for alcubierre
    ///the sqrt error calculation is significantly better for alcubierre, largely in terms of having no visual artifacts at all
    ///whereas the pow version is nearly 2x faster for kerr
    float next_ds = native_sqrt(((err * i_hate_computers) / diff));

    ///produces strictly worse results for kerr
    next_ds = 0.99f * current_ds * clamp(next_ds / current_ds, 0.3f, 2.f);

    next_ds = max(next_ds, MIN_STEP);

    *next_ds_out = next_ds;

    #ifdef SINGULARITY_DETECTION
    if(next_ds == MIN_STEP && (diff/i_hate_computers) > err * 10000)
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
                      __global int* restrict finished_count_out)
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
        calculate_metric_generic(position, g_metric);

        velocity = fix_light_velocity2(velocity, g_metric);
    }
    #endif // GENERIC_BIG_METRIC

    #ifdef IS_CONSTANT_THETA
    position.z = M_PIf/2;
    velocity.z = 0;
    acceleration.z = 0;
    #endif // IS_CONSTANT_THETA

    float next_ds = 0.00001;

    ///results:
    ///subambient_precision can't go above 0.5 much while in verlet mode without the size of the event horizon changing
    ///in euler mode this is actually already too low

    ///ambient precision however looks way too low at 0.01, testing up to 0.3 showed no noticable difference, needs more precise tests though
    ///only in the case without kruskals and event horizon crossings however, any precision > 0.01 is insufficient in that case
    float subambient_precision = 0.5;
    float ambient_precision = 0.2;

    float rs = 1;

    float uniform_coordinate_precision_divisor = max(max(W_V1, W_V2), max(W_V3, W_V4));

    int loop_limit = 64000;

    #ifdef DEVICE_SIDE_ENQUEUE
    loop_limit /= 125;
    #endif // DEVICE_SIDE_ENQUEUE

    #pragma unroll
    for(int i=0; i < loop_limit; i++)
    {
        #ifdef IS_CONSTANT_THETA
        position.z = M_PIf/2;
        velocity.z = 0;
        acceleration.z = 0;
        #endif // IS_CONSTANT_THETA

        float new_max = 10 * rs;
        float new_min = 3 * rs;

        float4 polar_position = generic_to_spherical(position);

        #ifdef IS_CONSTANT_THETA
        polar_position.z = M_PIf/2;
        #endif // IS_CONSTANT_THETA

        //float r_value = polar_position.y;

        float r_value = get_distance_to_object(polar_position);

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

        #ifndef SINGULAR
        if(fabs(polar_position.y) >= UNIVERSE_SIZE)
        #else
        if(fabs(polar_position.y) < rs*SINGULAR_TERMINATOR || fabs(polar_position.y) >= UNIVERSE_SIZE)
        #endif // SINGULAR
        {
            int out_id = atomic_inc(finished_count_out);

            float4 polar_velocity = generic_velocity_to_spherical_velocity(position, velocity);

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

        step_euler(position, velocity, ds, &next_position, &next_velocity);

        position = next_position;
        velocity = next_velocity;

        #endif // EULER_INTEGRATsION

        #ifdef VERLET_INTEGRATION_GENERIC

        float4 next_position, next_velocity, next_acceleration;

        step_verlet(position, velocity, acceleration, ds, &next_position, &next_velocity, &next_acceleration);

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

        /*if(sx == 700 && sy == 400)
        {
            float g_metric_big[16] = {};
            calculate_metric_generic_big(position, g_metric_big);
            printf("DS %f\n", dot_product_big(velocity, velocity, g_metric_big));
        }*/

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
                       __global int* generic_count_in, int geodesic_start, int width, int height,
                       float4 polar_camera_pos, float4 camera_quat, float2 base_angle)
{
    int id = geodesic_start;

    if(id >= *generic_count_in)
        return;

    if(get_global_id(0) > 1)
        return;

    __global struct lightray* ray = &generic_rays_in[id];

    float4 position = ray->position;
    float4 velocity = ray->velocity;
    float4 acceleration = ray->acceleration;

    int sx = ray->sx;
    int sy = ray->sy;

    #ifndef GENERIC_BIG_METRIC
    {
        float g_metric[4] = {0};
        calculate_metric_generic(position, g_metric);

        velocity = fix_light_velocity2(velocity, g_metric);
    }
    #endif // GENERIC_BIG_METRIC

    #ifdef IS_CONSTANT_THETA
    position.z = M_PIf/2;
    velocity.z = 0;
    acceleration.z = 0;
    #endif // IS_CONSTANT_THETA

    float next_ds = 0.00001;

    float subambient_precision = 0.5;
    float ambient_precision = 0.002;

    float rs = 1;

    int bufc = 0;

    float3 pixel_direction = calculate_pixel_direction(sx, sy, width, height, polar_camera_pos, camera_quat, base_angle);

    #pragma unroll
    for(int i=0; i < 64000; i++)
    {
        #ifdef IS_CONSTANT_THETA
        position.z = M_PIf/2;
        velocity.z = 0;
        acceleration.z = 0;
        #endif // IS_CONSTANT_THETA

        float new_max = 10 * rs;
        float new_min = 3 * rs;

        float4 polar_position = generic_to_spherical(position);

        #ifdef IS_CONSTANT_THETA
        polar_position.z = M_PIf/2;
        #endif // IS_CONSTANT_THETA

        //float r_value = polar_position.y;

        float r_value = get_distance_to_object(polar_position);

        float ds = linear_val(fabs(r_value), new_min, new_max, ambient_precision, subambient_precision);

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
            return;
        }

        float4 next_position, next_velocity, next_acceleration;

        step_verlet(position, velocity, acceleration, ds, &next_position, &next_velocity, &next_acceleration);

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

        float4 polar_out = generic_to_spherical(position);

        #if (defined(GENERIC_METRIC) && defined(GENERIC_CONSTANT_THETA)) || !defined(GENERIC_METRIC) || defined(DEBUG_CONSTANT_THETA)
        polar_out.yzw = get_texture_constant_theta_rotation(pixel_direction, polar_camera_pos, polar_out);
        #endif

        positions_out[bufc] = polar_out;
        bufc++;

        if(any(isnan(position)) || any(isnan(velocity)) || any(isnan(acceleration)))
        {
            return;
        }
    }
}

#ifdef DEVICE_SIDE_ENQUEUE
__kernel
void relauncher_generic(__global struct lightray* generic_rays_in, __global struct lightray* generic_rays_out,
                        __global struct lightray* finished_rays,
                        __global int* restrict generic_count_in, __global int* restrict generic_count_out,
                        __global int* finished_count_out,
                        int fallback)
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
                                         finished_count_out);
                   });

    enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_WAIT_KERNEL,
                   ndrange_1D(offset, one, oneoffset),
                   1, &f3, NULL,
                   ^{
                        relauncher_generic(generic_rays_out, generic_rays_in,
                                           finished_rays,
                                           generic_count_out, generic_count_in,
                                           finished_count_out, fallback + 1);
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

#if 1

#define NO_KRUSKAL
#define NO_EVENT_HORIZON_CROSSING

//#define EULER_INTEGRATION
#define VERLET_INTEGRATION

__kernel
void init_rays(float4 cartesian_camera_pos, float4 camera_quat, __global struct lightray* schwarzs_rays, __global struct lightray* kruskal_rays, __global int* schwarzs_count, __global int* kruskal_count, int width, int height)
{
    #define FOV 90

    float fov_rad = (FOV / 360.f) * 2 * M_PIf;

    int cx = get_global_id(0);
    int cy = get_global_id(1);

    if(cx >= width || cy >= height)
        return;

    float nonphysical_plane_half_width = width/2;
    float nonphysical_f_stop = nonphysical_plane_half_width / tan(fov_rad/2);

    float rs = 1;
    float c = 1;

    float3 pixel_direction = (float3){cx - width/2, cy - height/2, nonphysical_f_stop};

    pixel_direction = normalize(pixel_direction);
    pixel_direction = rot_quat(pixel_direction, camera_quat);

    float3 cartesian_velocity = normalize(pixel_direction);

    float3 new_basis_x = normalize(cartesian_velocity);
    float3 new_basis_y = normalize(-cartesian_camera_pos.yzw);

    new_basis_x = rejection(new_basis_x, new_basis_y);

    float3 new_basis_z = -normalize(cross(new_basis_x, new_basis_y));

    {
        float3 cartesian_camera_new_basis = unrotate_vector(new_basis_x, new_basis_y, new_basis_z, cartesian_camera_pos.yzw);
        float3 cartesian_velocity_new_basis = unrotate_vector(new_basis_x, new_basis_y, new_basis_z, cartesian_velocity);

        cartesian_camera_pos.yzw = cartesian_camera_new_basis;
        pixel_direction = normalize(cartesian_velocity_new_basis);
    }

    float3 polar_camera = cartesian_to_polar(cartesian_camera_pos.yzw);

    float4 lightray_velocity;
    float4 lightray_spacetime_position;

    float g_metric[4] = {};

    bool is_kruskal = polar_camera.x < 20;

    float4 camera;

    ///the reason that there aren't just two fully separate branches for is_kruskal and !is_kruskal
    ///is that for some reason it absolutely murders performance, possibly for register file allocation reasons
    ///but in reality i have very little idea. Its not branch divergence though, because all rays
    ///take the same branch here, this is basically a compile time switch, because its only dependent on camera position
    ///it may also be because it defeats some sort of compiler optimisation, or just honestly anything really
    if(is_kruskal)
        camera = (float4)(rt_to_T_krus(polar_camera.x, 0), rt_to_X_krus(polar_camera.x, 0), polar_camera.y, polar_camera.z);
    else
        camera = (float4)(0, polar_camera);

    if(is_kruskal)
        calculate_metric_krus_with_r(camera, polar_camera.x, g_metric);
    else
        calculate_metric((float4)(0, polar_camera), g_metric);

    float4 co_basis = (float4){native_sqrt(-g_metric[0]), native_sqrt(g_metric[1]), native_sqrt(g_metric[2]), native_sqrt(g_metric[3])};

    float4 bT = (float4)(1/co_basis.x, 0, 0, 0); ///or bt
    float4 bX = (float4)(0, 1/co_basis.y, 0, 0); ///or br
    float4 btheta = (float4)(0, 0, 1/co_basis.z, 0);
    float4 bphi = (float4)(0, 0, 0, 1/co_basis.w);

    float4 cX = btheta;
    float4 cY = bphi;
    float4 cZ = bX;

    float3 sVx;
    float3 sVy;
    float3 sVz;

    if(is_kruskal)
    {
        float Xpolar_r = TXdTdX_to_dr_with_r(camera.x, camera.y, cX.x, cX.y, polar_camera.x);
        float Hpolar_r = TXdTdX_to_dr_with_r(camera.x, camera.y, cY.x, cY.y, polar_camera.x);
        float Ppolar_r = TXdTdX_to_dr_with_r(camera.x, camera.y, cZ.x, cZ.y, polar_camera.x);

        sVx = (float3)(Xpolar_r, cX.zw);
        sVy = (float3)(Hpolar_r, cY.zw);
        sVz = (float3)(Ppolar_r, cZ.zw);
    }
    else
    {
        sVx = cX.yzw;
        sVy = cY.yzw;
        sVz = cZ.yzw;
    }

    float3 cartesian_cx = spherical_velocity_to_cartesian_velocity(polar_camera, sVx);
    float3 cartesian_cy = spherical_velocity_to_cartesian_velocity(polar_camera, sVy);
    float3 cartesian_cz = spherical_velocity_to_cartesian_velocity(polar_camera, sVz);

    pixel_direction = unrotate_vector(normalize(cartesian_cx), normalize(cartesian_cy), normalize(cartesian_cz), pixel_direction);

    float4 pixel_x = pixel_direction.x * cX;
    float4 pixel_y = pixel_direction.y * cY;
    float4 pixel_z = pixel_direction.z * cZ;
    float4 pixel_t = 1 * bT;

    float4 vec = pixel_x + pixel_y + pixel_z + pixel_t;

    float4 pixel_N = vec;
    pixel_N = fix_light_velocity2(pixel_N, g_metric);

    lightray_velocity = pixel_N;
    lightray_spacetime_position = camera;

    //lightray_velocity.y = -lightray_velocity.y;

    ///from kruskal > to kruskal
    #define FROM_KRUSKAL 1.05
    #define TO_KRUSKAL 1.049999

    bool dirty = false;

    #ifndef NO_KRUSKAL
    if(polar_camera.x >= rs * FROM_KRUSKAL && is_kruskal)
    #else
    if(is_kruskal)
    #endif // NO_KRUSKAL
    {
        is_kruskal = false;

        ///not 100% sure this is correct?
        float4 new_pos = kruskal_position_to_schwarzs_position_with_r(lightray_spacetime_position, polar_camera.x);
        float4 new_vel = kruskal_velocity_to_schwarzs_velocity_with_r(lightray_spacetime_position, lightray_velocity, polar_camera.x);

        lightray_spacetime_position = new_pos;
        lightray_velocity = new_vel;
    }

    float4 lightray_acceleration = (float4)(0,0,0,0);

    {
        float g_partials[16] = {0};

        {
            if(is_kruskal)
            {
                calculate_metric_krus(lightray_spacetime_position, g_metric);
                calculate_partial_derivatives_krus(lightray_spacetime_position, g_partials);
            }
            else
            {
                calculate_metric(lightray_spacetime_position, g_metric);
                calculate_partial_derivatives(lightray_spacetime_position, g_partials);
            }
        }

        lightray_velocity = fix_light_velocity2(lightray_velocity, g_metric);
        lightray_acceleration = calculate_acceleration(lightray_velocity, g_metric, g_partials);
    }

    struct lightray ray;
    ray.sx = cx;
    ray.sy = cy;
    ray.position = lightray_spacetime_position;
    ray.velocity = lightray_velocity;
    ray.acceleration = lightray_acceleration;

    if(is_kruskal)
    {
        int id = cy * width + cx;

        if(id == 0)
            *kruskal_count = (height - 1) * width + width - 1;

        kruskal_rays[id] = ray;
    }
    else
    {
        int id = cy * width + cx;

        if(id == 0)
            *schwarzs_count = (height - 1) * width + width-1;

        schwarzs_rays[id] = ray;
    }
}

__kernel
void do_kruskal_rays(__global struct lightray* schwarzs_rays_in, __global struct lightray* schwarzs_rays_out,
                      __global struct lightray* kruskal_rays_in, __global struct lightray* kruskal_rays_out,
                      __global struct lightray* finished_rays,
                      __global int* schwarzs_count_in, __global int* schwarzs_count_out,
                      __global int* kruskal_count_in, __global int* kruskal_count_out,
                      __global int* finished_count_out)
{
    int id = get_global_id(0);

    if(id >= *kruskal_count_in)
        return;

    __global struct lightray* ray = &kruskal_rays_in[id];

    float4 position = ray->position;
    float4 velocity = ray->velocity;
    float4 acceleration = ray->acceleration;
    int sx = ray->sx;
    int sy = ray->sy;

    #ifdef IS_CONSTANT_THETA
    position.z = M_PIf/2;
    velocity.z = 0;
    acceleration.z = 0;
    #endif // IS_CONSTANT_THETA

    {
        float g_metric[4] = {0};
        calculate_metric_krus(position, g_metric);

        velocity = fix_light_velocity2(velocity, g_metric);
    }

    float last_ds = 1000;

    float ambient_precision = 0.01;
    float subambient_precision = 0.5;

    float rs = 1;

    float krus_radius = FROM_KRUSKAL * rs;

    float T2_m_X2_transition = r_to_T2_m_X2(krus_radius);

    float4 last_position = 0;
    float4 last_velocity = 0;
    float4 intermediate_velocity = 0;

    for(int i=0; i < 64000/125; i++)
    {
        #ifdef IS_CONSTANT_THETA
        position.z = M_PIf/2;
        velocity.z = 0;
        intermediate_velocity.z = 0;
        acceleration.z = 0;
        #endif // IS_CONSTANT_THETA

        float r_value = TX_to_r_krus(position.x, position.y);
        float ds = linear_val(r_value, 0.5f * rs, 1 * rs, 0.001f, 0.01f);

        float g_metric[4] = {};
        float g_partials[16] = {};

        #ifdef NO_EVENT_HORIZON_CROSSING
        //if((position.x * position.x - position.y * position.y) >= 0)
        if(is_radius_leq_than(position, true, rs))
        #else
        if(is_radius_leq_than(position, true, 0.5 * rs))
        #endif // NO_EVENT_HORIZON_CROSSING
        {
            int out_id = atomic_inc(finished_count_out);

            float high_r = TX_to_r_krus_highprecision(position.x, position.y);

            struct lightray out_ray;
            out_ray.sx = sx;
            out_ray.sy = sy;
            out_ray.position = position;
            out_ray.velocity = velocity;
            out_ray.acceleration = (float4)(0,0,0,0);

            ///BIT HACKY INNIT
            out_ray.position.y = high_r;

            finished_rays[out_id] = out_ray;
            return;
        }

        {
            float T = position.x;
            float X = position.y;

            ///https://www.wolframalpha.com/input/?i=%281+-+r%29+*+e%5Er%2C+r+from+0+to+3
            ///if radius >= krus_radius
            #ifdef VERLET_INTEGRATION
            if(T*T - X*X < T2_m_X2_transition && i > 0)
            #else
            if(T*T - X*X < T2_m_X2_transition)
            #endif // VERLET_INTEGRATION
            {
                float high_r = TX_to_r_krus_highprecision(position.x, position.y);

                float4 new_pos = kruskal_position_to_schwarzs_position_with_r(position, high_r);
                float4 new_vel = kruskal_velocity_to_schwarzs_velocity_with_r(position, velocity, high_r);
                float4 new_acceleration = 0;

                #ifdef VERLET_INTEGRATION
                float4 ivel = kruskal_velocity_to_schwarzs_velocity_with_r(position, intermediate_velocity, high_r);

                float last_high_r = TX_to_r_krus_highprecision(last_position.x, last_position.y);

                float4 last_new_pos = kruskal_position_to_schwarzs_position_with_r(last_position, last_high_r);
                float4 last_new_vel = kruskal_velocity_to_schwarzs_velocity_with_r(last_position, last_velocity, last_high_r);

                //float4 old_lightray_acceleration = ((position - last_new_pos) - (last_velocity * last_ds)) / (0.5f * last_ds * last_ds);
                //float4 old_lightray_acceleration = ((velocity - last_new_vel) / last_ds);
                float4 old_lightray_acceleration = ((ivel - last_new_vel) / last_ds);
                new_acceleration = ((new_vel - last_new_vel) / (0.5f * last_ds)) - old_lightray_acceleration;
                #endif // VERLET_INTEGRATION

                struct lightray out_ray;
                out_ray.sx = sx;
                out_ray.sy = sy;
                out_ray.position = new_pos;
                out_ray.velocity = new_vel;
                out_ray.acceleration = new_acceleration;

                int fid = atomic_inc(schwarzs_count_in);

                schwarzs_rays_in[fid] = out_ray;

                return;
            }
        }

        #ifdef VERLET_INTEGRATION
        last_position = position;
        last_velocity = velocity;
        #endif // VERLET_INTEGRATION

        #ifdef EULER_INTEGRATION
        calculate_metric_krus(position, g_metric);
        calculate_partial_derivatives_krus(position, g_partials);

        velocity = fix_light_velocity2(velocity, g_metric);

        float4 lacceleration = calculate_acceleration(velocity, g_metric, g_partials);

        velocity += lacceleration * ds;

        position += velocity * ds;
        #endif // EULER_INTEGRATION

        #ifdef VERLET_INTEGRATION
        float4 next_position = position + velocity * ds + 0.5f * acceleration * ds * ds;
        float4 intermediate_next_velocity = velocity + acceleration * ds;

        calculate_metric_krus(next_position, g_metric);
        calculate_partial_derivatives_krus(next_position, g_partials);

        intermediate_next_velocity = fix_light_velocity2(intermediate_next_velocity, g_metric);

        float4 next_acceleration = calculate_acceleration(intermediate_next_velocity, g_metric, g_partials);
        float4 next_velocity = velocity + 0.5f * (acceleration + next_acceleration) * ds;

        #ifdef IS_CONSTANT_THETA
        next_position.z = 0;
        next_velocity.z = 0;
        intermediate_next_velocity.z = 0;
        next_acceleration.z = 0;
        #endif // IS_CONSTANT_THETA

        last_ds = ds;

        position = next_position;
        //velocity = fix_light_velocity2(next_velocity, g_metric);
        velocity = next_velocity;
        intermediate_velocity = intermediate_next_velocity;
        acceleration = next_acceleration;
        #endif // VERLET_INTEGRATION
    }

    int out_id = atomic_inc(kruskal_count_out);

    struct lightray out_ray;
    out_ray.sx = sx;
    out_ray.sy = sy;
    out_ray.position = position;
    out_ray.velocity = velocity;
    out_ray.acceleration = acceleration;

    kruskal_rays_out[out_id] = out_ray;
}

/*float ddt = 0;
float ddr = 0;
float ddp = 0;

{
    float dr = intermediate_next_velocity.y;
    float r = next_position.y;

    float t = next_position.x;
    float dt = intermediate_next_velocity.x;

    float p = next_position.w;
    float dp = intermediate_next_velocity.w;

    float Q = 0;
    float q = 0;

    ddt = dr * (q * r * Q + 2 * (Q * Q - r) * dt) / (r * ((r - 2) * r + Q * Q));
    ddr = (((r - 2) * r + Q * Q) * (q * r * Q * dt + r * r * r * r * dp * dp + (Q * Q - r) * dt * dt) / (r * r * r * r * r)) + (r - Q * Q) * dr * dr / (r * ((r - 2) * r + Q * Q));
    ddp = - 2 * dp * dr / r;
}

float4 next_acceleration = {ddt, ddr, 0, ddp};*/

__kernel
void do_schwarzs_rays(__global struct lightray* schwarzs_rays_in, __global struct lightray* schwarzs_rays_out,
                      __global struct lightray* kruskal_rays_in, __global struct lightray* kruskal_rays_out,
                      __global struct lightray* finished_rays,
                      __global int* schwarzs_count_in, __global int* schwarzs_count_out,
                      __global int* kruskal_count_in, __global int* kruskal_count_out,
                      __global int* finished_count_out)
{
    int id = get_global_id(0);

    if(id >= *schwarzs_count_in)
        return;

    __global struct lightray* ray = &schwarzs_rays_in[id];

    float4 position = ray->position;
    float4 velocity = ray->velocity;
    float4 acceleration = ray->acceleration;

    #ifdef IS_CONSTANT_THETA
    position.z = M_PIf/2;
    velocity.z = 0;
    acceleration.z = 0;
    #endif // IS_CONSTANT_THETA

    int sx = ray->sx;
    int sy = ray->sy;

    {
        float g_metric[4] = {0};
        calculate_metric(position, g_metric);

        velocity = fix_light_velocity2(velocity, g_metric);
    }

    float last_ds = 1000;

    ///results:
    ///subambient_precision can't go above 0.5 much while in verlet mode without the size of the event horizon changing
    ///in euler mode this is actually already too low

    ///ambient precision however looks way too low at 0.01, testing up to 0.3 showed no noticable difference, needs more precise tests though
    ///only in the case without kruskals and event horizon crossings however, any precision > 0.01 is insufficient in that case
    float ambient_precision = 0.01;
    float subambient_precision = 0.5;

    #ifdef NO_EVENT_HORIZON_CROSSING
    #ifdef NO_KRUSKAL
    ambient_precision = 0.5;
    #endif // NO_KRUSKAL
    #endif // NO_EVENT_HORIZON_CROSSING

    float rs = 1;

    float4 last_position = 0;
    float4 last_velocity = 0;
    float4 intermediate_velocity = 0;

    for(int i=0; i < 64000/125; i++)
    {
        #ifdef IS_CONSTANT_THETA
        position.z = M_PIf/2;
        velocity.z = 0;
        intermediate_velocity.z = 0;
        acceleration.z = 0;
        #endif // IS_CONSTANT_THETA

        float new_max = 6 * rs;
        float new_min = FROM_KRUSKAL * rs;

        float r_value = position.y;

        float ds = linear_val(r_value, new_min, new_max, ambient_precision, subambient_precision);

        #if 1
        if(r_value >= new_max)
        {
            //ds = 0.5 * pow(r_value - new_max, 0.8) + subambient_precision;

            //ds = 1 * sqrt(r_value - new_max) + subambient_precision;

            float end_max = 4000000;

            /*float mixd = linear_mix(r_value, new_max, end_max);
            ds = mix(subambient_precision, end_max/10, mixd);*/

            //ds = linear_val(r_value, new_max, end_max, subambient_precision, end_max/10);

            ds = 0.1 * (r_value - new_max) + subambient_precision;


            //ds = 0.1 * pow(r_value - new_max, 0.999) + subambient_precision;
            //ds = 0.05 * (r_value - new_max) + subambient_precision;
        }
        #endif // 0

        float g_metric[4] = {};
        float g_partials[16] = {};

        if(position.y >= 4000000 || position.y <= rs)
        {
            if(position.y <= rs)
                return;

            int out_id = atomic_inc(finished_count_out);

            struct lightray out_ray;
            out_ray.sx = sx;
            out_ray.sy = sy;
            out_ray.position = position;
            out_ray.velocity = velocity;
            out_ray.acceleration = 0;

            finished_rays[out_id] = out_ray;
            return;
        }

        #ifndef NO_KRUSKAL
        #ifdef VERLET_INTEGRATION
        if(position.y <= rs * TO_KRUSKAL && i > 0)
        #else
        if(position.y <= rs * TO_KRUSKAL)
        #endif // VERLET_INTEGRATION
        {
            float4 new_pos = schwarzs_position_to_kruskal_position((float4)(0.f, position.yzw));
            float4 new_vel = schwarzs_velocity_to_kruskal_velocity((float4)(0.f, position.yzw), velocity);
            float4 new_acceleration = 0;

            #ifdef VERLET_INTEGRATION
            float4 last_new_pos = schwarzs_position_to_kruskal_position((float4)(0, last_position.yzw));
            float4 last_new_vel = schwarzs_velocity_to_kruskal_velocity((float4)(0, last_position.yzw), last_velocity);

            float4 ivel = schwarzs_velocity_to_kruskal_velocity((float4)(0.f, position.yzw), intermediate_velocity);

            //float4 old_lightray_acceleration = ((lightray_spacetime_position - last_new_pos) - (last_new_vel * last_ds)) / (0.5 * last_ds * last_ds); ///worst
            //float4 old_lightray_acceleration = ((lightray_velocity - last_new_vel) / last_ds); ///second best, but seems fine
            float4 old_lightray_acceleration = ((ivel - last_new_vel) / last_ds); ///technically the best
            new_acceleration = ((new_vel - last_new_vel) / (0.5 * last_ds)) - old_lightray_acceleration;
            #endif // VERLET_INTEGRATION

            struct lightray out_ray;
            out_ray.sx = sx;
            out_ray.sy = sy;
            ///you know, could use old pos, vel, and acceleration, slightly less work
            out_ray.position = new_pos;
            out_ray.velocity = new_vel;
            out_ray.acceleration = new_acceleration;

            int kid = atomic_inc(kruskal_count_in);

            kruskal_rays_in[kid] = out_ray;

            return;
        }
        #endif // NO_KRUSKAL

        #ifdef VERLET_INTEGRATION
        last_position = position;
        last_velocity = velocity;
        #endif // VERLET_INTEGRATION

        #ifdef EULER_INTEGRATION
        calculate_metric(position, g_metric);
        calculate_partial_derivatives(position, g_partials);

        velocity = fix_light_velocity2(velocity, g_metric);

        float4 lacceleration = calculate_acceleration(velocity, g_metric, g_partials);
        velocity += lacceleration * ds;

        position += velocity * ds;
        #endif // EULER_INTEGRATION

        #ifdef VERLET_INTEGRATION
        float4 next_position = position + velocity * ds + 0.5f * acceleration * ds * ds;
        float4 intermediate_next_velocity = velocity + acceleration * ds;

        calculate_metric(next_position, g_metric);
        calculate_partial_derivatives(next_position, g_partials);

        ///1ms
        intermediate_next_velocity = fix_light_velocity2(intermediate_next_velocity, g_metric);

        float4 next_acceleration = calculate_acceleration(intermediate_next_velocity, g_metric, g_partials);
        float4 next_velocity = velocity + 0.5f * (acceleration + next_acceleration) * ds;

        #ifdef IS_CONSTANT_THETA
        next_position.z = 0;
        next_velocity.z = 0;
        intermediate_next_velocity.z = 0;
        next_acceleration.z = 0;
        #endif // IS_CONSTANT_THETA

        last_ds = ds;

        position = next_position;
        //velocity = fix_light_velocity2(next_velocity, g_metric);
        velocity = next_velocity;
        intermediate_velocity = intermediate_next_velocity;
        acceleration = next_acceleration;

        #endif // VERLET_INTEGRATION
    }

    int out_id = atomic_inc(schwarzs_count_out);

    struct lightray out_ray;
    out_ray.sx = sx;
    out_ray.sy = sy;
    out_ray.position = position;
    out_ray.velocity = velocity;
    out_ray.acceleration = acceleration;

    schwarzs_rays_out[out_id] = out_ray;
}

__kernel
void kruskal_launcher(__global struct lightray* schwarzs_rays_in, __global struct lightray* schwarzs_rays_out,
                      __global struct lightray* kruskal_rays_in, __global struct lightray* kruskal_rays_out,
                      __global struct lightray* finished_rays,
                      __global int* schwarzs_count_in, __global int* schwarzs_count_out,
                      __global int* kruskal_count_in, __global int* kruskal_count_out,
                      __global int* finished_count_out)
{
    int kruskal_count = *kruskal_count_in;

    if(kruskal_count == 0)
        return;

    enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_WAIT_KERNEL,
                   ndrange_1D(0, kruskal_count, 256),
                   0, NULL, NULL,
                   ^{
                        do_kruskal_rays(schwarzs_rays_in, schwarzs_rays_out,
                                        kruskal_rays_in, kruskal_rays_out,
                                        finished_rays,
                                        schwarzs_count_in, schwarzs_count_out,
                                        kruskal_count_in, kruskal_count_out,
                                        finished_count_out);
                   });
}

__kernel
void relauncher(__global struct lightray* schwarzs_rays_in, __global struct lightray* schwarzs_rays_out,
                      __global struct lightray* kruskal_rays_in, __global struct lightray* kruskal_rays_out,
                      __global struct lightray* finished_rays,
                      __global int* schwarzs_count_in, __global int* schwarzs_count_out,
                      __global int* kruskal_count_in, __global int* kruskal_count_out,
                      __global int* finished_count_out,
                      int width, int height, int fallback)
{
    ///failed to converge
    if(fallback > 125)
        return;

    if((*schwarzs_count_in) == 0 && (*kruskal_count_in) == 0)
        return;

    int schwarzs_count = *schwarzs_count_in;
    //int kruskal_count = *kruskal_count_in;

    int offset = 0;
    int loffset = 256;

    /*if((dim % loffset) != 0)
    {
        int rem = dim % loffset;

        dim -= rem;
        dim += loffset;
    }*/

    int one = 1;
    int oneoffset = 1;

    clk_event_t f1;

    enqueue_kernel(get_default_queue(),CLK_ENQUEUE_FLAGS_NO_WAIT,
                   ndrange_1D(offset, one, oneoffset),
                   0, NULL, &f1,
                   ^{
                       *schwarzs_count_out = 0;
                       *kruskal_count_out = 0;
                   });

    clk_event_t f3;

    enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_NO_WAIT,
                   ndrange_1D(offset, schwarzs_count, loffset),
                   1, &f1, &f3,
                   ^{
                        do_schwarzs_rays(schwarzs_rays_in, schwarzs_rays_out,
                                         kruskal_rays_in, kruskal_rays_out,
                                         finished_rays,
                                         schwarzs_count_in, schwarzs_count_out,
                                         kruskal_count_in, kruskal_count_out,
                                         finished_count_out);
                   });

    release_event(f1);

    clk_event_t f4;

    enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_NO_WAIT,
                   ndrange_1D(offset, 1, 1),
                   1, &f3, &f4,
                   ^{
                        kruskal_launcher(schwarzs_rays_out, schwarzs_rays_in,
                                        kruskal_rays_in, kruskal_rays_out,
                                        finished_rays,
                                        schwarzs_count_out, schwarzs_count_in,
                                        kruskal_count_in, kruskal_count_out,
                                        finished_count_out);
                   });

    enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_WAIT_KERNEL,
                   ndrange_1D(offset, one, oneoffset),
                   1, &f4, NULL,
                   ^{
                        relauncher(schwarzs_rays_out, schwarzs_rays_in,
                                   kruskal_rays_out, kruskal_rays_in,
                                   finished_rays,
                                   schwarzs_count_out, schwarzs_count_in,
                                   kruskal_count_out,kruskal_count_in,
                                   finished_count_out, width, height, fallback + 1);
                   });

    release_event(f3);
    release_event(f4);
}

__kernel
void calculate_texture_coordinates(__global struct lightray* finished_rays, __global int* finished_count_in, __global float2* texture_coordinates, int width, int height, float4 polar_camera_pos, float4 camera_quat, float2 base_angle)
{
    int id = get_global_id(0);

    if(id >= *finished_count_in)
        return;

    struct lightray* ray = &finished_rays[id];

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

    float3 pixel_direction = calculate_pixel_direction(sx, sy, width, height, polar_camera_pos, camera_quat, base_angle);

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

float4 read_mipmap(image2d_t mipmap1, image2d_t mipmap2, float position_y, sampler_t sam, float2 pos, float lod)
{
    return position_y >= 0 ? read_imagef(mipmap1, sam, pos, lod) : read_imagef(mipmap2, sam, pos, lod);
}

#define MIPMAP_CONDITIONAL(x) ((position.y >= 0) ? x(mip_background) : x(mip_background2))

__kernel
void render(__global struct lightray* finished_rays, __global int* finished_count_in, __write_only image2d_t out,
            __read_only image2d_t mip_background, __read_only image2d_t mip_background2,
            int width, int height, __global float2* texture_coordinates, sampler_t sam)
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

    int maxProbes = 8;

    iProbes = min(iProbes, maxProbes);

    if(iProbes < fProbes)
        minorRadius = 2 * majorRadius / (iProbes + 1);

    float levelofdetail = log2(minorRadius);

    int maxLod = MIPMAP_CONDITIONAL(get_image_num_mip_levels) - 1;

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

        end_result = read_mipmap(mip_background, mip_background2, position.y, sam, (float2){sxf, syf}, levelofdetail);
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

            float4 fval = read_mipmap(mip_background, mip_background2, position.y, sam, (float2){cu, cv}, levelofdetail);

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

    end_result.xyz = lin_to_srgb(lin_result);

    #endif // REDSHIFT

    #ifdef LINEAR_FRAMEBUFFER
    end_result.xyz = srgb_to_lin(end_result.xyz);
    #endif // LINEAR_FRAMEBUFFER

    write_imagef(out, (int2){sx, sy}, end_result);
}
#endif // 0

__kernel
void do_raytracing_multicoordinate(__write_only image2d_t out, float ds_, float4 cartesian_camera_pos, float4 camera_quat, __read_only image2d_t background)
{
    #define FOV 90

    float fov_rad = (FOV / 360.f) * 2 * M_PIf;

    int cx = get_global_id(0);
    int cy = get_global_id(1);

    float width = get_image_width(out);
    float height = get_image_height(out);

    if(cx >= width-1 || cy >= height-1)
        return;

    float nonphysical_plane_half_width = width/2;
    float nonphysical_f_stop = nonphysical_plane_half_width / tan(fov_rad/2);

    float rs = 1;
    float c = 1;

    float3 pixel_direction = (float3){cx - width/2, cy - height/2, nonphysical_f_stop};

    pixel_direction = fast_normalize(pixel_direction);
    pixel_direction = rot_quat(pixel_direction, camera_quat);

    float3 cartesian_velocity = fast_normalize(pixel_direction);

    float3 new_basis_x = fast_normalize(cartesian_velocity);
    float3 new_basis_y = fast_normalize(-cartesian_camera_pos.yzw);

    new_basis_x = rejection(new_basis_x, new_basis_y);

    float3 new_basis_z = -fast_normalize(cross(new_basis_x, new_basis_y));

    {
        float3 cartesian_camera_new_basis = unrotate_vector(new_basis_x, new_basis_y, new_basis_z, cartesian_camera_pos.yzw);
        float3 cartesian_velocity_new_basis = unrotate_vector(new_basis_x, new_basis_y, new_basis_z, cartesian_velocity);

        cartesian_camera_pos.yzw = cartesian_camera_new_basis;
        pixel_direction = fast_normalize(cartesian_velocity_new_basis);
    }

    float3 polar_camera = cartesian_to_polar(cartesian_camera_pos.yzw);

    float4 lightray_velocity;
    float4 lightray_spacetime_position;

    float g_metric[4] = {};

    bool is_kruskal = polar_camera.x < 20;

    float4 camera;

    ///the reason that there aren't just two fully separate branches for is_kruskal and !is_kruskal
    ///is that for some reason it absolutely murders performance, possibly for register file allocation reasons
    ///but in reality i have very little idea. Its not branch divergence though, because all rays
    ///take the same branch here, this is basically a compile time switch, because its only dependent on camera position
    ///it may also be because it defeats some sort of compiler optimisation, or just honestly anything really
    if(is_kruskal)
        camera = (float4)(rt_to_T_krus(polar_camera.x, 0), rt_to_X_krus(polar_camera.x, 0), polar_camera.y, polar_camera.z);
    else
        camera = (float4)(0, polar_camera);

    if(is_kruskal)
        calculate_metric_krus_with_r(camera, polar_camera.x, g_metric);
    else
        calculate_metric((float4)(0, polar_camera), g_metric);

    float4 co_basis = (float4){native_sqrt(-g_metric[0]), native_sqrt(g_metric[1]), native_sqrt(g_metric[2]), native_sqrt(g_metric[3])};

    float4 bT = (float4)(1/co_basis.x, 0, 0, 0); ///or bt
    float4 bX = (float4)(0, 1/co_basis.y, 0, 0); ///or br
    float4 btheta = (float4)(0, 0, 1/co_basis.z, 0);
    float4 bphi = (float4)(0, 0, 0, 1/co_basis.w);

    float4 cX = btheta;
    float4 cY = bphi;
    float4 cZ = bX;

    float3 sVx;
    float3 sVy;
    float3 sVz;

    if(is_kruskal)
    {
        float Xpolar_r = TXdTdX_to_dr_with_r(camera.x, camera.y, cX.x, cX.y, polar_camera.x);
        float Hpolar_r = TXdTdX_to_dr_with_r(camera.x, camera.y, cY.x, cY.y, polar_camera.x);
        float Ppolar_r = TXdTdX_to_dr_with_r(camera.x, camera.y, cZ.x, cZ.y, polar_camera.x);

        sVx = (float3)(Xpolar_r, cX.zw);
        sVy = (float3)(Hpolar_r, cY.zw);
        sVz = (float3)(Ppolar_r, cZ.zw);
    }
    else
    {
        sVx = cX.yzw;
        sVy = cY.yzw;
        sVz = cZ.yzw;
    }

    float3 cartesian_cx = spherical_velocity_to_cartesian_velocity(polar_camera, sVx);
    float3 cartesian_cy = spherical_velocity_to_cartesian_velocity(polar_camera, sVy);
    float3 cartesian_cz = spherical_velocity_to_cartesian_velocity(polar_camera, sVz);

    ///this is totally screwed in -r space
    pixel_direction = unrotate_vector(fast_normalize(cartesian_cx), fast_normalize(cartesian_cy), fast_normalize(cartesian_cz), pixel_direction);

    float4 pixel_x = pixel_direction.x * cX;
    float4 pixel_y = pixel_direction.y * cY;
    float4 pixel_z = pixel_direction.z * cZ;

    float4 vec = pixel_x + pixel_y + pixel_z;

    float4 pixel_N = vec / (dot(lower_index(vec, g_metric), vec));
    pixel_N = fix_light_velocity2(pixel_N, g_metric);

    lightray_velocity = pixel_N;
    lightray_spacetime_position = camera;

    //lightray_velocity.y = -lightray_velocity.y;

    float ambient_precision = 0.001;
    float subambient_precision = 0.5;

    ///TODO: need to use external observer time, currently using sim time!!
    float max_ds = 0.001;
    float min_ds = 0.001;

    #undef NO_EVENT_HORIZON_CROSSING
    #undef VERLET_INTEGRATION
    #undef EULER_INTEGRATION
    #undef NO_KRUSKAL

    //#define NO_EVENT_HORIZON_CROSSING

    #ifdef NO_EVENT_HORIZON_CROSSING
    ambient_precision = 0.01;
    max_ds = 0.01;
    min_ds = 0.01;
    #endif // NO_EVENT_HORIZON_CROSSING

    #define EULER_INTEGRATION
    //#define VERLET_INTEGRATION

    #ifdef VERLET_INTEGRATION
    #ifdef NO_EVENT_HORIZON_CROSSING
    ambient_precision = 0.05;
    max_ds = 0.05;
    min_ds = 0.05;
    #endif // NO_EVENT_HORIZON_CROSSING
    #endif // VERLET_INTEGRATION


    /*float min_radius = rs * 1.1;
    float max_radius = rs * 1.6;*/

    float min_radius = 0.7 * rs;
    float max_radius = 1.1 * rs;

    //#define NO_KRUSKAL

    ///from kruskal > to kruskal
    #define FROM_KRUSKAL 1.05
    #define TO_KRUSKAL 1.049999

    #ifndef NO_KRUSKAL
    if(polar_camera.x >= rs * FROM_KRUSKAL && is_kruskal)
    #else
    if(is_kruskal)
    #endif // NO_KRUSKAL
    {
        is_kruskal = false;

        lightray_spacetime_position.x = 0;

        ///not 100% sure this is correct?
        float4 new_pos = kruskal_position_to_schwarzs_position_with_r(lightray_spacetime_position, polar_camera.x);
        float4 new_vel = kruskal_velocity_to_schwarzs_velocity_with_r(lightray_spacetime_position, lightray_velocity, polar_camera.x);

        lightray_spacetime_position = new_pos;
        lightray_velocity = new_vel;
    }

    float4 lightray_acceleration = (float4)(0,0,0,0);

    //if(is_kruskal)
    {
        float g_partials[16] = {0};

        {
            if(is_kruskal)
                calculate_metric_krus(lightray_spacetime_position, g_metric);
            else
                calculate_metric(lightray_spacetime_position, g_metric);

            if(is_kruskal)
                calculate_partial_derivatives_krus(lightray_spacetime_position, g_partials);
            else
                calculate_partial_derivatives(lightray_spacetime_position, g_partials);
        }

        lightray_velocity = fix_light_velocity2(lightray_velocity, g_metric);
        lightray_acceleration = calculate_acceleration(lightray_velocity, g_metric, g_partials);
    }

    float4 last_position = lightray_spacetime_position;
    float4 last_velocity = lightray_velocity;
    float4 intermediate_velocity = lightray_velocity;
    //float4 last_acceleration = (float4)(0,0,0,0);
    float last_ds = 1;

    float krus_radius = FROM_KRUSKAL * rs;

    float T2_m_X2_transition = r_to_T2_m_X2(krus_radius);
    float krus_inner_cutoff = r_to_T2_m_X2(0.5 * rs);

    int bad_rays = 0;

    for(int it=0; it < 60000; it++)
    {
        float g_partials[16] = {0};

        #ifndef NO_KRUSKAL
        if(!is_kruskal)
        {
            if(lightray_spacetime_position.y < rs * TO_KRUSKAL)
            {
                is_kruskal = true;

                float4 new_pos = schwarzs_position_to_kruskal_position((float4)(0.f, lightray_spacetime_position.yzw));
                float4 new_vel = schwarzs_velocity_to_kruskal_velocity((float4)(0.f, lightray_spacetime_position.yzw), lightray_velocity);

                #ifdef VERLET_INTEGRATION
                float4 last_new_pos = schwarzs_position_to_kruskal_position((float4)(0, last_position.yzw));
                float4 last_new_vel = schwarzs_velocity_to_kruskal_velocity((float4)(0, last_position.yzw), last_velocity);

                float4 ivel = schwarzs_velocity_to_kruskal_velocity((float4)(0.f, lightray_spacetime_position.yzw), intermediate_velocity);
                #endif // VERLET_INTEGRATION

                lightray_spacetime_position = new_pos;
                lightray_velocity = new_vel;

                #ifdef VERLET_INTEGRATION
                last_position = last_new_pos;
                last_velocity = last_new_vel;

                //float4 old_lightray_acceleration = ((lightray_spacetime_position - last_position) - (last_velocity * last_ds)) / (0.5 * last_ds * last_ds); ///worst
                //float4 old_lightray_acceleration = ((lightray_velocity - last_velocity) / last_ds); ///second best, but seems fine
                float4 old_lightray_acceleration = ((ivel - last_velocity) / last_ds); ///technically the best
                lightray_acceleration = ((lightray_velocity - last_velocity) / (0.5 * last_ds)) - old_lightray_acceleration;
                #endif // VERLET_INTEGRATION
            }
        }
        #endif // NO_KRUSKAL

        if(is_kruskal)
        {
            float T = lightray_spacetime_position.x;
            float X = lightray_spacetime_position.y;

            ///https://www.wolframalpha.com/input/?i=%281+-+r%29+*+e%5Er%2C+r+from+0+to+3
            ///if radius >= krus_radius
            if(T*T - X*X < T2_m_X2_transition)
            {
                is_kruskal = false;

                float high_r = TX_to_r_krus_highprecision(lightray_spacetime_position.x, lightray_spacetime_position.y);

                float4 new_pos = kruskal_position_to_schwarzs_position_with_r(lightray_spacetime_position, high_r);
                float4 new_vel = kruskal_velocity_to_schwarzs_velocity_with_r(lightray_spacetime_position, lightray_velocity, high_r);

                #ifdef VERLET_INTEGRATION
                #ifndef NO_KRUSKAL
                float4 ivel = kruskal_velocity_to_schwarzs_velocity_with_r(lightray_spacetime_position, intermediate_velocity, high_r);
                #endif // NO_KRUSKAL
                #endif // VERLET_INTEGRATION

                lightray_spacetime_position = new_pos;
                lightray_velocity = new_vel;

                #ifdef VERLET_INTEGRATION
                #ifndef NO_KRUSKAL
                float last_high_r = TX_to_r_krus_highprecision(last_position.x, last_position.y);

                float4 last_new_pos = kruskal_position_to_schwarzs_position_with_r(last_position, last_high_r);
                float4 last_new_vel = kruskal_velocity_to_schwarzs_velocity_with_r(last_position, last_velocity, last_high_r);

                last_position = last_new_pos;
                last_velocity = last_new_vel;

                //float4 old_lightray_acceleration = ((lightray_spacetime_position - last_position) - (last_velocity * last_ds)) / (0.5 * last_ds * last_ds);
                //float4 old_lightray_acceleration = ((lightray_velocity - last_velocity) / last_ds);
                float4 old_lightray_acceleration = ((ivel - last_velocity) / last_ds);
                lightray_acceleration = ((lightray_velocity - last_velocity) / (0.5 * last_ds)) - old_lightray_acceleration;
                #endif // NO_KRUSKAL
                #endif // VERLET_INTEGRATION
            }
        }

        last_position = lightray_spacetime_position;
        last_velocity = lightray_velocity;

        #ifdef NO_EVENT_HORIZON_CROSSING
        if(is_kruskal)
        {
            float T = lightray_spacetime_position.x;
            float X = lightray_spacetime_position.y;

            if(T*T - X*X >= 0)
            {
                write_imagef(out, (int2)(cx, cy), (float4)(0,0,0,1));
                return;
            }
        }
        else
        {
            if(lightray_spacetime_position.y <= rs)
            {
                write_imagef(out, (int2)(cx, cy), (float4)(0,0,0,1));
                return;
            }
        }
        #endif

        float ds = 0;

        if(!is_kruskal)
        {
            float new_max = 4 * rs;
            float new_min = 1.1 * rs;

            float r_value = lightray_spacetime_position.y;

            float interp = clamp(r_value, new_min, new_max);
            float frac = (interp - new_min) / (new_max - new_min);

            ds = mix(ambient_precision, subambient_precision, frac);

            if(r_value >= new_max)
            {
                //float multiplier = linear_val(r_value, new_max, new_max * 10, 0.1, 1);

                ds = 0.1 * (r_value - new_max) + subambient_precision;

                //ds = (r_value - new_max) * multiplier + subambient_precision;
            }

            /*float ds_at_max = (new_max * 10) * 0.1 + subambient_precision;

            if(r_value >= new_max * 10)
            {
                ds = (r_value - new_max * 10) * 0.5 + ds_at_max;
            }*/

            if(r_value >= 80)
            {
                /*float nfrac = (r_value - 20) / 100;

                nfrac = clamp(nfrac, 0.f, 1.f);

                ds = mix(subambient_precision, 10, nfrac);*/

                //ds = r_value / 10;
            }
        }
        else
        {
            /*float interp = clamp(r_value, min_radius, max_radius);
            float frac = (interp - min_radius) / (max_radius - min_radius);
            ds = mix(max_ds, min_ds, frac);*/

            ds = min_ds;

            /*if(is_radius_leq_than(lightray_spacetime_position, is_kruskal, 0.2))
            {
                ds = min_ds/100;
            }

            if(is_radius_leq_than(lightray_spacetime_position, is_kruskal, 0.01))
            {
                ds = min_ds / 10000;
            }*/
        }

        //ds = 0.1;

        if(!is_radius_leq_than(lightray_spacetime_position, is_kruskal, 400000) || is_radius_leq_than(lightray_spacetime_position, is_kruskal, 0.5))
        {
            float r_value = 0;

            if(is_kruskal)
            {
                r_value = TX_to_r_krus_highprecision(lightray_spacetime_position.x, lightray_spacetime_position.y);
            }
            else
            {
                r_value = lightray_spacetime_position.y;
            }

            float3 cart_here = polar_to_cartesian((float3)(r_value, lightray_spacetime_position.zw));

            cart_here = rotate_vector(new_basis_x, new_basis_y, new_basis_z, cart_here);

            float3 npolar = cartesian_to_polar(cart_here);

            float thetaf = fmod(npolar.y, 2 * M_PIf);
            float phif = npolar.z;

            if(thetaf >= M_PIf)
            {
                phif += M_PIf;
                thetaf -= M_PIf;
            }

            phif = fmod(phif, 2 * M_PIf);

            float sx = (phif) / (2 * M_PIf);
            float sy = thetaf / M_PIf;

            sampler_t sam = CLK_NORMALIZED_COORDS_TRUE |
                            CLK_ADDRESS_REPEAT |
                            CLK_FILTER_LINEAR;

            float4 val = read_imagef(background, sam, (float2){sx, sy});

            if(r_value < 1)
            {
                val = (float4)(0,0,0,1);

                int x_half = fabs(fmod((sx + 1) * 10, 1)) > 0.5 ? 1 : 0;
                int y_half = fabs(fmod((sy + 1) * 10, 1)) > 0.5 ? 1 : 0;

                //val.x = (x_half + y_half) % 2;

                val.x = x_half;
                val.y = y_half;

                if(sy < 0.1 || sy >= 0.9)
                {
                    val.x = 0;
                    val.y = 0;
                    val.z = 1;
                }
            }

            write_imagef(out, (int2){cx, cy}, val);
            return;
        }

        #ifndef NO_EVENT_HORIZON_CROSSING
        if(is_kruskal)
        {
            /*float ftol = 0.001;

            if(fabs(g_partials[2 * 4 + 0]) < ftol && fabs(g_partials[2 * 4 + 1]) < ftol && fabs(g_partials[3 * 4 + 2]) < ftol)
                bad_rays++;

            if(bad_rays >= 3)
            {
                write_imagef(out, (int2)(cx, cy), (float4)(1, 0, 1, 1));
                return;
            }*/
        }
        #endif // NO_HORIZON_CROSSING

        #ifdef EULER_INTEGRATION
        {
            if(is_kruskal)
            {
                calculate_metric_krus(lightray_spacetime_position, g_metric);
                calculate_partial_derivatives_krus(lightray_spacetime_position, g_partials);
            }
            else
            {
                calculate_metric(lightray_spacetime_position, g_metric);
                calculate_partial_derivatives(lightray_spacetime_position, g_partials);
            }
        }

        float4 acceleration = calculate_acceleration(lightray_velocity, g_metric, g_partials);

        lightray_velocity += acceleration * ds;
        lightray_velocity = fix_light_velocity2(lightray_velocity, g_metric);

        lightray_spacetime_position += lightray_velocity * ds;

        #endif // EULER_INTEGRATION

        #ifdef VERLET_INTEGRATION
        float4 next_position = lightray_spacetime_position + lightray_velocity * ds + 0.5 * lightray_acceleration * ds * ds;
        float4 intermediate_next_velocity = lightray_velocity + lightray_acceleration * ds;

        ///try moving this out of the loop
        {
            if(is_kruskal)
            {
                calculate_metric_krus(next_position, g_metric);
                calculate_partial_derivatives_krus(next_position, g_partials);
            }
            else
            {
                calculate_metric(next_position, g_metric);
                calculate_partial_derivatives(next_position, g_partials);
            }
        }

        intermediate_next_velocity = fix_light_velocity2(intermediate_next_velocity, g_metric);

        float4 next_acceleration = calculate_acceleration(intermediate_next_velocity, g_metric, g_partials);
        float4 next_velocity = lightray_velocity + 0.5 * (lightray_acceleration + next_acceleration) * ds;

        last_ds = ds;

        lightray_spacetime_position = next_position;
        //lightray_velocity = fix_light_velocity2(next_velocity, g_metric);
        lightray_velocity = next_velocity;
        lightray_acceleration = next_acceleration;
        intermediate_velocity = intermediate_next_velocity;
        #endif // VERLET_INTEGRATION

        /*if(cx == width/2 && cy == height/2)
        {
            float ds = calculate_ds(lightray_velocity, g_metric);

            printf("DS %f\n", ds);
        }*/
    }

    write_imagef(out, (int2){cx, cy}, (float4){0, 1, 0, 1});
}
