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

float spacetime_metric_value(int i, int k, int l, float g_partial[16])
{
    if(i != k)
        return 0;

    return g_partial[l * 4 + i];
}

/*struct light_ray
{
    float4 spacetime_velocity;
    float4 spacetime_position;
};*/

/*float3 cartesian_to_schwarz(float3 in)
{

}*/

/*

*/

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

///ds2 = guv dx^u dx^v
float4 fix_light_velocity2(float4 v, float g_metric[])
{
    ///g_metric[1] * v[1]^2 + g_metric[2] * v[2]^2 + g_metric[3] * v[3]^2 = -g_metric[0] * v[0]^2

    float3 vmetric = {g_metric[1], g_metric[2], g_metric[3]};

    float tvl_2 = dot(vmetric, v.yzw * v.yzw) / -g_metric[0];

    v.x = native_sqrt(tvl_2);

    return v;
}

#define IS_CONSTANT_THETA

void calculate_metric(float4 spacetime_position, float g_metric_out[])
{
    float rs = 1;
    float c = 1;

    float r = spacetime_position.y;

    #ifndef IS_CONSTANT_THETA
    float theta = spacetime_position.z;
    #else
    float theta = M_PI/2;
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

float3 spherical_acceleration_to_cartesian_acceleration(float3 p, float3 dp, float3 ddp)
{
    float r = p.x;
    float dr = dp.x;
    float ddr = ddp.x;

    float x = p.y;
    float dx = dp.y;
    float ddx = ddp.y;

    float y = p.z;
    float dy = dp.z;
    float ddy = ddp.z;

    float v1 = -r * sin(x) * sin(y) * ddy + r * cos(x) * cos(y) * ddx + sin(x) * cos(y) * ddr - 2 * sin(x) * sin(y) * dr * dy + 2 * cos(x) * cos(y) * dr * dx - r * sin(x) * cos(y) * dx * dx - 2 * r * cos(x) * sin(y) * dx * dy - r * sin(x) * cos(y) * dy * dy;
    float v2 = sin(x) * sin(y) * ddr + r * sin(x) * cos(y) * ddy + r * cos(x) * sin(y) * ddx - r * sin(x) * sin(y) * dx * dx - r * sin(x) * sin(y) * dy * dy + 2 * r * cos(x) * cos(y) * dx * dy + 2 * cos(x) * sin(y) * dr * dx + 2 * sin(x) * cos(y) * dr * dy;
    float v3 = sin(x) * (-r * ddx - 2 * dr * dx) + cos(x) * (ddr - r * dx * dx);

    return (float3){v1, v2, v3};
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

///fast_normalized
float3 rejection(float3 my_vector, float3 basis)
{
    return fast_normalize(my_vector - dot(my_vector, basis) * basis);
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
    float3 sRGB = 0.585122381 * S1 + 0.783140355 * S2 - 0.368262736 * S3;

    return sRGB;
}

/*float lambert_w0(float x)
{
    const float c1 = 4.0 / 3.0;
    const float c2 = 7.0 / 3.0;
    const float c3 = 5.0 / 6.0;
    const float c4 = 2.0 / 3.0;
    float f;
    float temp;
    float wn;
    float y;
    float zn;

    f = log ( x );

    if ( x <= 0.7385 )
    {
        wn = x * ( 1.0 + c1 * x ) / ( 1.0 + x * ( c2 + c3 * x ) );
    }
    else
    {
        wn = f - 24.0 * ( ( f + 2.0 ) * f - 3.0 )
        / ( ( 0.7 * f + 58.0 ) * f + 127.0 );
    }

    zn = f - wn - log ( wn );
    temp = 1.0 + wn;
    y = 2.0 * temp * ( temp + c4 * zn ) - zn;
    float interm = zn * y / ( temp * ( y - zn ) );
    wn = wn * ( 1.0 + interm );

    return wn;
}*/

float lambert_w0_approx(float x)
{
  const float c1 = 4.0 / 3.0;
  const float c2 = 7.0 / 3.0;
  const float c3 = 5.0 / 6.0;
  const float c4 = 2.0 / 3.0;
  float f;
  float temp;
  float temp2;
  float wn;
  float y;
  float zn;

  f = log ( x );

  if ( x <= 6.46 )
  {
    wn = x * ( 1.0 + c1 * x ) / ( 1.0 + x * ( c2 + c3 * x ) );
    zn = f - wn - log ( wn );
  }
  else
  {
    wn = f;
    zn = - log ( wn );
  }

  temp = 1.0 + wn;
  y = 2.0 * temp * ( temp + c4 * zn ) - zn;
  wn = wn * ( 1.0 + zn * y / ( temp * ( y - zn ) ) );

  zn = f - wn - log ( wn );
  temp = 1.0 + wn;
  temp2 = temp + c4 * zn;
  float en2 = zn * temp2 / ( temp * temp2 - 0.5 * zn );
  wn = wn * ( 1.0 + en2 );

  return wn;
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

    //return lambert_w0_halley(x);

    //return lambert_w0_halley(x);
}


void calculate_metric_krus(float4 spacetime_position, float g_metric_out[])
{
    float rs = 1;
    float k = rs;

    #ifndef IS_CONSTANT_THETA
    float theta = spacetime_position.z;
    #else
    float theta = M_PI/2;
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
    float theta = M_PI/2;
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
    float theta = M_PI/2;
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
    float r = spacetime_position.y;

    float rs = 1;
    float c = 1;

    #ifndef IS_CONSTANT_THETA
    float theta = spacetime_position.z;
    #else
    float theta = M_PI/2;
    #endif // IS_CONSTANT_THETA

    g_metric_partials[0 * 4 + 1] = -c*c*rs/(r*r);
    g_metric_partials[1 * 4 + 1] = -rs / ((rs - r) * (rs - r));
    g_metric_partials[2 * 4 + 1] = 2 * r;
    g_metric_partials[3 * 4 + 1] = 2 * r * sin(theta) * sin(theta);
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

void get_lorenz_coeff(float4 time_basis, float g_metric[], float coeff_out[16])
{
    //float g_metric[] = {-g_metric_in[0], -g_metric_in[1], -g_metric_in[2], -g_metric_in[3]};

    float4 low_time_basis = lower_index(time_basis, g_metric);

    float tuu[] = {time_basis.x, time_basis.y, time_basis.z, time_basis.w};
    float tuv[] = {low_time_basis.x, low_time_basis.y, low_time_basis.z, low_time_basis.w};

    for(int i=0; i < 16; i++)
    {
        coeff_out[i] = 0;
    }

    float obvs[] = {1, 0, 0, 0};
    float4 lobvsv = lower_index((float4){1, 0, 0, 0}, g_metric);

    float lobvs[4] = ARRAY4(lobvsv);

    float gamma = -tuv[0] * obvs[0];

    coeff_out[0 * 4 + 0] = 1;
    coeff_out[1 * 4 + 1] = 1;
    coeff_out[2 * 4 + 2] = 1;
    coeff_out[3 * 4 + 3] = 1;

    for(int u=0; u < 4; u++)
    {
        for(int v=0; v < 4; v++)
        {
            float val = -1/(1 + gamma) * (tuu[u] + obvs[u]) * (-tuv[v] - lobvs[v]) - 2 * obvs[u] * tuv[v];

            coeff_out[u * 4 + v] += val;
        }
    }
}

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
    float christoff[64] = {0};

    ///diagonal of the metric, because it only has diagonals
    float g_inv[4] = {1/g_metric[0], 1/g_metric[1], 1/g_metric[2], 1/g_metric[3]};

    {
        for(int i=0; i < 4; i++)
        {
            float ginvii = 0.5 * g_inv[i];

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

    for(int uu=0; uu < 4; uu++)
    {
        float sum = 0;

        for(int aa = 0; aa < 4; aa++)
        {
            for(int bb = 0; bb < 4; bb++)
            {
                sum += (velocity_arr[aa]) * (velocity_arr[bb]) * christoff[uu * 16 + aa*4 + bb*1];
            }
        }

        christ_result[uu] = sum;
    }

    float4 acceleration = {christ_result[0], christ_result[1], christ_result[2], christ_result[3]};

    return -acceleration;
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
    //return mix(mixd, val_at_min, val_at_max);
}

struct lightray
{
    float4 position;
    float4 velocity;
    float4 acceleration;
    int sx, sy;
};

#if 1

#define NO_KRUSKAL
#define NO_EVENT_HORIZON_CROSSING

//#define EULER_INTEGRATION
#define VERLET_INTEGRATION

__kernel
void init_rays(float4 cartesian_camera_pos, float4 camera_quat, __global struct lightray* schwarzs_rays, __global struct lightray* kruskal_rays, __global int* schwarzs_count, __global int* kruskal_count, int width, int height)
{
    #define FOV 90

    float fov_rad = (FOV / 360.f) * 2 * M_PI;

    int cx = get_global_id(0);
    int cy = get_global_id(1);

    if(cx >= width || cy >= height)
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

    float lorenz[16] = {};
    get_lorenz_coeff(bT, g_metric, lorenz);

    float4 cX = tensor_contract(lorenz, btheta);
    float4 cY = tensor_contract(lorenz, bphi);
    float4 cZ = tensor_contract(lorenz, bX);

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

                //float4 old_lightray_acceleration = ((position - last_new_pos) - (last_velocity * last_ds)) / (0.5 * last_ds * last_ds);
                //float4 old_lightray_acceleration = ((velocity - last_new_vel) / last_ds);
                float4 old_lightray_acceleration = ((ivel - last_new_vel) / last_ds);
                new_acceleration = ((new_vel - last_new_vel) / (0.5 * last_ds)) - old_lightray_acceleration;
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
        float4 next_position = position + velocity * ds + 0.5 * acceleration * ds * ds;
        float4 intermediate_next_velocity = velocity + acceleration * ds;

        calculate_metric_krus(next_position, g_metric);
        calculate_partial_derivatives_krus(next_position, g_partials);

        intermediate_next_velocity = fix_light_velocity2(intermediate_next_velocity, g_metric);

        float4 next_acceleration = calculate_acceleration(intermediate_next_velocity, g_metric, g_partials);
        float4 next_velocity = velocity + 0.5 * (acceleration + next_acceleration) * ds;

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
    float ambient_precision = 0.01;
    float subambient_precision = 0.5;

    float rs = 1;

    float4 last_position = 0;
    float4 last_velocity = 0;
    float4 intermediate_velocity = 0;

    for(int i=0; i < 64000/125; i++)
    {
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
void calculate_texture_coordinates(__global struct lightray* finished_rays, __global int* finished_count_in, __global float2* texture_coordinates, int width, int height, float4 cartesian_camera_pos, float4 camera_quat)
{
    int id = get_global_id(0);

    if(id >= *finished_count_in)
        return;

    struct lightray* ray = &finished_rays[id];

    int pos = ray->sy * width + ray->sx;
    int sx = ray->sx;
    int sy = ray->sy;

    float4 position = ray->position;

    float rs = 1;
    float r_value = position.y;

    #ifdef NO_EVENT_HORIZON_CROSSING
    if(r_value <= rs)
    {
        return;
    }
    #endif // NO_EVENT_HORIZON_CROSSINGS

    float3 cart_here = polar_to_cartesian((float3)(r_value, position.zw));

    #define FOV 90

    float fov_rad = (FOV / 360.f) * 2 * M_PI;

    float nonphysical_plane_half_width = width/2;
    float nonphysical_f_stop = nonphysical_plane_half_width / tan(fov_rad/2);

    float3 pixel_direction = (float3){sx - width/2, sy - height/2, nonphysical_f_stop};

    pixel_direction = fast_normalize(pixel_direction);
    pixel_direction = rot_quat(pixel_direction, camera_quat);

    float3 cartesian_velocity = fast_normalize(pixel_direction);

    float3 new_basis_x = fast_normalize(cartesian_velocity);
    float3 new_basis_y = fast_normalize(-cartesian_camera_pos.yzw);

    new_basis_x = rejection(new_basis_x, new_basis_y);

    float3 new_basis_z = -fast_normalize(cross(new_basis_x, new_basis_y));

    cart_here = rotate_vector(new_basis_x, new_basis_y, new_basis_z, cart_here);

    float3 npolar = cartesian_to_polar(cart_here);

    float thetaf = fmod(npolar.y, 2 * M_PI);
    float phif = npolar.z;

    if(thetaf >= M_PI)
    {
        phif += M_PI;
        thetaf -= M_PI;
    }

    phif = fmod(phif, 2 * M_PI);

    float sxf = (phif) / (2 * M_PI);
    float syf = thetaf / M_PI;

    sxf += 0.5f;

    /*if(sx == width/2 && sy == height/2)
    {
        printf("COORD %f %f\n", sxf, syf);
    }*/

    texture_coordinates[pos] = (float2)(sxf, syf);
}

float4 read_mip(float2 pos, int level, __read_only image2d_t background0,
            __read_only image2d_t background1,
            __read_only image2d_t background2,
            __read_only image2d_t background3)
{
    sampler_t sam = CLK_NORMALIZED_COORDS_TRUE |
                    CLK_ADDRESS_REPEAT |
                    CLK_FILTER_LINEAR;

    if(level == 0)
        return read_imagef(background0, sam, pos);
    if(level == 1)
        return read_imagef(background1, sam, pos);
    if(level == 2)
        return read_imagef(background2, sam, pos);
    if(level == 3)
        return read_imagef(background3, sam, pos);

    return (float4)(1, 0, 1, 1);
}

__kernel
void render(float4 cartesian_camera_pos, float4 camera_quat, __global struct lightray* finished_rays, __global int* finished_count_in, __write_only image2d_t out,
            __read_only image2d_t mip_background,
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

    float rs = 1;
    float r_value = position.y;

    #ifdef NO_EVENT_HORIZON_CROSSING
    if(r_value <= rs)
    {
        write_imagef(out, (int2){sx, sy}, (float4)(0, 0, 0, 1));
        return;
    }
    #endif // NO_EVENT_HORIZON_CROSSINGS

    float sxf = texture_coordinates[sy * width + sx].x;
    float syf = texture_coordinates[sy * width + sx].y;

    #ifndef NO_EVENT_HORIZON_CROSSING
    if(r_value <= rs)
    {
        val = (float4)(0,0,0,1);

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
    }
    #endif // NO_EVENT_HORIZON_CROSSINGS

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

    float2 dx_vtc = (tr - tl);
    float2 dy_vtc = (bl - tl);

    float2 dx_vtc2 = (tr - tl - 1);
    float2 dy_vtc2 = (bl - tl - 1);

    /*if(any(dx_vtc > 0.001))
        return;

    if(any(dy_vtc > 0.001))
        return;*/

    //dx_vtc = min(tr - tl - 1, dx_vtc);
    //dy_vtc = min(bl - tl - 1, dy_vtc);

    if(dx == -1)
    {
        dx_vtc = -dx_vtc;
        dx_vtc2 = -dx_vtc2;
    }

    if(dy == -1)
    {
        dy_vtc = -dy_vtc;
        dy_vtc2 = -dy_vtc2;
    }

    /*if(fabs(dx_vtc2.x) < fabs(dx_vtc.x))
        dx_vtc.x = dx_vtc2.x;

    if(fabs(dx_vtc2.y) < fabs(dx_vtc.y))
        dx_vtc.y = dx_vtc2.y;

    if(fabs(dy_vtc2.x) < fabs(dy_vtc.x))
        dy_vtc.x = dy_vtc2.x;

    if(fabs(dy_vtc2.y) < fabs(dy_vtc.y))
        dy_vtc.y = dy_vtc2.y;*/

    if(fast_length(dx_vtc) > 0.1 || fast_length(dy_vtc) > 0.1)
    {
        return;

        dx_vtc = dx_vtc2;
    }

    //#define TRILINEAR
    #ifdef TRILINEAR
    dx_vtc.x *= get_image_width(mip_background);
    dy_vtc.x *= get_image_width(mip_background);

    dx_vtc.y *= get_image_height(mip_background);
    dy_vtc.y *= get_image_height(mip_background);

    //dx_vtc.x /= 10.f;
    //dy_vtc.x /= 10.f;


    dx_vtc /= 2.f;
    dy_vtc /= 2.f;

    float delta_max_sqr = max(dot(dx_vtc, dx_vtc), dot(dy_vtc, dy_vtc));

    float mip_level = 0.5 * log2(delta_max_sqr);

    //mip_level -= 0.5;

    float mip_clamped = clamp(mip_level, 0.f, 5.f);

    float4 end_result = read_imagef(mip_background, sam, (float2){sxf, syf}, mip_clamped);
    #else

    dx_vtc.x *= get_image_width(mip_background);
    dy_vtc.x *= get_image_width(mip_background);

    dx_vtc.y *= get_image_height(mip_background);
    dy_vtc.y *= get_image_height(mip_background);

    ///http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.1002.1336&rep=rep1&type=pdf
    float dv_dx = dx_vtc.y;
    float dv_dy = dy_vtc.y;

    float du_dx = dx_vtc.x;
    float du_dy = dy_vtc.x;

    float Ann = dv_dx * dv_dx + dv_dy * dv_dy;
    float Bnn = -2 * (du_dx * dv_dx + du_dy * dv_dy);
    float Cnn = du_dx * du_dx + du_dy * du_dy; ///only tells lies

    ///hecc
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

    float majorRadius = sqrt(1/a_prime);
    float minorRadius = sqrt(1/c_prime);

    float theta = atan2(B, (A - C)/2);

    majorRadius = max(majorRadius, 1.f);
    minorRadius = max(minorRadius, 1.f);

    float fProbes = 2 * (majorRadius / minorRadius) - 1;
    int iProbes = floor(fProbes + 0.5f);

    int maxProbes = 8;

    iProbes = min(iProbes, maxProbes);

    if(iProbes < fProbes)
        minorRadius = 2 * majorRadius / (iProbes + 1);

    float levelofdetail = log2(minorRadius);

    int maxLod = 9;

    /*if(sx == width/2 && sy == height/2)
    {
        printf("HI %f %i\n", levelofdetail, iProbes);
    }*/

    if(levelofdetail > maxLod)
    {
        levelofdetail = maxLod;
        iProbes = 1;
    }

    if(iProbes == 1)
    {
        float4 end_result = read_imagef(mip_background, sam, (float2){sxf, syf}, levelofdetail);
        write_imagef(out, (int2){sx, sy}, end_result);
        return;
    }

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

    float sU = du / get_image_width(mip_background);
    float sV = dv / get_image_height(mip_background);

    for(int cnt = 0; cnt < iProbes; cnt++)
    {
        float d_2 = (currentN * currentN / 4.f) * (du * du + dv * dv) / (majorRadius * majorRadius);

        float relativeWeight = native_exp(-alpha * d_2);

        float centreu = tl.x;
        float centrev = tl.y;

        float cu = centreu + (currentN / 2.f) * sU;
        float cv = centrev + (currentN / 2.f) * sV;

        float4 fval = read_imagef(mip_background, sam, (float2){cu, cv}, levelofdetail);

        totalWeight += relativeWeight * fval;
        accumulatedProbes += relativeWeight;

        currentN += 2;
    }

    float4 end_result = totalWeight / accumulatedProbes;

    #endif // TRILINEAR

    //float4 end_result = read_imagef(mip_background, sam, (float2){sxf, syf}, dx_vtc, dy_vtc);

    #else
    float4 end_result = read_imagef(mip_background, sam, (float2){sxf, syf}, 0);
    #endif // MIPMAPPING

    //end_result.x = sxf;
    //end_result.y = syf;

    write_imagef(out, (int2){sx, sy}, end_result);
}
#endif // 0

__kernel
void do_raytracing_multicoordinate(__write_only image2d_t out, float ds_, float4 cartesian_camera_pos, float4 camera_quat, __read_only image2d_t background)
{
    #define FOV 90

    float fov_rad = (FOV / 360.f) * 2 * M_PI;

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

    float lorenz[16] = {};
    get_lorenz_coeff(bT, g_metric, lorenz);

    float4 cX = tensor_contract(lorenz, btheta);
    float4 cY = tensor_contract(lorenz, bphi);
    float4 cZ = tensor_contract(lorenz, bX);

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

            float thetaf = fmod(npolar.y, 2 * M_PI);
            float phif = npolar.z;

            if(thetaf >= M_PI)
            {
                phif += M_PI;
                thetaf -= M_PI;
            }

            phif = fmod(phif, 2 * M_PI);

            float sx = (phif) / (2 * M_PI);
            float sy = thetaf / M_PI;

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
