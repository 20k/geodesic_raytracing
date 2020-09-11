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
    //float theta = atan2(sqrt(in.x * in.x + in.y * in.y), in.z);
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
    float pdot = (p.z * (p.x * v.x + p.y * v.y) - (p.x * p.x + p.y * p.y) * v.z) / ((p.x * p.x + p.y * p.y + p.z * p.z) * sqrt(p.x * p.x + p.y * p.y));*/

    float r = length(p);

    float repeated_eq = r * sqrt(1 - (p.z*p.z / (r * r)));

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
float4 fix_light_velocity(float4 velocity, float g_metric[])
{
    //velocity.yzw = normalize(velocity.yzw);

    /*velocity.x = -1/sqrt(-g_metric[0]);
    velocity.y = velocity.y / sqrt(g_metric[1]);
    velocity.z = velocity.z / sqrt(g_metric[2]);
    velocity.w = velocity.w / sqrt(g_metric[3]);*/

    float v[4] = {velocity.x, velocity.y, velocity.z, velocity.w};

    ///so. g_metric[0] is negative. velocity_arr[0] is 1

    ///so rewritten, ds2 = Eu Ev dxu * dx v

    ///g_metric[1] * v[1]^2 + g_metric[2] * v[2]^2 + g_metric[3] * v[3]^2 = -g_metric[0] * v[0]^2 * scale

    float time_scale = (g_metric[1] * v[1] * v[1] + g_metric[2] * v[2] * v[2] + g_metric[3] * v[3] * v[3]) / (-g_metric[0] * v[0] * v[0]);

    ///g_metric[1] * v[1]^2 / scale + g_metric[2] * v[2]^2 / scale + g_metric[3] * v[3]^2 / scale = -g_metric[0] * v[0]^2

    /*v[1] /= sqrt(time_scale);
    v[2] /= sqrt(time_scale);
    v[3] /= sqrt(time_scale);*/

    v[0] *= sqrt(time_scale);

    ///should print 0
    /*float fds = calculate_ds((float4){v[0], v[1], v[2], v[3]}, g_metric);
    printf("%f fds\n", fds);*/

    return (float4){v[0], v[1], v[2], v[3]};
}

float4 fix_light_velocity2(float4 v, float g_metric[])
{
    ///g_metric[1] * v[1]^2 + g_metric[2] * v[2]^2 + g_metric[3] * v[3]^2 = -g_metric[0] * v[0]^2

    float tvl_2 = (g_metric[1] * v.y * v.y + g_metric[2] * v.z * v.z + g_metric[3] * v.w * v.w) / -g_metric[0];

    v.x = sqrt(tvl_2);

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

float4 unrotate_vector4(float4 bx, float4 by, float4 bz, float4 bw, float4 v)
{

}

///normalized
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
    float3 S1 = sqrt(val);
    float3 S2 = sqrt(S1);
    float3 S3 = sqrt(S2);
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

/*float lambert_w0(float x)
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
}*/

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

void calculate_partial_derivatives_krus(float4 spacetime_position, float g_metric_partials[])
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
        return sqrt(r/k - 1) * exp(0.5 * r/k) * sinh(0.5 * t/k);
    else
        return sqrt(1 - r/k) * exp(0.5 * r/k) * cosh(0.5 * t/k);
}

float rt_to_X_krus(float r, float t)
{
    float rs = 1;
    float k = rs;

    if(r > rs)
        return sqrt(r/k - 1) * exp(0.5 * r/k) * cosh(0.5 * t/k);
    else
        return sqrt(1 - r/k) * exp(0.5 * r/k) * sinh(0.5 * t/k);
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
        return exp(0.5 * r/k) * (dt * (0.5 * r - 0.5 * k) * sinh((0.5 * t) / k) + 0.5 * r * dr * cosh((0.5 * t) / k)) / (k * k * sqrt(r/k - 1));
    else
        return exp(0.5 * r/k) * (dt * (0.5 * k - 0.5 * r) * cosh((0.5 * t) / k) - 0.5 * r * dr * sinh((0.5 * t) / k)) / (k * k * sqrt(1 - k/r));
}

float trdtdr_to_dT(float t, float r, float dt, float dr)
{
    float rs = 1;
    float k = rs;

    if(r > rs)
        return exp(0.5 * r/k) * (dt * (0.5 * r - 0.5 * k) * cosh((0.5 * t) / k) + 0.5 * r * dr * sinh((0.5 * t) / k)) / (k * k * sqrt(r/k - 1));
    else
        return exp(0.5 * r/k) * (dt * (0.5 * k - 0.5 * r) * sinh((0.5 * t) / k) - 0.5 * r * dr * cosh((0.5 * t) / k)) / (k * k * sqrt(1 - k/r));
}

float TX_to_t(float T, float X)
{
    float rs = 1;

    if(T * T - X * X < 0)
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
    float X = krus.y;
    float T = krus.x;

    float rs = 1;
    float r = TX_to_r_krus(T, X);

    float t = TX_to_t(T, X);

    return (float4)(t, r, krus.zw);
}

float4 kruskal_velocity_to_schwarzs_velocity(float4 krus, float4 dkrus)
{
    float dr = TXdTdX_to_dr(krus.x, krus.y, dkrus.x, dkrus.y);
    float dt = TXdTdX_to_dt(krus.x, krus.y, dkrus.x, dkrus.y);

    return (float4)(dt, dr, dkrus.zw);
}

float4 kruskal_position_to_schwarzs_position_with_r(float4 krus, float r)
{
    float X = krus.y;
    float T = krus.x;

    float rs = 1;

    float t = TX_to_t(T, X);

    return (float4)(t, r, krus.zw);
}

float4 kruskal_velocity_to_schwarzs_velocity_with_r(float4 krus, float4 dkrus, float r)
{
    float dr = TXdTdX_to_dr_with_r(krus.x, krus.y, dkrus.x, dkrus.y, r);
    float dt = TXdTdX_to_dt(krus.x, krus.y, dkrus.x, dkrus.y);

    return (float4)(dt, dr, dkrus.zw);
}

float4 schwarzs_position_to_kruskal_position(float4 pos)
{
    float X = rt_to_X_krus(pos.y, pos.x);
    float T = rt_to_T_krus(pos.y, pos.x);

    return (float4)(T, X, pos.zw);
}

float4 schwarzs_velocity_to_kruskal_velocity(float4 pos, float4 dpos)
{
    float dX = trdtdr_to_dX(pos.x, pos.y, dpos.x, dpos.y);
    float dT = trdtdr_to_dX(pos.x, pos.y, dpos.x, dpos.y);

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
    ///diagonal of the metric, because it only has diagonals
    float g_inv[4] = {1/g_metric[0], 1/g_metric[1], 1/g_metric[2], 1/g_metric[3]};

    float christoff[64] = {0};

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

        /*if(!isfinite(sum))
        {
            write_imagef(out, (int2){cx, cy}, (float4){1, 0, 0, 1});
            return;
        }*/

        christ_result[uu] = sum;
    }

    float4 acceleration = {-christ_result[0], -christ_result[1], -christ_result[2], -christ_result[3]};

    return acceleration;
}

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

    float4 co_basis = (float4){sqrt(-g_metric[0]), sqrt(g_metric[1]), sqrt(g_metric[2]), sqrt(g_metric[3])};

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

    pixel_direction = unrotate_vector(normalize(cartesian_cx), normalize(cartesian_cy), normalize(cartesian_cz), pixel_direction);

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

    #define NO_EVENT_HORIZON_CROSSING

    #ifdef NO_EVENT_HORIZON_CROSSING
    ambient_precision = 0.01;
    max_ds = 0.01;
    min_ds = 0.01;
    #endif // NO_EVENT_HORIZON_CROSSING


    /*float min_radius = rs * 1.1;
    float max_radius = rs * 1.6;*/

    float min_radius = 0.7 * rs;
    float max_radius = 1.1 * rs;

    //#define NO_KRUSKAL

    #ifndef NO_KRUSKAL
    if(polar_camera.x >= rs * 1.15 && is_kruskal)
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
    //float4 last_acceleration = (float4)(0,0,0,0);
    float last_ds = 1;

    ///T is 0 because time coordinate is 0
    //float T_at_transition_radius = rt_to_T_krus(rs * 1.15, 0);

    float krus_radius = 1.15 * rs;

    float T2_m_X2_transition = r_to_T2_m_X2(krus_radius);
    float krus_inner_cutoff = r_to_T2_m_X2(0.5 * rs);

    int bad_rays = 0;

    for(int it=0; it < 60000; it++)
    {
        float g_partials[16] = {0};

        #ifndef NO_KRUSKAL
        if(!is_kruskal)
        {
            if(lightray_spacetime_position.y < rs * 1.1)
            {
                is_kruskal = true;

                float4 new_pos = schwarzs_position_to_kruskal_position(lightray_spacetime_position);
                float4 new_vel = schwarzs_velocity_to_kruskal_velocity(lightray_spacetime_position, lightray_velocity);

                lightray_spacetime_position = new_pos;
                lightray_velocity = new_vel;

                /*calculate_metric_krus(new_pos, g_metric);
                calculate_partial_derivatives_krus(new_pos, g_partials);

                lightray_acceleration = calculate_acceleration(lightray_velocity, g_metric, g_partials);*/

                //lightray_acceleration = ((lightray_spacetime_position - last_position) - (last_velocity * last_ds)) / (0.5 * last_ds * last_ds);
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

                lightray_spacetime_position = new_pos;
                lightray_velocity = new_vel;

                /*calculate_metric(new_pos, g_metric);
                calculate_partial_derivatives(new_pos, g_partials);

                lightray_acceleration = calculate_acceleration(lightray_velocity, g_metric, g_partials);*/
                //lightray_acceleration = ((lightray_spacetime_position - last_position) - (last_velocity * last_ds)) / (0.5 * last_ds * last_ds);
            }
        }

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

            if(r_value >= 20)
            {
                /*float nfrac = (r_value - 20) / 100;

                nfrac = clamp(nfrac, 0.f, 1.f);

                ds = mix(subambient_precision, 10, nfrac);*/

                ds = r_value / 10;
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

        if(!is_radius_leq_than(lightray_spacetime_position, is_kruskal, 40) || is_radius_leq_than(lightray_spacetime_position, is_kruskal, 0.5))
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

                int x_half = fabs(fmod(sx * 10, 1)) > 0.5 ? 1 : 0;
                int y_half = fabs(fmod(sy * 10, 1)) > 0.5 ? 1 : 0;

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
            float ftol = 0.001;

            if(fabs(g_partials[2 * 4 + 0]) < ftol && fabs(g_partials[2 * 4 + 1]) < ftol && fabs(g_partials[3 * 4 + 2]) < ftol)
                bad_rays++;

            if(bad_rays >= 3)
            {
                write_imagef(out, (int2)(cx, cy), (float4)(1, 0, 1, 1));
                return;
            }
        }
        #endif // NO_HORIZON_CROSSING


        ///euler

        #define EULER_INTEGRATION
        #ifdef EULER_INTEGRATION
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

        float4 acceleration = calculate_acceleration(lightray_velocity, g_metric, g_partials);
        lightray_velocity += acceleration * ds;
        lightray_velocity = fix_light_velocity2(lightray_velocity, g_metric);
        lightray_spacetime_position += lightray_velocity * ds;
        #endif // EULER_INTEGRATION

        //#define VERLET_INTEGRATION
        #ifdef VERLET_INTEGRATION
        float4 next_position = lightray_spacetime_position + lightray_velocity * ds + 0.5 * lightray_acceleration * ds * ds;
        float4 intermediate_next_velocity = lightray_velocity + lightray_acceleration * ds;

        ///try moving this out of the loop
        {
            if(is_kruskal)
                calculate_metric_krus(next_position, g_metric);
            else
                calculate_metric(next_position, g_metric);

            if(is_kruskal)
                calculate_partial_derivatives_krus(next_position, g_partials);
            else
                calculate_partial_derivatives(next_position, g_partials);
        }

        float4 next_acceleration = calculate_acceleration(intermediate_next_velocity, g_metric, g_partials);
        float4 next_velocity = lightray_velocity + 0.5 * (lightray_acceleration + next_acceleration) * ds;

        last_position = lightray_spacetime_position;
        last_velocity = lightray_velocity;
        last_ds = ds;

        lightray_spacetime_position = next_position;
        lightray_velocity = fix_light_velocity2(next_velocity, g_metric);
        lightray_acceleration = next_acceleration;
        #endif // VERLET_INTEGRATION
    }

    write_imagef(out, (int2){cx, cy}, (float4){0, 1, 0, 1});
}
