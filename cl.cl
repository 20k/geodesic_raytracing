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

//#define IS_CONSTANT_THETA

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

float lambert_w0(float x)
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

    float f10 = (2 * k * X * lambert_interior) / ((X*X - T*T) * (lambert_interior + 1));
    float f01 = (2 * k * T * lambert_interior) / ((T*T - X*X) * (lambert_interior + 1));

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

__kernel
void do_raytracing(__write_only image2d_t out, float ds_, float4 cartesian_camera_pos, float4 camera_quat, __read_only image2d_t background)
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

    ///need to rotate by camera angle

    ///this position is incredibly wrong
    //float3 pixel_virtual_pos = (float3){cx - width/2, cy - height/2, nonphysical_f_stop};

    //pixel_virtual_pos = normalize(pixel_virtual_pos) / 299792458.f;

    //pixel_virtual_pos = rot_quat(pixel_virtual_pos, camera_quat);

    /*float3 cartesian_velocity = normalize(pixel_virtual_pos);

    float3 new_basis_x = normalize(cartesian_velocity);
    float3 new_basis_y = normalize(-cartesian_camera_pos.yzw);

    new_basis_x = rejection(new_basis_x, new_basis_y);

    float3 new_basis_z = -normalize(cross(new_basis_x, new_basis_y));

    float3 cartesian_camera_new_basis = unrotate_vector(new_basis_x, new_basis_y, new_basis_z, cartesian_camera_pos.yzw);
    float3 cartesian_velocity_new_basis = unrotate_vector(new_basis_x, new_basis_y, new_basis_z, cartesian_velocity);

    float3 polar_velocity = cartesian_velocity_to_polar_velocity(cartesian_camera_new_basis, cartesian_velocity_new_basis);*/

    /*float3 polar_velocity = cartesian_velocity_to_polar_velocity(cartesian_camera_pos.yzw, normalize(pixel_virtual_pos));

    float rs = 1;
    float c = 1;

    float4 spacetime_polar_velocity = (float4)(1, polar_velocity.xyz);

    float4 lightray_polar_position = (float4)(0, cartesian_to_polar(cartesian_camera_pos.yzw));*/

    float rs = 1;
    float c = 1;

    float3 polar_camera = cartesian_to_polar(cartesian_camera_pos.yzw);

    float4 krus_camera = (float4)(rt_to_T_krus(polar_camera.x, 0), rt_to_X_krus(polar_camera.x, 0), polar_camera.y, polar_camera.z);

    float g_metric[4] = {};
    calculate_metric_krus(krus_camera, g_metric);

    float3 pixel_direction = (float3){cx - width/2, cy - height/2, -nonphysical_f_stop};
    pixel_direction = normalize(pixel_direction);

    float local_r = polar_camera.x;

    /*float4 bT = (float4)(1/(sqrt(4 * rs * rs * rs / local_r) * exp(-local_r/rs)), 0, 0, 0);
    float4 bX = (float4)(0, 1/(sqrt(4 * rs * rs * rs / local_r) * exp(-local_r/rs)), 0, 0);
    float4 btheta = (float4)(0, 0, 1/local_r, 0);
    float4 bphi = (float4)(0, 0, 0, 1/(local_r * sin(krus_camera.z)));*/

    float4 co_basis = (float4){sqrt(-g_metric[0]), sqrt(g_metric[1]), sqrt(g_metric[2]), sqrt(g_metric[3])};

    float4 bT = (float4)(1/co_basis.x, 0, 0, 0);
    float4 bX = (float4)(0, 1/co_basis.y, 0, 0);
    float4 btheta = (float4)(0, 0, 1/co_basis.z, 0);
    float4 bphi = (float4)(0, 0, 0, 1/co_basis.w);

    float lorenz[16] = {};

    get_lorenz_coeff(bT, g_metric, lorenz);

    /*float4 cX = tensor_contract(lorenz, btheta);
    float4 cY = tensor_contract(lorenz, bphi);
    float4 cZ = tensor_contract(lorenz, bX);*/

    float4 cX = btheta;
    float4 cY = bphi;
    float4 cZ = bX;

    float4 pixel_x = pixel_direction.x * cX;
    float4 pixel_y = pixel_direction.y * cY;
    float4 pixel_z = pixel_direction.z * cZ;

    float4 vec = pixel_x + pixel_y + pixel_z;

    float4 pixel_N = vec / (dot(lower_index(vec, g_metric), vec));

    pixel_N.yzw = rot_quat(pixel_N.yzw, camera_quat);

    //pixel_N = fix_light_velocity(pixel_N, g_metric);

    float4 lightray_velocity = pixel_N;
    float4 lightray_spacetime_position = krus_camera;

    /*float3 polar_camera = cartesian_to_polar(cartesian_camera_pos.yzw);

    float4 bT = (float4)(1/(1 - rs/polar_camera.x), -sqrt(rs/polar_camera.x), 0, 0);
    float4 bR = (float4)(-sqrt(rs/polar_camera.x) / (1 - rs/polar_camera.x), 1, 0, 0);
    float4 btheta = (float4)(0, 0, 1/polar_camera.x, 0);
    float4 bphi = (float4)(0, 0, 0, 1/(polar_camera.x * sin(polar_camera.y)));

    float g_metric_p[4] = {};

    calculate_metric((float4)(0, polar_camera.xyz), g_metric_p);

    float3 pixel_direction = (float3){cx - width/2, cy - height/2, -nonphysical_f_stop};

    pixel_direction = normalize(pixel_direction);

    float lorenz[16] = {};

    get_lorenz_coeff(bT, g_metric_p, lorenz);
    float4 ftime = tensor_contract(lorenz, bT);

    float4 cX = tensor_contract(lorenz, btheta);
    float4 cY = tensor_contract(lorenz, bphi);
    float4 cZ = tensor_contract(lorenz, bR);

    float4 pixel_x = pixel_direction.x * cX;
    float4 pixel_y = pixel_direction.y * cY;
    float4 pixel_z = pixel_direction.z * cZ;

    float4 vec = pixel_x + pixel_y + pixel_z;

    float4 pixel_N = vec / (dot(lower_index(vec, g_metric_p), vec));

    pixel_N = fix_light_velocity(pixel_N, g_metric_p);

    pixel_N.yzw = rot_quat(pixel_N.yzw, camera_quat);

    float4 spacetime_polar_velocity = pixel_N;
    float4 lightray_polar_position = (float4)(0, polar_camera.xyz);

    float start_T = rt_to_T_krus(lightray_polar_position.y, lightray_polar_position.x);
    float start_X = rt_to_X_krus(lightray_polar_position.y, lightray_polar_position.x);

    float4 lightray_spacetime_position = (float4)(start_T, start_X, lightray_polar_position.z, lightray_polar_position.w);

    float g_metric[4] = {0};

    calculate_metric_krus(lightray_spacetime_position, g_metric);

    float dX = trdtdr_to_dX(0, lightray_polar_position.y, spacetime_polar_velocity.x, spacetime_polar_velocity.y);
    float dT = trdtdr_to_dT(0, lightray_polar_position.y, spacetime_polar_velocity.x, spacetime_polar_velocity.y);

    //float4 lightray_velocity = fix_light_velocity((float4)(dT, dX, spacetime_polar_velocity.zw), g_metric);

    float4 lightray_velocity = (float4)(-dX, dX, spacetime_polar_velocity.zw);*/

    //float4 lightray_velocity = (float4)(1, dX, polar_velocity.yz);

    //printf("START_R %f\n", TX_to_r_krus(start_T, start_X));

    //float4 lightray_velocity = (float4)(1, bad_light_velocity.xyz);

    //write_imagef(out, (int2){cx, cy}, (float4){0, 0, 0, 1});

    if(cx == width/2 && cy == height/2)
    {
        float lds = calculate_ds(lightray_velocity, g_metric);
        printf("LDS %f\n", lds);

        printf("VEC %f %f %f %f\n", lightray_velocity.x, lightray_velocity.y, lightray_velocity.z, lightray_velocity.w);
        printf("MET %f %f %f %f\n", g_metric[0], g_metric[1], g_metric[2], g_metric[3]);
    }

    float ambient_precision = 0.1;

    ///TODO: need to use external observer time, currently using sim time!!
    float max_ds = 0.1;
    float min_ds = ambient_precision;

    float min_radius = rs * 1.1;
    float max_radius = rs * 1.6;

    for(int it=0; it < 32000; it++)
    {
        float kT = lightray_spacetime_position.x;
        float kX = lightray_spacetime_position.y;

        float r_value = TX_to_r_krus(kT, kX);

        float interp = clamp(r_value, min_radius, max_radius);

        float frac = (r_value - min_radius) / (max_radius - min_radius);

        float ds = mix(max_ds, min_ds, frac);

        #if 1
        /*if(r_value < (rs + rs * 0.00000001))
        {
            //printf("RVAL %f %f %f\n", kX, kT, r_value);

            write_imagef(out, (int2){cx, cy}, (float4){0,0,1,1});
            return;
        }*/

        if(r_value > 20)
        {
            float3 cart_here = polar_to_cartesian((float3)(r_value, lightray_spacetime_position.zw));

            //cart_here = rotate_vector(new_basis_x, new_basis_y, new_basis_z, cart_here);

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

            write_imagef(out, (int2){cx, cy}, val);
            return;
        }

        calculate_metric_krus(lightray_spacetime_position, g_metric);

        float christoff[64] = {0};

        #ifndef IS_CONSTANT_THETA
        float theta = lightray_spacetime_position.z;
        #else
        float theta = M_PI/2;
        #endif // IS_CONSTANT_THETA

        float g_partials[16] = {0};

        calculate_partial_derivatives_krus(lightray_spacetime_position, g_partials);

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

            if(!isfinite(sum))
            {
                write_imagef(out, (int2){cx, cy}, (float4){1, 0, 0, 1});
                return;
            }

            christ_result[uu] = sum;
        }

        float4 acceleration = {-christ_result[0], -christ_result[1], -christ_result[2], -christ_result[3]};
        #endif // 0

        lightray_velocity += acceleration * ds;
        lightray_spacetime_position += lightray_velocity * ds;

        //lightray_velocity = fix_light_velocity(lightray_velocity, g_metric);

        if((cx == width/2 && cy == height/2) || (cx == width-2 && cy == height/2) || (cx == 0 && cy == height/2))
        {
            //float3 lp = lightray_spacetime_position.yzw;

            float3 lp = (float3)(r_value, lightray_spacetime_position.zw);

            lp.x *= 10;

            float3 world_pos = (float3){lp.x * sin(lp.y) * cos(lp.z),
                                        lp.x * sin(lp.y) * sin(lp.z),
                                        lp.x * cos(lp.y)};

            //world_pos = rotate_vector(new_basis_x, new_basis_y, new_basis_z, world_pos);

            world_pos.x += width/2;
            world_pos.y += height/2;
            world_pos.z += height/2;

            float4 write_col = (float4){1,0,1,1};

            world_pos = round(world_pos) + 0.5;

            write_col = pow(write_col, 1/2.2);

            //printf("%f r\n", lightray_spacetime_position.y);

            //printf("world pos %f\n", lp.x);

            if(all(world_pos.xz > 0) && all(world_pos.xz < (float2){width-2, height-2}))
                write_imagef(out, (int2){world_pos.x, world_pos.z}, write_col);

            write_imagef(out, (int2){width/2, height/2}, (float4){1, 0, 0, 1});
        }
    }

    write_imagef(out, (int2){cx, cy}, (float4){0, 1, 0, 1});
}

__kernel
void do_raytracing_old(__write_only image2d_t out, float ds_, float4 cartesian_camera_pos, float4 camera_quat, __read_only image2d_t background)
{
    /*
    so t = -(1- rs / r) * c^2
    then r = (1 - rs/ r)^-1
    theta = r^2
    phi = r^2 * (sin theta)^2
    */

    ///DT
    /*
    0
    0
    0
    0
    */

    ///DR
    /*
    -c^2 * rs / r^2
    -rs/((rs - x)^2)
    2r
    2r * (sin theta)^2
    */

    ///DTHETA
    /*
    0
    0
    0
    2 r^2 * sin(theta) * cos(theta)
    */

    ///DPHI
    /*
    0
    0
    0
    0*/

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

    float c = 1;
    float rs = 1;

    ///need to rotate by camera angle

    ///this position is incredibly wrong
    float3 pixel_virtual_pos = (float3){cx - width/2, cy - height/2, nonphysical_f_stop};

    //pixel_virtual_pos = normalize(pixel_virtual_pos) / 299792458.f;

    /*pixel_virtual_pos = rot_quat(pixel_virtual_pos, camera_quat);

    float3 cartesian_velocity = normalize(pixel_virtual_pos);

    float3 new_basis_x = normalize(cartesian_velocity);
    float3 new_basis_y = normalize(-cartesian_camera_pos.yzw);

    new_basis_x = rejection(new_basis_x, new_basis_y);

    float3 new_basis_z = -normalize(cross(new_basis_x, new_basis_y));

    float3 cartesian_camera_new_basis = unrotate_vector(new_basis_x, new_basis_y, new_basis_z, cartesian_camera_pos.yzw);
    float3 cartesian_velocity_new_basis = unrotate_vector(new_basis_x, new_basis_y, new_basis_z, cartesian_velocity);

    float3 polar_velocity = cartesian_velocity_to_polar_velocity(cartesian_camera_new_basis, cartesian_velocity_new_basis);

    float4 lightray_start_position = (float4)(0, cartesian_to_polar(cartesian_camera_new_basis));

    float4 lightray_spacetime_position = lightray_start_position;

    float g_metric[4];

    calculate_metric(lightray_spacetime_position, g_metric);

    float4 lightray_velocity = fix_light_velocity((float4)(-1, polar_velocity.xyz), g_metric);*/

    #if 1
    //{
        float3 polar_camera = cartesian_to_polar(cartesian_camera_pos.yzw);

        float4 bT = (float4)(1/(1 - rs/polar_camera.x), -sqrt(rs/polar_camera.x), 0, 0);
        float4 bR = (float4)(-sqrt(rs/polar_camera.x) / (1 - rs/polar_camera.x), 1, 0, 0);
        float4 btheta = (float4)(0, 0, 1/polar_camera.x, 0);
        float4 bphi = (float4)(0, 0, 0, 1/(polar_camera.x * sin(polar_camera.y)));

        float g_metric[4] = {};

        calculate_metric((float4)(0, polar_camera.xyz), g_metric);

        float3 pixel_direction = (float3){cx - width/2, cy - height/2, -nonphysical_f_stop};

        pixel_direction = normalize(pixel_direction);

        float lorenz[16] = {};

        get_lorenz_coeff(bT, g_metric, lorenz);

        float4 cX = tensor_contract(lorenz, btheta);
        float4 cY = tensor_contract(lorenz, bphi);
        float4 cZ = tensor_contract(lorenz, bR);

        float4 pixel_x = pixel_direction.x * cX;
        float4 pixel_y = pixel_direction.y * cY;
        float4 pixel_z = pixel_direction.z * cZ;

        float4 vec = pixel_x + pixel_y + pixel_z;

        float4 pixel_N = vec / (dot(lower_index(vec, g_metric), vec));

        pixel_N = fix_light_velocity(pixel_N, g_metric);

        pixel_N.yzw = rot_quat(pixel_N.yzw, camera_quat);

        float4 lightray_velocity = pixel_N;
        float4 lightray_spacetime_position = (float4)(0, polar_camera.xyz);
    //}
    #endif // 0

    //float4 lightray_velocity = (float4)(1, bad_light_velocity.xyz);

    //write_imagef(out, (int2){cx, cy}, (float4){0, 0, 0, 1});

    float ambient_precision = 0.1;

    float max_ds = 0.001;
    float min_ds = ambient_precision;

    float min_radius = rs * 1.1;
    float max_radius = rs * 1.6;

    for(int it=0; it < 32000; it++)
    {
        float interp = clamp(lightray_spacetime_position.y, min_radius, max_radius);

        float frac = (interp - min_radius) / (max_radius - min_radius);

        float ds = mix(max_ds, min_ds, frac);

        #if 1
        if(lightray_spacetime_position.y < (rs + rs * 0.0000001))
        {
            write_imagef(out, (int2){cx, cy}, (float4){0,0,0,1});
            return;
        }

        if(lightray_spacetime_position.y > 20)
        {
            float3 cart_here = polar_to_cartesian(lightray_spacetime_position.yzw);

            //float thetaf = fmod(lightray_spacetime_position.z, 2 * M_PI);
            //float phif = lightray_spacetime_position.w;

            //cart_here = rotate_vector(new_basis_x, new_basis_y, new_basis_z, cart_here);

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


            //#define BLUESHIFT
            #ifdef BLUESHIFT
            float3 start_col = {1,1,1};
            float3 blueshift_col = {0, 0, 1};

            /*so, blueshift is
            ///linf / le = 1/sqrt(a - rs/re)
            ///so, obviously infinities are bad, and infinite shifting is hard to represent
            ///so lets say instead that at the event horizon, its maximally blue

            ///so, the blueshift equation can be expressed as 1/sqrt(1 - x)
            ///if we remap that equation to be between 0 and 1, we can instead get
            ///1/(sqrt(1 - 0.75 * x)) - 1
            ///this gives a well behaved blueshift-like curve*/

            float x_val = rs / lightray_start_position.y;

            float shift_fraction = (1/(sqrt(1 - 0.75 * x_val))) - 1;

            shift_fraction = clamp(shift_fraction, 0.f, 1.f);

            float3 my_val = mix(start_col, blueshift_col, shift_fraction);

            float3 linear_val = srgb_to_lin(val.xyz);

            float radiant_energy = linear_val.x * 0.2125 + linear_val.y*0.7154 + linear_val.z*0.0721;

            float3 rotated_val = mix(linear_val, radiant_energy * (blueshift_col / 0.0721), shift_fraction);

            rotated_val = clamp(rotated_val, 0.f, 1.f);

            val.xyz = lin_to_srgb(rotated_val);
            #endif // BLUESHIFT

            write_imagef(out, (int2){cx, cy}, val);
            return;
        }

        calculate_metric(lightray_spacetime_position, g_metric);

        /*if(cx == width/2 && cy == height/2)
        {
            float fds = calculate_ds(lightray_velocity, g_metric);

            printf("FDS %f\n", fds);
        }*/

        /*
        g_metric_out[0] = -c * c * (1 - rs / r);
        g_metric_out[1] = 1/(1 - rs / r);
        g_metric_out[2] = r * r;
        g_metric_out[3] = r * r * sin(theta) * sin(theta);*/

        ///so metric 3 is degenerate around the poles, because 1/(r*r*sin(theta)*sin(theta)) -> infinity

        float christoff[64] = {0};

        float r = lightray_spacetime_position.y;

        #ifndef IS_CONSTANT_THETA
        float theta = lightray_spacetime_position.z;
        #else
        float theta = M_PI/2;
        #endif // IS_CONSTANT_THETA

        float g_partials[16] = {0};

        g_partials[0 * 4 + 1] = -c*c*rs/(r*r);
        g_partials[1 * 4 + 1] = -rs / ((rs - r) * (rs - r));
        g_partials[2 * 4 + 1] = 2 * r;
        g_partials[3 * 4 + 1] = 2 * r * sin(theta) * sin(theta);
        g_partials[3 * 4 + 2] = 2 * r * r * sin(theta) * cos(theta);

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

            if(!isfinite(sum))
            {
                write_imagef(out, (int2){cx, cy}, (float4){1, 0, 0, 1});
                return;
            }

            christ_result[uu] = sum;
        }

        /*float curvature = 0;

        {
            for(int a = 0; a < 4; a++)
            {
                int b = a;

                float cur = g_inv[a];


                float accum = 0;

                for(int c=0; c < 4; c++)
                {
                    accum += christoff[]
                }
            }
        }*/

        float curvature = 0;

        //for(int i=0)


        float4 acceleration = {-christ_result[0], -christ_result[1], -christ_result[2], -christ_result[3]};
        #endif // 0

        lightray_spacetime_position += lightray_velocity * ds;

        lightray_velocity += acceleration * ds;
        lightray_velocity = fix_light_velocity(lightray_velocity, g_metric);

        /*if((cx == width/2 && cy == height/2) || (cx == width-2 && cy == height/2) || (cx == 0 && cy == height/2))
        {
            float3 lp = lightray_spacetime_position.yzw;

            lp.x *= 10;

            float3 world_pos = (float3){lp.x * sin(lp.y) * cos(lp.z),
                                        lp.x * sin(lp.y) * sin(lp.z),
                                        lp.x * cos(lp.y)};

            world_pos = rotate_vector(new_basis_x, new_basis_y, new_basis_z, world_pos);

            world_pos.x += width/2;
            world_pos.y += height/2;
            world_pos.z += height/2;

            float4 write_col = (float4){1,0,1,1};

            world_pos = round(world_pos) + 0.5;

            write_col = pow(write_col, 1/2.2);

            //printf("%f r\n", lightray_spacetime_position.y);

            //printf("world pos %f\n", lp.x);

            if(all(world_pos.xz > 0) && all(world_pos.xz < (float2){width-2, height-2}))
                write_imagef(out, (int2){world_pos.x, world_pos.z}, write_col);

            write_imagef(out, (int2){width/2, height/2}, (float4){1, 0, 0, 1});
        }*/
    }


    /*float frac = (float)cx / width;

    write_imagef(out, (int2){cx, cy}, (float4){frac, 0, 0, 1});*/

    /*float pixel_angle = dot(cartesian_velocity, (float3){0, 0, 1});

    write_imagef(out, (int2){cx, cy}, (float4){pixel_angle, 0, 0, 1});*/



    write_imagef(out, (int2){cx, cy}, (float4){0, 1, 0, 1});
}
