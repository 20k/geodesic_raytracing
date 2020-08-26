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

    v[1] /= sqrt(time_scale);
    v[2] /= sqrt(time_scale);
    v[3] /= sqrt(time_scale);

    //v[0] *= sqrt(time_scale);

    ///should print 0
    /*float fds = calculate_ds((float4){v[0], v[1], v[2], v[3]}, g_metric);
    printf("%f fds\n", fds);*/

    return (float4){v[0], v[1], v[2], v[3]};
}

void calculate_metric(float4 spacetime_position, float g_metric_out[])
{
    float rs = 1;
    float c = 1;

    float r = spacetime_position.y;
    float theta = spacetime_position.z;

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

__kernel
void do_raytracing(__write_only image2d_t out, float ds_, float4 cartesian_camera_pos, float4 polar_camera_pos, float4 camera_quat, __read_only image2d_t background)
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

    ///need to rotate by camera angle

    ///this position is incredibly wrong
    float3 pixel_virtual_pos = (float3){cx - width/2, cy - height/2, nonphysical_f_stop};

    //pixel_virtual_pos = normalize(pixel_virtual_pos) / 299792458.f;

    pixel_virtual_pos = rot_quat(pixel_virtual_pos, camera_quat);

    float3 cartesian_velocity = normalize(pixel_virtual_pos);


    //printf("CVEL %f %f %f\n", cartesian_velocity.x, cartesian_velocity.y, cartesian_velocity.z);

    float3 polar_velocity = cartesian_velocity_to_polar_velocity(cartesian_camera_pos.yzw, cartesian_velocity);

    //float pvx = pixel_virtual_pos.x / (width/2);
    //float pvy = pixel_virtual_pos.y / (height/2);

    //float3 polar_velocity = {-1, pvx, pvy};

    //float3 polar_velocity = (float3){-1, 0, 0};

    float3 bad_light_velocity = polar_velocity;

    //float4 spacetime = (float4)(100,2,M_PI/2,M_PI);
    float4 lightray_start_position = polar_camera_pos;

    float4 lightray_spacetime_position = lightray_start_position;

    float rs = 1;
    float c = 1;

    float g_metric[4];

    calculate_metric(lightray_spacetime_position, g_metric);

    float4 lightray_velocity = fix_light_velocity((float4)(-1, bad_light_velocity.xyz), g_metric);

    //float4 lightray_velocity = (float4)(1, bad_light_velocity.xyz);

    //write_imagef(out, (int2){cx, cy}, (float4){0, 0, 0, 1});

    float ambient_precision = 0.1;

    float max_ds = 0.001;
    float min_ds = ambient_precision;

    float min_radius = 1.1;
    float max_radius = 1.6;

    float traced_quantity = 0;
    float total_ds = 0;

    for(int it=0; it < 120000; it++)
    {
        float interp = clamp(lightray_spacetime_position.y, min_radius, max_radius);

        float frac = (interp - min_radius) / (max_radius - min_radius);

        float ds = mix(max_ds, min_ds, frac);

        //float ds = ambient_precision;

        if(lightray_spacetime_position.z < M_PI/8)
        {
            ds = mix(0.0001, ambient_precision, lightray_spacetime_position.z / (M_PI/8));
        }

        if(lightray_spacetime_position.z >= (M_PI - M_PI/8))
        {
            float ffrac = (M_PI - lightray_spacetime_position.z) / (M_PI/8);
            ds = mix(0.0001, ambient_precision, ffrac);
        }

        if(lightray_spacetime_position.z < M_PI/(4096))
        {
            write_imagef(out, (int2){cx, cy}, (float4){0,0,0,1});
            return;
            //ds = mix(0.1, 0.1, lightray_spacetime_position.z / (M_PI/32));
        }

        if(lightray_spacetime_position.z >= M_PI - M_PI/(4096))
        {
            write_imagef(out, (int2){cx, cy}, (float4){0,0,0,1});
            return;
        }

        #if 1
        if(lightray_spacetime_position.y < (rs + rs * 0.0001))
        {
            /*if(total_ds == 0)
                return;

            write_imagef(out, (int2){cx, cy}, (float4){traced_quantity / total_ds, 0, 0, 1});*/

            write_imagef(out, (int2){cx, cy}, (float4){0,0,0,1});
            return;
        }

        if(lightray_spacetime_position.y > 20)
        {
            float thetaf = fmod(lightray_spacetime_position.z, 2 * M_PI);
            float phif = lightray_spacetime_position.w;

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

            if(total_ds == 0)
                total_ds = 1;

            //float quantity = clamp(traced_quantity / 64, 0.f, 1.f);
            float quantity = (traced_quantity / (1)) / (512);

            quantity = clamp(quantity, 0.f, 1.f);

            /*val = (float4){0,0,0,1};
            val.x = quantity;*/

            write_imagef(out, (int2){cx, cy}, val);
            return;
        }

        calculate_metric(lightray_spacetime_position, g_metric);

        /*
        g_metric_out[0] = -c * c * (1 - rs / r);
        g_metric_out[1] = 1/(1 - rs / r);
        g_metric_out[2] = r * r;
        g_metric_out[3] = r * r * sin(theta) * sin(theta);*/

        ///so metric 3 is degenerate around the poles, because 1/(r*r*sin(theta)*sin(theta)) -> infinity

        float christoff[64] = {0};

        float r = lightray_spacetime_position.y;
        float theta = lightray_spacetime_position.z;

        float g_partials[16] = {0};

        g_partials[0 * 4 + 1] = -c*c*rs/(r*r);
        g_partials[1 * 4 + 1] = -rs / ((rs - r) * (rs - r));
        g_partials[2 * 4 + 1] = 2 * r;
        g_partials[3 * 4 + 1] = 2 * r * sin(theta) * sin(theta);
        g_partials[3 * 4 + 2] = 2 * r * r * sin(theta) * cos(theta);

        {
            ///diagonal of the metric, because it only has diagonals
            float g_inv[4] = {1/g_metric[0], 1/g_metric[1], 1/g_metric[2], 0};

            for(int i=0; i < 3; i++)
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

            int i = 3;

            float result = tan(theta);

            if(fabs(result) < 0.0000001)
            {
                result = 0.0000001 * sign(result);
            }

            float third = 1/result;

            float half_partials_divided[4] = {0, 1 / r, third, 0.f};

            for(int m=0; m < 4; m++)
            {
                float adding = half_partials_divided[m];

                christoff[i * 16 + i * 4 + m] += adding;
                christoff[i * 16 + m * 4 + i] += adding;
                ///g_partials[m * 4 + i] = 0
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

        //float4 acceleration = 0;

        //lightray_spacetime_position += lightray_velocity * ds + 0.5 * acceleration * ds * ds;

        //lightray_velocity += ds * (last_acceleration + acceleration)/2;

        //float3 cartesian_acceleration = spherical_acceleration_to_cartesian_acceleration(lightray_spacetime_position.yzw, lightray_velocity.yzw, acceleration.yzw);

        /*float deformity = length(cartesian_acceleration);

        traced_quantity += deformity;
        total_ds += ds;*/

        //deformity = sqrt(deformity);

        /*if(ds == ambient_precision)
        {
            ds = mix(min_ds, max_ds, clamp(deformity, 0.f, 1.f));
        }*/

        //float3 old_velocity = spherical_velocity_to_cartesian_velocity(lightray_spacetime_position.yzw, lightray_velocity.yzw);
        //float old_time = lightray_velocity.x;

        lightray_spacetime_position += lightray_velocity * ds;
        lightray_velocity += acceleration * ds;

        /*float3 new_velocity = spherical_velocity_to_cartesian_velocity(lightray_spacetime_position.yzw, lightray_velocity.yzw);

        float4 accel = sqrt(fabs(dot((float4)(old_time, old_velocity), (float4)(lightray_velocity.x, new_velocity)))) / ds;

        //float deformity = length(accel) / 100;

        float base_deformity = 2 * sqrt(rs * (lightray_spacetime_position.y - rs));

        float maximum_deformity = 2 * sqrt(rs * (20 - rs));

        float deformity = (maximum_deformity - base_deformity) / maximum_deformity;*/




        //deformity /= 10;


        //float deformity = sqrt(fabs(dot(old_velocity, new_velocity))) / 4;
        //float deformity = length(acceleration) * 10;//fabs(dot(old_velocity, new_velocity));

        //traced_quantity += deformity;
        //total_ds += ds;

        /*if(cx == width/2 && cy == width/2)
        {
            printf("%f d\n", deformity / total_ds);
        }*/

        //float spatial_deformity = length(acceleration);

        /*if(i == 0 && cx == width / 2 && cy == height/2)
        {
            printf("%f light", lightray_spacetime_position.y);
        }*/

        /*if(lightray_spacetime_position.y < (rs + rs * 0.0001))
        {
            //write_imagef(out, (int2){cx, cy}, (float4){0, 0, 0, 1});
            return;
        }*/

        /*if((cx == width/2 && cy == height/2) || (cx == width-2 && cy == height/2) || (cx == 0 && cy == height/2))
        {
            float3 lp = lightray_spacetime_position.yzw;

            lp.x *= 10;

            float3 world_pos = (float3){lp.x * sin(lp.y) * cos(lp.z),
                                        lp.x * sin(lp.y) * sin(lp.z),
                                        lp.x * cos(lp.y)};

            world_pos.x += width/2;
            world_pos.y += height/2;
            world_pos.z += height/2;

            float4 write_col = (float4){1,0,1,1};

            float spatial_deformity = deformity;

            //spatial_deformity = clamp((spatial_deformity/16), 0.f, 1.f);

            write_col = (float4){0, spatial_deformity, 0, 1};

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
