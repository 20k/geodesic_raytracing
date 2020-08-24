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
    float v[4] = {velocity.x, velocity.y, velocity.z, velocity.w};

    /*
    float ds = 0;

    ds += g_metric[0] * velocity_arr[0] * velocity_arr[0];
    ds += g_metric[1] * velocity_arr[1] * velocity_arr[1];
    ds += g_metric[2] * velocity_arr[2] * velocity_arr[2];
    ds += g_metric[3] * velocity_arr[3] * velocity_arr[3];
    */

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
    float rs = 0.5;
    float c = 1;

    float r = spacetime_position.y;
    float theta = spacetime_position.z;

    g_metric_out[0] = -c * c * (1 - rs / r);
    g_metric_out[1] = 1/(1 - rs / r);
    g_metric_out[2] = r * r;
    g_metric_out[3] = r * r * pow(sin(theta), 2);
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

__kernel
void do_raytracing(__write_only image2d_t out, float ds, float4 cartesian_camera_pos, float4 polar_camera_pos)//, __global struct light_ray* inout_rays, __global struct light_ray* inout_deltas)
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

    ///inverse of diagonal matrix is 1/ all the entries

    /*float mat[64] = {0};

    for(int i=0; i < 4; i++)
    {
        for(int k=0; k < 4; k++)
        {
            for(int l=0; l < 4; l++)
            {
                float sum = 0;

                #define INDEX(i,


                sum += g_inv[i] * g_partial()
            }
        }
    }*/

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

    pixel_virtual_pos = normalize(pixel_virtual_pos) / 299792458.f;

    float3 cartesian_velocity = fast_normalize(pixel_virtual_pos);

    //printf("CVEL %f %f %f\n", cartesian_velocity.x, cartesian_velocity.y, cartesian_velocity.z);

    float3 polar_velocity = cartesian_velocity_to_polar_velocity(pixel_virtual_pos + cartesian_camera_pos.yzw, cartesian_velocity);

    float3 bad_light_velocity = polar_velocity;

    //float4 spacetime = (float4)(100,2,M_PI/2,M_PI);
    float4 lightray_start_position = polar_camera_pos;

    float4 lightray_spacetime_position = lightray_start_position;

    float rs = 0.5;
    float c = 1;

    float g_metric[4];

    calculate_metric(lightray_spacetime_position, g_metric);

    float4 lightray_velocity = fix_light_velocity((float4)(1, bad_light_velocity.xyz), g_metric);

    //write_imagef(out, (int2){cx, cy}, (float4){0, 0, 0, 1});

    for(int i=0; i < 2048; i++)
    {
        if(lightray_spacetime_position.y < (rs + rs * 0.05))
        {
            //write_imagef(out, (int2){cx, cy}, (float4){1, 0, 0, 1});
            return;
        }

        calculate_metric(lightray_spacetime_position, g_metric);

        //fix_light_velocity(lightray_velocity, g_metric);

        ///diagonal of the metric, because it only has diagonals
        /*float g_inv[4] = {1/g_metric[0], 1/g_metric[1], 1/g_metric[2], 1/g_metric[3]};

        float g_partial[16] = {0,0,0,0,
            -c*c * rs / (r*r), -rs/((rs - r) * (rs - r)), 2 * r, 2 * r * (sin(theta) * sin(theta)),
            0,0,0,2 * r * r * sin(theta) * cos(theta),
            0,0,0,0
        };*/

        float r = lightray_spacetime_position.y;
        float theta = lightray_spacetime_position.z;

        float christoff[64] = {
            0,+ (-1 / ((1 - rs / r) * c * c)) * (-rs * c * c / (r * r)),0,0,+ (-1 / ((1 - rs / r) * c * c)) * (-rs * c * c / (r * r)),0,0,0,0,0,0,0,0,0,0,0,- (1 - rs / r) * (-rs * c * c / (r * r)),0,0,0,0,+ (1 - rs / r) * (-rs / pow(r - rs, 2)) + (1 - rs / r) * (-rs / pow(r - rs, 2)) + - (1 - rs / r) * (-rs / pow(r - rs, 2)),0,0,0,0,- (1 - rs / r) * (2 * r),0,0,0,0,- (1 - rs / r) * (2 * r * pow(sin(theta), 2)),0,0,0,0,0,0,+ (1 / (r * r)) * (2 * r),0,0,+ (1 / (r * r)) * (2 * r),0,0,0,0,0,- (1 / (r * r)) * (2 * r * r * sin(theta) * cos(theta)),0,0,0,0,0,0,0,+ (1 / pow(r * sin(theta), 2)) * (2 * r * pow(sin(theta), 2)),0,0,0,+ (1 / pow(r * sin(theta), 2)) * (2 * r * r * sin(theta) * cos(theta)),0,+ (1 / pow(r * sin(theta), 2)) * (2 * r * pow(sin(theta), 2)),+ (1 / pow(r * sin(theta), 2)) * (2 * r * r * sin(theta) * cos(theta)),0
        };

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

            if(isnan(sum))
                return;

            christ_result[uu] = sum;
        }

        float4 acceleration = {-christ_result[0], -christ_result[1], -christ_result[2], -christ_result[3]};

        lightray_velocity += acceleration * ds;
        lightray_spacetime_position += lightray_velocity * ds;

        /*if(i == 0 && cx == width / 2 && cy == height/2)
        {
            printf("%f light", lightray_spacetime_position.y);
        }*/

        if(cx == width/2 && cy == height/2 || (cx == width-2 && cy == height/2) || (cx == 0 && cy == height/2))
        {
            float3 lp = lightray_spacetime_position.yzw;

            lp.x *= 10;

            float3 world_pos = (float3){lp.x * sin(lp.y) * cos(lp.z),
                                        lp.x * sin(lp.y) * sin(lp.z),
                                        lp.x * cos(lp.y)};

            world_pos.x += width/2;
            world_pos.y += height/2;
            world_pos.z += height/2;

            //printf("%f r\n", lightray_spacetime_position.y);

            //printf("world pos %f\n", lp.x);

            if(all(world_pos.xz > 0) && all(world_pos.xz < (float2){width-2, height-2}))
                write_imagef(out, (int2){world_pos.x, world_pos.z}, (float4){1, 0, 1, 1});
        }
    }



    //return;

    /*float frac = (float)cx / width;

    write_imagef(out, (int2){cx, cy}, (float4){frac, 0, 0, 1});*/

    /*float pixel_angle = dot(cartesian_velocity, (float3){0, 0, 1});

    write_imagef(out, (int2){cx, cy}, (float4){pixel_angle, 0, 0, 1});*/
    //write_imagef(out, (int2){cx, cy}, (float4){0, 1, 0, 1});
}
