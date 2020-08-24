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
    float r = sqrt(in.x * in.x + in.y * in.y + in.z * in.z);
    //float theta = atan2(sqrt(in.x * in.x + in.y * in.y), in.z);
    float theta = acos(in.z / r);
    float phi = atan2(in.y, in.x);

    return (float3){r, theta, phi};
}

float3 cartesian_velocity_to_polar_velocity(float3 cartesian_position, float3 cartesian_velocity)
{
    float3 p = cartesian_position;
    float3 v = cartesian_velocity;

    float rdot = (p.x * v.x + p.y * v.y + p.z * v.z) / sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
    float tdot = (v.x * p.y - p.x * v.y) / (p.x * p.x + p.y * p.y);
    float pdot = (p.z * (p.x * v.x + p.y * v.y) - (p.x * p.x + p.y * p.y) * v.z) / ((p.x * p.x + p.y * p.y + p.z * p.z) * sqrt(p.x * p.x + p.y * p.y));

    return (float3){rdot, tdot, pdot};
}

float4 fix_light_velocity(float4 velocity)
{

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

    #define FOV 90

    float fov_rad = (FOV / 360.f) * 2 * M_PI;

    int cx = get_global_id(0);
    int cy = get_global_id(1);

    float width = get_image_width(out);
    float height = get_image_height(out);

    float nonphysical_plane_half_width = width/2;
    float nonphysical_f_stop = nonphysical_plane_half_width / tan(fov_rad/2);

    ///need to rotate by camera angle
    float3 pixel_virtual_pos = (float3){cx - width/2, cy - height/2, nonphysical_f_stop};

    float3 cartesian_velocity = fast_normalize(pixel_virtual_pos);

    float3 polar_velocity = cartesian_velocity_to_polar_velocity(pixel_virtual_pos + cartesian_camera_pos, cartesian_velocity);

    float4 bad_light_velocity = polar_velocity;

    float4 lightray_velocity = fix_velocity(bad_light_velocity);

    //float4 spacetime = (float4)(100,2,M_PI/2,M_PI);
    float4 spacetime = polar_camera_pos;

    float4 lightray_spacetime_position = spacetime;

    float rs = 1;
    float c = 1;

    float r = lightray_spacetime_position.y;
    float theta = lightray_spacetime_position.z;

    float g_metric[] = {-c * c * (1 - rs / r),
                     1/(1 - rs / r),
                     r * r,
                     r * r * pow(sin(theta), 2)};

    ///diagonal of the metric, because it only has diagonals
    float g_inv[] = {1/g_metric[0], 1/g_metric[1], 1/g_metric[2], 1/g_metric[3]};

    float g_partial[16] = {0,0,0,0,
        -c*c * rs / (r*r), -rs/((rs - r) * (rs - r)), 2 * r, 2 * r * (sin(theta) * sin(theta)),
        0,0,0,2 * r * r * sin(theta) * cos(theta),
        0,0,0,0
    };

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

    float mat[64] = {
    0,+ (-1 / ((1 - rs / r) * c * c)) * (-rs * c * c / (r * r)),0,0,+ (-1 / ((1 - rs / r) * c * c)) * (-rs * c * c / (r * r)),0,0,0,0,0,0,0,0,0,0,0,- (1 - rs / r) * (-rs * c * c / (r * r)),0,0,0,0,+ (1 - rs / r) * (-rs / pow(r - rs, 2)) + (1 - rs / r) * (-rs / pow(r - rs, 2)) + - (1 - rs / r) * (-rs / pow(r - rs, 2)),0,0,0,0,- (1 - rs / r) * (2 * r),0,0,0,0,- (1 - rs / r) * (2 * r * pow(sin(theta), 2)),0,0,0,0,0,0,+ (1 / (r * r)) * (2 * r),0,0,+ (1 / (r * r)) * (2 * r),0,0,0,0,0,- (1 / (r * r)) * (2 * r * r * sin(theta) * cos(theta)),0,0,0,0,0,0,0,+ (1 / pow(r * sin(theta), 2)) * (2 * r * pow(sin(theta), 2)),0,0,0,+ (1 / pow(r * sin(theta), 2)) * (2 * r * r * sin(theta) * cos(theta)),0,+ (1 / pow(r * sin(theta), 2)) * (2 * r * pow(sin(theta), 2)),+ (1 / pow(r * sin(theta), 2)) * (2 * r * r * sin(theta) * cos(theta)),0
    };

    float frac = (float)cx / width;

    write_imagef(out, (int2){cx, cy}, (float4){frac, 0, 0, 1});
}
