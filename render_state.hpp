#ifndef RENDER_STATE_HPP_INCLUDED
#define RENDER_STATE_HPP_INCLUDED

#include <CL/cl.h>
#include <array>
#include <toolkit/texture.hpp>

struct lightray
{
    cl_float4 position;
    cl_float4 velocity;
    cl_float4 initial_quat;
    cl_float4 acceleration;
    cl_float ku_uobsu;
    cl_float running_dlambda_dnew;
    cl_int terminated;
};

struct render_state
{
    cl::buffer g_camera_pos_cart;
    cl::buffer g_camera_quat;
    cl::buffer g_camera_pos_generic;
    cl::buffer g_camera_pos_polar_readback;
    cl::buffer g_geodesic_basis_speed;

    std::array<cl::buffer, 4> tetrad;

    cl::buffer rays_in;
    cl::buffer rays_count_in;

    cl::buffer tri_intersections;
    cl::buffer tri_intersections_count;

    cl::buffer termination_buffer;

    cl::buffer texture_coordinates;

    cl::buffer accel_ray_time_min;
    cl::buffer accel_ray_time_max;

    cl::buffer stored_rays;
    cl::buffer stored_ray_counts;
    cl_int max_stored = 90;

    cl::buffer stored_mins;
    cl::buffer stored_maxs;

    cl::buffer chunked_mins;
    cl::buffer chunked_maxs;

    cl::buffer tri_list1;
    cl::buffer tri_list_counts1;

    cl::buffer tri_list2;
    cl::buffer tri_list_counts2;

    cl::buffer computed_tris;
    cl::buffer computed_tri_count;

    int width = 0;
    int height = 0;

    render_state(cl::context& ctx, cl::command_queue& cqueue) :
        g_camera_pos_cart(ctx), g_camera_quat(ctx),
        g_camera_pos_generic(ctx), g_camera_pos_polar_readback(ctx), g_geodesic_basis_speed(ctx),
        tetrad{ctx, ctx, ctx, ctx},
        rays_in(ctx),
        rays_count_in(ctx),
        tri_intersections(ctx), tri_intersections_count(ctx),
        termination_buffer(ctx),
        texture_coordinates(ctx),
        accel_ray_time_min(ctx), accel_ray_time_max(ctx),
        stored_rays(ctx), stored_ray_counts(ctx),
        stored_mins(ctx), stored_maxs(ctx),
        chunked_mins(ctx), chunked_maxs(ctx),
        tri_list1(ctx), tri_list_counts1(ctx),
        tri_list2(ctx), tri_list_counts2(ctx),
        computed_tris(ctx), computed_tri_count(ctx)
    {
        g_camera_pos_cart.alloc(sizeof(cl_float4));
        g_camera_quat.alloc(sizeof(cl_float4));
        g_camera_pos_generic.alloc(sizeof(cl_float4));
        g_camera_pos_polar_readback.alloc(sizeof(cl_float4));
        g_geodesic_basis_speed.alloc(sizeof(cl_float4));

        for(auto& i : tetrad)
        {
            i.alloc(sizeof(cl_float4));
            i.set_to_zero(cqueue);
        }

        rays_count_in.alloc(sizeof(cl_int));

        //tri_intersections.alloc(sizeof());
        tri_intersections_count.alloc(sizeof(cl_int));

        accel_ray_time_min.alloc(sizeof(cl_int));
        accel_ray_time_max.alloc(sizeof(cl_int));

        computed_tris.alloc(1024 * 1024 * 102);
        computed_tri_count.alloc(sizeof(cl_int));
    }

    void realloc(uint32_t _width, uint32_t _height)
    {
        width = _width;
        height = _height;

        uint32_t ray_count = width * height;

        rays_in.alloc(ray_count * sizeof(lightray));

        termination_buffer.alloc(width * height * sizeof(cl_int));
        texture_coordinates.alloc(width * height * sizeof(float) * 2);

        tri_intersections.alloc(sizeof(cl_int) * width * height * 10);

        stored_rays.alloc(sizeof(cl_float4) * width * height * max_stored);
        stored_ray_counts.alloc(sizeof(cl_int) * width * height);

        stored_mins.alloc(sizeof(cl_float4) * width * height);
        stored_maxs.alloc(sizeof(cl_float4) * width * height);

        ///width, height / workgroup_size in reality
        chunked_mins.alloc(sizeof(cl_float4) * width * height);
        chunked_maxs.alloc(sizeof(cl_float4) * width * height);

        ///i need to do this properly, can't get away with the memory fudge so much
        tri_list1.alloc(sizeof(cl_int) * width * height * 50);
        tri_list_counts1.alloc(sizeof(cl_int) * width * height);

        ///i need to do this properly, can't get away with the memory fudge so much
        tri_list2.alloc(sizeof(cl_int) * width * height * 50);
        tri_list_counts2.alloc(sizeof(cl_int) * width * height);
    }
};

#endif // RENDER_STATE_HPP_INCLUDED
