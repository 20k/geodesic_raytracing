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
        accel_ray_time_min(ctx), accel_ray_time_max(ctx)
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
    }
};

#endif // RENDER_STATE_HPP_INCLUDED
