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
    cl_int sx;
    cl_int sy;
};

struct render_data_struct
{
    cl_float2 tex_coord;
    cl_float z_shift;
    cl_int sx;
    cl_int sy;
    cl_int terminated;
    cl_int side;
    cl_float ku_uobsu;
};

struct single_render_state
{
    cl::buffer stored_rays;

    cl::buffer tri_list_allocator;
    cl::buffer tri_list_offsets;
    cl::buffer tri_list1;
    cl::buffer tri_list_counts1;

    cl::buffer computed_tris;
    cl::buffer computed_tri_count;
    int max_tris = 1024 * 1024 * 200;

    bool allocated = false;

    uint32_t last_width = 0;
    uint32_t last_height = 0;
    cl_int last_max_stored_rays = 0;

    single_render_state(cl::context& ctx) :
        stored_rays(ctx),
        tri_list_allocator(ctx), tri_list_offsets(ctx),
        tri_list1(ctx), tri_list_counts1(ctx),
        computed_tris(ctx), computed_tri_count(ctx)
    {

    }

    void deallocate(cl::context& ctx)
    {
        *this = single_render_state(ctx);
    }

    void lazy_allocate(uint32_t width, uint32_t height, int max_stored)
    {
        max_stored = clamp(max_stored, 1, 1000);

        if(!allocated || last_width != width || last_height != height)
        {
            computed_tris.alloc(1024 * 1024 * 800);
            computed_tri_count.alloc(sizeof(cl_int));
            tri_list_allocator.alloc(sizeof(cl_ulong));

            tri_list_offsets.alloc(sizeof(cl_ulong) * width * height);
            tri_list1.alloc(sizeof(cl_int) * max_tris);
            tri_list_counts1.alloc(sizeof(cl_int) * width * height);

            stored_rays.alloc(sizeof(cl_float4) * width * height * max_stored);
        }

        if(!allocated || last_max_stored_rays != max_stored)
        {
            stored_rays.alloc(sizeof(cl_float4) * width * height * max_stored);
        }

        last_width = width;
        last_height = height;
        last_max_stored_rays = max_stored;

        allocated = true;
    }
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

    cl::buffer rays_adaptive;
    cl::buffer rays_adaptive_count;

    cl::buffer render_data;
    cl::buffer render_data_count;

    cl::buffer termination_buffer;

    cl::buffer texture_coordinates;

    cl::buffer accel_ray_time_min;
    cl::buffer accel_ray_time_max;

    cl::buffer stored_ray_counts;

    cl::buffer stored_mins;
    cl::buffer stored_maxs;

    cl::buffer chunked_mins;
    cl::buffer chunked_maxs;

    cl::buffer already_rendered;

    int width = 0;
    int height = 0;

    render_state(cl::context& ctx, cl::command_queue& cqueue) :
        g_camera_pos_cart(ctx), g_camera_quat(ctx),
        g_camera_pos_generic(ctx), g_camera_pos_polar_readback(ctx), g_geodesic_basis_speed(ctx),
        tetrad{ctx, ctx, ctx, ctx},
        rays_in(ctx),
        rays_count_in(ctx),
        rays_adaptive(ctx),
        rays_adaptive_count(ctx),
        render_data(ctx),
        render_data_count(ctx),
        termination_buffer(ctx),
        texture_coordinates(ctx),
        accel_ray_time_min(ctx), accel_ray_time_max(ctx),
        stored_ray_counts(ctx),
        stored_mins(ctx), stored_maxs(ctx),
        chunked_mins(ctx), chunked_maxs(ctx),

        already_rendered(ctx)
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
        rays_adaptive_count.alloc(sizeof(cl_int));
        render_data_count.alloc(sizeof(cl_int));

        accel_ray_time_min.alloc(sizeof(cl_int));
        accel_ray_time_max.alloc(sizeof(cl_int));
    }

    void realloc(uint32_t _width, uint32_t _height)
    {
        width = _width;
        height = _height;

        uint32_t ray_count = width * height;

        rays_in.alloc(ray_count * sizeof(lightray));
        rays_adaptive.alloc(ray_count * sizeof(lightray));
        render_data.alloc(ray_count * sizeof(render_data_struct));

        termination_buffer.alloc(width * height * sizeof(cl_int));
        texture_coordinates.alloc(width * height * sizeof(float) * 2);

        stored_ray_counts.alloc(sizeof(cl_int) * width * height);

        stored_mins.alloc(sizeof(cl_float4) * width * height);
        stored_maxs.alloc(sizeof(cl_float4) * width * height);

        ///width, height / workgroup_size in reality
        chunked_mins.alloc(sizeof(cl_float4) * width * height);
        chunked_maxs.alloc(sizeof(cl_float4) * width * height);

        already_rendered.alloc(sizeof(cl_int) * width * height);
    }
};

#endif // RENDER_STATE_HPP_INCLUDED
