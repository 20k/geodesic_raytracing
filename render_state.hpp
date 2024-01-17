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
    cl_uint sx, sy;
    cl_float ku_uobsu;
    cl_int early_terminate;
    cl_float running_dlambda_dnew;
};

struct render_state
{
    //cl::buffer g_camera_pos_cart;
    cl::buffer g_camera_pos_generic;
    cl::buffer g_camera_pos_polar_readback;
    //cl::buffer g_camera_quat;
    cl::buffer g_geodesic_basis_speed;

    std::array<cl::buffer, 4> tetrad;

    cl::buffer rays_in;
    cl::buffer rays_out;
    cl::buffer rays_finished;
    cl::buffer rays_prepass;

    cl::buffer rays_count_in;
    cl::buffer rays_count_out;
    cl::buffer rays_count_finished;
    cl::buffer rays_count_prepass;

    cl::buffer tri_intersections;
    cl::buffer tri_intersections_count;

    cl::buffer termination_buffer;

    cl::buffer texture_coordinates;

    texture tex;
    cl::gl_rendertexture rtex;

    cl::buffer accel_ray_time_min;
    cl::buffer accel_ray_time_max;

    render_state(cl::context& ctx, cl::command_queue& cqueue) :
        g_camera_pos_generic(ctx), g_camera_pos_polar_readback(ctx), g_geodesic_basis_speed(ctx),
        tetrad{ctx, ctx, ctx, ctx},
        rays_in(ctx), rays_out(ctx), rays_finished(ctx), rays_prepass(ctx),
        rays_count_in(ctx), rays_count_out(ctx), rays_count_finished(ctx), rays_count_prepass(ctx),
        tri_intersections(ctx), tri_intersections_count(ctx),
        termination_buffer(ctx),
        texture_coordinates(ctx),
        rtex(ctx),
        accel_ray_time_min(ctx), accel_ray_time_max(ctx)
    {
        //g_camera_pos_cart.alloc(sizeof(cl_float4));
        g_camera_pos_generic.alloc(sizeof(cl_float4));
        g_camera_pos_polar_readback.alloc(sizeof(cl_float4));
        //g_camera_quat.alloc(sizeof(cl_float4));
        g_geodesic_basis_speed.alloc(sizeof(cl_float4));

        for(auto& i : tetrad)
        {
            i.alloc(sizeof(cl_float4));
            i.set_to_zero(cqueue);
        }

        rays_count_in.alloc(sizeof(cl_int));
        rays_count_out.alloc(sizeof(cl_int));
        rays_count_finished.alloc(sizeof(cl_int));
        rays_count_prepass.alloc(sizeof(cl_int));

        //tri_intersections.alloc(sizeof());
        tri_intersections_count.alloc(sizeof(cl_int));

        accel_ray_time_min.alloc(sizeof(cl_int));
        accel_ray_time_max.alloc(sizeof(cl_int));

        {
            cl_float4 camera_start_pos = {0, 0, -4, 0};

            quat camera_start_quat;
            camera_start_quat.load_from_axis_angle({1, 0, 0, -M_PI/2});

            //g_camera_pos_cart.write(cqueue, std::span{&camera_start_pos, 1});

            cl_float4 as_cl_camera_quat = {camera_start_quat.q.x(), camera_start_quat.q.y(), camera_start_quat.q.z(), camera_start_quat.q.w()};

            //g_camera_quat.write(cqueue, std::span{&as_cl_camera_quat, 1});
        }
    }

    void realloc(uint32_t width, uint32_t height)
    {
        uint32_t ray_count = width * height;

        rays_in.alloc(ray_count * sizeof(lightray));
        rays_out.alloc(ray_count * sizeof(lightray));
        rays_finished.alloc(ray_count * sizeof(lightray));
        rays_prepass.alloc(ray_count * sizeof(lightray));

        texture_settings new_sett;
        new_sett.width = width;
        new_sett.height = height;
        new_sett.is_srgb = false;
        new_sett.generate_mipmaps = false;

        tex.load_from_memory(new_sett, nullptr);
        rtex.create_from_texture(tex.handle);

        termination_buffer.alloc(width * height * sizeof(cl_int));
        texture_coordinates.alloc(width * height * sizeof(float) * 2);

        tri_intersections.alloc(sizeof(cl_int) * width * height * 10);
    }
};

#endif // RENDER_STATE_HPP_INCLUDED
