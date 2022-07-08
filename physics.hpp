#ifndef PHYSICS_HPP_INCLUDED
#define PHYSICS_HPP_INCLUDED

#include <vec/vec.hpp>
#include <toolkit/opencl.hpp>
#include "triangle_manager.hpp"

std::string pull_kernel = R"(
struct object
{
    float4 pos;
};

__kernel void pull(__global struct object* current_pos, __global float4* geodesic_out, int object_count)
{
    int id = get_global_id(0);

    if(id >= object_count)
        return;

    geodesic_out[id] = current_pos[id].pos;
}
)";

struct physics
{
    int max_path_length = 16000;
    cl::buffer geodesic_paths;
    cl::buffer positions;
    cl::buffer counts;
    cl::buffer basis_speeds;

    cl::buffer gpu_object_count;

    std::array<cl::buffer, 4> tetrads;
    cl::buffer polar_positions;
    cl::buffer timelike_vectors;

    cl::program prog;
    cl::kernel pull;

    int object_count = 0;

    physics(cl::context& ctx) : geodesic_paths(ctx), positions(ctx), counts(ctx), basis_speeds(ctx),
                                gpu_object_count(ctx),
                                tetrads{ctx, ctx, ctx, ctx}, polar_positions(ctx), timelike_vectors(ctx),
                                prog(ctx, pull_kernel, false)
    {
        ctx.register_program(prog);

        gpu_object_count.alloc(sizeof(cl_int));
    }

    void setup(cl::command_queue& cqueue, triangle_rendering::manager& manage)
    {
        object_count = manage.gpu_object_count;

        ///need to pull geodesic initial position from gpu tris
        geodesic_paths.alloc(manage.gpu_object_count * sizeof(cl_float4) * max_path_length);
        positions.alloc(manage.gpu_object_count * sizeof(cl_float4));
        counts.alloc(manage.gpu_object_count * sizeof(cl_int));
        basis_speeds.alloc(manage.gpu_object_count * sizeof(cl_float3));
        basis_speeds.set_to_zero(cqueue);

        for(int i=0; i < 4; i++)
        {
            tetrads[i].alloc(manage.gpu_object_count * sizeof(cl_float4));
        }

        polar_positions.alloc(manage.gpu_object_count * sizeof(cl_float4));
        timelike_vectors.alloc(manage.gpu_object_count * 1024); ///approximate because don't want to import gpu lightray definition

        counts.set_to_zero(cqueue);

        init_positions(cqueue, manage);
    }

    void init_positions(cl::command_queue& cqueue, triangle_rendering::manager& manage)
    {
        cl::args args;
        args.push_back(manage.objects);
        args.push_back(positions);
        args.push_back(manage.gpu_object_count);

        cqueue.exec("pull", args, {manage.gpu_object_count}, {256});
    }

    void trace(cl::command_queue& cqueue, cl::buffer& dynamic_config)
    {
        counts.set_to_zero(cqueue);

        cl_float4 cartesian_basis_speed = {0,0,0,0};

        {
            cl_float clflip = 0;

            cl::args to_polar;
            to_polar.push_back(positions);
            to_polar.push_back(polar_positions);
            to_polar.push_back(object_count);
            to_polar.push_back(clflip);

            cqueue.exec("cart_to_polar_kernel", to_polar, {object_count}, {256});
        }

        {
            cl::args tetrad_args;
            tetrad_args.push_back(polar_positions);
            tetrad_args.push_back(cartesian_basis_speed);

            for(auto& i : tetrads)
            {
                tetrad_args.push_back(i);
            }

            tetrad_args.push_back(dynamic_config);

            cqueue.exec("init_basis_vectors", tetrad_args, {object_count}, {256});
        }

        ///would boost here if we were going to
        {
            gpu_object_count.set_to_zero(cqueue);

            cl::args args;
            args.push_back(polar_positions);
            args.push_back(object_count);
            args.push_back(timelike_vectors);
            args.push_back(gpu_object_count);

            for(auto& i : tetrads)
            {
                args.push_back(i);
            }

            args.push_back(basis_speeds);
            args.push_back(dynamic_config);

            cqueue.exec("init_inertial_ray", args, {object_count}, {256});
        }

        {
            cl::args snapshot_args;
            snapshot_args.push_back(timelike_vectors);
            snapshot_args.push_back(geodesic_paths);
            snapshot_args.push_back(nullptr); // velocity
            snapshot_args.push_back(nullptr); // dT_ds
            snapshot_args.push_back(nullptr); // ds
            snapshot_args.push_back(gpu_object_count);
            snapshot_args.push_back(max_path_length);
            snapshot_args.push_back(dynamic_config);
            snapshot_args.push_back(counts);

            cqueue.exec("get_geodesic_path", snapshot_args, {object_count}, {256});
        }
    }
};

#endif // PHYSICS_HPP_INCLUDED
