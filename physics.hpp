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
    cl::buffer geodesics;
    cl::buffer positions;
    cl::buffer counts;

    cl::program prog;
    cl::kernel pull;

    physics(cl::context& ctx) : geodesics(ctx), positions(ctx), counts(ctx), prog(ctx, pull_kernel, false)
    {
        ctx.register_program(prog);
    }

    void setup(cl::command_queue& cqueue, triangle_rendering::manager& manage)
    {
        ///need to pull geodesic initial position from gpu tris
        geodesics.alloc(manage.gpu_object_count * sizeof(cl_float4) * max_path_length);
        positions.alloc(manage.gpu_object_count * sizeof(cl_float4));
        counts.alloc(manage.gpu_object_count * sizeof(cl_int));

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

    void trace(cl::command_queue& cqueue)
    {
        counts.set_to_zero(cqueue);
    }
};

#endif // PHYSICS_HPP_INCLUDED
