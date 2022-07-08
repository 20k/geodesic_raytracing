#ifndef PHYSICS_HPP_INCLUDED
#define PHYSICS_HPP_INCLUDED

#include <vec/vec.hpp>
#include <toolkit/opencl.hpp>
#include "triangle_manager.hpp"

struct physics
{
    int max_path_length = 16000;
    cl::buffer geodesics;

    physics(cl::context& ctx) : geodesics(ctx)
    {

    }

    void setup(triangle_rendering::manager& manage)
    {
        ///need to pull geodesic initial position from gpu tris
        geodesics.alloc(manage.gpu_object_count * sizeof(cl_float4) * max_path_length);
    }
};

#endif // PHYSICS_HPP_INCLUDED
