#ifndef RENDER_STATE_HPP_INCLUDED
#define RENDER_STATE_HPP_INCLUDED

#include <CL/cl.h>
#include <array>
#include <toolkit/texture.hpp>

struct render_state
{
    cl::buffer g_camera_pos_cart;
    cl::buffer g_camera_pos_generic;
    cl::buffer g_camera_pos_polar_readback;
    cl::buffer g_camera_quat;

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
};

#endif // RENDER_STATE_HPP_INCLUDED
