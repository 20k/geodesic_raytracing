#ifndef PHYSICS_HPP_INCLUDED
#define PHYSICS_HPP_INCLUDED

#include <vec/vec.hpp>
#include <toolkit/opencl.hpp>
#include "triangle_manager.hpp"

struct physics
{
    int max_path_length = 16000;
    cl::buffer geodesic_paths;
    cl::buffer geodesic_velocities;
    cl::buffer geodesic_ds;
    cl::buffer positions;
    cl::buffer counts;
    cl::buffer basis_speeds;

    cl::buffer gpu_object_count;

    std::array<cl::buffer, 4> tetrads;
    std::array<cl::buffer, 4> parallel_transported_tetrads;
    std::array<cl::buffer, 4> inverted_tetrads;
    cl::buffer polar_positions;
    cl::buffer timelike_vectors;

    int object_count = 0;

    bool needs_trace = false;

    physics(cl::context& ctx) : geodesic_paths(ctx), geodesic_velocities(ctx), geodesic_ds(ctx), positions(ctx), counts(ctx), basis_speeds(ctx),
                                gpu_object_count(ctx),
                                tetrads{ctx, ctx, ctx, ctx}, parallel_transported_tetrads{ctx, ctx, ctx, ctx}, inverted_tetrads{ctx, ctx, ctx, ctx}, polar_positions(ctx), timelike_vectors(ctx)
    {
        gpu_object_count.alloc(sizeof(cl_int));
    }

    void setup(cl::command_queue& cqueue, triangle_rendering::manager& manage)
    {
        int clamped_count = std::max(manage.gpu_object_count, 1);

        object_count = manage.gpu_object_count;

        ///need to pull geodesic initial position from gpu tris
        geodesic_paths.alloc(clamped_count * sizeof(cl_float4) * max_path_length);
        geodesic_velocities.alloc(clamped_count * sizeof(cl_float4) * max_path_length);
        geodesic_ds.alloc(clamped_count * sizeof(cl_float) * max_path_length);
        positions.alloc(clamped_count * sizeof(cl_float4));
        counts.alloc(clamped_count * sizeof(cl_int));
        basis_speeds.alloc(clamped_count * sizeof(cl_float3));
        basis_speeds.set_to_zero(cqueue);

        for(int i=0; i < 4; i++)
        {
            tetrads[i].alloc(clamped_count * sizeof(cl_float4));
            parallel_transported_tetrads[i].alloc(clamped_count * sizeof(cl_float4) * max_path_length);
            inverted_tetrads[i].alloc(clamped_count * sizeof(cl_float4) * max_path_length);
        }

        polar_positions.alloc(clamped_count * sizeof(cl_float4));
        timelike_vectors.alloc(clamped_count * 1024); ///approximate because don't want to import gpu lightray definition

        counts.set_to_zero(cqueue);

        needs_trace = true;
    }

    void init_positions(cl::command_queue& cqueue, triangle_rendering::manager& manage)
    {
        cl::args args;
        args.push_back(manage.objects);
        args.push_back(positions);
        args.push_back(manage.gpu_object_count);

        cqueue.exec("pull_object_positions", args, {manage.gpu_object_count}, {256});
    }

    void trace(cl::command_queue& cqueue, triangle_rendering::manager& manage, cl::buffer& dynamic_config, cl::buffer& dynamic_feature_buffer)
    {
        if(!needs_trace)
            return;

        #define SMEARED
        #ifdef SMEARED
        manage.force_update_objects(cqueue);
        #endif // PHYSICS_HPP_INCLUDED

        init_positions(cqueue, manage);

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
            tetrad_args.push_back(object_count);
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
            cl::args lorentz;
            lorentz.push_back(polar_positions);
            lorentz.push_back(object_count);
            lorentz.push_back(manage.objects_velocity);

            for(auto& i : tetrads)
            {
                lorentz.push_back(i);
            }

            lorentz.push_back(dynamic_config);

            cqueue.exec("boost_tetrad", lorentz, {object_count}, {256});
        }

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
            snapshot_args.push_back(geodesic_paths); // position
            snapshot_args.push_back(geodesic_velocities);        // velocity
            snapshot_args.push_back(nullptr);        // dT_ds
            snapshot_args.push_back(geodesic_ds);    // ds
            snapshot_args.push_back(gpu_object_count);
            snapshot_args.push_back(max_path_length);
            snapshot_args.push_back(dynamic_config);
            snapshot_args.push_back(dynamic_feature_buffer);
            snapshot_args.push_back(counts);

            cqueue.exec("get_geodesic_path", snapshot_args, {object_count}, {256});
        }

        for(int i=0; i < 4; i++)
        {
            cl::args pt_args;
            pt_args.push_back(geodesic_paths);
            pt_args.push_back(geodesic_velocities);
            pt_args.push_back(geodesic_ds);
            pt_args.push_back(tetrads[i]);
            pt_args.push_back(counts);
            pt_args.push_back(object_count);
            pt_args.push_back(parallel_transported_tetrads[i]);
            pt_args.push_back(dynamic_config);

            cqueue.exec("parallel_transport_quantity", pt_args, {object_count}, {256});
        }


        cl::args invert_args;
        invert_args.push_back(counts);
        invert_args.push_back(object_count);

        for(int i=0; i < 4; i++)
        {
            invert_args.push_back(parallel_transported_tetrads[i]);
        }

        for(int i=0; i < 4; i++)
        {
            invert_args.push_back(inverted_tetrads[i]);
        }

        cqueue.exec("calculate_tetrad_inverse", invert_args, {object_count}, {256});

        needs_trace = false;
    }

    void push_object_positions(cl::command_queue& cqueue, triangle_rendering::manager& manage, cl::buffer& dynamic_config, float target_time)
    {
        cl::args args;
        args.push_back(geodesic_paths);
        args.push_back(counts);
        args.push_back(manage.objects);
        args.push_back(target_time);
        args.push_back(object_count);
        args.push_back(dynamic_config);

        cqueue.exec("push_object_positions", args, {object_count}, {256});
    }
};

#endif // PHYSICS_HPP_INCLUDED
