#ifndef TRIANGLE_MANAGER_HPP_INCLUDED
#define TRIANGLE_MANAGER_HPP_INCLUDED

#include <memory>
#include <toolkit/opencl.hpp>
#include "triangle.hpp"

///so. This is never going to be a full on 3d renderer
///highest tri count we might get is 100k tris
///each object represents something that is on the same time coordinate, and also may follow a geodesic
///need to subtriangulate and deduplicate individually
namespace triangle_rendering
{
    struct object
    {
        vec4f pos;
        std::vector<triangle> tris;

        int gpu_offset = -1;

        void set_pos(vec4f in)
        {
            pos = in;

            dirty = true;
        }

        bool dirty = false;
    };

    namespace impl
    {
        struct sub_point
        {
            cl_float x, y, z;
            cl_int parent;
            cl_int object_parent;
        };
    }

    struct gpu_object
    {
        vec4f pos;

        gpu_object(const object& o)
        {
            pos = o.pos;
        }
    };

    struct manager
    {
        std::vector<std::shared_ptr<object>> cpu_objects;

        int tri_count = 0;
        cl::buffer objects;
        cl::buffer tris;

        int fill_point_count = 0;
        cl::buffer fill_points;

        bool acceleration_needs_rebuild = false;

        manager(cl::context& ctx) : objects(ctx), tris(ctx), fill_points(ctx)
        {

        }

        std::shared_ptr<object> make_new();

        void build(cl::command_queue& cqueue, float acceleration_voxel_size);
        void update_objects(cl::command_queue& cqueue);
    };

    struct acceleration
    {
        cl::buffer offsets;
        cl::buffer counts;
        cl::buffer memory;
        cl::buffer memory_count;

        vec3i offset_size = {128, 128, 128};
        float offset_width = 20;

        acceleration(cl::context& ctx) : offsets(ctx), counts(ctx), memory(ctx), memory_count(ctx)
        {
            memory_count.alloc(sizeof(cl_int));

            int cells = offset_size.x() * offset_size.y() * offset_size.z();

            offsets.alloc(sizeof(cl_int) * cells);
            counts.alloc(sizeof(cl_int) * cells);
            memory.alloc(1024 * 1024 * 128 * sizeof(cl_int));
        }

        void build(cl::command_queue& cqueue, manager& tris)
        {
            if(!tris.acceleration_needs_rebuild)
                return;

            tris.acceleration_needs_rebuild = false;

            memory_count.set_to_zero(cqueue);

            {
                cl::args aclear;
                aclear.push_back(counts);
                aclear.push_back(offset_size.x());

                cqueue.exec("clear_accel_counts", aclear, {offset_size.x(), offset_size.y(), offset_size.z()}, {8, 8, 1});
            }

            {
                cl::args count_args;
                count_args.push_back(tris.fill_points);
                count_args.push_back(tris.fill_point_count);
                count_args.push_back(tris.objects);
                count_args.push_back(counts);
                count_args.push_back(offset_width);
                count_args.push_back(offset_size.x());

                cqueue.exec("generate_acceleration_counts", count_args, {tris.fill_point_count}, {256});
            }

            {
                cl::args accel;
                accel.push_back(offsets);
                accel.push_back(counts);
                accel.push_back(offset_size.x());
                accel.push_back(memory_count);

                cqueue.exec("alloc_acceleration", accel, {offset_size.x(), offset_size.y(), offset_size.z()}, {8, 8, 1});
            }

            {
                cl::args gen;
                gen.push_back(tris.fill_points);
                gen.push_back(tris.fill_point_count);
                gen.push_back(tris.objects);
                gen.push_back(offsets);
                gen.push_back(counts);
                gen.push_back(memory);
                gen.push_back(offset_width);
                gen.push_back(offset_size.x());

                cqueue.exec("generate_acceleration_data", gen, {tris.fill_point_count}, {256});
            }
        }
    };
}


#endif // TRIANGLE_MANAGER_HPP_INCLUDED
