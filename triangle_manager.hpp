#ifndef TRIANGLE_MANAGER_HPP_INCLUDED
#define TRIANGLE_MANAGER_HPP_INCLUDED

#include <memory>
#include <toolkit/opencl.hpp>
#include "triangle.hpp"

struct physics;

///so. This is never going to be a full on 3d renderer
///highest tri count we might get is 100k tris
///each object represents something that is on the same time coordinate, and also may follow a geodesic
///need to subtriangulate and deduplicate individually
namespace triangle_rendering
{
    std::vector<triangle> load_tris_from_model(const std::string& model_name);

    struct object
    {
        vec4f pos;
        vec3f velocity;
        std::vector<triangle> tris;
        float scale = 1;

        vec4f last_pos;
        vec3f last_velocity;

        int gpu_offset = -1;
        int physics_offset = -1;

        void set_pos(vec4f in)
        {
            pos = in;
        }

        std::vector<triangle> get_tris_with_scale()
        {
            if(scale == 1)
                return tris;

            auto ret = tris;

            for(triangle& t : ret)
            {
                vec3f v0 = t.get_vert(0);
                vec3f v1 = t.get_vert(1);
                vec3f v2 = t.get_vert(2);

                v0 *= scale;
                v1 *= scale;
                v2 *= scale;

                t.set_vert(0, v0);
                t.set_vert(1, v1);
                t.set_vert(2, v2);
            }

            return ret;
        }
    };

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
        int gpu_object_count = 0;
        cl::buffer objects;
        cl::buffer objects_velocity;
        cl::buffer tris;

        manager(cl::context& ctx) : objects(ctx), objects_velocity(ctx), tris(ctx)
        {

        }

        std::shared_ptr<object> make_new();
        ///split into dir and name
        std::shared_ptr<object> make_new(const std::string& model_name);

        void build(cl::command_queue& cqueue);
        void update_objects(cl::command_queue& cqueue);
        void force_update_objects(cl::command_queue& cqueue);
    };
}

#endif // TRIANGLE_MANAGER_HPP_INCLUDED
