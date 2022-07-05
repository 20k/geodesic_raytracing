#ifndef TRIANGLE_MANAGER_HPP_INCLUDED
#define TRIANGLE_MANAGER_HPP_INCLUDED

#include <CL/cl.h>
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
        std::array<subtriangle, 4> subtriangulate(const subtriangle& t)
        {
            vec3f v0 = t.get_vert(0);
            vec3f v1 = t.get_vert(1);
            vec3f v2 = t.get_vert(2);

            vec3f h01 = (v0 + v1)/2;
            vec3f h12 = (v1 + v2)/2;
            vec3f h20 = (v2 + v0)/2;

            std::array<subtriangle, 4> res;

            for(subtriangle& o : res)
            {
                o.parent = t.parent;
            }

            std::vector<vec3f> st0 = {v0, h01, h20};
            std::vector<vec3f> st1 = {v1, h12, h01};
            std::vector<vec3f> st2 = {v2, h20, h12};
            std::vector<vec3f> st3 = {h01, h12, h20};

            for(int i=0; i < 3; i++)
            {
                res[0].set_vert(i, st0[i]);
                res[1].set_vert(i, st1[i]);
                res[2].set_vert(i, st2[i]);
                res[3].set_vert(i, st3[i]);
            }

            return res;
        }

        struct sub_point
        {
            cl_float x, y, z;
            cl_int parent;
            cl_int object_parent;
        };

        inline
        std::vector<subtriangle> triangulate_those_bigger_than(const std::vector<subtriangle>& in, float size)
        {
            std::vector<subtriangle> ret;

            bool any = false;

            for(const subtriangle& t : in)
            {
                vec3f v0 = t.get_vert(0);
                vec3f v1 = t.get_vert(1);
                vec3f v2 = t.get_vert(2);

                float l0 = (v1 - v0).length();
                float l1 = (v2 - v1).length();
                float l2 = (v0 - v2).length();

                if(l0 >= size || l1 >= size || l2 >= size)
                {
                    auto res = subtriangulate(t);

                    for(auto& i : res)
                    {
                        any = true;

                        ret.push_back(i);
                    }
                }
                else
                {
                    ret.push_back(t);
                }
            }

            if(any)
                return triangulate_those_bigger_than(ret, size);

            return ret;
        }

        inline
        std::vector<subtriangle> triangulate_those_bigger_than(const std::vector<triangle>& in, float size)
        {
            std::vector<subtriangle> ret;

            for(int i=0; i < (int)in.size(); i++)
            {
                subtriangle stri(i, in[i]);

                ret.push_back(stri);
            }

            return triangulate_those_bigger_than(ret, size);
        }
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

        std::shared_ptr<object> make_new()
        {
            std::shared_ptr<object> obj = std::make_shared<object>();

            cpu_objects.push_back(obj);

            return obj;
        }

        void build(cl::command_queue& cqueue, float acceleration_voxel_size)
        {
            acceleration_needs_rebuild = true;

            std::vector<triangle> linear_tris;
            std::vector<gpu_object> gpu_objects;

            for(auto& i : cpu_objects)
            {
                int parent = gpu_objects.size();

                for(triangle& t : i->tris)
                {
                    t.parent = parent;
                }

                linear_tris.insert(linear_tris.end(), i->tris.begin(), i->tris.end());

                gpu_object obj(*i);

                i->gpu_offset = parent;

                gpu_objects.push_back(obj);
            }

            objects.alloc(sizeof(gpu_object) * gpu_objects.size());
            objects.write(cqueue, gpu_objects);

            tri_count = linear_tris.size();

            using namespace impl;

            std::vector<std::pair<vec3f, int>> global_subtri_as_points;

            int global_running_tri_index = 0;

            for(auto& i : cpu_objects)
            {
                std::vector<subtriangle> sub = triangulate_those_bigger_than(i->tris, acceleration_voxel_size);

                std::vector<std::pair<vec3f, int>> local_subtri_as_points;

                for(subtriangle& t : sub)
                {
                    local_subtri_as_points.push_back({t.get_vert(0), t.parent + global_running_tri_index});
                    local_subtri_as_points.push_back({t.get_vert(1), t.parent + global_running_tri_index});
                    local_subtri_as_points.push_back({t.get_vert(2), t.parent + global_running_tri_index});
                }

                global_running_tri_index += i->tris.size();

                for(auto& [point, p] : local_subtri_as_points)
                {
                    float scale = acceleration_voxel_size;

                    vec3f vox = point / scale;

                    vox = floor(vox);

                    point = vox * scale;
                }

                std::sort(local_subtri_as_points.begin(), local_subtri_as_points.end(), [](auto& i1, auto& i2)
                {
                    return std::tie(i1.first.z(), i1.first.y(), i1.first.x(), i1.second) < std::tie(i2.first.z(), i2.first.y(), i2.first.x(), i2.second);
                });

                local_subtri_as_points.erase(std::unique(local_subtri_as_points.begin(), local_subtri_as_points.end()), local_subtri_as_points.end());

                global_subtri_as_points.insert(global_subtri_as_points.end(), local_subtri_as_points.begin(), local_subtri_as_points.end());
            }

            std::cout << "FIN POINTS " << global_subtri_as_points.size() << std::endl;

            std::vector<sub_point> gpu;

            for(auto& p : global_subtri_as_points)
            {
                sub_point point;
                point.x = p.first.x();
                point.y = p.first.y();
                point.z = p.first.z();
                point.parent = p.second;
                point.object_parent = linear_tris[point.parent].parent;

                gpu.push_back(point);
            }

            tris.alloc(sizeof(triangle) * tri_count);
            tris.write(cqueue, linear_tris);

            fill_point_count = gpu.size();
            fill_points.alloc(sizeof(sub_point) * gpu.size());
            fill_points.write(cqueue, gpu);
        }

        void update_objects(cl::command_queue& cqueue)
        {
            for(std::shared_ptr<object>& obj : cpu_objects)
            {
                gpu_object gobj(*obj);

                if(obj->gpu_offset == -1)
                    continue;

                if(!obj->dirty)
                    continue;

                objects.write(cqueue, (const char*)&gobj, sizeof(gpu_object), sizeof(gpu_object) * obj->gpu_offset);

                obj->dirty = false;

                acceleration_needs_rebuild = true;
            }
        }
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
