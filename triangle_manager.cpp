#include "triangle_manager.hpp"
#include "physics.hpp"
#include "print.hpp"
#include <tinyobjloader/tiny_obj_loader.h>

struct sub_point
{
    cl_float x, y, z;
    cl_int parent;
    cl_int object_parent;
};

std::array<subtriangle, 4> subtriangulate(const subtriangle& t)
{
    vec3f v0 = t.get_vert(0);
    vec3f v1 = t.get_vert(1);
    vec3f v2 = t.get_vert(2);

    vec3f h01 = (v0 + v1)/2.f;
    vec3f h12 = (v1 + v2)/2.f;
    vec3f h20 = (v2 + v0)/2.f;

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

std::shared_ptr<triangle_rendering::object> triangle_rendering::manager::make_new()
{
    std::shared_ptr<object> obj = std::make_shared<object>();

    cpu_objects.push_back(obj);

    return obj;
}

std::vector<triangle> triangle_rendering::load_tris_from_model(const std::string& model_name)
{
    tinyobj::ObjReaderConfig reader_config;
    reader_config.mtl_search_path = "./"; // Path to material files

    tinyobj::ObjReader reader;

    if(!reader.ParseFromFile(model_name, reader_config))
    {
        if(!reader.Error().empty())
        {
            std::cerr << "TinyObjReader: " << reader.Error();
        }

        throw std::runtime_error("Could not load object " + model_name);
    }

    if(!reader.Warning().empty())
    {
        std::cout << "TinyObjReader: " << reader.Warning();
    }

    auto& attrib = reader.GetAttrib();
    auto& shapes = reader.GetShapes();
    //auto& materials = reader.GetMaterials();

    std::vector<triangle> ret;

    // Loop over shapes
    for(const tinyobj::shape_t& s : shapes)
    {
        // Loop over faces(polygon)
        size_t index_offset = 0;

        for(size_t fv : s.mesh.num_face_vertices)
        {
            if(fv != 3)
                throw std::runtime_error("Must be used with triangulation");

            triangle tri;

            // Loop over vertices in the face.
            for(size_t v = 0; v < fv; v++)
            {
                // access to vertex
                tinyobj::index_t idx = s.mesh.indices[index_offset + v];
                tinyobj::real_t vx = attrib.vertices[3*size_t(idx.vertex_index)+0];
                tinyobj::real_t vy = attrib.vertices[3*size_t(idx.vertex_index)+1];
                tinyobj::real_t vz = attrib.vertices[3*size_t(idx.vertex_index)+2];

                tri.set_vert(v, {vx, vy, vz});

                // Check if `normal_index` is zero or positive. negative = no normal data
                /*if(idx.normal_index >= 0)
                {
                    tinyobj::real_t nx = attrib.normals[3*size_t(idx.normal_index)+0];
                    tinyobj::real_t ny = attrib.normals[3*size_t(idx.normal_index)+1];
                    tinyobj::real_t nz = attrib.normals[3*size_t(idx.normal_index)+2];
                }

                // Check if `texcoord_index` is zero or positive. negative = no texcoord data
                if(idx.texcoord_index >= 0)
                {
                    tinyobj::real_t tx = attrib.texcoords[2*size_t(idx.texcoord_index)+0];
                    tinyobj::real_t ty = attrib.texcoords[2*size_t(idx.texcoord_index)+1];
                }*/

                // Optional: vertex colors
                // tinyobj::real_t red   = attrib.colors[3*size_t(idx.vertex_index)+0];
                // tinyobj::real_t green = attrib.colors[3*size_t(idx.vertex_index)+1];
                // tinyobj::real_t blue  = attrib.colors[3*size_t(idx.vertex_index)+2];
            }

            ret.push_back(tri);

            index_offset += fv;

            // per-face material
            //shapes[s].mesh.material_ids[f];
        }
    }

    return ret;
}

std::shared_ptr<triangle_rendering::object> triangle_rendering::manager::make_new(const std::string& model_name)
{
    std::shared_ptr<object> obj = std::make_shared<object>();

    obj->tris = load_tris_from_model(model_name);

    cpu_objects.push_back(obj);

    return obj;
}

void triangle_rendering::manager::build(cl::command_queue& cqueue, float acceleration_voxel_size)
{
    acceleration_needs_rebuild = true;

    std::vector<triangle> linear_tris;
    std::vector<gpu_object> gpu_objects;
    std::vector<cl_float3> gpu_velocities;

    for(auto& i : cpu_objects)
    {
        int parent = gpu_objects.size();

        for(triangle& t : i->tris)
        {
            t.parent = parent;
        }

        auto with_scale = i->get_tris_with_scale();
        linear_tris.insert(linear_tris.end(), with_scale.begin(), with_scale.end());

        gpu_object obj(*i);

        i->gpu_offset = parent;

        gpu_objects.push_back(obj);
        gpu_velocities.push_back({i->velocity.x(), i->velocity.y(), i->velocity.z()});
    }

    objects.alloc(sizeof(gpu_object) * gpu_objects.size());
    objects.write(cqueue, gpu_objects);

    objects_velocity.alloc(sizeof(cl_float3) * gpu_objects.size());
    objects_velocity.write(cqueue, gpu_velocities);

    gpu_object_count = gpu_objects.size();

    tri_count = linear_tris.size();

    std::vector<std::pair<vec3f, int>> global_subtri_as_points;

    int global_running_tri_index = 0;

    for(auto& i : cpu_objects)
    {
        std::vector<subtriangle> sub = triangulate_those_bigger_than(i->get_tris_with_scale(), acceleration_voxel_size);

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

    printj("FIN POINTS ", global_subtri_as_points.size());

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

void triangle_rendering::manager::update_objects(cl::command_queue& cqueue)
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

triangle_rendering::acceleration::acceleration(cl::context& ctx, cl::command_queue& cqueue) : offsets(ctx), counts(ctx), unculled_counts(ctx), memory(ctx), memory_count(ctx), cell_time_min(ctx), cell_time_max(ctx), global_min(ctx), global_max(ctx)
{
    memory_count.alloc(sizeof(cl_int));

    int cells = offset_size.x() * offset_size.y() * offset_size.z() * offset_size.w();

    offsets.alloc(sizeof(cl_int) * cells);
    counts.alloc(sizeof(cl_int) * cells);
    unculled_counts.alloc(sizeof(cl_int) * cells);
    memory.alloc(max_memory_size);

    cl_int min_set = INT_MAX;
    cl_int max_set = INT_MIN;

    cell_time_min.alloc(sizeof(cl_int) * cells);
    cell_time_max.alloc(sizeof(cl_int) * cells);

    global_min.alloc(sizeof(cl_int));
    global_max.alloc(sizeof(cl_int));

    cell_time_min.fill(cqueue, min_set);
    cell_time_max.fill(cqueue, max_set);

    global_min.fill(cqueue, min_set);
    global_max.fill(cqueue, max_set);
}

void triangle_rendering::acceleration::build(cl::command_queue& cqueue, manager& tris, physics& phys, cl::buffer& dynamic_config)
{
    //if(!tris.acceleration_needs_rebuild)
    //    return;

    tris.acceleration_needs_rebuild = false;

    memory_count.set_to_zero(cqueue);

    {
        int count = counts.alloc_size / sizeof(cl_int);

        cl::args aclear;
        aclear.push_back(counts);
        aclear.push_back(count);

        cqueue.exec("clear_accel_counts", aclear, {count}, {256});
    }

    {
        int count = unculled_counts.alloc_size / sizeof(cl_int);

        cl::args aclear;
        aclear.push_back(unculled_counts);
        aclear.push_back(count);

        cqueue.exec("clear_accel_counts", aclear, {count}, {256});
    }

    {
        int should_store = 0;
        int generate_unculled = 1;

        cl::args gen;
        gen.push_back(tris.fill_points);
        gen.push_back(tris.fill_point_count);
        gen.push_back(tris.objects);
        gen.push_back(tris.gpu_object_count);
        gen.push_back(tris.tris);
        gen.push_back(offsets);
        gen.push_back(counts);
        gen.push_back(memory);
        gen.push_back(unculled_counts);
        gen.push_back(cell_time_min);
        gen.push_back(cell_time_max);
        gen.push_back(phys.geodesic_paths);
        gen.push_back(phys.counts);
        gen.push_back(offset_width);
        gen.push_back(offset_size.x());
        gen.push_back(should_store);
        gen.push_back(generate_unculled);
        gen.push_back(dynamic_config);

        cqueue.exec("generate_smeared_acceleration", gen, {tris.fill_point_count}, {256});
    }

    {
        cl::args accel;
        accel.push_back(offsets);
        accel.push_back(counts);
        accel.push_back(offset_size.x());
        accel.push_back(memory_count);
        accel.push_back(max_memory_size);

        cqueue.exec("alloc_acceleration", accel, {offset_size.x(), offset_size.y(), offset_size.z()}, {8, 8, 1});
    }

    {
        int should_store = 1;
        int generate_unculled = 0;

        cl::args gen;
        gen.push_back(tris.fill_points);
        gen.push_back(tris.fill_point_count);
        gen.push_back(tris.objects);
        gen.push_back(tris.gpu_object_count);
        gen.push_back(tris.tris);
        gen.push_back(offsets);
        gen.push_back(counts);
        gen.push_back(memory);
        gen.push_back(unculled_counts);
        gen.push_back(cell_time_min);
        gen.push_back(cell_time_max);
        gen.push_back(phys.geodesic_paths);
        gen.push_back(phys.counts);
        gen.push_back(offset_width);
        gen.push_back(offset_size.x());
        gen.push_back(should_store);
        gen.push_back(generate_unculled);
        gen.push_back(dynamic_config);

        cqueue.exec("generate_smeared_acceleration", gen, {tris.fill_point_count}, {256});
    }

    #if 0
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
        accel.push_back(max_memory_size);

        cqueue.exec("alloc_acceleration", accel, {offset_size.x(), offset_size.y(), offset_size.z()}, {8, 8, 1});
    }

    {
        cl::args gen;
        gen.push_back(tris.fill_points);
        gen.push_back(tris.fill_point_count);
        gen.push_back(tris.objects);
        gen.push_back(tris.tris);
        gen.push_back(offsets);
        gen.push_back(counts);
        gen.push_back(memory);
        gen.push_back(offset_width);
        gen.push_back(offset_size.x());

        cqueue.exec("generate_acceleration_data", gen, {tris.fill_point_count}, {256});
    }
    #endif
}
