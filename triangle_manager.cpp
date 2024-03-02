#include "triangle_manager.hpp"
#include <tinyobjloader/tiny_obj_loader.h>
#include "print.hpp"
#include "physics.hpp"

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

    return ret;

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

void triangle_rendering::manager::build(cl::command_queue& cqueue)
{
    std::vector<triangle> linear_tris;
    std::vector<gpu_object> gpu_objects;
    std::vector<cl_float3> gpu_velocities;

    for(auto& i : cpu_objects)
    {
        i->last_pos = i->pos;
        i->last_velocity = i->velocity;

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

    ///should the opencl library handle 0 sized allocations?
    objects.alloc(sizeof(gpu_object) * std::max(gpu_objects.size(), 1ull));
    objects.write(cqueue, gpu_objects);

    objects_velocity.alloc(sizeof(cl_float3) * std::max(gpu_objects.size(), 1ull));
    objects_velocity.write(cqueue, gpu_velocities);

    gpu_object_count = gpu_objects.size();

    tri_count = linear_tris.size();

    tris.alloc(sizeof(triangle) * std::max(tri_count, 1));
    tris.write(cqueue, linear_tris);
}

void triangle_rendering::manager::update_objects(cl::command_queue& cqueue)
{
    for(std::shared_ptr<object>& obj : cpu_objects)
    {
        if(obj->gpu_offset == -1)
            continue;

        if(obj->pos != obj->last_pos && obj->velocity != obj->last_velocity)
            continue;

        obj->last_pos = obj->pos;
        obj->last_velocity = obj->velocity;

        gpu_object gobj(*obj);
        cl_float3 next_velocity = {obj->velocity.x(), obj->velocity.y(), obj->velocity.z()};

        objects.write(cqueue, (const char*)&gobj, sizeof(gpu_object), sizeof(gpu_object) * obj->gpu_offset);
        objects_velocity.write(cqueue, (const char*)&next_velocity, sizeof(cl_float3), sizeof(cl_float3) * obj->gpu_offset);
    }
}

void triangle_rendering::manager::force_update_objects(cl::command_queue& cqueue)
{
    update_objects(cqueue);
}
