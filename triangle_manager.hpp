#ifndef TRIANGLE_MANAGER_HPP_INCLUDED
#define TRIANGLE_MANAGER_HPP_INCLUDED

///so. This is never going to be a full on 3d renderer
///highest tri count we might get is 100k tris
///each object represents something that is on the same time coordinate, and also may follow a geodesic
namespace cpu
{
    struct object
    {
        vec4f pos;

        std::vector<triangle> tris;
    };
}

namespace gpu
{
    struct object
    {
        vec4f pos;
    };
}

struct triangle_manager
{
    std::vector<cpu::object> objects;
};

#endif // TRIANGLE_MANAGER_HPP_INCLUDED
