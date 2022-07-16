#ifndef TRIANGLE_HPP_INCLUDED
#define TRIANGLE_HPP_INCLUDED

#include <vec/vec.hpp>

///so: core assumption, triangles are not smeared across diferent times
///when objects come, each object will have its own unique time coordinate
struct triangle
{
    int parent = -1;
    float time = 0;

    float v0x = 0, v0y = 0, v0z = 0;
    float v1x = 0, v1y = 0, v1z = 0;
    float v2x = 0, v2y = 0, v2z = 0;

    void set_vert(int which, vec3f pos)
    {
        if(which == 0)
        {
            v0x = pos.x();
            v0y = pos.y();
            v0z = pos.z();
        }

        if(which == 1)
        {
            v1x = pos.x();
            v1y = pos.y();
            v1z = pos.z();
        }

        if(which == 2)
        {
            v2x = pos.x();
            v2y = pos.y();
            v2z = pos.z();
        }
    }

    vec3f get_vert(int which) const
    {
        if(which == 0)
        {
            return {v0x, v0y, v0z};
        }

        if(which == 1)
        {
            return {v1x, v1y, v1z};
        }

        if(which == 2)
        {
            return {v2x, v2y, v2z};
        }

        assert(false);
    }
};

struct subtriangle
{
    subtriangle(){}

    subtriangle(int p, const triangle& t)
    {
        parent = p;

        v0x = t.v0x;
        v0y = t.v0y;
        v0z = t.v0z;

        v1x = t.v1x;
        v1y = t.v1y;
        v1z = t.v1z;

        v2x = t.v2x;
        v2y = t.v2y;
        v2z = t.v2z;
    }

    void set_vert(int which, vec3f pos)
    {
        if(which == 0)
        {
            v0x = pos.x();
            v0y = pos.y();
            v0z = pos.z();
        }

        if(which == 1)
        {
            v1x = pos.x();
            v1y = pos.y();
            v1z = pos.z();
        }

        if(which == 2)
        {
            v2x = pos.x();
            v2y = pos.y();
            v2z = pos.z();
        }
    }

    vec3f get_vert(int which) const
    {
        if(which == 0)
        {
            return {v0x, v0y, v0z};
        }

        if(which == 1)
        {
            return {v1x, v1y, v1z};
        }

        if(which == 2)
        {
            return {v2x, v2y, v2z};
        }

        assert(false);
    }

    int parent = 0;

    float v0x = 0, v0y = 0, v0z = 0;
    float v1x = 0, v1y = 0, v1z = 0;
    float v2x = 0, v2y = 0, v2z = 0;
};

#endif // TRIANGLE_HPP_INCLUDED
