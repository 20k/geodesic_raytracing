#ifndef CAMERA_HPP_INCLUDED
#define CAMERA_HPP_INCLUDED

#include <vec/vec.hpp>

struct camera
{
    vec4f pos;
    quat rot;

    camera()
    {
        pos = {0, 0, -4, 0};
        rot.load_from_axis_angle({1, 0, 0, -M_PI/2});
    }

    void handle_input(vec2f mouse_delta, vec4f keyboard_input, float universe_size)
    {
        ///translation is: .x is forward - back, .y = right - left, .z = down - up
        ///totally arbitrary
        quat local_camera_quat = rot;

        if(mouse_delta.x() != 0)
        {
            quat q;
            q.load_from_axis_angle((vec4f){0, 0, -1, mouse_delta.x()});

            local_camera_quat = q * local_camera_quat;
        }

        {
            vec3f right = rot_quat((vec3f){1, 0, 0}, local_camera_quat);

            if(mouse_delta.y() != 0)
            {
                quat q;
                q.load_from_axis_angle({right.x(), right.y(), right.z(), mouse_delta.y()});

                local_camera_quat = q * local_camera_quat;
            }
        }

        vec4f local_camera_pos_cart = pos;

        vec3f up = {0, 0, -1};
        vec3f right = rot_quat((vec3f){1, 0, 0}, local_camera_quat);
        vec3f forw = rot_quat((vec3f){0, 0, 1}, local_camera_quat);

        vec3f offset = {0,0,0};

        offset += forw * keyboard_input.x();
        offset += right * keyboard_input.y();
        offset += up * keyboard_input.z();

        local_camera_pos_cart.x() += keyboard_input.w();
        local_camera_pos_cart.y() += offset.x();
        local_camera_pos_cart.z() += offset.y();
        local_camera_pos_cart.w() += offset.z();

        {
            float rad = local_camera_pos_cart.yzw().length();

            if(rad > universe_size * 0.99f)
            {
                vec3f next = local_camera_pos_cart.yzw().norm() * universe_size * 0.99f;

                local_camera_pos_cart.y() = next.x();
                local_camera_pos_cart.z() = next.y();
                local_camera_pos_cart.w() = next.z();
            }
        }

        pos = local_camera_pos_cart;
        rot = local_camera_quat;
    }
};

#endif // CAMERA_HPP_INCLUDED
