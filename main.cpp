#include <iostream>
#include <toolkit/render_window.hpp>
#include <toolkit/texture.hpp>
#include <vec/vec.hpp>
#include <GLFW/glfw3.h>
#include <SFML/Graphics.hpp>

/*struct light_ray
{
    vec4f spacetime_velocity;
    vec4f spacetime_position;
};


vec4f calculate_lightbeam_spacetime_velocity_schwarz(vec3f position)
{

}*/

///not a good mapping
vec4f cartesian_to_schwarz(vec4f position)
{
    vec3f polar = cartesian_to_polar((vec3f){position.y(), position.z(), position.w()});

    return (vec4f){position.x(), polar.x(), polar.y(), polar.z()};
}

int main()
{
    render_settings sett;
    sett.width = 1500;
    sett.height = 800;
    sett.opencl = true;
    sett.no_double_buffer = true;

    render_window win(sett, "Geodesics");

    assert(win.clctx);

    opencl_context& clctx = *win.clctx;

    cl::program prog(clctx.ctx, "cl.cl");
    prog.build(clctx.ctx, "");

    clctx.ctx.register_program(prog);

    texture_settings tsett;
    tsett.width = sett.width;
    tsett.height = sett.height;
    tsett.is_srgb = false;

    texture tex;
    tex.load_from_memory(tsett, nullptr);

    cl::gl_rendertexture rtex(clctx.ctx);
    rtex.create_from_texture(tex.handle);

    sf::Image img;
    img.loadFromFile("background_med.png");

    cl::image clbackground(clctx.ctx);
    //clbackground.alloc(sizeof(uint8_t) * img.getSize().x * img.getSize().y);

    std::vector<vec4f> as_float;

    for(int y=0; y < img.getSize().y; y++)
    {
        for(int x=0; x < img.getSize().x; x++)
        {
            auto col = img.getPixel(x, y);

            vec4f val = {col.r / 255.f, col.g / 255.f, col.b / 255.f, col.a / 255.f};

            as_float.push_back(val);
        }
    }

    clbackground.alloc({img.getSize().x, img.getSize().y}, {CL_RGBA, CL_FLOAT});

    vec<2, size_t> origin = {0,0};
    vec<2, size_t> region = {img.getSize().x, img.getSize().y};

    clbackground.write(clctx.cqueue, (const char*)&as_float[0], origin, region);

    /*int pixels_width = win.get_window_size().x();
    int pixels_height = win.get_window_size().y();

    cl::buffer rays;
    rays.alloc(sizeof(light_ray) * pixels_width * pixels_height);

    cl::buffer deltas;
    deltas.alloc(sizeof(light_ray) * pixels_width * pixels_height);

    {
        vec3f camera = {0, -10000, 0};
        vec3f look_vector = {0, 0, 1};

        std::vector<light_ray> rays_data;
        std::vector<light_ray> deltas_data;

        for(int y=0; y < pixels_height; y++)
        {
            for(int x = 0; x < pixels_width; x++)
            {
                vec3f pos_3d = camera + (vec3f){x - pixels_width/2, y - pixels_height/2, camera.z};

                light_ray ray;
                ray.spacetime_position = (vec4f){0, pos_3d.x(), pos_3d.y(), pos_3d.z()};
                ray.spacetime_velocity = {1, 0, 0, 1};

                rays_data.push_back(ray);

                light_ray delta;
                delta.spacetime_position = ray.spacetime_position - (vec4f){0, 0, 0, 0};
                delta.spacetime_velocity = {0, 0, 0, 0};

                deltas_data.push_back(delta);
            }
        }

        rays.write(clctx.ctx, rays_data);
        deltas.write(clctx.ctx, deltas_data);
    }*/

    ///t, x, y, z
    vec4f camera = {0, 0.01, -0.024, -5};
    quat camera_quat;
    //camera_quat.load_from_matrix(axis_angle_to_mat({0, 0, 0}, 0));

    vec3f forward_axis = {0, 0, 1};
    vec3f up_axis = {0, 1, 0};

    bool kruskal = false;

    //vec4f camera =

    sf::Clock clk;

    while(!win.should_close())
    {
        win.poll();

        rtex.acquire(clctx.cqueue);

        float ds = 0.01;

        float speed = 0.001;

        if(ImGui::IsKeyDown(GLFW_KEY_LEFT_SHIFT))
            speed = 0.1;

        if(ImGui::IsKeyDown(GLFW_KEY_LEFT_CONTROL))
            speed = 0.00001;

        if(ImGui::IsKeyDown(GLFW_KEY_LEFT_ALT))
            speed /= 1000;

        if(ImGui::IsKeyDown(GLFW_KEY_Z))
            speed *= 100;

        if(ImGui::IsKeyDown(GLFW_KEY_X))
            speed *= 100;

        if(ImGui::IsKeyPressed(GLFW_KEY_B))
        {
            camera = {0, 0, 0, -100};
        }

        /*camera.y() += (ImGui::IsKeyDown(GLFW_KEY_D) - ImGui::IsKeyDown(GLFW_KEY_A)) * speed;
        camera.z() += (ImGui::IsKeyDown(GLFW_KEY_W) - ImGui::IsKeyDown(GLFW_KEY_S)) * speed;
        camera.w() += (ImGui::IsKeyDown(GLFW_KEY_E) - ImGui::IsKeyDown(GLFW_KEY_Q)) * speed;*/

        if(ImGui::IsKeyDown(GLFW_KEY_RIGHT))
        {
            mat3f m = mat3f().ZRot(M_PI/128);

            quat q;
            q.load_from_matrix(m);

            camera_quat = q * camera_quat;
        }

        if(ImGui::IsKeyDown(GLFW_KEY_LEFT))
        {
            mat3f m = mat3f().ZRot(-M_PI/128);

            quat q;
            q.load_from_matrix(m);

            camera_quat = q * camera_quat;
        }

        if(ImGui::IsKeyPressed(GLFW_KEY_T))
            kruskal = !kruskal;

        vec3f up = {0, 0, -1};
        vec3f right = rot_quat({1, 0, 0}, camera_quat);
        vec3f forward_axis = rot_quat({0, 0, 1}, camera_quat);

        if(ImGui::IsKeyDown(GLFW_KEY_DOWN))
        {
            quat q;
            q.load_from_axis_angle({right.x(), right.y(), right.z(), M_PI/128});

            camera_quat = q * camera_quat;
        }

        if(ImGui::IsKeyDown(GLFW_KEY_UP))
        {
            quat q;
            q.load_from_axis_angle({right.x(), right.y(), right.z(), -M_PI/128});

            camera_quat = q * camera_quat;
        }

        vec3f offset = {0,0,0};

        offset += forward_axis * ((ImGui::IsKeyDown(GLFW_KEY_W) - ImGui::IsKeyDown(GLFW_KEY_S)) * speed);
        offset += right * (ImGui::IsKeyDown(GLFW_KEY_D) - ImGui::IsKeyDown(GLFW_KEY_A)) * speed;
        offset += up * (ImGui::IsKeyDown(GLFW_KEY_E) - ImGui::IsKeyDown(GLFW_KEY_Q)) * speed;

        camera.y() += offset.x();
        camera.z() += offset.y();
        camera.w() += offset.z();

        //printf("%f camera\n", camera.z());

        vec4f scamera = cartesian_to_schwarz(camera);

        printf("Polar vals %f %f %f\n", scamera.y(), scamera.z(), scamera.w());

        //printf("scamera %f\n", scamera.y());

        cl::args clr;
        clr.push_back(rtex);

        clctx.cqueue.exec("clear", clr, {win.get_window_size().x(), win.get_window_size().y()}, {16, 16});

        cl::args args;
        args.push_back(rtex);
        args.push_back(ds);
        args.push_back(camera);
        args.push_back(camera_quat);
        args.push_back(clbackground);

        clctx.cqueue.exec("do_raytracing_multicoordinate", args, {win.get_window_size().x(), win.get_window_size().y()}, {16, 16});

        rtex.unacquire(clctx.cqueue);

        glFinish();
        clctx.cqueue.block();
        glFinish();

        {
            ImDrawList* lst = ImGui::GetBackgroundDrawList();

            ImVec2 screen_pos = ImGui::GetMainViewport()->Pos;

            ImVec2 tl = {0,0};
            ImVec2 br = {win.get_window_size().x(),win.get_window_size().y()};

            if(win.get_render_settings().viewports)
            {
                tl.x += screen_pos.x;
                tl.y += screen_pos.y;

                br.x += screen_pos.x;
                br.y += screen_pos.y;
            }

            lst->AddImage((void*)rtex.texture_id, tl, br);
        }

        win.display();

        double time = clk.restart().asMicroseconds() / 1000.;

        std::cout << "FRAMETIME " << time << std::endl;
    }

    return 0;
}
