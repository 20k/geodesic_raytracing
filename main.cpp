#include <iostream>
#include <toolkit/render_window.hpp>
#include <toolkit/texture.hpp>
#include <vec/vec.hpp>
#include <GLFW/glfw3.h>
#include <SFML/Graphics.hpp>
#include <CL/cl_ext.h>

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


struct lightray
{
    vec4f position;
    vec4f velocity;
    vec4f acceleration;
    int sx, sy;
};

int get_mipmap_size(int width, int height, int levels)
{
    int sum = 0;

    for(int i=levels-1; i >= 0; i--)
    {
        sum += width * height * 4;
        width /= 2;
        height /= 2;
    }

    return sum + 1;
}

int main()
{
    render_settings sett;
    sett.width = 1800;
    sett.height = 900;
    sett.opencl = true;
    sett.no_double_buffer = true;

    render_window win(sett, "Geodesics");

    assert(win.clctx);

    opencl_context& clctx = *win.clctx;

    cl::program prog(clctx.ctx, "cl.cl");
    prog.build(clctx.ctx, "-O3 -cl-std=CL2.2");

    clctx.ctx.register_program(prog);

    int supersample_mult = 2;

    int supersample_width = sett.width * supersample_mult;
    int supersample_height = sett.height * supersample_mult;

    texture_settings tsett;
    tsett.width = supersample_width;
    tsett.height = supersample_height;
    tsett.is_srgb = false;


    /*texture tex;
    tex.load_from_memory(tsett, nullptr);

    cl::gl_rendertexture rtex(clctx.ctx);
    rtex.create_from_texture(tex.handle);*/

    std::array<texture, 2> tex;
    tex[0].load_from_memory(tsett, nullptr);
    tex[1].load_from_memory(tsett, nullptr);

    std::array<cl::gl_rendertexture, 2> rtex{clctx.ctx, clctx.ctx};
    rtex[0].create_from_texture(tex[0].handle);
    rtex[1].create_from_texture(tex[1].handle);

    int which_buffer = 0;

    sf::Image img;
    img.loadFromFile("background_med.png");

    cl::image clbackground(clctx.ctx);
    //clbackground.alloc(sizeof(uint8_t) * img.getSize().x * img.getSize().y);

    std::vector<vec4f> as_float;
    std::vector<uint8_t> as_uint8;

    for(int y=0; y < img.getSize().y; y++)
    {
        for(int x=0; x < img.getSize().x; x++)
        {
            auto col = img.getPixel(x, y);

            vec4f val = {col.r / 255.f, col.g / 255.f, col.b / 255.f, col.a / 255.f};

            as_float.push_back(val);
            as_uint8.push_back(col.r);
            as_uint8.push_back(col.g);
            as_uint8.push_back(col.b);
            as_uint8.push_back(col.a);
        }
    }

    clbackground.alloc({img.getSize().x, img.getSize().y}, {CL_RGBA, CL_FLOAT});

    vec<2, size_t> origin = {0,0};
    vec<2, size_t> region = {img.getSize().x, img.getSize().y};

    clbackground.write(clctx.cqueue, (const char*)&as_float[0], origin, region);


    /*cl::image_with_mipmaps clmippedbackground(clctx.ctx);
    clmippedbackground.alloc((vec2i){img.getSize().x, img.getSize().y}, 4, {CL_RGBA, CL_FLOAT});

    clmippedbackground.write(clctx.cqueue, (const char*)&as_float[0], origin, region, 0);*/

    texture_settings bsett;
    bsett.width = img.getSize().x;
    bsett.height = img.getSize().y;
    bsett.is_srgb = false;

    texture background_with_mips;
    background_with_mips.load_from_memory(bsett, &as_uint8[0]);

    #define MIP_LEVELS 6

    /*cl::gl_rendertexture background_mipped[4] = {clctx.ctx, clctx.ctx, clctx.ctx, clctx.ctx};

    glFinish();

    for(int i=0; i < MIP_LEVELS; i++)
    {
        background_mipped[i].create_from_texture_with_mipmaps(background_with_mips.handle, i);

        background_mipped[i].acquire(clctx.cqueue);
    }*/

    cl::image_with_mipmaps background_mipped(clctx.ctx);
    background_mipped.alloc((vec2i){img.getSize().x, img.getSize().y}, 4, {CL_RGBA, CL_FLOAT});

    int swidth = img.getSize().x;
    int sheight = img.getSize().y;

    for(int i=0; i < MIP_LEVELS; i++)
    {
        printf("I is %i\n", i);

        //int cwidth = img.getSize().x / pow(2, i);
        //int cheight = img.getSize().y / pow(2, i);

        int cwidth = swidth;
        int cheight = sheight;

        swidth /= 2;
        sheight /= 2;

        cl::gl_rendertexture temp(clctx.ctx);
        temp.create_from_texture_with_mipmaps(background_with_mips.handle, i);
        temp.acquire(clctx.cqueue);

        std::vector<cl_uchar4> res = temp.read<2, cl_uchar4>(clctx.cqueue, (vec<2, size_t>){0,0}, (vec<2, size_t>){cwidth, cheight});

        temp.unacquire(clctx.cqueue);

        std::vector<cl_float4> converted;
        converted.reserve(res.size());

        for(auto& i : res)
        {
            converted.push_back({i.s[0] / 255.f, i.s[1] / 255.f, i.s[2] / 255.f, i.s[3] / 255.f});
        }

        background_mipped.write(clctx.cqueue, (char*)&converted[0], vec<2, size_t>{0, 0}, vec<2, size_t>{cwidth, cheight}, i);
    }

    cl::command_queue read_queue(clctx.ctx, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
    cl::device_command_queue dqueue(clctx.ctx);

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
    vec4f camera = {0, 0.01, -0.024, -5.5};
    quat camera_quat;
    //camera_quat.load_from_matrix(axis_angle_to_mat({0, 0, 0}, 0));

    vec3f forward_axis = {0, 0, 1};
    vec3f up_axis = {0, 1, 0};

    sf::Clock clk;

    int ray_count = supersample_width * supersample_height;

    cl::buffer schwarzs_1(clctx.ctx);
    cl::buffer schwarzs_2(clctx.ctx);
    cl::buffer kruskal_1(clctx.ctx);
    cl::buffer kruskal_2(clctx.ctx);
    cl::buffer finished_1(clctx.ctx);

    cl::buffer schwarzs_count_1(clctx.ctx);
    cl::buffer schwarzs_count_2(clctx.ctx);
    cl::buffer kruskal_count_1(clctx.ctx);
    cl::buffer kruskal_count_2(clctx.ctx);
    cl::buffer finished_count_1(clctx.ctx);

    cl::buffer texture_coordinates[2] = {clctx.ctx, clctx.ctx};

    for(int i=0; i < 2; i++)
    {
        texture_coordinates[i].alloc(supersample_width * supersample_height * sizeof(float) * 2);
        texture_coordinates[i].set_to_zero(clctx.cqueue);
    }

    schwarzs_1.alloc(sizeof(lightray) * ray_count * 3);
    schwarzs_2.alloc(sizeof(lightray) * ray_count * 3);
    kruskal_1.alloc(sizeof(lightray) * ray_count * 3);
    kruskal_2.alloc(sizeof(lightray) * ray_count * 3);
    finished_1.alloc(sizeof(lightray) * ray_count * 3);

    schwarzs_count_1.alloc(sizeof(int));
    schwarzs_count_2.alloc(sizeof(int));
    kruskal_count_1.alloc(sizeof(int));
    kruskal_count_2.alloc(sizeof(int));
    finished_count_1.alloc(sizeof(int));

    cl_sampler_properties sampler_props[] = {
    CL_SAMPLER_NORMALIZED_COORDS, CL_TRUE,
    CL_SAMPLER_ADDRESSING_MODE, CL_ADDRESS_REPEAT,
    CL_SAMPLER_FILTER_MODE, CL_FILTER_LINEAR,
    CL_SAMPLER_MIP_FILTER_MODE_KHR, CL_FILTER_LINEAR,
    //CL_SAMPLER_LOD_MIN_KHR, 0.0f,
    //CL_SAMPLER_LOD_MAX_KHR, FLT_MAX,
    0
    };

    cl_sampler sam = clCreateSamplerWithProperties(clctx.ctx.native_context.data, sampler_props, nullptr);

    std::optional<cl::event> last_event;

    std::cout << "Supports shared events? " << cl::supports_extension(clctx.ctx, "cl_khr_gl_event") << std::endl;

    bool supersample = false;

    while(!win.should_close())
    {
        win.poll();

        glFinish();
        rtex[which_buffer].acquire(clctx.cqueue);

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

        if(ImGui::IsKeyPressed(GLFW_KEY_N))
        {
            camera = {0, 0, 0, -1.16};
        }

        if(ImGui::IsKeyPressed(GLFW_KEY_M))
        {
            camera = {0, 0, 0, 1.16};
        }

        if(ImGui::IsKeyPressed(GLFW_KEY_R))
        {
            camera = {0, 0, 22, 0};
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

        vec4f scamera = cartesian_to_schwarz(camera);

        float time = clk.restart().asMicroseconds() / 1000.;

        ImGui::Begin("DBG", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

        ImGui::DragFloat3("Pos", &scamera.v[1]);

        ImGui::DragFloat("Time", &time);

        ImGui::Checkbox("Supersample", &supersample);

        ImGui::End();

        int width = win.get_window_size().x();
        int height = win.get_window_size().y();

        if(supersample)
        {
            width *= supersample_mult;
            height *= supersample_mult;
        }

        cl::args clr;
        clr.push_back(rtex[which_buffer]);

        clctx.cqueue.exec("clear", clr, {width, height}, {16, 16});

        #if 0
        cl::args args;
        args.push_back(rtex[0]);
        args.push_back(ds);
        args.push_back(camera);
        args.push_back(camera_quat);
        args.push_back(clbackground);

        clctx.cqueue.exec("do_raytracing_multicoordinate", args, {win.get_window_size().x(), win.get_window_size().y()}, {16, 16});

        rtex[0].unacquire(clctx.cqueue);

        glFinish();
        clctx.cqueue.block();
        glFinish();
        #endif // OLD_AND_GOOD

        /*schwarzs_1.set_to_zero(clctx.cqueue);
        schwarzs_2.set_to_zero(clctx.cqueue);
        kruskal_1.set_to_zero(clctx.cqueue);
        kruskal_2.set_to_zero(clctx.cqueue);
        finished_1.set_to_zero(clctx.cqueue);*/

        #if 1
        schwarzs_count_1.set_to_zero(clctx.cqueue);
        schwarzs_count_2.set_to_zero(clctx.cqueue);
        kruskal_count_1.set_to_zero(clctx.cqueue);
        kruskal_count_2.set_to_zero(clctx.cqueue);
        finished_count_1.set_to_zero(clctx.cqueue);

        int fallback = 0;

        cl::buffer* b1 = &schwarzs_1;
        cl::buffer* b2 = &schwarzs_2;
        cl::buffer* c1 = &schwarzs_count_1;
        cl::buffer* c2 = &schwarzs_count_2;

        cl::event next;

        {
            cl::args init_args;
            init_args.push_back(camera);
            init_args.push_back(camera_quat);
            init_args.push_back(*b1);
            init_args.push_back(kruskal_1); ///temp
            init_args.push_back(*c1);
            init_args.push_back(kruskal_count_1); ///temp
            init_args.push_back(width);
            init_args.push_back(height);

            clctx.cqueue.exec("init_rays", init_args, {width, height}, {16, 16});

            //#define CPU_CONTROL
            #ifdef CPU_CONTROL
            for(int i=0; i < 100; i++)
            {
                c2->set_to_zero(clctx.cqueue);

                cl::args run_args;
                run_args.push_back(*b1);
                run_args.push_back(*b2);
                run_args.push_back(kruskal_1);
                run_args.push_back(kruskal_2);
                run_args.push_back(finished_1);
                run_args.push_back(*c1);
                run_args.push_back(*c2);
                run_args.push_back(kruskal_count_1);
                run_args.push_back(kruskal_count_2);
                run_args.push_back(finished_count_1);

                clctx.cqueue.exec("do_schwarzs_rays", run_args, {width * height}, {256});

                std::swap(b1, b2);
                std::swap(c1, c2);
            }

            #else

            cl::args run_args;
            run_args.push_back(*b1);
            run_args.push_back(*b2);
            run_args.push_back(kruskal_1);
            run_args.push_back(kruskal_2);
            run_args.push_back(finished_1);
            run_args.push_back(*c1);
            run_args.push_back(*c2);
            run_args.push_back(kruskal_count_1);
            run_args.push_back(kruskal_count_2);
            run_args.push_back(finished_count_1);
            run_args.push_back(width);
            run_args.push_back(height);
            run_args.push_back(fallback);

            clctx.cqueue.exec("relauncher", run_args, {1}, {1});

            #endif // CPU_CONTROL

            cl::args texture_args;
            texture_args.push_back(finished_1);
            texture_args.push_back(finished_count_1);
            texture_args.push_back(texture_coordinates[which_buffer]);
            texture_args.push_back(width);
            texture_args.push_back(height);
            texture_args.push_back(camera);
            texture_args.push_back(camera_quat);

            clctx.cqueue.exec("calculate_texture_coordinates", texture_args, {width * height}, {256});

            cl::args render_args;
            render_args.push_back(camera);
            render_args.push_back(camera_quat);
            render_args.push_back(finished_1);
            render_args.push_back(finished_count_1);
            render_args.push_back(rtex[which_buffer]);
            render_args.push_back(background_mipped);
            render_args.push_back(width);
            render_args.push_back(height);
            render_args.push_back(texture_coordinates[which_buffer]);
            render_args.push_back(sam);

            next = clctx.cqueue.exec("render", render_args, {width * height}, {256});
        }

        clctx.cqueue.flush();

        rtex[which_buffer].unacquire(clctx.cqueue);

        which_buffer = (which_buffer + 1) % 2;

        if(last_event.has_value())
            last_event.value().block();

        last_event = next;
        #endif

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

            if(!supersample)
                lst->AddImage((void*)rtex[which_buffer].texture_id, tl, br, ImVec2(0, 0), ImVec2(1.f/supersample_mult, 1.f/supersample_mult));
            else
                lst->AddImage((void*)rtex[which_buffer].texture_id, tl, br, ImVec2(0, 0), ImVec2(1, 1));
        }

        win.display();
    }

    last_event = std::nullopt;

    clctx.cqueue.block();

    return 0;
}
