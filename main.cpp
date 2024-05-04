#include <iostream>
#include <toolkit/render_window.hpp>
#include <toolkit/texture.hpp>
#include <vec/vec.hpp>
#include <GLFW/glfw3.h>
#include <SFML/Graphics.hpp>
#include <CL/cl_ext.h>
#include "metric.hpp"
#include "chromaticity.hpp"
#include <imgui/misc/freetype/imgui_freetype.h>
#include <imgui/imgui_internal.h>
#include "js_interop.hpp"
#include <toolkit/fs_helpers.hpp>
#include <filesystem>
#include "workshop/steam_ugc_manager.hpp"
#include "content_manager.hpp"
#include "equation_context.hpp"
#include "fullscreen_window_manager.hpp"
#include "raw_input.hpp"
#include "metric_manager.hpp"
#include "graphics_settings.hpp"
#include <networking/serialisable.hpp>
#include "input_manager.hpp"
#include "print.hpp"
#include "triangle.hpp"
#include "triangle_manager.hpp"
#include "physics.hpp"
#include "dynamic_feature_config.hpp"
#include "render_state.hpp"
#include <thread>
#include "common.cl"
//#include "dual_complex.hpp"

/**
Big list of general relativity references so i can shut some browser tabs
https://arxiv.org/pdf/1601.02063.pdf - GPU renderer
https://arxiv.org/pdf/1511.06025.pdf - CPU renderer - primary source from which this raytracer is derived
https://en.wikipedia.org/wiki/Frame_fields_in_general_relativity#Example:_Static_observers_in_Schwarzschild_vacuum - frame fields
https://www.spacetimetravel.org/wurmlochflug/wurmlochflug.html - renderings of wormholes
https://www.damtp.cam.ac.uk/user/hsr1000/lecturenotes_2012.pdf - lecture notes for relativity, misc everything
https://arxiv.org/pdf/0904.4184.pdf - spacetime catalogue, contains a bunch of metrics and tetrads
https://arxiv.org/pdf/1104.4829.pdf - gram schmidt orthonormalisation in a relativistic context
https://arxiv.org/pdf/1702.05802.pdf - double kerr (massless strut)
https://arxiv.org/ftp/arxiv/papers/1008/1008.3244.pdf - double kerr (massles strut)
https://arxiv.org/pdf/1702.02209.pdf - rotating double kerr with a massless strut
https://arxiv.org/pdf/1905.05273.pdf - janis-newman-winicour rendering + accretion disk
https://arxiv.org/pdf/1408.6041.pdf - alternative formulation of janis-newman-winicour line element that's less singularity inducing
http://www.roma1.infn.it/teongrav/VALERIA/TEACHING/ONDE_GRAV_STELLE_BUCHINERI/AA2012_13/Kerr.pdf - kerr info
http://cloud.yukterez.net/relativistic.raytracer/kerr.90.1720.png - kerr reference picture
https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods - runge kutta
https://physics.stackexchange.com/questions/409106/finding-the-metric-tensor-from-a-line-element - line element / metric tensor
https://arxiv.org/pdf/0706.0622.pdf - kerr spacetime coordinate systems
http://www.roma1.infn.it/teongrav/leonardo/bh/bhcap3.pdf - kerr, again
https://www.wolframalpha.com/input/?i=%28x%5E2+%2B+y%5E2%29+%2F+%28r%5E2+%2B+a%5E2%29+%2B+z%5E2%2Fr%5E2+%3D+1%2C+solve+for+r - solving for r in kerr-schild
https://arxiv.org/pdf/0807.0734.pdf - symplectic integrators
https://iopscience.iop.org/article/10.1088/0067-0049/184/2/387/pdf - radiative transport
https://javierrubioblog.files.wordpress.com/2015/12/chapter4.pdf - coordinate transforms
https://arxiv.org/pdf/1308.2298.pdf - double balanced kerr
https://www.researchgate.net/figure/Shadows-of-the-double-Schwarzschild-BH-solution-with-equal-masses-0-and-different-BH_fig1_323026854 - binary schwarzs
https://www2.mpia-hd.mpg.de/homes/tmueller/pdfs/catalogue_2014-05-21.pdf - loads of good information about everything, metric catalogue
https://www.sciencedirect.com/science/article/pii/S0370269319304563 - binary kerr
https://www.sciencedirect.com/science/article/pii/S0370269320307206 - binary kerr newman, equal
https://arxiv.org/abs/1502.03809 - the interstellar paper
https://arxiv.org/pdf/2110.00679.pdf - unequal binary kerr newman
https://arxiv.org/pdf/2110.04879.pdf - binary kerr newman, unequal, extremal

https://www.reed.edu/physics/courses/Physics411/html/page2/page2.html - some useful info
https://www.uio.no/studier/emner/matnat/astro/nedlagte-emner/AST1100/h11/undervisningsmateriale/lecture15.pdf - useful basic info
https://theconfused.me/blog/numerical-integration-of-light-paths-in-a-schwarzschild-metric/ - simple schwarzschild raytracer

https://github.com/stranger80/GraviRayTraceSharp/blob/master/GraviRayTraceSharp/ - raytracer with runge kutta integrator
https://en.wikipedia.org/wiki/Interior_Schwarzschild_metric - metric for the inside of a body
https://en.wikipedia.org/wiki/Vaidya_metric - radiating metric
https://en.wikipedia.org/wiki/Category:Exact_solutions_in_general_relativity - more exact solutions

https://en.wikipedia.org/wiki/Tetrad_formalism - tetrads/coordinate basis
https://arxiv.org/abs/gr-qc/0507014v1 - numerical relativity
https://arxiv.org/pdf/gr-qc/0104063.pdf - numerical relativity

https://en.wikipedia.org/wiki/Two-body_problem_in_general_relativity#Schwarzschild_solution - useful references on numerical relativity
https://en.wikipedia.org/wiki/Kerr%E2%80%93Newman_metric - kerr-newman (charged + rotating)

https://en.wikipedia.org/wiki/Cash%E2%80%93Karp_method - runge-kutta with adaptive error
https://drum.lib.umd.edu/bitstream/handle/1903/2202/2004-berry-healy-jas.pdf;jsessionid=B20F478B9DB479C86B9DD179A24331F3?sequence=7 - integration

https://www.pp.rhul.ac.uk/~cowan/ph2150/kepler_xy.pdf - good explanation of integration

https://core.ac.uk/download/pdf/1321518.pdf - numerical relativity phd
https://arxiv.org/pdf/gr-qc/9509020.pdf - numerical relativity

https://www.cec.uchile.cl/cinetica/pcordero/MC_libros/NumericalRecipesinC.pdf - 710

https://arxiv.org/pdf/0712.4333.pdf - coordinate system choices for schwarzschild and kerr (hyperboloidal)
https://iopscience.iop.org/article/10.1088/1361-6382/ab6e3e/pdf - another

http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.1002.1336&rep=rep1&type=pdf

http://yukterez.net/ - loads of good stuff
https://scholarworks.rit.edu/cgi/viewcontent.cgi?article=11286&context=theses - lots more good stuff, numerical relativity
https://physics.stackexchange.com/questions/51915/can-one-raise-indices-on-covariant-derivative-and-products-thereof - transforming covariant derivative indices
http://ccom.ucsd.edu/~lindblom/Talks/Milwaukee_14October2011.pdf - simple introduction to numerical relativity
http://ccom.ucsd.edu/~lindblom/Talks/NRBeijing1.pdf - seems to be more up to date

https://www.aanda.org/articles/aa/pdf/2012/09/aa19599-12.pdf - radiative transfer
https://arxiv.org/pdf/0704.0986.pdf - tetrad info
https://www.researchgate.net/figure/View-of-a-static-observer-located-at-x-0-y-4-in-the-positive-y-direction-for-t_fig2_225428633 - alcubierre. Successfully managed to replicate this https://imgur.com/a/48SONjV. This paper is an absolute goldmine of useful information

https://arxiv.org/pdf/astro-ph/9707230.pdf - neutron star numerical relativity
https://www.aanda.org/articles/aa/full_html/2012/07/aa19209-12/aa19209-12.html - a* with a thin disk
https://gyoto.obspm.fr/GyotoManual.pdf - gyoto, general relativity tracer
https://core.ac.uk/download/pdf/25279526.pdf - binary black hole approximation?

https://hal.archives-ouvertes.fr/hal-01862911/document - natario warp drive

https://arxiv.org/pdf/1908.10757.pdf - good tetrad reference
https://academic.oup.com/ptep/article/2015/4/043E02/1524372 - interesting kerr coordinate system

"how do i convert rgb to wavelengths"
https://github.com/colour-science/smits1999
https://github.com/appleseedhq/appleseed/blob/54ce23fc940087180511cb5659d8a7aac33712fb/src/appleseed/foundation/image/colorspace.h#L956
https://github.com/wip-/RgbToSpectrum/blob/master/Spectra/SimpleSpectrum.cs
https://en.wikipedia.org/wiki/Dominant_wavelength
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.40.9608&rep=rep1&type=pdf
https://www.fourmilab.ch/documents/specrend/specrend.c
https://www.researchgate.net/publication/308305862_Relationship_between_peak_wavelength_and_dominant_wavelength_of_light_sources_based_on_vector-based_dominant_wavelength_calculation_method
https://www.semrock.com/how-to-calculate-luminosity-dominant-wavelength-and-excitation-purity.aspx
*/

///perfectly fine
vec4f cartesian_to_schwarz(vec4f position)
{
    vec3f polar = cartesian_to_polar((vec3f){position.y(), position.z(), position.w()});

    return (vec4f){position.x(), polar.x(), polar.y(), polar.z()};
}

#define GENERIC_METRIC

void execute_kernel(graphics_settings& sett,
                    cl::command_queue& cqueue, cl::buffer& rays_in,
                    cl::buffer& count_in,
                    cl::buffer& ray_time_min, cl::buffer& ray_time_max,
                    //cl::buffer& visual_path, cl::buffer& visual_ray_counts,
                    triangle_rendering::manager& manage,
                    physics& phys,
                    int num_rays,
                    bool use_device_side_enqueue,
                    dynamic_feature_config& dfg,
                    cl::buffer& dynamic_config,
                    cl::buffer& dynamic_feature_config,
                    int width, int height,
                    render_state& st,
                    single_render_state& single_state,
                    cl::event evt)
{
    if(use_device_side_enqueue)
    {
        int fallback = 0;

        cl::args run_args;
        run_args.push_back(rays_in);
        run_args.push_back(count_in);
        run_args.push_back(fallback);
        run_args.push_back(dynamic_config);

        cqueue.exec("relauncher_generic", run_args, {1}, {1});
    }
    else
    {
        st.stored_ray_counts.set_to_zero(cqueue);

        count_in.write_async(cqueue, (const char*)&num_rays, sizeof(int));

        if(dfg.get_feature<bool>("use_triangle_rendering"))
        {
            cl_int my_min = INT_MAX;
            cl_int my_max = INT_MIN;

            ray_time_min.fill(cqueue, my_min);
            ray_time_max.fill(cqueue, my_max);
        }

        int mouse_x = ImGui::GetMousePos().x;
        int mouse_y = ImGui::GetMousePos().y;

        cl::args run_args;
        run_args.push_back(rays_in);
        run_args.push_back(count_in);
        run_args.push_back(ray_time_min);
        run_args.push_back(ray_time_max);
        run_args.push_back(dynamic_config);
        run_args.push_back(dynamic_feature_config);
        run_args.push_back(width);
        run_args.push_back(height);
        run_args.push_back(mouse_x);
        run_args.push_back(mouse_y);

        if(dfg.get_feature<bool>("use_triangle_rendering"))
            run_args.push_back(single_state.stored_rays);
        else
            run_args.push_back(nullptr);

        run_args.push_back(st.stored_ray_counts);
        run_args.push_back(single_state.max_stored);

        cqueue.exec("do_generic_rays", run_args, {width, height}, {sett.workgroup_size[0], sett.workgroup_size[1]}, {evt});
    }
}

int calculate_ray_count(int width, int height)
{
    return height * width;
}

struct main_menu
{
    enum windows
    {
        MAIN,
        SETTINGS,
        QUIT
    };

    graphics_settings sett;
    background_settings background_sett;

    int state = MAIN;
    bool should_open = false;
    bool is_open = true;
    bool dirty_settings = false;
    bool should_quit = false;
    bool already_started = false;

    bool load()
    {
        background_sett.path1 = "./backgrounds/nasa.png";
        background_sett.path2 = "./backgrounds/nasa.png";

        background_sett.load();

        bool loaded_settings = false;

        if(file::exists("settings.json"))
        {
            try
            {
                nlohmann::json js = nlohmann::json::parse(file::read("settings.json", file::mode::BINARY));

                deserialise<graphics_settings>(js, sett, serialise_mode::DISK);

                loaded_settings = true;
            }
            catch(std::exception& ex)
            {
                std::cout << "Failed to load settings.json " << ex.what() << std::endl;
            }
        }

        return loaded_settings;
    }

    bool is_first_time_main_menu_open()
    {
        return is_open && !already_started;
    }

    void display_main_menu()
    {
        std::string start_string = already_started ? "Continue" : "Start";

        if(ImGui::Button(start_string.c_str()))
        {
            already_started = true;

            close();
        }

        if(ImGui::Button("Settings"))
        {
            state = SETTINGS;
        }

        if(ImGui::Button("Quit"))
        {
            state = QUIT;
        }
    }

    void display_settings_menu(render_window& win, input_manager& input, background_images& bi)
    {
        if(ImGui::BeginTabBar("Tab Bar"))
        {
            if(ImGui::BeginTabItem("Video"))
            {
                dirty_settings |= sett.display_video_settings();

                ImGui::EndTabItem();
            }

            if(ImGui::BeginTabItem("Controls"))
            {
                dirty_settings |= sett.display_control_settings();

                ImGui::EndTabItem();
            }

            if(ImGui::BeginTabItem("Keybinds"))
            {
                input.display_key_rebindings(win);

                ImGui::EndTabItem();
            }

            if(ImGui::BeginTabItem("Background"))
            {
                background_sett.display(bi);

                ImGui::EndTabItem();
            }

            ImGui::EndTabBar();
        }

        if(ImGui::Button("Back"))
        {
            state = MAIN;
        }
    }

    void display_quit_menu()
    {
        ImGui::Text("Are you sure?");

        ImGui::SameLine();

        if(ImGui::Button("Yes"))
        {
            should_quit = true;
        }

        ImGui::SameLine();

        if(ImGui::Button("No"))
        {
            state = MAIN;
        }
    }

    void close()
    {
        is_open = false;
    }

    void open()
    {
        should_open = true;
    }

    void poll_open()
    {
        if(should_open)
        {
            //ImGui::OpenPopup("Main Menu");
            is_open = true;
        }

        should_open = false;
    }

    bool in_settings()
    {
        return state == SETTINGS && is_open;
    }

    void display(render_window& win, input_manager& input, background_images& bi)
    {
        if(state == SETTINGS)
        {
            vec2i dim = win.get_window_size();

            vec2i menu_size = dim/2;

            vec2i menu_tl = (dim / 2) - menu_size/2;

            ImGui::SetNextWindowSize(ImVec2(dim.x()/2, dim.y()/2), ImGuiCond_Always);

            ImVec2 tl = ImGui::GetMainViewport()->Pos;

            ImGui::SetNextWindowPos(ImVec2(tl.x + menu_tl.x(), tl.y + menu_tl.y()), ImGuiCond_Always);
        }

        ImGui::Begin("Main Menu", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);

        //if(ImGui::BeginPopupModal("Main Menu", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse))
        {
            int current_state = state;

            if(current_state == MAIN)
            {
                display_main_menu();
            }

            if(current_state == SETTINGS)
            {
                display_settings_menu(win, input, bi);
            }

            if(current_state == QUIT)
            {
                display_quit_menu();
            }

            if(ImGui::IsKeyPressed(GLFW_KEY_ESCAPE))
            {
                if(current_state == MAIN)
                {
                    close();
                }

                state = MAIN;
            }

            ImVec2 viewport_size = ImGui::GetMainViewport()->Size;
            ImVec2 viewport_pos = ImGui::GetMainViewport()->Pos;

            ImVec2 window_size = ImGui::GetWindowSize();

            ImGui::SetWindowPos({viewport_pos.x + viewport_size.x/2 - window_size.x/2, viewport_pos.y + viewport_size.y/2 - window_size.y/2});

            ImGui::End();

            //ImGui::EndPopup();
        }
    }
};

template<typename T>
struct read_queue
{
    struct element
    {
        T data = T{};
        cl::event evt;
        cl::buffer gpu_buffer;

        element(cl::context& ctx) : gpu_buffer(ctx){}
    };

    std::vector<element*> q;

    void start_read(cl::context& ctx, cl::command_queue& async_q, cl::buffer&& buf, std::vector<cl::event> wait_on)
    {
        element* e = new element(ctx);

        e->gpu_buffer = std::move(buf);

        e->evt = e->gpu_buffer.read_async(async_q, (char*)&e->data, sizeof(T), wait_on);

        q.push_back(e);
    }

    std::vector<std::pair<T, cl::buffer>> fetch()
    {
        std::vector<std::pair<T, cl::buffer>> ret;

        for(int i=0; i < (int)q.size(); i++)
        {
            element* e = q[i];

            if(e->evt.is_finished())
            {
                ret.push_back({e->data, std::move(e->gpu_buffer)});

                q.erase(q.begin() + i);
                i--;
                delete e;
                continue;
            }
        }

        return ret;
    }
};

template<typename T>
struct read_queue_pool
{
    read_queue<T> q;

    std::vector<cl::buffer> pool;

    cl::buffer get_buffer(cl::context& ctx)
    {
        if(pool.size() > 4)
        {
            cl::buffer next = pool.front();
            pool.erase(pool.begin());
            return next;
        }

        cl::buffer buf(ctx);
        buf.alloc(sizeof(T));

        return buf;
    }

    void start_read(cl::context& ctx, cl::command_queue& async_q, cl::buffer&& buf, std::vector<cl::event> wait_on)
    {
        q.start_read(ctx, async_q, std::move(buf), wait_on);
    }

    std::vector<T> fetch()
    {
        std::vector<T> ret;

        std::vector<std::pair<T, cl::buffer>> impl_fetch = q.fetch();

        for(auto& [d, buf] : impl_fetch)
        {
            ret.push_back(std::move(d));
            pool.push_back(buf);
        }

        return ret;
    }
};

std::vector<triangle> make_cube(vec3f pos)
{
    std::vector<triangle> tris;

    float d = 0.5f;

    {
        triangle& t = tris.emplace_back();
        t.set_vert(0, {0, 0, 0});
        t.set_vert(1, {0+d, 0, 0});
        t.set_vert(2, {0, 0, d});
    }

    {
        triangle& t = tris.emplace_back();
        t.set_vert(0, {0, 0, d});
        t.set_vert(1, {0+d, 0, 0});
        t.set_vert(2, {0+d, 0, d});
    }

    {
        triangle& t = tris.emplace_back();
        t.set_vert(0, {0,     d, 0});
        t.set_vert(1, {0+d, d, 0});
        t.set_vert(2, {0,     d, d});
    }

    {
        triangle& t = tris.emplace_back();
        t.set_vert(0, {0,     d, d});
        t.set_vert(1, {0+d, d, 0});
        t.set_vert(2, {0+d, d, d});
    }

    {
        triangle& t = tris.emplace_back();
        t.set_vert(0, {0, 0, 0});
        t.set_vert(1, {0, 0, d});
        t.set_vert(2, {0, d, 0});
    }

    {
        triangle& t = tris.emplace_back();
        t.set_vert(0, {0, d, d});
        t.set_vert(1, {0, 0, d});
        t.set_vert(2, {0, d, 0});
    }

    {
        triangle& t = tris.emplace_back();
        t.set_vert(0, {0+d, d, d});
        t.set_vert(1, {0+d, 0, d});
        t.set_vert(2, {0+d, d, 0});
    }

    {
        triangle& t = tris.emplace_back();
        t.set_vert(0, {0+d, 0, 0});
        t.set_vert(1, {0+d, 0, d});
        t.set_vert(2, {0+d, d, 0});
    }

    {
        triangle& t = tris.emplace_back();
        t.set_vert(0, {0, 0, 0});
        t.set_vert(1, {0+d, 0, 0});
        t.set_vert(2, {0, d, 0});
    }

    {
        triangle& t = tris.emplace_back();
        t.set_vert(0, {0+d, d, 0});
        t.set_vert(1, {0+d, 0, 0});
        t.set_vert(2, {0, d, 0});
    }

    {
        triangle& t = tris.emplace_back();
        t.set_vert(0, {0, 0, d});
        t.set_vert(1, {0+d, 0, d});
        t.set_vert(2, {0, d, d});
    }

    {
        triangle& t = tris.emplace_back();
        t.set_vert(0, {0+d, d, d});
        t.set_vert(1, {0+d, 0, d});
        t.set_vert(2, {0, d, d});
    }

    for(triangle& t : tris)
    {
        t.v0x += pos.x();
        t.v1x += pos.x();
        t.v2x += pos.x();

        t.v0y += pos.y();
        t.v1y += pos.y();
        t.v2y += pos.y();

        t.v0z += pos.z();
        t.v1z += pos.z();
        t.v2z += pos.z();
    }

    return tris;
}

void DragFloatCol(const std::string& name, cl_float4& val, int highlight)
{
    ImGui::BeginGroup();

    ImGui::PushMultiItemsWidths(4, ImGui::CalcItemWidth());

    for(int i=0; i < 4; i++)
    {
        if(highlight == i)
            ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.8, 0.1, 0.1, 1));

        ImGui::DragFloat(("###_" + std::to_string(i) + "_" + name).c_str(), &val.s[i]);

        if(highlight == i && ImGui::IsItemHovered())
        {
            ImGui::SetTooltip("Timelike");
        }

        if(highlight == i)
            ImGui::PopStyleColor(1);

        ImGui::PopItemWidth();

        ImGui::SameLine(0, ImGui::GetStyle().ItemInnerSpacing.x);
    }

    ImGui::Text("%s", name.c_str());

    ImGui::EndGroup();
}

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

template<typename T>
struct async_executor
{
    std::mutex mut;
    std::vector<T> yields;

    int peek()
    {
        std::lock_guard guard(mut);

        return yields.size();
    }

    void add(T&& in)
    {
        std::lock_guard guard(mut);

        yields.push_back(std::move(in));
    }

    std::optional<T> produce(bool at_least_one = false)
    {
        if(!at_least_one)
        {
            std::lock_guard guard(mut);

            if(yields.size() == 0)
                return std::nullopt;

            auto val = std::move(yields.front());
            yields.erase(yields.begin());
            return val;
        }
        else
        {
            while(1)
            {
                std::lock_guard guard(mut);

                if(yields.size() == 0)
                    continue;

                auto val = std::move(yields.front());
                yields.erase(yields.begin());
                return val;
            }

            return std::nullopt;
        }
    }
};

struct gl_image_shared
{
    texture tex;
    cl::gl_rendertexture rtex;

    gl_image_shared(cl::context& ctx) : rtex(ctx){}

    void make(int width, int height)
    {
        texture_settings new_sett;
        new_sett.width = width;
        new_sett.height = height;
        new_sett.is_srgb = false;
        new_sett.generate_mipmaps = false;

        tex.load_from_memory(new_sett, nullptr);
        rtex.create_from_texture(tex.handle);
    }
};


template<typename T>
struct spsc
{
    std::mutex mut;
    std::vector<T> dat;

    void add(T&& in)
    {
        std::lock_guard guard(mut);
        dat.push_back(std::move(in));
    }

    std::optional<T> produce()
    {
        std::lock_guard guard(mut);

        if(dat.size() == 0)
            return std::nullopt;

        if(dat.size() > 0)
        {
            auto val = std::move(dat[0]);
            dat.erase(dat.begin());
            return std::move(val);
        }

        return std::nullopt;
    }
};

struct gl_image_shared_queue
{
    std::vector<gl_image_shared> free_images;
    std::mutex mut;

    void push_free(gl_image_shared&& gl)
    {
        std::scoped_lock guard(mut);

        free_images.push_back(std::move(gl));
    }

    gl_image_shared pop_free_or_make_new(cl::context& ctx, int width, int height)
    {
        std::scoped_lock guard(mut);

        if(free_images.size() > 4)
        {
            auto size = free_images.front().rtex.size<2>();

            if(size.x() == width && size.y() == height)
            {
                auto next = std::move(free_images.front());

                free_images.erase(free_images.begin());

                return next;
            }
            else
            {
                free_images.clear();
            }
        }

        gl_image_shared shared(ctx);
        shared.make(width, height);

        return shared;
    }
};

void test_overlap()
{
    static_assert(periodic_range_overlaps(0, 2, 0, 1, 2));
    static_assert(periodic_range_overlaps(0, 1, 0, 8, 2));

    static_assert(periodic_range_overlaps(0, 1, 2, 3, 2));

    static_assert(!periodic_range_overlaps(0, 1, 3.1, 3.9, 2));

    static_assert(periodic_range_overlaps(0.1, 0.3, 3.1, 4.2, 2));
    static_assert(periodic_range_overlaps(3.1, 4.2, 0.1, 0.3, 2));
}

///i need the ability to have dynamic parameters
int main(int argc, char* argv[])
{
    test_overlap();

    #ifdef REDIRECT_STDOUT
    *stdout = *fopen("debug.txt","w");
    #endif // REDIRECT_STDOUT

    std::optional<std::string> start_metric;
    bool should_print_frametime = false;

    for(int i=1; i < argc - 1; i++)
    {
        std::string current = argv[i];
        std::string next = argv[i + 1];

        if(current == "-bench")
        {
            start_metric = next;
            should_debug = false;
            should_print_frametime = true;
        }

        if(current == "-start")
        {
            start_metric = next;
        }
    }

    bool has_new_content = false;

    steam_info steam;

    ugc_view workshop;
    workshop.only_get_subscribed();

    steam_callback_executor exec;

    workshop.fetch(steam, exec, [&](){has_new_content = true;});

    //dual_types::test_operation();

    main_menu menu;
    bool loaded_settings = menu.load();

    render_settings sett;
    sett.width = 1280;
    sett.height = 720;
    sett.opencl = true;
    sett.no_double_buffer = false;
    sett.is_srgb = true;
    sett.no_decoration = true;
    sett.viewports = false;

    #define REMEMBER_SIZE

    if(loaded_settings)
    {
        #ifdef REMEMBER_SIZE
        sett.width = menu.sett.width;
        sett.height = menu.sett.height;
        #endif
    }

    render_window win(sett, "Geodesics");

    #ifdef REMEMBER_SIZE
    win.backend->set_window_position({menu.sett.pos_x, menu.sett.pos_y});
    #endif // REMEMBER_SIZE

    if(loaded_settings)
    {
        win.set_vsync(menu.sett.vsync_enabled);
        //win.backend->set_is_maximised(current_settings.fullscreen);

        if(menu.sett.fullscreen)
        {
            win.backend->clear_demaximise_cache();
        }
    }
    else
    {
        win.backend->set_is_maximised(true);
        win.backend->clear_demaximise_cache();
    }

    #ifndef REMEMBER_SIZE
    win.backend->set_is_maximised(false);
    #endif

    assert(win.clctx);

    //std::cout << "extensions " << cl::get_extensions(win.clctx->ctx) << std::endl;

    ImFontAtlas* atlas = ImGui::GetIO().Fonts;
    atlas->FontBuilderFlags = ImGuiFreeTypeBuilderFlags_LCD | ImGuiFreeTypeBuilderFlags_FILTER_DEFAULT | ImGuiFreeTypeBuilderFlags_LoadColor;

    ImFontConfig font_cfg;
    font_cfg.GlyphExtraSpacing = ImVec2(0, 0);
    font_cfg.FontBuilderFlags = ImGuiFreeTypeBuilderFlags_LCD | ImGuiFreeTypeBuilderFlags_FILTER_DEFAULT | ImGuiFreeTypeBuilderFlags_LoadColor;

    ImGuiIO& io = ImGui::GetIO();

    io.Fonts->Clear();
    io.Fonts->AddFontFromFileTTF("VeraMono.ttf", 14, &font_cfg);

    opencl_context& clctx = *win.clctx;

    cl::command_queue async_queue(clctx.ctx, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);

    std::filesystem::path scripts_dir{"./scripts"};

    std::vector<std::string> scripts;

    for(const auto& entry : std::filesystem::directory_iterator{scripts_dir})
    {
        std::string name = entry.path().string();

        if(name.ends_with(".js"))
        {
            scripts.push_back(name);
        }
    }

    content_manager all_content;

    all_content.add_content_folder("./scripts");

    cl::buffer dynamic_feature_buffer(clctx.ctx);

    dynamic_feature_config dfg;
    dfg.add_feature<bool>("use_triangle_rendering");
    dfg.set_feature("use_triangle_rendering", false);

    dfg.add_feature<bool>("redshift");
    dfg.set_feature("redshift", false);

    dfg.add_feature<float>("universe_size");
    dfg.set_feature("universe_size", 20.f);

    dfg.add_feature<float>("max_acceleration_change");
    dfg.set_feature("max_acceleration_change", 0.01f);

    dfg.add_feature<float>("max_precision_radius");
    dfg.set_feature("max_precision_radius", 10.f);

    dfg.add_feature<bool>("reparameterisation");
    dfg.set_feature("reparameterisation", false);

    dfg.add_feature<float>("min_step");
    dfg.set_feature("min_step", 0.000001f);

    //print("WLs %f %f %f\n", chromaticity::srgb_to_wavelength({1, 0, 0}), chromaticity::srgb_to_wavelength({0, 1, 0}), chromaticity::srgb_to_wavelength({0, 0, 1}));

    int last_supersample_mult = 2;

    int start_width = sett.width;
    int start_height = sett.height;

    /*cl::image background_mipped(clctx.ctx);
    cl::image background_mipped2(clctx.ctx);

    {
        sf::Image img_1 = load_image("background.png");

        background_mipped = load_mipped_image(img_1, clctx);

        sf::Image img_2 = load_image("background2.png");

        bool is_eq = false;

        if(img_1.getSize().x == img_2.getSize().x && img_1.getSize().y == img_2.getSize().y)
        {
            size_t len = size_t{img_1.getSize().x} * size_t{img_1.getSize().y} * 4;

            if(memcmp(img_1.getPixelsPtr(), img_2.getPixelsPtr(), len) == 0)
            {
                background_mipped2 = background_mipped;

                is_eq = true;
            }
        }

        if(!is_eq)
            background_mipped2 = load_mipped_image(img_2, clctx);
    }*/

    #ifdef USE_DEVICE_SIDE_QUEUE
    print("Pre dqueue\n");

    cl::device_command_queue dqueue(clctx.ctx);

    print("Post dqueue\n");
    #endif // USE_DEVICE_SIDE_QUEUE

    /*
    ///t, x, y, z
    //vec4f camera = {0, -2, -2, 0};
    //vec4f camera = {0, -2, -8, 0};
    vec4f camera = {0, 0, -4, 0};
    //vec4f camera = {0, 0.01, -0.024, -5.5};
    //vec4f camera = {0, 0, -4, 0};
    quat camera_quat;

    quat q;
    q.load_from_axis_angle({1, 0, 0, -M_PI/2});

    camera_quat = q * camera_quat;*/

    sf::Clock clk;

    cl::buffer dynamic_config(clctx.ctx);

    print("Post buffer declarations\n");

    cl::buffer geodesic_count_buffer(clctx.ctx);
    geodesic_count_buffer.alloc(sizeof(cl_int));

    int max_trace_length = 64000;

    cl::buffer geodesic_trace_buffer(clctx.ctx);
    geodesic_trace_buffer.alloc(max_trace_length * sizeof(cl_float4));

    cl::buffer geodesic_vel_buffer(clctx.ctx);
    geodesic_vel_buffer.alloc(max_trace_length * sizeof(cl_float4));

    cl::buffer geodesic_ds_buffer(clctx.ctx);
    geodesic_ds_buffer.alloc(max_trace_length * sizeof(cl_float));

    geodesic_count_buffer.set_to_zero(clctx.cqueue);
    geodesic_trace_buffer.set_to_zero(clctx.cqueue);
    geodesic_vel_buffer.set_to_zero(clctx.cqueue);
    geodesic_ds_buffer.set_to_zero(clctx.cqueue);

    cl::buffer generic_geodesic_buffer(clctx.ctx);
    generic_geodesic_buffer.alloc(sizeof(cl_float4) * 1024);

    cl::buffer generic_geodesic_count(clctx.ctx);
    generic_geodesic_count.alloc(sizeof(cl_int));
    generic_geodesic_count.set_to_zero(clctx.cqueue);

    cl::buffer g_geodesic_basis_speed(clctx.ctx);
    g_geodesic_basis_speed.alloc(sizeof(cl_float4));
    g_geodesic_basis_speed.set_to_zero(clctx.cqueue);

    std::array<cl::buffer, 4> parallel_transported_tetrads{clctx.ctx, clctx.ctx, clctx.ctx, clctx.ctx};

    for(int i=0; i < 4; i++)
    {
        parallel_transported_tetrads[i].alloc(max_trace_length * sizeof(cl_float4));
        parallel_transported_tetrads[i].set_to_zero(clctx.cqueue);
    }

    print("Alloc trace buffer\n");

    std::vector<cl_float4> current_geodesic_path;
    std::vector<cl_float> current_geodesic_dT_ds;

    read_queue_pool<cl_float4> camera_q;
    read_queue_pool<cl_float4> geodesic_q;
    read_queue_pool<cl_int> timelike_q;

    print("Finished async read queue init\n");

    printj("Supports shared events? ", cl::supports_extension(clctx.ctx, "cl_khr_gl_event"));

    bool last_supersample = false;
    bool should_take_screenshot = false;

    bool time_progresses = false;
    float time_progression_factor = 1.f;
    bool flip_sign = false;
    float current_geodesic_time = 0;
    bool camera_on_geodesic = false;
    bool camera_time_progresses = false;
    float camera_geodesic_time_progression_speed = 1.f;
    float set_camera_time = 0;
    bool parallel_transport_observer = true;
    cl_float4 cartesian_basis_speed = {0,0,0,0};
    float linear_basis_speed = 0.f;
    float object_chuck_speed = 0.f;

    bool has_geodesic = false;

    print("Pre fullscreen\n");

    steady_timer workshop_poll;
    steady_timer frametime_timer;

    bool open_main_menu_trigger = true;

    background_images back_images(clctx.ctx, clctx.cqueue);
    back_images.load(menu.background_sett.path1, menu.background_sett.path2);

    bool hide_ui = false;

    fullscreen_window_manager fullscreen("Relativity Workshop");

    ImGuiStyle& style = ImGui::GetStyle();

    style.FrameRounding = 0;
    style.WindowRounding = 0;
    style.ChildRounding = 0;
    style.ChildBorderSize = 0;
    style.FrameBorderSize = 0;
    //style.PopupBorderSize = 0;
    style.WindowBorderSize = 1;

    raw_input raw_input_manager;
    input_manager input;

    metric_manager metric_manage;

    menu.sett.width = win.get_window_size().x();
    menu.sett.height = win.get_window_size().y();
    menu.sett.vsync_enabled = win.backend->is_vsync();
    menu.sett.fullscreen = win.backend->is_maximised();

    print("Pre main\n");

    triangle_rendering::manager tris(clctx.ctx);

    /*for(int z=-5; z <= 5; z++)
    {
        for(int y=-5; y <= 5; y++)
        {
            for(int x=-5; x <= 5; x++)
            {
                float adist = fabs(x) + fabs(y) + fabs(z);

                if(adist <= 10)
                    continue;

                std::shared_ptr<triangle_rendering::object> obj = tris.make_new();

                obj->tris = make_cube({0, 0, 0});
                obj->pos = {0, x, y, z};
            }
        }
    }*/

    //#define TEAPOTS
    #ifdef TEAPOTS
    //#define INDIVIDUAL_GEODESICS
    #ifndef INDIVIDUAL_GEODESICS
    auto obj = tris.make_new("./models/newell_teaset/teapot.obj");
    obj->pos = {0, -5, 0, 0};
    obj->velocity = {0, 0.35f, 0};
    obj->scale = 0.45f;
    #else
    std::vector<triangle> teapot_tris = triangle_rendering::load_tris_from_model("./models/newell_teaset/teapot.obj");

    for(triangle t : teapot_tris)
    {
        float scale = 0.45f;

        vec3f v0 = t.get_vert(0) * scale;
        vec3f v1 = t.get_vert(1) * scale;
        vec3f v2 = t.get_vert(2) * scale;

        vec3f centre = (v0 + v1 + v2) / 3.f;

        t.set_vert(0, v0 - centre);
        t.set_vert(1, v1 - centre);
        t.set_vert(2, v2 - centre);

        auto obj = tris.make_new();
        obj->tris = {t};
        obj->pos = {0, -5, 0, 0};

        obj->pos.y() += centre.x();
        obj->pos.z() += centre.y();
        obj->pos.w() += centre.z();

        obj->velocity = {0, 0.05f, 0};

        //obj->scale = 0.45f;
    }
    #endif // INDIVIDUAL_GEODESICS
    #endif // TEAPOTS

    /*for(int z=-5; z <= 5; z++)
    {
        for(int y=-5; y <= 5; y++)
        {
            for(int x=-5; x <= 5; x++)
            {
                float adist = fabs(x) + fabs(y) + fabs(z);

                if(adist <= 14)
                    continue;

                std::shared_ptr<triangle_rendering::object> obj = tris.make_new();

                obj->tris = make_cube({0, 0, 0});
                obj->pos = {0, x, y, z};
            }
        }
    }*/

    /*std::shared_ptr<triangle_rendering::object> obj = tris.make_new();

    obj->tris = make_cube({0, 0, 0});
    obj->pos = {0, -5, 0, 0};*/

    //#define CUBE_INTO_HORIZON
    #ifdef CUBE_INTO_HORIZON
    ///event horizon tester
    std::shared_ptr<triangle_rendering::object> obj = tris.make_new();

    obj->tris = make_cube({0, 0, 0});
    obj->pos = {-26, -5, -1, 0};
    obj->velocity = {0, 0.1f, 0};
    #endif // CUBE_INTO_HORIZON

    //#define DEBUG_FAST_CUBE
    #ifdef DEBUG_FAST_CUBE
    std::shared_ptr<triangle_rendering::object> obj = tris.make_new();

    obj->tris = make_cube({0, 0, 0});
    obj->pos = {-10, -5, -1, 0};
    obj->velocity = {-0.8f, 0.0f, 0};
    #endif // DEBUG_FAST_CUBE

    //#define TRI_STANDARD_CASE
    #ifdef TRI_STANDARD_CASE
    std::shared_ptr<triangle_rendering::object> obj = tris.make_new("./models/newell_teaset/smallteapot.obj");

    //obj->tris = make_cube({0, 0, 0});
    obj->pos = {-60, -5, -1, 0};
    obj->velocity = {-0.2f, 0.2f, 0};
    obj->scale = 0.1f;
    #endif // TRI_STANDARD_CASE

    physics phys(clctx.ctx);

    if(dfg.get_feature<bool>("use_triangle_rendering"))
    {
        tris.build(clctx.cqueue);
        phys.setup(clctx.cqueue, tris);
    }

    print("Pre main\n");

    #ifdef UNSTRUCTURED
    std::array<cl::command_queue, 3> cqueues{clctx.ctx, clctx.ctx, clctx.ctx};
    #else
    cl::command_queue mqueue(clctx.ctx, 1<<9);
    #endif

    std::vector<render_state> states;

    for(int i=0; i < 3; i++)
    {
        states.emplace_back(clctx.ctx, clctx.cqueue);
        states[i].realloc(start_width, start_height);
    }

    single_render_state single_state(clctx.ctx);

    bool reset_camera = true;
    bool once = false;

    std::optional<cl_float4> last_camera_pos;
    std::optional<cl_float4> last_geodesic_velocity;
    std::optional<cl_int> last_timelike_coordinate;

    auto save_graphics = [&]()
    {
        file::write_atomic("./settings.json", serialise(menu.sett, serialise_mode::DISK).dump(), file::mode::BINARY);
    };

    clctx.cqueue.block();

    camera cam;

    gl_image_shared_queue glisq;
    async_executor<gl_image_shared> glexec;
    async_executor<gl_image_shared> unacquired;

    std::optional<gl_image_shared> last_frame_opt;

    cl::event last_event;
    cl::event last_last_event;

    int which_state = 0;

    async_executor<gl_image_shared> glsq;

    std::jthread acquire_thread([&](std::stop_token stoken)
    {
        std::vector<cl::command_queue> circ;

        for(int i=0; i < 4; i++)
            circ.emplace_back(clctx.ctx);

        std::atomic_uint which_circ{0};

        std::vector<std::tuple<gl_image_shared, cl::command_queue, cl::event>> unfinished;

        while(1)
        {
            if(stoken.stop_requested())
                return;

            if(auto opt = glsq.produce(); opt.has_value())
            {
                auto& gl = opt.value();
                auto cqueue = circ[which_circ % circ.size()];
                which_circ++;

                auto evt = gl.rtex.unacquire(cqueue);

                cqueue.flush();

                unfinished.push_back({std::move(gl), cqueue, evt});
            }

            if(unfinished.size() > 0)
            {
                if(std::get<2>(unfinished.front()).is_finished())
                {
                    std::get<1>(unfinished.front()).block();
                    glexec.add(std::move(std::get<0>(unfinished.front())));
                    unfinished.erase(unfinished.begin());
                }
            }
        }
    });

    std::jthread release_thread([&](std::stop_token stoken)
    {
        std::vector<cl::command_queue> circ;

        for(int i=0; i < 4; i++)
            circ.emplace_back(clctx.ctx);

        std::atomic_uint which_circ{0};

        std::vector<std::tuple<gl_image_shared, cl::command_queue, cl::event>> unfinished;

        while(1)
        {
            if(stoken.stop_requested())
                return;

            if(auto opt = unacquired.produce(); opt.has_value())
            {
                auto& gl = opt.value();
                auto cqueue = circ[which_circ % circ.size()];
                which_circ++;

                auto evt = gl.rtex.acquire(cqueue);

                cqueue.flush();

                unfinished.push_back({std::move(gl), cqueue, evt});
            }

            if(unfinished.size() > 0)
            {
                if(std::get<2>(unfinished.front()).is_finished())
                {
                    std::get<1>(unfinished.front()).block();
                    glisq.push_free(std::move(std::get<0>(unfinished.front())));
                    unfinished.erase(unfinished.begin());
                }
            }
        }
    });

    #define START_CLOSED
    #ifdef START_CLOSED
    open_main_menu_trigger = false;
    #endif

    int unprocessed_frames = 0;

    while(!win.should_close() && !menu.should_quit && fullscreen.open)
    {
        render_state& st = states[which_state];
        #ifdef UNSTRUCTURED
        cl::command_queue& mqueue = cqueues[which_state];
        #endif

        which_state = (which_state + 1) % states.size();

        dfg.alloc_and_write_gpu_buffer(mqueue, dynamic_feature_buffer);

        if(menu.dirty_settings)
        {
            if((vec2i){menu.sett.width, menu.sett.height} != win.get_window_size())
            {
                win.resize({menu.sett.width, menu.sett.height});
            }

            if(win.backend->is_vsync() != menu.sett.vsync_enabled)
            {
                win.backend->set_vsync(menu.sett.vsync_enabled);
            }

            if(win.backend->is_maximised() != menu.sett.fullscreen)
            {
                win.backend->set_is_maximised(menu.sett.fullscreen);
            }

            if(win.backend->get_window_position() != (vec2i){menu.sett.pos_x, menu.sett.pos_y})
            {
                win.backend->set_window_position({menu.sett.pos_x, menu.sett.pos_y});
            }

            save_graphics();
        }

        if(((vec2i){menu.sett.width, menu.sett.height} != win.get_window_size() ||
           menu.sett.vsync_enabled != win.backend->is_vsync() ||
           menu.sett.fullscreen != win.backend->is_maximised() ||
           (vec2i){menu.sett.pos_x, menu.sett.pos_y} != win.backend->get_window_position())
           && !menu.dirty_settings && !menu.in_settings())
        {
            menu.sett.width = win.get_window_size().x();
            menu.sett.height = win.get_window_size().y();
            menu.sett.vsync_enabled = win.backend->is_vsync();
            menu.sett.fullscreen = win.backend->is_maximised();
            menu.sett.pos_x = win.backend->get_window_position().x();
            menu.sett.pos_y = win.backend->get_window_position().y();

            save_graphics();
       }

        menu.dirty_settings = false;

        exec.poll();

        if(workshop_poll.get_elapsed_time_s() > 20)
        {
            workshop.fetch(steam, exec, [&](){has_new_content = true;});

            workshop_poll.restart();
        }

        if(has_new_content)
        {
            for(const ugc_details& det : workshop.items)
            {
                std::string path = det.absolute_content_path;

                if(path == "")
                    continue;

                printj("Added content ", path);

                all_content.add_content_folder(path);
            }

            has_new_content = false;
        }

        float frametime_s = frametime_timer.restart();

        float controls_multiplier = 1.f;

        if(menu.sett.time_adjusted_controls)
        {
            ///16.f simulates the only camera speed at 16ms/frame
            controls_multiplier = (1000/16.f) * clamp(frametime_s, 0.f, 100.f);
        }

        win.poll();

        ImGui::PushAllowKeyboardFocus(false);

        if(input.is_key_pressed("toggle_mouse"))
            raw_input_manager.set_enabled(win, !raw_input_manager.is_enabled);

        if(open_main_menu_trigger)
        {
            menu.open();

            open_main_menu_trigger = false;
        }

        menu.poll_open();

        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0,0,0,0));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0);

        fullscreen.start(win);

        bool should_recompile = false;

        std::vector<std::string> metric_names;
        std::vector<content*> parent_directories;

        for(content& c : all_content.content_directories)
        {
            for(int idx = 0; idx < (int)c.metrics.size(); idx++)
            {
                std::string friendly_name = "Error looking up metric config";

                std::optional<metrics::metric_config*> config_opt = c.get_config_of_filename(c.metrics[idx]);

                if(config_opt.has_value())
                    friendly_name = config_opt.value()->name;

                metric_names.push_back(friendly_name);
                parent_directories.push_back(&c);
            }
        }

        if(start_metric && metric_manage.selected_idx == -1)
        {
            for(int selected = 0; selected < (int)metric_names.size(); selected++)
            {
                std::string name = metric_names[selected];

                if(name == start_metric.value())
                {
                    metric_manage.selected_idx = selected;
                    should_recompile = true;
                    break;
                }
            }

            start_metric = std::nullopt;
        }

        if(!hide_ui)
        {
            if(ImGui::BeginMenuBar())
            {
                ///steam fps padder
                ImGui::Indent();
                ImGui::Indent();

                ImGui::Text("Metric: ");

                bool valid_selected_idx = metric_manage.selected_idx >= 0 && metric_manage.selected_idx < metric_names.size();

                std::string preview = "None";

                if(valid_selected_idx)
                    preview = metric_names[metric_manage.selected_idx];

                if(ImGui::BeginCombo("##Metrics Box", preview.c_str()))
                {
                    for(int selected = 0; selected < (int)metric_names.size(); selected++)
                    {
                        std::string name = metric_names[selected];

                        if(ImGui::Selectable(name.c_str(), selected == metric_manage.selected_idx))
                        {
                            metric_manage.selected_idx = selected;
                            should_recompile = true;
                        }

                        if(ImGui::IsItemHovered())
                        {
                            std::string name = metric_names[selected];

                            content* c = parent_directories[selected];

                            auto path_opt = c->lookup_path_to_metric_file(name);

                            if(path_opt.has_value())
                            {
                                std::optional<metrics::metric_config*> config_opt = c->get_config_of_filename(path_opt.value());

                                if(config_opt.has_value())
                                {
                                    metrics::metric_config* cfg = config_opt.value();

                                    ImGui::SetTooltip("%s", cfg->description.c_str());
                                }
                            }
                        }

                    }

                    ImGui::EndCombo();
                }

                ImGui::Text("Mouselook:");

                if(raw_input_manager.is_enabled)
                    ImGui::Text("Y");
                else
                    ImGui::Text("N");

                ImGui::Text("(Tab to toggle)");

                ImGui::EndMenuBar();
            }
        }

        fullscreen.stop();

        ImGui::PopStyleVar(1);
        ImGui::PopStyleColor(1);

        if(ImGui::IsKeyDown(GLFW_KEY_LEFT_ALT) && ImGui::IsKeyPressed(GLFW_KEY_ENTER))
        {
            win.backend->set_is_maximised(!win.backend->is_maximised());
        }

        if(input.is_key_pressed("hide_ui"))
        {
            hide_ui = !hide_ui;
        }

        if(!menu.is_open && ImGui::IsKeyPressed(GLFW_KEY_ESCAPE))
        {
            raw_input_manager.set_enabled(win, false);
            menu.open();
        }

        if(menu.is_open)
        {
            hide_ui = false;
            menu.display(win, input, back_images);
        }

        {
            auto buffer_size = (vec<2, size_t>){st.width, st.height};

            bool taking_screenshot = should_take_screenshot;
            should_take_screenshot = false;

            bool should_snapshot_geodesic = false;

            vec<2, size_t> super_adjusted_width = menu.sett.supersample ? (buffer_size / menu.sett.supersample_factor) : buffer_size;

            if((vec2i){super_adjusted_width.x(), super_adjusted_width.y()} != win.get_window_size() || taking_screenshot || last_supersample != menu.sett.supersample || last_supersample_mult != menu.sett.supersample_factor || menu.dirty_settings)
            {
                int width = 16;
                int height = 16;

                if(!taking_screenshot)
                {
                    width = win.get_window_size().x();
                    height = win.get_window_size().y();

                    if(menu.sett.supersample)
                    {
                        width *= menu.sett.supersample_factor;
                        height *= menu.sett.supersample_factor;
                    }
                }
                else
                {
                    width = menu.sett.screenshot_width * menu.sett.supersample_factor;
                    height = menu.sett.screenshot_height * menu.sett.supersample_factor;
                }

                width = std::max(width, 16 * menu.sett.supersample_factor);
                height = std::max(height, 16 * menu.sett.supersample_factor);

                for(auto& i : states)
                    i.realloc(width, height);

                last_supersample = menu.sett.supersample;
                last_supersample_mult = menu.sett.supersample_factor;
            }

            gl_image_shared glis = glisq.pop_free_or_make_new(clctx.ctx, st.width, st.height);

            if(!glis.rtex.acquired)
            {
                glis.rtex.acquire(mqueue);
            }

            float time = clk.restart().asMicroseconds() / 1000.;

            {
                std::vector<cl_float4> cam_data = camera_q.fetch();
                std::vector<cl_float4> geodesic_data = geodesic_q.fetch();
                std::vector<cl_int> timelike_data = timelike_q.fetch();

                if(cam_data.size() > 0)
                {
                    last_camera_pos = cam_data.back();
                }

                if(geodesic_data.size() > 0)
                {
                    last_geodesic_velocity = geodesic_data.back();
                }

                if(timelike_data.size() > 0)
                {
                    last_timelike_coordinate = timelike_data.back();
                }
            }

            bool should_soft_recompile = false;
            bool should_update_camera_time = false;
            bool should_set_observer_velocity = false;
            bool should_chuck_object = false;

            if(!taking_screenshot && !hide_ui && !menu.is_first_time_main_menu_open())
            {
                ImGui::Begin("Settings and Information", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

                if(ImGui::BeginTabBar("Tabbity tab tabs"))
                {
                    if(ImGui::BeginTabItem("General"))
                    {
                        if(last_camera_pos.has_value())
                        {
                            DragFloatCol("Camera Position", last_camera_pos.value(), last_timelike_coordinate.value_or(-1));
                        }

                        if(ImGui::DragFloat("Camera Time", &set_camera_time, 0.5f))
                        {
                            should_update_camera_time = true;
                        }

                        ImGui::DragFloat("Frametime", &time);

                        ImGui::Checkbox("Time Progresses", &time_progresses);

                        ImGui::SliderFloat("Time Progression Factor", &time_progression_factor, 0.01f, 10.f);

                        ImGui::Checkbox("Put Camera into negative space", &flip_sign);

                        if(ImGui::Button("Screenshot"))
                            should_take_screenshot = true;

                        ImGui::EndTabItem();
                    }

                    if(ImGui::BeginTabItem("Metric Settings"))
                    {
                        ImGui::Text("Dynamic Options");

                        ImGui::Indent();

                        if(metric_manage.current_metric)
                        {
                            if(metric_manage.current_metric->sand.cfg.display())
                            {
                                int dyn_config_bytes = metric_manage.current_metric->sand.cfg.current_values.size() * sizeof(cl_float);

                                if(dyn_config_bytes < 4)
                                    dyn_config_bytes = 4;

                                dynamic_config.alloc(dyn_config_bytes);

                                std::vector<float> vars = metric_manage.current_metric->sand.cfg.current_values;

                                if(vars.size() == 0)
                                    vars.resize(1);

                                dynamic_config.write_async(mqueue, vars);
                                should_soft_recompile = true;
                            }
                        }

                        ImGui::Unindent();

                        ImGui::Text("Compile Options");

                        ImGui::Indent();

                        bool has_redshift = dfg.get_feature<bool>("redshift");

                        bool has_reparam = dfg.get_feature<bool>("reparameterisation");

                        if(ImGui::Checkbox("Ray Reparam", &has_reparam))
                            dfg.set_feature("reparameterisation", has_reparam);

                        if(ImGui::Checkbox("Redshift", &has_redshift))
                            dfg.set_feature("redshift", has_redshift);

                        float max_acceleration_change = dfg.get_feature<float>("max_acceleration_change");
                        ImGui::InputFloat("Error Tolerance", &max_acceleration_change, 0.0000001f, 0.00001f, "%.8f");
                        dfg.set_feature("max_acceleration_change", max_acceleration_change);

                        float min_step = dfg.get_feature<float>("min_step");
                        ImGui::InputFloat("Minimum Step", &min_step, 0.0000001f, 0.0001f, "%.8f");
                        dfg.set_feature("min_step", min_step);

                        float universe_size = dfg.get_feature<float>("universe_size");
                        ImGui::DragFloat("Universe Size", &universe_size, 1, 1, 0, "%.1f");
                        dfg.set_feature("universe_size", universe_size);

                        float max_precision_radius = dfg.get_feature<float>("max_precision_radius");
                        ImGui::DragFloat("Precision Radius", &max_precision_radius, 1, 0.0001f, dfg.get_feature<float>("universe_size"), "%.1f");

                        if(max_precision_radius < 1)
                            max_precision_radius = 1;

                        dfg.set_feature("max_precision_radius", max_precision_radius);

                        if(ImGui::IsItemHovered())
                        {
                            ImGui::SetTooltip("Radius at which lightrays raise their precision checking unconditionally");
                        }

                        if(dfg.is_dirty)
                            should_soft_recompile = true;

                        ImGui::Unindent();

                        ImGui::EndTabItem();
                    }

                    if(ImGui::BeginTabItem("Paths"))
                    {
                        if(ImGui::Button("Snapshot Camera Geodesic"))
                        {
                            should_snapshot_geodesic = true;
                        }

                        if(has_geodesic)
                        {
                            ImGui::SameLine();

                            if(ImGui::Button("Clear"))
                            {
                                has_geodesic = false;
                            }
                        }

                        if(ImGui::DragFloat("Observer Velocity", &linear_basis_speed, 0.001f, -0.9999999f, 0.9999999f))
                        {
                            should_set_observer_velocity = true;
                        }

                        if(has_geodesic)
                        {
                            ImGui::Separator();

                            if(last_camera_pos.has_value())
                            {
                                DragFloatCol("Geodesic Position", last_camera_pos.value(), last_timelike_coordinate.value_or(-1));
                            }

                            if(last_geodesic_velocity.has_value())
                            {
                                ImGui::DragFloat4("Geodesic Velocity", &last_geodesic_velocity.value().s[0]);
                            }

                            ImGui::DragFloat("Proper Time", &current_geodesic_time, 0.1, 0.f, 0.f);

                            ImGui::Checkbox("Proper Time Progresses", &camera_time_progresses);

                            ImGui::SliderFloat("Proper Time Progession Rate", &camera_geodesic_time_progression_speed, 0.f, 4.f, "%.2f");

                            ImGui::Checkbox("Attach Camera to Geodesic", &camera_on_geodesic);

                            ImGui::Checkbox("Transport observer along geodesic", &parallel_transport_observer);
                        }

                        ImGui::EndTabItem();
                    }

                    if(ImGui::BeginTabItem("Object Physics"))
                    {
                        bool use_triangle_rendering = dfg.get_feature<bool>("use_triangle_rendering");
                        ImGui::Checkbox("Use Triangle Rendering", &use_triangle_rendering);
                        dfg.set_feature("use_triangle_rendering", use_triangle_rendering);

                        if(dfg.is_dirty && use_triangle_rendering)
                        {
                            tris.build(mqueue);
                            phys.setup(mqueue, tris);
                        }

                        if(dfg.is_dirty)
                            should_soft_recompile = true;

                        for(int idx = 0; idx < (int)tris.cpu_objects.size(); idx++)
                        {
                            std::shared_ptr<triangle_rendering::object> obj = tris.cpu_objects[idx];

                            std::string name = "Object " + std::to_string(idx);

                            ImGui::SetNextTreeNodeOpen(true, ImGuiCond_Appearing);

                            if(ImGui::TreeNode(name.c_str()))
                            {
                                bool dirty = false;

                                dirty |= ImGui::DragFloat4("Pos", &obj->pos.v[0], 0.1f, 0.f, 0.f);
                                dirty |= ImGui::DragFloat3("Vel", &obj->velocity.v[0], 0.01f, -1.f, 1.f);

                                if(dirty)
                                {
                                    phys.needs_trace = true;
                                }

                                float vel_max = 0.9999f;

                                if(obj->velocity.length() > vel_max)
                                {
                                    obj->velocity = obj->velocity.norm() * vel_max;
                                }

                                ImGui::TreePop();
                            }
                        }

                        if(ImGui::Button("Rebuild"))
                        {
                            phys.needs_trace = true;
                        }

                        if(ImGui::Button("Chuck Box"))
                        {
                            should_chuck_object = true;
                        }

                        ImGui::SliderFloat("Object Chuck Speed", &object_chuck_speed, 0.f, 0.99999f);

                        ImGui::EndTabItem();
                    }

                    ImGui::EndTabBar();
                }

                ImGui::End();
            }

            //if(time_progresses)
            //    camera.v[0] += time / 1000.f;

            if(camera_time_progresses)
                current_geodesic_time += camera_geodesic_time_progression_speed * time / 1000.f;

            if(!has_geodesic)
            {
                camera_on_geodesic = false;
                camera_time_progresses = false;
            }

            if(dfg.is_static_dirty)
            {
                should_recompile = true;
                dfg.is_static_dirty = false;
            }

            if(metric_manage.check_recompile(should_recompile, should_soft_recompile, parent_directories,
                                          all_content, metric_names, dynamic_config, mqueue, dfg,
                                          sett, clctx.ctx))
            {
                phys.needs_trace = true;

                ///clearing the termination buffers is necessary
                for(auto& i : states)
                {
                    i.termination_buffer.set_to_zero(mqueue);
                }
            }

            dfg.alloc_and_write_gpu_buffer(mqueue, dynamic_feature_buffer);

            if(dfg.get_feature<bool>("use_triangle_rendering") && should_chuck_object)
            {
                std::shared_ptr<triangle_rendering::object> obj = tris.make_new();

                obj->tris = make_cube({0, 0, 0});
                obj->pos = cam.pos;

                vec3f dir = {0, 0, 1};

                vec3f chuck_dir = rot_quat(dir, cam.rot);

                obj->velocity = chuck_dir * object_chuck_speed;

                tris.build(mqueue);
                phys.setup(mqueue, tris);

                should_chuck_object = false;
            }

            metric_manage.check_substitution(clctx.ctx);

            if(time_progresses)
            {
                set_camera_time += time_progression_factor * time / 1000.f;

                cam.pos.x() = set_camera_time;
            }

            if(should_update_camera_time)
            {
                should_update_camera_time = false;

                cam.pos.x() = set_camera_time;
            }

            float speed = 0.1;

            if(!menu.is_open && !ImGui::GetIO().WantCaptureKeyboard)
            {
                if(input.is_key_down("speed_10x"))
                    speed *= 10;

                if(input.is_key_down("speed_superslow"))
                    speed = 0.00001;

                if(input.is_key_down("speed_100th"))
                    speed /= 100;

                if(input.is_key_down("speed_100x"))
                    speed *= 100;

                if(input.is_key_down("camera_reset"))
                {
                    reset_camera = true;
                    set_camera_time = 0;
                }

                if(input.is_key_down("camera_centre"))
                {
                    reset_camera = true;
                    set_camera_time = 0;
                }

                if(input.is_key_pressed("toggle_wormhole_space"))
                    flip_sign = !flip_sign;

                if(input.is_key_pressed("toggle_geodesic_play"))
                    camera_time_progresses = !camera_time_progresses;

                if(input.is_key_down("play_speed_minus"))
                {
                    camera_geodesic_time_progression_speed -= 1.f * frametime_s;
                    camera_geodesic_time_progression_speed = clamp(camera_geodesic_time_progression_speed, 0.f, 4.f);
                }

                if(input.is_key_down("play_speed_plus"))
                {
                    camera_geodesic_time_progression_speed += 1.f * frametime_s;
                    camera_geodesic_time_progression_speed = clamp(camera_geodesic_time_progression_speed, 0.f, 4.f);
                }

                vec2f delta;

                if(!raw_input_manager.is_enabled)
                {
                    delta.x() = (float)input.is_key_down("camera_turn_right") - (float)input.is_key_down("camera_turn_left");
                    delta.y() = (float)input.is_key_down("camera_turn_up") - (float)input.is_key_down("camera_turn_down");
                }
                else
                {
                    delta = {ImGui::GetIO().MouseDelta.x, ImGui::GetIO().MouseDelta.y};

                    delta.y() = -delta.y();

                    float mouse_sensitivity_mult = 0.05;

                    delta *= mouse_sensitivity_mult;
                }

                delta *= menu.sett.mouse_sensitivity * M_PI/128;

                vec4f translation_delta = {input.is_key_down("forward") - input.is_key_down("back"),
                                           input.is_key_down("right") - input.is_key_down("left"),
                                           input.is_key_down("down") - input.is_key_down("up"),
                                           input.is_key_down("time_forwards") - input.is_key_down("time_backwards")};

                translation_delta *= menu.sett.keyboard_sensitivity * controls_multiplier * speed;

                if(translation_delta.x() != 0 || translation_delta.y() != 0 || translation_delta.z() != 0 || translation_delta.w() != 0 || delta.x() != 0 || delta.y() != 0)
                {
                    float universe_size = dfg.get_feature<float>("universe_size");

                    cam.handle_input(delta, translation_delta, universe_size);
                }
            }

            cl::buffer& g_camera_pos_cart = st.g_camera_pos_cart;
            cl::event camera_pos_event = g_camera_pos_cart.write_async(mqueue, std::span{&cam.pos.v[0], 4});

            cl::buffer& g_camera_quat = st.g_camera_quat;
            cl::event camera_quat_event = g_camera_quat.write_async(mqueue, std::span{&cam.rot.q.v[0], 4});

            cl::event geodesic_event;
            cl::buffer& g_geodesic_basis_speed = st.g_geodesic_basis_speed;

            //if(should_set_observer_velocity)
            {
                vec3f base = {0, 0, 1};

                vec3f rotated = rot_quat(base, cam.rot).norm() * linear_basis_speed;

                vec4f geodesic_basis_speed = (vec4f){rotated.x(), rotated.y(), rotated.z(), 0.f};

                geodesic_event = g_geodesic_basis_speed.write_async(mqueue, std::span{&geodesic_basis_speed.v[0], 4});
            }

            if(camera_on_geodesic)
            {
                cl::buffer next_geodesic_velocity = geodesic_q.get_buffer(clctx.ctx);

                cl_int transport = parallel_transport_observer;

                cl::args interpolate_args;
                interpolate_args.push_back(geodesic_trace_buffer);
                interpolate_args.push_back(geodesic_vel_buffer);
                interpolate_args.push_back(geodesic_ds_buffer);
                interpolate_args.push_back(st.g_camera_pos_generic);

                for(auto& i : parallel_transported_tetrads)
                {
                    interpolate_args.push_back(i);
                }

                for(auto& i : st.tetrad)
                {
                    interpolate_args.push_back(i);
                }

                interpolate_args.push_back(current_geodesic_time);
                interpolate_args.push_back(geodesic_count_buffer);
                interpolate_args.push_back(transport);
                interpolate_args.push_back(g_geodesic_basis_speed);
                interpolate_args.push_back(next_geodesic_velocity);
                interpolate_args.push_back(dynamic_config);

                cl::event evt = mqueue.exec("handle_interpolating_geodesic", interpolate_args, {1}, {1}, {geodesic_event});

                if(!menu.sett.no_gpu_reads)
                    geodesic_q.start_read(clctx.ctx, async_queue, std::move(next_geodesic_velocity), {evt});
            }

            if(!camera_on_geodesic)
            {
                {
                    cl_float clflip = flip_sign;

                    cl::args args;
                    args.push_back(g_camera_pos_cart,
                                   st.g_camera_pos_generic,
                                   (cl_int)1,
                                   clflip,
                                   dynamic_config);

                    mqueue.exec("cart_to_generic_kernel", args, {1}, {1}, {camera_pos_event});
                }

                {
                    int count = 1;

                    cl::args tetrad_args;
                    tetrad_args.push_back(st.g_camera_pos_generic);
                    tetrad_args.push_back(count);
                    tetrad_args.push_back(cartesian_basis_speed);

                    for(int i=0; i < 4; i++)
                    {
                        tetrad_args.push_back(st.tetrad[i]);
                    }

                    tetrad_args.push_back(dynamic_config);

                    mqueue.exec("init_basis_vectors", tetrad_args, {1}, {1});
                }
            }

            if(!menu.sett.no_gpu_reads)
            {
                {
                    cl::buffer next_generic_camera = camera_q.get_buffer(clctx.ctx);
                    cl::event copy_generic = cl::copy(mqueue, st.g_camera_pos_generic, next_generic_camera);

                    camera_q.start_read(clctx.ctx, async_queue, std::move(next_generic_camera), {copy_generic});
                }

                {
                    cl::buffer next_timelike_coordinate = timelike_q.get_buffer(clctx.ctx);

                    cl::args args;
                    args.push_back(st.g_camera_pos_generic);
                    args.push_back(dynamic_config);
                    args.push_back(next_timelike_coordinate);

                    cl::event evt = mqueue.exec("calculate_timelike_coordinate", args, {1}, {1});

                    timelike_q.start_read(clctx.ctx, async_queue, std::move(next_timelike_coordinate), {evt});
                }
            }

            cl::event produce_event;

            {
                int width = glis.rtex.size<2>().x();
                int height = glis.rtex.size<2>().y();

                if(dfg.get_feature<bool>("use_triangle_rendering"))
                {
                    tris.update_objects(mqueue);

                    phys.trace(mqueue, tris, dynamic_config, dynamic_feature_buffer);
                    single_state.lazy_allocate(width, height);
                }
                else
                {
                    tris = triangle_rendering::manager(clctx.ctx);
                    phys = physics(clctx.ctx);
                    single_state.deallocate(clctx.ctx);
                }

                ///should invert geodesics is unused for the moment
                int isnap = 0;

                cl_int prepass_width = width/16;
                cl_int prepass_height = height/16;

                /*if(metric_manage.current_metric->metric_cfg.use_prepass)
                {
                    st.termination_buffer.set_to_zero(mqueue);
                }*/

                if(metric_manage.current_metric->metric_cfg.use_prepass)
                {
                    cl::args clear_args;
                    clear_args.push_back(st.termination_buffer);
                    clear_args.push_back(prepass_width);
                    clear_args.push_back(prepass_height);

                    mqueue.exec("clear_termination_buffer", clear_args, {prepass_width*prepass_height}, {256});

                    cl::args init_args_prepass;

                    init_args_prepass.push_back(st.g_camera_pos_generic);
                    init_args_prepass.push_back(g_camera_quat);
                    init_args_prepass.push_back(st.rays_in);
                    init_args_prepass.push_back(st.rays_count_in);
                    init_args_prepass.push_back(prepass_width);
                    init_args_prepass.push_back(prepass_height);
                    init_args_prepass.push_back(st.termination_buffer);
                    init_args_prepass.push_back(prepass_width);
                    init_args_prepass.push_back(prepass_height);
                    init_args_prepass.push_back(isnap);

                    for(auto& i : st.tetrad)
                    {
                        init_args_prepass.push_back(i);
                    }

                    init_args_prepass.push_back(dynamic_config);

                    mqueue.exec("init_rays_generic", init_args_prepass, {prepass_width*prepass_height}, {256}, {camera_quat_event, last_last_event});

                    int rays_num = calculate_ray_count(prepass_width, prepass_height);

                    execute_kernel(menu.sett, mqueue, st.rays_in, st.rays_count_in, st.accel_ray_time_min, st.accel_ray_time_max, tris, phys, rays_num, false, dfg, dynamic_config, dynamic_feature_buffer, st.width, st.height, st, single_state, last_event);

                    cl::args singular_args;
                    singular_args.push_back(st.rays_in);
                    singular_args.push_back(st.rays_count_in);
                    singular_args.push_back(st.termination_buffer);
                    singular_args.push_back(prepass_width);
                    singular_args.push_back(prepass_height);

                    mqueue.exec("calculate_singularities", singular_args, {prepass_width*prepass_height}, {256});
                }

                cl::args init_args;
                init_args.push_back(st.g_camera_pos_generic);
                init_args.push_back(g_camera_quat);
                init_args.push_back(st.rays_in);
                init_args.push_back(st.rays_count_in);
                init_args.push_back(width);
                init_args.push_back(height);
                init_args.push_back(st.termination_buffer);
                init_args.push_back(prepass_width);
                init_args.push_back(prepass_height);
                init_args.push_back(isnap);

                for(auto& i : st.tetrad)
                {
                    init_args.push_back(i);
                }

                init_args.push_back(dynamic_config);

                mqueue.exec("init_rays_generic", init_args, {width*height}, {16*16}, {camera_quat_event, last_last_event});

                int rays_num = calculate_ray_count(width, height);

                execute_kernel(menu.sett, mqueue, st.rays_in, st.rays_count_in, st.accel_ray_time_min, st.accel_ray_time_max, tris, phys, rays_num, false, dfg, dynamic_config, dynamic_feature_buffer, st.width, st.height, st, single_state, last_event);

                cl::args texture_args;
                texture_args.push_back(st.rays_in);
                texture_args.push_back(st.rays_count_in);
                texture_args.push_back(st.texture_coordinates);
                texture_args.push_back(width);
                texture_args.push_back(height);
                texture_args.push_back(dynamic_config);
                texture_args.push_back(dynamic_feature_buffer);

                mqueue.exec("calculate_texture_coordinates", texture_args, {width, height}, {16, 16});

                cl::args render_args;
                render_args.push_back(st.rays_in);
                render_args.push_back(st.rays_count_in);
                render_args.push_back(glis.rtex);
                render_args.push_back(back_images.i1);
                render_args.push_back(back_images.i2);
                render_args.push_back(width);
                render_args.push_back(height);
                render_args.push_back(st.texture_coordinates);
                render_args.push_back(menu.sett.anisotropy);
                render_args.push_back(dynamic_config);
                render_args.push_back(dynamic_feature_buffer);

                last_last_event.block();

                produce_event = mqueue.exec("render", render_args, {width, height}, {16, 16});

                /*{
                    cl::args dbg;
                    dbg.push_back(rtex);

                    mqueue.exec("interpolate_debug", dbg, {width, height}, {16, 16});
                }*/

                /*cl::args dbg;
                dbg.push_back(rtex);

                mqueue.exec("interpolate_debug2", dbg, {width, height}, {16, 16});*/
            }

            if(dfg.get_feature<bool>("use_triangle_rendering"))
            {
                int chunk_x = menu.sett.workgroup_size[0];
                int chunk_y = menu.sett.workgroup_size[1];

                int chunk_x_num = get_chunk_size(menu.sett.width, chunk_x);
                int chunk_y_num = get_chunk_size(menu.sett.height, chunk_y);

                int chunks = chunk_x_num * chunk_y_num;

                ///completely standalone
                {
                    single_state.computed_tri_count.set_to_zero(mqueue);

                    cl::args args;
                    args.push_back(tris.tris);
                    args.push_back(tris.tri_count);
                    args.push_back(phys.object_count);
                    args.push_back(phys.subsampled_paths);
                    args.push_back(phys.subsampled_counts);

                    for(int i=0; i < 4; i++)
                        args.push_back(phys.subsampled_parallel_transported_tetrads[i]);

                    args.push_back(single_state.computed_tris);
                    args.push_back(single_state.computed_tri_count);
                    args.push_back(st.accel_ray_time_min);
                    args.push_back(st.accel_ray_time_max);
                    args.push_back(dynamic_config);

                    mqueue.exec("generate_computed_tris", args, {tris.tri_count}, {128});
                }

                st.already_rendered.set_to_zero(mqueue);

                for(int kk=0; kk < 4; kk++)
                {
                    {
                        cl::args args;
                        args.push_back(single_state.stored_rays);
                        args.push_back(st.stored_ray_counts);
                        args.push_back(single_state.max_stored);
                        args.push_back(st.stored_mins);
                        args.push_back(st.stored_maxs);
                        args.push_back(st.width);
                        args.push_back(st.height);
                        args.push_back(st.chunked_mins);
                        args.push_back(st.chunked_maxs);
                        args.push_back(kk);
                        args.push_back(cl::local_memory(sizeof(cl_float4) * chunk_x * chunk_y));
                        args.push_back(cl::local_memory(sizeof(cl_float4) * chunk_x * chunk_y));
                        args.push_back(cl::local_memory(sizeof(cl_char) * chunk_x * chunk_y));

                        mqueue.exec("generate_clip_regions", args, {st.width, st.height}, {chunk_x, chunk_y});
                    }

                    {
                        single_state.tri_list_allocator.set_to_zero(mqueue);

                        cl::args args;
                        args.push_back(single_state.computed_tris);
                        args.push_back(single_state.computed_tri_count);

                        args.push_back(single_state.tri_list1);
                        args.push_back(single_state.tri_list_counts1);
                        args.push_back(single_state.tri_list_allocator);
                        args.push_back(single_state.tri_list_offsets);
                        args.push_back(single_state.max_tris);
                        args.push_back(st.chunked_mins);
                        args.push_back(st.chunked_maxs);
                        args.push_back(chunk_x);
                        args.push_back(chunk_y);
                        args.push_back(st.width);
                        args.push_back(st.height);
                        args.push_back(dynamic_config);

                        ///we could actually work this out, because computed tris are only generated once in theory
                        mqueue.exec("generate_tri_lists2", args, {chunk_x_num, chunk_y_num}, {8, 8});
                        //mqueue.exec("generate_tri_lists", args, {1024 * 1024 * 10}, {128});
                    }

                    {
                        cl::args args;
                        args.push_back(single_state.computed_tris);
                        args.push_back(single_state.computed_tri_count);
                        args.push_back(phys.object_count);
                        args.push_back(glis.rtex);
                        args.push_back(single_state.tri_list1);
                        args.push_back(single_state.tri_list_counts1);
                        args.push_back(single_state.tri_list_offsets);
                        args.push_back(st.width);
                        args.push_back(st.height);
                        args.push_back(chunk_x);
                        args.push_back(chunk_y);
                        args.push_back(single_state.stored_rays);
                        args.push_back(st.stored_ray_counts);
                        args.push_back(phys.subsampled_paths);
                        args.push_back(phys.subsampled_counts);

                        for(int i=0; i < 4; i++)
                            args.push_back(phys.subsampled_parallel_transported_tetrads[i]);

                        for(int i=0; i < 4; i++)
                            args.push_back(phys.subsampled_inverted_tetrads[i]);

                        args.push_back(st.stored_mins);
                        args.push_back(st.stored_maxs);
                        args.push_back(st.already_rendered);

                        float mx = ImGui::GetIO().MousePos.x;
                        float my = ImGui::GetIO().MousePos.y;

                        args.push_back(dynamic_config);
                        args.push_back(mx);
                        args.push_back(my);
                        args.push_back(kk);

                        produce_event = mqueue.exec("render_chunked_tris", args, {st.width, st.height}, {chunk_x, chunk_y});
                    }
                }
                /*cl::args intersect_args;
                intersect_args.push_back(st.tri_intersections);
                intersect_args.push_back(st.tri_intersections_count);
                intersect_args.push_back(accel.memory);
                intersect_args.push_back(glis.rtex);

                produce_event = mqueue.exec("render_intersections", intersect_args, {glis.rtex.size<2>().x() * glis.rtex.size<2>().y()}, {256});*/
            }

            last_last_event = last_event;
            last_event = produce_event;

            /*{
                cl::args tri_args;
                tri_args.push_back(gpu_tris);
                tri_args.push_back(tri_count);
                tri_args.push_back(finished_1);
                tri_args.push_back(finished_count_1);
                tri_args.push_back(visual_ray_path);
                tri_args.push_back(visual_ray_counts);
                tri_args.push_back(width);
                tri_args.push_back(height);
                tri_args.push_back(rtex);
                tri_args.push_back(dynamic_config);

                mqueue.exec("render_tris", tri_args, {width, height}, {16, 16});
            }*/

            if(should_snapshot_geodesic)
            {
                current_geodesic_time = 0;
                has_geodesic = true;
                camera_on_geodesic = true;

                ///lorentz boost tetrad by the basis speed
                ///g_geodesic_basis_speed is defined locally
                ///so first i calculate a timelike vector from the existing tetrad from the speed
                ///and then interpret this is an observer velocity, and boost the tetrads
                {
                    int count_in = 1;

                    cl::args lorentz;
                    lorentz.push_back(st.g_camera_pos_generic);
                    lorentz.push_back(count_in);
                    lorentz.push_back(g_geodesic_basis_speed);

                    for(auto& i : st.tetrad)
                    {
                        lorentz.push_back(i);
                    }

                    lorentz.push_back(dynamic_config);

                    mqueue.exec("boost_tetrad", lorentz, {1}, {1}, {geodesic_event});
                }

                {
                    generic_geodesic_count.set_to_zero(mqueue);

                    int count_in = 1;

                    cl::args args;
                    args.push_back(st.g_camera_pos_generic);
                    args.push_back(count_in);
                    args.push_back(generic_geodesic_buffer);
                    args.push_back(generic_geodesic_count);

                    for(auto& i : st.tetrad)
                    {
                        args.push_back(i);
                    }

                    args.push_back(g_geodesic_basis_speed);
                    args.push_back(dynamic_config);

                    mqueue.exec("init_inertial_ray", args, {count_in}, {1});
                }

                geodesic_trace_buffer.set_to_zero(mqueue);
                geodesic_count_buffer.set_to_zero(mqueue);
                geodesic_vel_buffer.set_to_zero(mqueue);
                geodesic_ds_buffer.set_to_zero(mqueue);

                cl::args snapshot_args;
                snapshot_args.push_back(generic_geodesic_buffer);
                snapshot_args.push_back(geodesic_trace_buffer);
                snapshot_args.push_back(geodesic_vel_buffer);
                snapshot_args.push_back(geodesic_ds_buffer);
                snapshot_args.push_back(generic_geodesic_count);

                snapshot_args.push_back(max_trace_length);
                snapshot_args.push_back(dynamic_config);
                snapshot_args.push_back(dynamic_feature_buffer);
                snapshot_args.push_back(geodesic_count_buffer);

                mqueue.exec("get_geodesic_path", snapshot_args, {1}, {1});

                for(int i=0; i < (int)parallel_transported_tetrads.size(); i++)
                {
                    int count = 1;

                    cl::args pt_args;
                    pt_args.push_back(geodesic_trace_buffer);
                    pt_args.push_back(geodesic_vel_buffer);
                    pt_args.push_back(geodesic_ds_buffer);
                    pt_args.push_back(st.tetrad[i]);
                    pt_args.push_back(geodesic_count_buffer);
                    pt_args.push_back(count);
                    pt_args.push_back(parallel_transported_tetrads[i]);
                    pt_args.push_back(dynamic_config);

                    mqueue.exec("parallel_transport_quantity", pt_args, {1}, {1});
                }
            }

            if(!taking_screenshot)
            {
                unprocessed_frames++;
                glsq.add(std::move(glis));
            }

            if(taking_screenshot)
            {
                glis.rtex.unacquire(mqueue);

                print("Taking screenie\n");

                mqueue.block();

                int high_width = menu.sett.screenshot_width * menu.sett.supersample_factor;
                int high_height = menu.sett.screenshot_height * menu.sett.supersample_factor;

                print("Blocked\n");

                std::cout << "WIDTH " << high_width << " HEIGHT "<< high_height << std::endl;

                std::vector<vec4f> pixels = glis.tex.read(0);

                std::cout << "pixels size " << pixels.size() << std::endl;

                assert(pixels.size() == (high_width * high_height));

                sf::Image img;
                img.create(high_width, high_height);

                for(int y=0; y < high_height; y++)
                {
                    for(int x=0; x < high_width; x++)
                    {
                        vec4f current_pixel = pixels[y * high_width + x];

                        current_pixel = clamp(current_pixel, 0.f, 1.f);
                        current_pixel = lin_to_srgb(current_pixel);
                        current_pixel = clamp(current_pixel, 0.f, 1.f);

                        img.setPixel(x, y, sf::Color(current_pixel.x() * 255.f, current_pixel.y() * 255.f, current_pixel.z() * 255.f, current_pixel.w() * 255.f));
                    }
                }

                std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();
                auto duration = now.time_since_epoch();
                auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

                file::mkdir("./screenshots");

                std::string fname = "./screenshots/" + metric_manage.current_metric->metric_cfg.name + "_" + std::to_string(millis) + ".png";

                img.saveToFile(fname);

                bool add_to_steam_library = menu.sett.use_steam_screenshots;

                if(steam.is_enabled() && add_to_steam_library)
                {
                    std::vector<vec<3, char>> as_rgb;
                    as_rgb.resize(high_width * high_height);

                    for(int y=0; y < high_height; y++)
                    {
                        for(int x=0; x < high_width; x++)
                        {
                            sf::Color c = img.getPixel(x, y);

                            as_rgb[y * high_width + x] = {c.r, c.g, c.b};
                        }
                    }

                    ISteamScreenshots* iss = SteamAPI_SteamScreenshots();

                    SteamAPI_ISteamScreenshots_WriteScreenshot(iss, as_rgb.data(), sizeof(vec<3, char>) * as_rgb.size(), high_width, high_height);
                }

                unacquired.add(std::move(glis));
            }
        }

        if(menu.sett.max_frames_ahead == 0)
            mqueue.block();

        if(glexec.peek() || unprocessed_frames >= menu.sett.max_frames_ahead)
        {
            ///so, if the number of unprocessed frames is large enough, consume *all* unprocessed frames
            ///this is to account for draining the queue if menu.sett.max_frames_ahead changes
            ///otherwise, we eat a single frame, as peek has come through
            int frames_to_consume = 1;

            if(unprocessed_frames >= menu.sett.max_frames_ahead && menu.sett.max_frames_ahead != 0)
            {
                frames_to_consume = (unprocessed_frames - menu.sett.max_frames_ahead) + 1;
            }

            for(int i=0; i < frames_to_consume; i++)
            {
                auto opt = glexec.produce(true);

                assert(opt.has_value());

                unprocessed_frames--;

                if(last_frame_opt.has_value())
                    unacquired.add(std::move(last_frame_opt.value()));

                last_frame_opt = std::move(opt.value());
            }
        }

        if(last_frame_opt.has_value())
        {
            ImDrawList* lst = hide_ui ?
                              ImGui::GetForegroundDrawList(ImGui::GetMainViewport()) :
                              ImGui::GetBackgroundDrawList(ImGui::GetMainViewport());

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

            lst->AddImage((void*)last_frame_opt.value().tex.handle, tl, br, ImVec2(0, 0), ImVec2(1, 1));
        }

        ImGui::PopAllowKeyboardFocus();

        win.display();

        if(should_print_frametime)
        {
            float frametime_ms = frametime_s * 1000;

            ///Don't replace this printf
            ///do not change this string
            printf("Frametime Elapsed: %f\n", frametime_ms);
        }

        if(frametime_s > 20 && once)
            return 0;

        once = true;
    }

    release_thread.request_stop();
    acquire_thread.request_stop();

    //mqueue.block();
    clctx.cqueue.block();

    return 0;
}
