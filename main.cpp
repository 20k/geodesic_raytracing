#include <iostream>
#include <toolkit/render_window.hpp>
#include <toolkit/texture.hpp>
#include <vec/vec.hpp>
#include <GLFW/glfw3.h>
#include <SFML/Graphics.hpp>
#include <CL/cl_ext.h>
#include "dual.hpp"
#include "dual_value.hpp"
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

struct lightray
{
    cl_float4 position;
    cl_float4 velocity;
    cl_float4 acceleration;
    cl_float4 initial_quat;
    cl_uint sx, sy;
    cl_float ku_uobsu;
    cl_float original_theta;
    cl_int early_terminate;
};

#define GENERIC_METRIC

cl::image load_mipped_image(const std::string& fname, opencl_context& clctx)
{
    sf::Image img;
    img.loadFromFile(fname);

    std::vector<uint8_t> as_uint8;

    for(int y=0; y < (int)img.getSize().y; y++)
    {
        for(int x=0; x < (int)img.getSize().x; x++)
        {
            auto col = img.getPixel(x, y);

            as_uint8.push_back(col.r);
            as_uint8.push_back(col.g);
            as_uint8.push_back(col.b);
            as_uint8.push_back(col.a);
        }
    }

    texture_settings bsett;
    bsett.width = img.getSize().x;
    bsett.height = img.getSize().y;
    bsett.is_srgb = false;

    texture opengl_tex;
    opengl_tex.load_from_memory(bsett, &as_uint8[0]);

    #define MIP_LEVELS 10

    int max_mips = floor(log2(std::min(img.getSize().x, img.getSize().y))) + 1;

    max_mips = std::min(max_mips, MIP_LEVELS);

    cl::image image_mipped(clctx.ctx);
    image_mipped.alloc((vec3i){img.getSize().x, img.getSize().y, max_mips}, {CL_RGBA, CL_FLOAT}, cl::image_flags::ARRAY);

    ///and all of THIS is to work around a bug in AMDs drivers, where you cant write to a specific array level!
    int swidth = img.getSize().x;
    int sheight = img.getSize().y;

    std::vector<std::vector<vec4f>> all_mip_levels;
    all_mip_levels.reserve(max_mips);

    for(int i=0; i < max_mips; i++)
    {
        all_mip_levels.push_back(opengl_tex.read(i));
    }

    std::vector<std::vector<vec4f>> uniformly_padded;

    for(int i=0; i < max_mips; i++)
    {
        int cwidth = swidth / pow(2, i);
        int cheight = sheight / pow(2, i);

        assert(cwidth * cheight == all_mip_levels[i].size());

        std::vector<vec4f> replacement;
        replacement.resize(swidth * sheight);

        for(int y = 0; y < sheight; y++)
        {
            for(int x=0; x < swidth; x++)
            {
                ///clamp to border
                int lx = std::min(x, cwidth - 1);
                int ly = std::min(y, cheight - 1);

                replacement[y * swidth + x] = all_mip_levels[i][ly * cwidth + lx];
            }
        }

        uniformly_padded.push_back(replacement);
    }

    std::vector<vec4f> as_uniform;

    for(auto& i : uniformly_padded)
    {
        for(auto& j : i)
        {
            as_uniform.push_back(j);
        }
    }

    vec<3, size_t> origin = {0, 0, 0};
    vec<3, size_t> region = {swidth, sheight, max_mips};

    image_mipped.write(clctx.cqueue, (char*)as_uniform.data(), origin, region);

    return image_mipped;
}

void execute_kernel(cl::command_queue& cqueue, cl::buffer& rays_in, cl::buffer& rays_out,
                                               cl::buffer& rays_finished,
                                               cl::buffer& count_in, cl::buffer& count_out,
                                               cl::buffer& count_finished,
                                               //cl::buffer& visual_path, cl::buffer& visual_ray_counts,
                                               triangle_rendering::manager& manage, cl::buffer& intersections, cl::buffer& intersections_count,
                                               cl::buffer& potential_intersections, cl::buffer& potential_intersections_count,
                                               triangle_rendering::acceleration& accel,
                                               int num_rays,
                                               bool use_device_side_enqueue,
                                               cl::buffer& dynamic_config)
{
    if(use_device_side_enqueue)
    {
        int fallback = 0;

        cl::args run_args;
        run_args.push_back(rays_in);
        run_args.push_back(rays_out);
        run_args.push_back(rays_finished);
        run_args.push_back(count_in);
        run_args.push_back(count_out);
        run_args.push_back(count_finished);
        run_args.push_back(fallback);
        run_args.push_back(dynamic_config);

        cqueue.exec("relauncher_generic", run_args, {1}, {1});
    }
    else
    {
        count_in.write_async(cqueue, (const char*)&num_rays, sizeof(int));
        count_out.set_to_zero(cqueue);
        count_finished.set_to_zero(cqueue);
        intersections_count.set_to_zero(cqueue);
        potential_intersections_count.set_to_zero(cqueue);

        cl::args run_args;
        run_args.push_back(rays_in);
        run_args.push_back(rays_out);
        run_args.push_back(rays_finished);
        run_args.push_back(count_in);
        run_args.push_back(count_out);
        run_args.push_back(count_finished);
        //run_args.push_back(visual_path);
        //run_args.push_back(visual_ray_counts);
        run_args.push_back(manage.tris);
        run_args.push_back(manage.tri_count);
        run_args.push_back(intersections);
        run_args.push_back(intersections_count);
        run_args.push_back(potential_intersections);
        run_args.push_back(potential_intersections_count);
        run_args.push_back(accel.counts);
        run_args.push_back(accel.offsets);
        run_args.push_back(accel.memory);
        run_args.push_back(accel.offset_width);
        run_args.push_back(accel.offset_size.x());
        run_args.push_back(manage.objects);
        run_args.push_back(dynamic_config);

        cqueue.exec("do_generic_rays", run_args, {num_rays}, {256});
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

    int state = MAIN;
    bool should_open = false;
    bool is_open = true;
    bool dirty_settings = false;
    bool should_quit = false;
    bool already_started = false;

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

    void display_settings_menu(render_window& win, input_manager& input)
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

    void display(render_window& win, input_manager& input)
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
                display_settings_menu(win, input);
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

    void start_read(cl::context& ctx, cl::command_queue& async_q, cl::buffer&& buf, cl::event wait_on)
    {
        element* e = new element(ctx);

        e->gpu_buffer = std::move(buf);

        e->evt = e->gpu_buffer.read_async(async_q, (char*)&e->data, sizeof(T), {wait_on});

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
        if(pool.size() > 0)
        {
            cl::buffer next = pool.back();
            pool.pop_back();
            return next;
        }

        cl::buffer buf(ctx);
        buf.alloc(sizeof(T));

        return buf;
    }

    void start_read(cl::context& ctx, cl::command_queue& async_q, cl::buffer&& buf, cl::event wait_on)
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


///i need the ability to have dynamic parameters
int main(int argc, char* argv[])
{
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
    }

    bool has_new_content = false;

    steam_info steam;

    ugc_view workshop;
    workshop.only_get_subscribed();

    steam_callback_executor exec;

    workshop.fetch(steam, exec, [&](){has_new_content = true;});

    //dual_types::test_operation();

    graphics_settings current_settings;

    bool loaded_settings = false;

    if(file::exists("settings.json"))
    {
        try
        {
            nlohmann::json js = nlohmann::json::parse(file::read("settings.json", file::mode::BINARY));

            deserialise<graphics_settings>(js, current_settings, serialise_mode::DISK);

            loaded_settings = true;
        }
        catch(std::exception& ex)
        {
            std::cout << "Failed to load settings.json " << ex.what() << std::endl;
        }
    }

    render_settings sett;
    sett.width = 800;
    sett.height = 600;
    sett.opencl = true;
    sett.no_double_buffer = false;
    sett.is_srgb = true;
    sett.no_decoration = true;
    sett.viewports = false;

    if(loaded_settings)
    {
        sett.width = current_settings.width;
        sett.height = current_settings.height;
    }

    render_window win(sett, "Geodesics");

    if(loaded_settings)
    {
        win.set_vsync(current_settings.vsync_enabled);
        win.backend->set_is_maximised(current_settings.fullscreen);

        if(current_settings.fullscreen)
        {
            win.backend->clear_demaximise_cache();
        }
    }
    else
    {
        win.backend->set_is_maximised(true);
        win.backend->clear_demaximise_cache();
    }

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

    metrics::config cfg;
    ///necessary for double schwarzs
    cfg.universe_size = 20;
    cfg.use_device_side_enqueue = false;
    //cfg.error_override = 100.f;
    //cfg.error_override = 0.000001f;
    //cfg.error_override = 0.000001f;
    //cfg.error_override = 0.00001f;
    //cfg.redshift = true;

    metrics::config current_cfg = cfg;

    //print("WLs %f %f %f\n", chromaticity::srgb_to_wavelength({1, 0, 0}), chromaticity::srgb_to_wavelength({0, 1, 0}), chromaticity::srgb_to_wavelength({0, 0, 1}));

    int last_supersample_mult = 2;

    int start_width = sett.width;
    int start_height = sett.height;

    texture_settings tsett;
    tsett.width = start_width;
    tsett.height = start_height;
    tsett.is_srgb = false;
    tsett.generate_mipmaps = false;

    texture tex;
    tex.load_from_memory(tsett, nullptr);

    cl::gl_rendertexture rtex{clctx.ctx};
    rtex.create_from_texture(tex.handle);

    cl::image background_mipped = load_mipped_image("background.png", clctx);
    cl::image background_mipped2 = load_mipped_image("background2.png", clctx);

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

    ///in polar, were .x = t
    cl::buffer g_camera_pos_cart(clctx.ctx);
    cl::buffer g_camera_pos_polar(clctx.ctx);
    cl::buffer g_camera_quat(clctx.ctx);

    g_camera_pos_cart.alloc(sizeof(cl_float4));
    g_camera_pos_polar.alloc(sizeof(cl_float4));
    g_camera_quat.alloc(sizeof(cl_float4));

    {
        cl_float4 camera_start_pos = {0, 0, -4, 0};

        quat camera_start_quat;
        camera_start_quat.load_from_axis_angle({1, 0, 0, -M_PI/2});

        g_camera_pos_cart.write(clctx.cqueue, std::span{&camera_start_pos, 1});

        cl_float4 as_cl_camera_quat = {camera_start_quat.q.x(), camera_start_quat.q.y(), camera_start_quat.q.z(), camera_start_quat.q.w()};

        g_camera_quat.write(clctx.cqueue, std::span{&as_cl_camera_quat, 1});
    }

    //camera_quat.load_from_matrix(axis_angle_to_mat({0, 0, 0}, 0));

    sf::Clock clk;

    int ray_count = start_width * start_height;

    print("Pre buffer declarations\n");

    cl::buffer schwarzs_1(clctx.ctx);
    cl::buffer schwarzs_scratch(clctx.ctx);
    cl::buffer schwarzs_prepass(clctx.ctx);
    cl::buffer finished_1(clctx.ctx);

    cl::buffer schwarzs_count_1(clctx.ctx);
    cl::buffer schwarzs_count_scratch(clctx.ctx);
    cl::buffer schwarzs_count_prepass(clctx.ctx);
    cl::buffer finished_count_1(clctx.ctx);

    cl::buffer termination_buffer(clctx.ctx);

    cl::buffer dynamic_config(clctx.ctx);

    print("Post buffer declarations\n");

    termination_buffer.alloc(start_width * start_height * sizeof(cl_int));

    print("Allocated termination buffer\n");

    termination_buffer.set_to_zero(clctx.cqueue);

    print("Zero termination buffer\n");

    cl::buffer geodesic_count_buffer(clctx.ctx);
    geodesic_count_buffer.alloc(sizeof(cl_int));

    int max_trace_length = 64000;

    cl::buffer geodesic_trace_buffer(clctx.ctx);
    geodesic_trace_buffer.alloc(max_trace_length * sizeof(cl_float4));

    cl::buffer geodesic_vel_buffer(clctx.ctx);
    geodesic_vel_buffer.alloc(max_trace_length * sizeof(cl_float4));

    cl::buffer geodesic_dT_ds_buffer(clctx.ctx);
    geodesic_dT_ds_buffer.alloc(max_trace_length * sizeof(cl_float));

    cl::buffer geodesic_ds_buffer(clctx.ctx);
    geodesic_ds_buffer.alloc(max_trace_length * sizeof(cl_float));

    geodesic_count_buffer.set_to_zero(clctx.cqueue);
    geodesic_trace_buffer.set_to_zero(clctx.cqueue);
    geodesic_vel_buffer.set_to_zero(clctx.cqueue);
    geodesic_dT_ds_buffer.set_to_zero(clctx.cqueue);
    geodesic_ds_buffer.set_to_zero(clctx.cqueue);

    cl::buffer generic_geodesic_buffer(clctx.ctx);
    generic_geodesic_buffer.alloc(sizeof(cl_float4) * 1024);

    cl::buffer generic_geodesic_count(clctx.ctx);
    generic_geodesic_count.alloc(sizeof(cl_int));
    generic_geodesic_count.set_to_zero(clctx.cqueue);

    cl::buffer g_geodesic_basis_speed(clctx.ctx);
    g_geodesic_basis_speed.alloc(sizeof(cl_float4));
    g_geodesic_basis_speed.set_to_zero(clctx.cqueue);

    std::array<cl::buffer, 4> tetrad{clctx.ctx, clctx.ctx, clctx.ctx, clctx.ctx};
    std::array<cl::buffer, 4> geodesic_tetrad{clctx.ctx, clctx.ctx, clctx.ctx, clctx.ctx};

    for(int i=0; i < 4; i++)
    {
        tetrad[i].alloc(sizeof(cl_float4));
        tetrad[i].set_to_zero(clctx.cqueue);

        geodesic_tetrad[i].alloc(sizeof(cl_float4));
        geodesic_tetrad[i].set_to_zero(clctx.cqueue);
    }

    std::array<cl::buffer, 4> parallel_transported_tetrads{clctx.ctx, clctx.ctx, clctx.ctx, clctx.ctx};

    for(int i=0; i < 4; i++)
    {
        parallel_transported_tetrads[i].alloc(max_trace_length * sizeof(cl_float4));
        parallel_transported_tetrads[i].set_to_zero(clctx.cqueue);
    }

    cl::buffer timelike_geodesic_pos{clctx.ctx};
    timelike_geodesic_pos.alloc(sizeof(cl_float4));
    timelike_geodesic_pos.set_to_zero(clctx.cqueue);

    cl::buffer timelike_geodesic_vel{clctx.ctx};
    timelike_geodesic_vel.alloc(sizeof(cl_float4));
    timelike_geodesic_vel.set_to_zero(clctx.cqueue);

    print("Alloc trace buffer\n");

    std::vector<cl_float4> current_geodesic_path;
    std::vector<cl_float> current_geodesic_dT_ds;

    print("Pre texture coordinates\n");

    cl::buffer texture_coordinates{clctx.ctx};

    texture_coordinates.alloc(start_width * start_height * sizeof(float) * 2);
    texture_coordinates.set_to_zero(clctx.cqueue);

    print("Post texture coordinates\n");

    schwarzs_1.alloc(sizeof(lightray) * ray_count);
    schwarzs_scratch.alloc(sizeof(lightray) * ray_count);
    schwarzs_prepass.alloc(sizeof(lightray) * ray_count);
    finished_1.alloc(sizeof(lightray) * ray_count);

    schwarzs_count_1.alloc(sizeof(int));
    schwarzs_count_scratch.alloc(sizeof(int));
    schwarzs_count_prepass.alloc(sizeof(int));
    finished_count_1.alloc(sizeof(int));

    triangle_rendering::acceleration accel(clctx.ctx);

    int potential_intersection_size = 10;

    cl::buffer potential_intersections(clctx.ctx);
    potential_intersections.alloc(potential_intersection_size * 1024 * 1024 * 160);

    cl::buffer potential_intersection_count(clctx.ctx);
    potential_intersection_count.alloc(sizeof(cl_int));


    read_queue_pool<cl_float4> camera_q;
    read_queue_pool<cl_float4> geodesic_q;

    print("Finished async read queue init\n");

    printj("Supports shared events? ", cl::supports_extension(clctx.ctx, "cl_khr_gl_event"));

    bool last_supersample = false;
    bool should_take_screenshot = false;

    bool time_progresses = false;
    bool flip_sign = false;
    float current_geodesic_time = 0;
    bool camera_on_geodesic = false;
    bool camera_time_progresses = false;
    float camera_geodesic_time_progression_speed = 1.f;
    float set_camera_time = 0;
    bool parallel_transport_observer = true;
    cl_float4 cartesian_basis_speed = {0,0,0,0};
    float linear_basis_speed = 0.f;

    bool has_geodesic = false;

    print("Pre fullscreen\n");

    steady_timer workshop_poll;
    steady_timer frametime_timer;

    bool open_main_menu_trigger = true;
    main_menu menu;

    bool hide_ui = false;

    fullscreen_window_manager fullscreen;

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

    current_settings.width = win.get_window_size().x();
    current_settings.height = win.get_window_size().y();
    current_settings.vsync_enabled = win.backend->is_vsync();
    current_settings.fullscreen = win.backend->is_maximised();

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

    //#define INDIVIDUAL_GEODESICS
    #ifndef INDIVIDUAL_GEODESICS
    auto obj = tris.make_new("./models/newell_teaset/teapot.obj");
    obj->pos = {0, -5, 0, 0};
    obj->scale = 0.05f;
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

        //obj->scale = 0.45f;
    }
    #endif // INDIVIDUAL_GEODESICS

    tris.build(clctx.cqueue, accel.offset_width / accel.offset_size.x());

    physics phys(clctx.ctx);
    phys.setup(clctx.cqueue, tris);

    print("Pre main\n");

    cl::buffer gpu_intersections(clctx.ctx);
    gpu_intersections.alloc(sizeof(cl_int2) * start_width * start_height * 10);

    cl::buffer gpu_intersections_count(clctx.ctx);
    gpu_intersections_count.alloc(sizeof(cl_int));

    bool reset_camera = true;

    std::optional<cl_float4> last_camera_pos;
    std::optional<cl_float4> last_geodesic_velocity;

    while(!win.should_close() && !menu.should_quit && fullscreen.open)
    {
        if(menu.dirty_settings)
        {
            current_settings = menu.sett;

            if((vec2i){current_settings.width, current_settings.height} != win.get_window_size())
            {
                win.resize({current_settings.width, current_settings.height});
            }

            if(win.backend->is_vsync() != current_settings.vsync_enabled)
            {
                win.backend->set_vsync(current_settings.vsync_enabled);
            }

            if(win.backend->is_maximised() != current_settings.fullscreen)
            {
                win.backend->set_is_maximised(current_settings.fullscreen);
            }

            file::write_atomic("./settings.json", serialise(current_settings, serialise_mode::DISK).dump(), file::mode::BINARY);
        }

        if(!menu.is_open || menu.dirty_settings)
        {
            vec2i real_dim = win.get_window_size();

            menu.sett = current_settings;

            menu.sett.width = real_dim.x();
            menu.sett.height = real_dim.y();
            menu.sett.vsync_enabled = win.backend->is_vsync();
            menu.sett.fullscreen = win.backend->is_maximised();
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

                std::cout << "Added content " << path << std::endl;

                all_content.add_content_folder(path);
            }

            has_new_content = false;
        }

        float frametime_s = frametime_timer.restart();

        float controls_multiplier = 1.f;

        if(current_settings.time_adjusted_controls)
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
            menu.display(win, input);
        }

        {
            auto buffer_size = rtex.size<2>();

            bool taking_screenshot = should_take_screenshot;
            should_take_screenshot = false;

            bool should_snapshot_geodesic = false;

            vec<2, size_t> super_adjusted_width = current_settings.supersample ? (buffer_size / current_settings.supersample_factor) : buffer_size;

            if((vec2i){super_adjusted_width.x(), super_adjusted_width.y()} != win.get_window_size() || taking_screenshot || last_supersample != current_settings.supersample || last_supersample_mult != current_settings.supersample_factor || menu.dirty_settings)
            {
                int width = 16;
                int height = 16;

                if(!taking_screenshot)
                {
                    width = win.get_window_size().x();
                    height = win.get_window_size().y();

                    if(current_settings.supersample)
                    {
                        width *= current_settings.supersample_factor;
                        height *= current_settings.supersample_factor;
                    }
                }
                else
                {
                    width = current_settings.screenshot_width * current_settings.supersample_factor;
                    height = current_settings.screenshot_height * current_settings.supersample_factor;
                }

                width = max(width, 16 * current_settings.supersample_factor);
                height = max(height, 16 * current_settings.supersample_factor);

                ray_count = width * height;

                texture_settings new_sett;
                new_sett.width = width;
                new_sett.height = height;
                new_sett.is_srgb = false;
                new_sett.generate_mipmaps = false;

                tex.load_from_memory(new_sett, nullptr);
                rtex.create_from_texture(tex.handle);

                termination_buffer.alloc(width * height * sizeof(cl_int));
                termination_buffer.set_to_zero(clctx.cqueue);

                schwarzs_1.alloc(sizeof(lightray) * ray_count);
                schwarzs_scratch.alloc(sizeof(lightray) * ray_count);
                schwarzs_prepass.alloc(sizeof(lightray) * ray_count);
                finished_1.alloc(sizeof(lightray) * ray_count);

                texture_coordinates.alloc(width * height * sizeof(float) * 2);
                texture_coordinates.set_to_zero(clctx.cqueue);

                gpu_intersections.alloc(sizeof(cl_int) * ray_count * 10);

                last_supersample = current_settings.supersample;
                last_supersample_mult = current_settings.supersample_factor;
            }

            rtex.acquire(clctx.cqueue);

            float time = clk.restart().asMicroseconds() / 1000.;

            {
                std::vector<cl_float4> cam_data = camera_q.fetch();
                std::vector<cl_float4> geodesic_data = geodesic_q.fetch();

                if(cam_data.size() > 0)
                {
                    last_camera_pos = cam_data.back();
                }

                if(geodesic_data.size() > 0)
                {
                    last_geodesic_velocity = geodesic_data.back();
                }
            }

            bool should_soft_recompile = false;
            bool should_update_camera_time = false;
            bool should_set_observer_velocity = false;

            if(!taking_screenshot && !hide_ui && !menu.is_first_time_main_menu_open())
            {
                ImGui::Begin("Settings and Information", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

                if(ImGui::BeginTabBar("Tabbity tab tabs"))
                {
                    if(ImGui::BeginTabItem("General"))
                    {
                        if(!current_settings.no_gpu_reads)
                        {
                            if(last_camera_pos.has_value())
                            {
                                ImGui::DragFloat4("Camera Position", &last_camera_pos.value().s[0]);
                            }
                        }

                        if(ImGui::SliderFloat("Camera Time", &set_camera_time, 0.f, 100.f))
                        {
                            should_update_camera_time = true;
                        }

                        ImGui::DragFloat("Frametime", &time);

                        ImGui::Checkbox("Time Progresses", &time_progresses);

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

                                dynamic_config.write(clctx.cqueue, vars);
                                should_soft_recompile = true;
                            }
                        }

                        ImGui::Unindent();

                        ImGui::Text("Compile Options");

                        ImGui::Indent();

                        ImGui::Checkbox("Redshift", &cfg.redshift);

                        ImGui::InputFloat("Error Tolerance", &cfg.error_override, 0.0000001f, 0.00001f, "%.8f");

                        ImGui::DragFloat("Universe Size", &cfg.universe_size, 1, 1, 0, "%.1f");;

                        ImGui::DragFloat("Precision Radius", &cfg.max_precision_radius, 1, 0.0001f, cfg.universe_size, "%.1f");

                        if(ImGui::IsItemHovered())
                        {
                            ImGui::SetTooltip("Radius at which lightrays raise their precision checking unconditionally");
                        }

                        should_recompile |= ImGui::Button("Update");

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

                            if(!current_settings.no_gpu_reads)
                            {
                                if(last_camera_pos.has_value())
                                {
                                    ImGui::DragFloat4("Geodesic Position", &last_camera_pos.value().s[0]);
                                }

                                if(last_geodesic_velocity.has_value())
                                {
                                    ImGui::DragFloat4("Geodesic Velocity", &last_geodesic_velocity.value().s[0]);
                                }
                            }

                            ImGui::DragFloat("Proper Time", &current_geodesic_time, 0.1, 0.f, 0.f);

                            ImGui::Checkbox("Proper Time Progresses", &camera_time_progresses);

                            ImGui::SliderFloat("Proper Time Progession Rate", &camera_geodesic_time_progression_speed, 0.f, 4.f, "%.2f");

                            ImGui::Checkbox("Attach Camera to Geodesic", &camera_on_geodesic);

                            ImGui::Checkbox("Transport observer along geodesic", &parallel_transport_observer);
                        }

                        ImGui::EndTabItem();
                    }

                    if(ImGui::BeginTabItem("Physics"))
                    {
                        if(ImGui::Button("Rebuild"))
                        {
                            phys.needs_trace = true;
                            phys.trace(clctx.cqueue, tris, dynamic_config);
                        }

                        ImGui::EndTabItem();
                    }

                    ImGui::EndTabBar();
                }

                ImGui::End();
            }

            if(should_recompile)
            {
                current_cfg = cfg;
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

            metric_manage.check_recompile(should_recompile, should_soft_recompile, parent_directories,
                                          all_content, metric_names, dynamic_config, clctx.cqueue, cfg,
                                          sett, clctx.ctx, termination_buffer);

            metric_manage.check_substitution(clctx.ctx);

            if(time_progresses)
            {
                set_camera_time += time / 1000.f;

                cl::args args;
                args.push_back(g_camera_pos_cart);
                args.push_back(set_camera_time);

                clctx.cqueue.exec("set_time", args, {1}, {1});
            }

            if(should_update_camera_time)
            {
                cl::args args;
                args.push_back(g_camera_pos_cart);
                args.push_back(set_camera_time);

                clctx.cqueue.exec("set_time", args, {1}, {1});
                should_update_camera_time = false;
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
                    g_camera_pos_cart.write(clctx.cqueue, std::vector<cl_float4>{{0, 0, 0, -4}});
                }

                if(input.is_key_down("camera_centre"))
                {
                    reset_camera = true;
                    set_camera_time = 0;
                    g_camera_pos_cart.write(clctx.cqueue, std::vector<cl_float4>{{0, 0, 0, 0}});
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

                delta *= current_settings.mouse_sensitivity * M_PI/128;

                vec3f translation_delta = {input.is_key_down("forward") - input.is_key_down("back"),
                                           input.is_key_down("right") - input.is_key_down("left"),
                                           input.is_key_down("down") - input.is_key_down("up")};

                translation_delta *= current_settings.keyboard_sensitivity * controls_multiplier * speed;

                cl_float2 cl_mouse = {delta.x(), delta.y()};
                cl_float4 cl_translation = {translation_delta.x(), translation_delta.y(), translation_delta.z(), 0};

                cl::args controls_args;
                controls_args.push_back(g_camera_pos_cart);
                controls_args.push_back(g_camera_quat);
                controls_args.push_back(cl_mouse);
                controls_args.push_back(cl_translation);
                controls_args.push_back(current_cfg.universe_size);
                controls_args.push_back(dynamic_config);

                clctx.cqueue.exec("handle_controls_free", controls_args, {1}, {1});
            }

            //if(should_set_observer_velocity)
            {
                cl::args args;
                args.push_back(g_camera_quat);
                args.push_back(linear_basis_speed);
                args.push_back(g_geodesic_basis_speed);

                clctx.cqueue.exec("init_basis_speed", args, {1}, {1});
            }

            if(camera_on_geodesic)
            {
                cl::buffer next_geodesic_velocity = geodesic_q.get_buffer(clctx.ctx);

                cl_int transport = parallel_transport_observer;

                cl::args interpolate_args;
                interpolate_args.push_back(geodesic_trace_buffer);
                interpolate_args.push_back(geodesic_vel_buffer);
                interpolate_args.push_back(geodesic_dT_ds_buffer);
                interpolate_args.push_back(geodesic_ds_buffer);
                interpolate_args.push_back(g_camera_pos_polar);

                for(auto& i : parallel_transported_tetrads)
                {
                    interpolate_args.push_back(i);
                }

                for(auto& i : tetrad)
                {
                    interpolate_args.push_back(i);
                }

                interpolate_args.push_back(current_geodesic_time);
                interpolate_args.push_back(geodesic_count_buffer);
                interpolate_args.push_back(transport);
                interpolate_args.push_back(g_geodesic_basis_speed);
                interpolate_args.push_back(next_geodesic_velocity);
                interpolate_args.push_back(dynamic_config);

                cl::event evt = clctx.cqueue.exec("handle_interpolating_geodesic", interpolate_args, {1}, {1});

                geodesic_q.start_read(clctx.ctx, async_queue, std::move(next_geodesic_velocity), evt);
            }

            if(!camera_on_geodesic)
            {
                {
                    int count = 1;
                    cl_float clflip = flip_sign;

                    cl::args args;

                    args.push_back(g_camera_pos_cart);
                    args.push_back(g_camera_pos_polar);
                    args.push_back(count);
                    args.push_back(clflip);

                    clctx.cqueue.exec("cart_to_polar_kernel", args, {1}, {1});
                }

                {
                    int count = 1;

                    cl::args tetrad_args;
                    tetrad_args.push_back(g_camera_pos_polar);
                    tetrad_args.push_back(count);
                    tetrad_args.push_back(cartesian_basis_speed);

                    for(int i=0; i < 4; i++)
                    {
                        tetrad_args.push_back(tetrad[i]);
                    }

                    tetrad_args.push_back(dynamic_config);

                    clctx.cqueue.exec("init_basis_vectors", tetrad_args, {1}, {1});
                }
            }

            {
                cl::buffer next_generic_camera = camera_q.get_buffer(clctx.ctx);

                cl::args args;
                args.push_back(g_camera_pos_polar);
                args.push_back(next_generic_camera);
                args.push_back(dynamic_config);

                cl::event evt = clctx.cqueue.exec("camera_polar_to_generic", args, {1}, {1});

                camera_q.start_read(clctx.ctx, async_queue, std::move(next_generic_camera), evt);
            }

            int width = rtex.size<2>().x();
            int height = rtex.size<2>().y();

            cl::args clr;
            clr.push_back(rtex);

            clctx.cqueue.exec("clear", clr, {width, height}, {16, 16});

            {
                tris.update_objects(clctx.cqueue);

                phys.trace(clctx.cqueue, tris, dynamic_config);
                phys.push_object_positions(clctx.cqueue, tris, dynamic_config, set_camera_time);

                accel.build(clctx.cqueue, tris);
            }

            {
                ///should invert geodesics is unused for the moment
                int isnap = 0;

                float on_geodesic = camera_on_geodesic ? 1 : 0;

                cl_int prepass_width = width/16;
                cl_int prepass_height = height/16;

                if(metric_manage.current_metric->metric_cfg.use_prepass && tris.cpu_objects.size() == 0)
                {
                    cl::args clear_args;
                    clear_args.push_back(termination_buffer);
                    clear_args.push_back(prepass_width);
                    clear_args.push_back(prepass_height);

                    clctx.cqueue.exec("clear_termination_buffer", clear_args, {prepass_width*prepass_height}, {256});

                    cl::args init_args_prepass;

                    init_args_prepass.push_back(g_camera_pos_polar);
                    init_args_prepass.push_back(g_camera_quat);
                    init_args_prepass.push_back(schwarzs_prepass);
                    init_args_prepass.push_back(schwarzs_count_prepass);
                    init_args_prepass.push_back(prepass_width);
                    init_args_prepass.push_back(prepass_height);
                    init_args_prepass.push_back(termination_buffer);
                    init_args_prepass.push_back(prepass_width);
                    init_args_prepass.push_back(prepass_height);
                    init_args_prepass.push_back(isnap);

                    for(auto& i : tetrad)
                    {
                        init_args_prepass.push_back(i);
                    }

                    init_args_prepass.push_back(dynamic_config);

                    clctx.cqueue.exec("init_rays_generic", init_args_prepass, {prepass_width*prepass_height}, {256});

                    int rays_num = calculate_ray_count(prepass_width, prepass_height);

                    execute_kernel(clctx.cqueue, schwarzs_prepass, schwarzs_scratch, finished_1, schwarzs_count_prepass, schwarzs_count_scratch, finished_count_1, tris, gpu_intersections, gpu_intersections_count, potential_intersections, potential_intersection_count, accel, rays_num, cfg.use_device_side_enqueue, dynamic_config);

                    cl::args singular_args;
                    singular_args.push_back(finished_1);
                    singular_args.push_back(finished_count_1);
                    singular_args.push_back(termination_buffer);
                    singular_args.push_back(prepass_width);
                    singular_args.push_back(prepass_height);

                    clctx.cqueue.exec("calculate_singularities", singular_args, {prepass_width*prepass_height}, {256});
                }

                cl::args init_args;
                init_args.push_back(g_camera_pos_polar);
                init_args.push_back(g_camera_quat);
                init_args.push_back(schwarzs_1);
                init_args.push_back(schwarzs_count_1);
                init_args.push_back(width);
                init_args.push_back(height);
                init_args.push_back(termination_buffer);
                init_args.push_back(prepass_width);
                init_args.push_back(prepass_height);
                init_args.push_back(isnap);

                for(auto& i : tetrad)
                {
                    init_args.push_back(i);
                }

                init_args.push_back(dynamic_config);

                clctx.cqueue.exec("init_rays_generic", init_args, {width*height}, {16*16});

                int rays_num = calculate_ray_count(width, height);

                execute_kernel(clctx.cqueue, schwarzs_1, schwarzs_scratch, finished_1, schwarzs_count_1, schwarzs_count_scratch, finished_count_1, tris, gpu_intersections, gpu_intersections_count, potential_intersections, potential_intersection_count, accel, rays_num, cfg.use_device_side_enqueue, dynamic_config);

                cl::args texture_args;
                texture_args.push_back(finished_1);
                texture_args.push_back(finished_count_1);
                texture_args.push_back(texture_coordinates);
                texture_args.push_back(width);
                texture_args.push_back(height);
                texture_args.push_back(g_camera_pos_polar);
                texture_args.push_back(g_camera_quat);

                clctx.cqueue.exec("calculate_texture_coordinates", texture_args, {width * height}, {256});

                cl::args render_args;
                render_args.push_back(finished_1);
                render_args.push_back(finished_count_1);
                render_args.push_back(rtex);
                render_args.push_back(background_mipped);
                render_args.push_back(background_mipped2);
                render_args.push_back(width);
                render_args.push_back(height);
                render_args.push_back(texture_coordinates);
                render_args.push_back(current_settings.anisotropy);

                clctx.cqueue.exec("render", render_args, {width * height}, {256});
            }

            #if 0
            {
                cl::args intersect_args;
                intersect_args.push_back(potential_intersections);
                intersect_args.push_back(potential_intersection_count);
                intersect_args.push_back(accel_counts);
                intersect_args.push_back(accel_offsets);
                intersect_args.push_back(accel_generic_buffer);
                intersect_args.push_back(offset_width);
                intersect_args.push_back(offset_size.x());
                intersect_args.push_back(gpu_tris);
                intersect_args.push_back(rtex);

                clctx.cqueue.exec("render_potential_intersections", intersect_args, {1920 * 1080 * 10}, {256});
            }
            #endif // 0

            {
                cl::args intersect_args;
                intersect_args.push_back(gpu_intersections);
                intersect_args.push_back(gpu_intersections_count);
                intersect_args.push_back(rtex);

                clctx.cqueue.exec("render_intersections", intersect_args, {width * height}, {256});
            }

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

                clctx.cqueue.exec("render_tris", tri_args, {width, height}, {16, 16});
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
                    cl::args lorentz;
                    lorentz.push_back(g_camera_pos_polar);
                    lorentz.push_back(g_geodesic_basis_speed);

                    for(auto& i : tetrad)
                    {
                        lorentz.push_back(i);
                    }

                    lorentz.push_back(dynamic_config);

                    clctx.cqueue.exec("boost_tetrad", lorentz, {1}, {1});
                }

                {
                    generic_geodesic_count.set_to_zero(clctx.cqueue);

                    int count_in = 1;

                    cl::args args;
                    args.push_back(g_camera_pos_polar);
                    args.push_back(count_in);
                    args.push_back(generic_geodesic_buffer);
                    args.push_back(generic_geodesic_count);

                    for(auto& i : tetrad)
                    {
                        args.push_back(i);
                    }

                    args.push_back(g_geodesic_basis_speed);
                    args.push_back(dynamic_config);

                    clctx.cqueue.exec("init_inertial_ray", args, {count_in}, {1});
                }

                geodesic_trace_buffer.set_to_zero(clctx.cqueue);
                geodesic_dT_ds_buffer.set_to_zero(clctx.cqueue);
                geodesic_count_buffer.set_to_zero(clctx.cqueue);
                geodesic_vel_buffer.set_to_zero(clctx.cqueue);
                geodesic_ds_buffer.set_to_zero(clctx.cqueue);

                cl::args snapshot_args;
                snapshot_args.push_back(generic_geodesic_buffer);
                snapshot_args.push_back(geodesic_trace_buffer);
                snapshot_args.push_back(geodesic_vel_buffer);
                snapshot_args.push_back(geodesic_dT_ds_buffer);
                snapshot_args.push_back(geodesic_ds_buffer);
                snapshot_args.push_back(generic_geodesic_count);

                snapshot_args.push_back(max_trace_length);
                snapshot_args.push_back(dynamic_config);
                snapshot_args.push_back(geodesic_count_buffer);

                clctx.cqueue.exec("get_geodesic_path", snapshot_args, {1}, {1});

                for(int i=0; i < (int)tetrad.size(); i++)
                {
                    cl::copy(clctx.cqueue, tetrad[i], geodesic_tetrad[i]);
                }

                for(int i=0; i < (int)parallel_transported_tetrads.size(); i++)
                {
                    cl::args pt_args;
                    pt_args.push_back(geodesic_trace_buffer);
                    pt_args.push_back(geodesic_vel_buffer);
                    pt_args.push_back(geodesic_ds_buffer);
                    pt_args.push_back(geodesic_tetrad[i]);
                    pt_args.push_back(geodesic_count_buffer);
                    pt_args.push_back(parallel_transported_tetrads[i]);
                    pt_args.push_back(dynamic_config);

                    clctx.cqueue.exec("parallel_transport_quantity", pt_args, {1}, {1});
                }
            }

            rtex.unacquire(clctx.cqueue);

            if(taking_screenshot)
            {
                print("Taking screenie\n");

                clctx.cqueue.block();

                int high_width = current_settings.screenshot_width * current_settings.supersample_factor;
                int high_height = current_settings.screenshot_height * current_settings.supersample_factor;

                print("Blocked\n");

                std::cout << "WIDTH " << high_width << " HEIGHT "<< high_height << std::endl;

                std::vector<vec4f> pixels = tex.read(0);

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

                bool add_to_steam_library = current_settings.use_steam_screenshots;

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
            }
        }

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

            lst->AddImage((void*)rtex.texture_id, tl, br, ImVec2(0, 0), ImVec2(1, 1));
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
    }


    clctx.cqueue.block();

    return 0;
}
