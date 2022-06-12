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
    cl_uint sx, sy;
    cl_float ku_uobsu;
    cl_float original_theta;
    cl_int early_terminate;
};

#define GENERIC_METRIC

vec4f interpolate_geodesic(const std::vector<cl_float4>& geodesic, const std::vector<cl_float>& geodesic_dT_dt, float coordinate_time)
{
    assert(geodesic.size() == geodesic_dT_dt.size());

    float current_proper_time = 0;

    for(int i=0; i < (int)geodesic.size() - 1; i++)
    {
        vec4f cur = {geodesic[i].s[0], geodesic[i].s[1], geodesic[i].s[2], geodesic[i].s[3]};
        vec4f next = {geodesic[i + 1].s[0], geodesic[i + 1].s[1], geodesic[i + 1].s[2], geodesic[i + 1].s[3]};

        if(next.x() < cur.x())
            std::swap(next, cur);

        float dt = next.x() - cur.x();

        float next_proper_time = current_proper_time + geodesic_dT_dt[i] * dt;

        #ifdef COORDINATE_TIME_GEODESICS
        if(coordinate_time >= cur.x() && coordinate_time < next.x())
        #else
        if(coordinate_time >= current_proper_time && coordinate_time < next_proper_time)
        #endif
        {
            vec3f as_cart1 = polar_to_cartesian<float>({fabs(cur.y()), cur.z(), cur.w()});
            vec3f as_cart2 = polar_to_cartesian<float>({fabs(next.y()), next.z(), next.w()});

            float r1 = cur.y();
            float r2 = next.y();

            ///this might be why things bug out, the division here could easily be singular
            #ifdef COORDINATE_TIME_GEODESICS
            float dx = (coordinate_time - cur.x()) / (next.x() - cur.x());
            #else
            float dx = (coordinate_time - current_proper_time) / (next_proper_time - current_proper_time);
            #endif

            vec3f next_cart = cartesian_to_polar(mix(as_cart1, as_cart2, dx));
            float next_r = mix(r1, r2, dx);

            next_cart.x() = next_r;

            float resulting_coordinate_time = mix(cur.x(), next.x(), dx);

            return {resulting_coordinate_time, next_cart.x(), next_cart.y(), next_cart.z()};
        }

        current_proper_time = next_proper_time;
    }

    if(geodesic.size() == 0)
        return {0,0,0,0};

    cl_float4 selected_geodesic = {0,0,0,0};

    #ifdef COORDINATE_TIME_GEODESICS
    if(coordinate_time >= geodesic.back().s[0])
        selected_geodesic = geodesic.back();
    else
        selected_geodesic = geodesic.front();
    #else
    if(coordinate_time >= current_proper_time)
        selected_geodesic = geodesic.back();
    else
        selected_geodesic = geodesic.front();
    #endif

    return {selected_geodesic.s[0], selected_geodesic.s[1], selected_geodesic.s[2], selected_geodesic.s[3]};
}

vec2f get_geodesic_intersection(const metrics::metric& met, const std::vector<cl_float4>& geodesic, const std::vector<cl_float>& geodesic_dT_dt)
{
    for(int i=0; i < (int)geodesic.size() - 2; i++)
    {
        vec4f cur = {geodesic[i].s[0], geodesic[i].s[1], geodesic[i].s[2], geodesic[i].s[3]};
        vec4f next = {geodesic[i + 1].s[0], geodesic[i + 1].s[1], geodesic[i + 1].s[2], geodesic[i + 1].s[3]};

        if(signum(geodesic[i].s[1]) != signum(geodesic[i + 1].s[1]))
        {
            float total_r = fabs(geodesic[i].s[1]) + fabs(geodesic[i + 1].s[1]);

            float dx = fabs(geodesic[i].s[1]) / total_r;

            vec3f as_cart1 = polar_to_cartesian<float>({fabs(cur.y()), cur.z(), cur.w()});
            vec3f as_cart2 = polar_to_cartesian<float>({fabs(next.y()), next.z(), next.w()});

            vec3f next_cart = cartesian_to_polar(mix(as_cart1, as_cart2, dx));

            return {next_cart.y(), next_cart.z()};
        }
    }

    return {M_PI/2, 0};
}

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

        cl::args run_args;
        run_args.push_back(rays_in);
        run_args.push_back(rays_out);
        run_args.push_back(rays_finished);
        run_args.push_back(count_in);
        run_args.push_back(count_out);
        run_args.push_back(count_finished);
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

///i need the ability to have dynamic parameters
int main()
{
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
    sett.no_double_buffer = true;
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

    std::cout << "extensions " << cl::get_extensions(win.clctx->ctx) << std::endl;

    ImFontAtlas* atlas = ImGui::GetIO().Fonts;
    atlas->FontBuilderFlags = ImGuiFreeTypeBuilderFlags_LCD | ImGuiFreeTypeBuilderFlags_FILTER_DEFAULT | ImGuiFreeTypeBuilderFlags_LoadColor;

    ImFontConfig font_cfg;
    font_cfg.GlyphExtraSpacing = ImVec2(0, 0);
    font_cfg.FontBuilderFlags = ImGuiFreeTypeBuilderFlags_LCD | ImGuiFreeTypeBuilderFlags_FILTER_DEFAULT | ImGuiFreeTypeBuilderFlags_LoadColor;

    ImGuiIO& io = ImGui::GetIO();

    io.Fonts->Clear();
    io.Fonts->AddFontFromFileTTF("VeraMono.ttf", 14, &font_cfg);

    opencl_context& clctx = *win.clctx;

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

    //printf("WLs %f %f %f\n", chromaticity::srgb_to_wavelength({1, 0, 0}), chromaticity::srgb_to_wavelength({0, 1, 0}), chromaticity::srgb_to_wavelength({0, 0, 1}));

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
    printf("Pre dqueue\n");

    cl::device_command_queue dqueue(clctx.ctx);

    printf("Post dqueue\n");
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

    printf("Pre buffer declarations\n");

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

    printf("Post buffer declarations\n");

    termination_buffer.alloc(start_width * start_height * sizeof(cl_int));

    printf("Allocated termination buffer\n");

    termination_buffer.set_to_zero(clctx.cqueue);

    printf("Zero termination buffer\n");

    cl::buffer geodesic_count_buffer(clctx.ctx);
    geodesic_count_buffer.alloc(sizeof(cl_int));

    cl::buffer geodesic_trace_buffer(clctx.ctx);
    geodesic_trace_buffer.alloc(64000 * sizeof(cl_float4));

    cl::buffer geodesic_dT_dt_buffer(clctx.ctx);
    geodesic_dT_dt_buffer.alloc(64000 * sizeof(cl_float));

    printf("Alloc trace buffer\n");

    std::vector<cl_float4> current_geodesic_path;
    std::vector<cl_float> current_geodesic_dT_dt;

    printf("Pre texture coordinates\n");

    cl::buffer texture_coordinates{clctx.ctx};

    texture_coordinates.alloc(start_width * start_height * sizeof(float) * 2);
    texture_coordinates.set_to_zero(clctx.cqueue);

    printf("Post texture coordinates\n");

    schwarzs_1.alloc(sizeof(lightray) * ray_count);
    schwarzs_scratch.alloc(sizeof(lightray) * ray_count);
    schwarzs_prepass.alloc(sizeof(lightray) * ray_count);
    finished_1.alloc(sizeof(lightray) * ray_count);

    schwarzs_count_1.alloc(sizeof(int));
    schwarzs_count_scratch.alloc(sizeof(int));
    schwarzs_count_prepass.alloc(sizeof(int));
    finished_count_1.alloc(sizeof(int));

    printf("Alloc rays and counts\n");

    std::optional<cl::event> last_event;

    std::cout << "Supports shared events? " << cl::supports_extension(clctx.ctx, "cl_khr_gl_event") << std::endl;

    bool last_supersample = false;
    bool should_take_screenshot = false;

    bool time_progresses = false;
    bool flip_sign = false;
    float current_geodesic_time = 0;
    bool camera_on_geodesic = false;
    bool camera_time_progresses = false;
    float camera_geodesic_time_progression_speed = 1.f;
    bool camera_geodesics_go_foward = true;
    vec2f base_angle = {M_PI/2, 0};
    float set_camera_time = 0;

    printf("Pre main\n");

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

    printf("Prog1\n");

    cl::program util(clctx.ctx, "util.cl");
    util.build(clctx.ctx, "-cl-std=CL1.2 -cl-fast-relaxed-math ");

    cl::kernel handle_controls(util, "handle_controls");
    cl::kernel camera_cart_to_polar(util, "camera_cart_to_polar");
    cl::kernel advance_time(util, "advance_time");
    cl::kernel set_time(util, "set_time");

    printf("Prog2\n");

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
                if(last_event.has_value())
                    last_event.value().block();

                last_event = std::nullopt;

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

                last_supersample = current_settings.supersample;
                last_supersample_mult = current_settings.supersample_factor;
            }

            rtex.acquire(clctx.cqueue);

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
                    set_camera_time = 0;
                    g_camera_pos_cart.write(clctx.cqueue, std::vector<cl_float4>{{0, 0, 0, -4}});
                }

                if(input.is_key_down("camera_centre"))
                {
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

                handle_controls.set_args(controls_args);

                clctx.cqueue.exec(handle_controls, {1}, {1});

                /*if(delta.x() != 0.f)
                {
                    quat q;
                    q.load_from_axis_angle({0, 0, -1, current_settings.mouse_sensitivity * delta.x() * M_PI/128});

                    camera_quat = q * camera_quat;
                }

                if(input.is_key_pressed("toggle_wormhole_space"))
                {
                    flip_sign = !flip_sign;
                }

                vec3f up = {0, 0, -1};
                vec3f right = rot_quat({1, 0, 0}, camera_quat);
                vec3f forward_axis = rot_quat({0, 0, 1}, camera_quat);

                if(delta.y() != 0.f)
                {
                    quat q;
                    q.load_from_axis_angle({right.x(), right.y(), right.z(), current_settings.mouse_sensitivity * delta.y() * M_PI/128});

                    camera_quat = q * camera_quat;
                }

                vec3f offset = {0,0,0};

                offset += current_settings.keyboard_sensitivity * controls_multiplier * forward_axis * (() * speed);
                offset += current_settings.keyboard_sensitivity * controls_multiplier * right * () * speed;
                offset += current_settings.keyboard_sensitivity * controls_multiplier * up * () * speed;

                camera.y() += offset.x();
                camera.z() += offset.y();
                camera.w() += offset.z();*/
            }

            /*#define CLAMP_CAMERA
            #ifdef CLAMP_CAMERA
            {
                float rad = camera.length();

                if(rad > current_cfg.universe_size * 0.99f)
                {
                    camera = camera.norm() * current_cfg.universe_size * 0.99f;
                }
            }
            #endif // CLAMP_CAMERA*/

            /*vec4f scamera = cartesian_to_schwarz(camera);

            if(flip_sign)
                scamera.y() = -scamera.y();

            if(camera_on_geodesic)
            {
                scamera = interpolate_geodesic(current_geodesic_path, current_geodesic_dT_dt, current_geodesic_time);

                if(metric_manage.current_metric)
                {
                    base_angle = get_geodesic_intersection(*metric_manage.current_metric, current_geodesic_path, current_geodesic_dT_dt);
                }
            }
            else
            {
                base_angle = {M_PI/2, 0.f};
            }*/

            float time = clk.restart().asMicroseconds() / 1000.;

            {
                cl::args args;

                args.push_back(g_camera_pos_polar);
                args.push_back(g_camera_pos_cart);

                cl_float clflip = flip_sign;

                args.push_back(clflip);

                camera_cart_to_polar.set_args(args);

                clctx.cqueue.exec(camera_cart_to_polar, {1}, {1});
            }

            bool should_soft_recompile = false;

            if(!taking_screenshot && !hide_ui && !menu.is_first_time_main_menu_open())
            {
                ImGui::Begin("Settings and Information", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

                if(ImGui::BeginTabBar("Tabbity tab tabs"))
                {
                    if(ImGui::BeginTabItem("General"))
                    {
                        //ImGui::DragFloat3("Polar Pos", &scamera.v[1]);
                        //ImGui::DragFloat3("Cart Pos", &camera.v[1]);
                        if(ImGui::SliderFloat("Camera Time", &set_camera_time, 0.f, 100.f))
                        {
                            cl::args args;
                            args.push_back(g_camera_pos_cart);
                            args.push_back(set_camera_time);

                            clctx.cqueue.exec("set_time", args, {1}, {1});
                        }

                        ImGui::DragFloat("Frametime", &time);

                        ImGui::Checkbox("Time Progresses", &time_progresses);

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
                        ImGui::DragFloat("Geodesic Camera Time", &current_geodesic_time, 0.1, 0.f, 0.f);

                        ImGui::Checkbox("Use Camera Geodesic", &camera_on_geodesic);

                        ImGui::Checkbox("Camera Time Progresses Along Geodesic", &camera_time_progresses);

                        ImGui::SliderFloat("Camera Time Progression Speed", &camera_geodesic_time_progression_speed, 0.f, 4.f, "%.2f");

                        if(ImGui::Button("Snapshot Camera Geodesic"))
                        {
                            should_snapshot_geodesic = true;
                        }

                        ImGui::Checkbox("Camera Snapshot Geodesic goes forward", &camera_geodesics_go_foward);

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

            if(time_progresses)
            {
                set_camera_time += time / 1000.f;

                cl::args args;
                args.push_back(g_camera_pos_cart);
                args.push_back(set_camera_time);

                set_time.set_args(args);

                clctx.cqueue.exec(set_time, {1}, {1});
            }

            if(camera_time_progresses)
                current_geodesic_time += camera_geodesic_time_progression_speed * time / 1000.f;

            metric_manage.check_recompile(should_recompile, should_soft_recompile, parent_directories,
                                          all_content, metric_names, dynamic_config, clctx.cqueue, cfg,
                                          sett, clctx.ctx, termination_buffer);

            metric_manage.check_substitution(clctx.ctx);

            int width = rtex.size<2>().x();
            int height = rtex.size<2>().y();

            cl::args clr;
            clr.push_back(rtex);

            clctx.cqueue.exec("clear", clr, {width, height}, {16, 16});

            cl::event next;

            {
                int isnap = should_snapshot_geodesic;

                if(should_snapshot_geodesic)
                {
                    if(camera_geodesics_go_foward)
                    {
                        isnap = 1;
                    }
                    else
                    {
                        isnap = 0;
                    }
                }

                cl_int prepass_width = width/16;
                cl_int prepass_height = height/16;

                if(metric_manage.current_metric->metric_cfg.use_prepass)
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
                    init_args_prepass.push_back(base_angle);
                    init_args_prepass.push_back(dynamic_config);

                    clctx.cqueue.exec("init_rays_generic", init_args_prepass, {prepass_width*prepass_height}, {256});

                    int rays_num = calculate_ray_count(prepass_width, prepass_height);

                    execute_kernel(clctx.cqueue, schwarzs_prepass, schwarzs_scratch, finished_1, schwarzs_count_prepass, schwarzs_count_scratch, finished_count_1, rays_num, cfg.use_device_side_enqueue, dynamic_config);

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
                init_args.push_back(base_angle);
                init_args.push_back(dynamic_config);

                clctx.cqueue.exec("init_rays_generic", init_args, {width*height}, {16*16});

                /*if(should_snapshot_geodesic)
                {
                    int idx = (height/2) * width + width/2;

                    geodesic_trace_buffer.set_to_zero(clctx.cqueue);
                    geodesic_dT_dt_buffer.set_to_zero(clctx.cqueue);
                    geodesic_count_buffer.set_to_zero(clctx.cqueue);

                    cl::args snapshot_args;
                    snapshot_args.push_back(schwarzs_1);
                    snapshot_args.push_back(geodesic_trace_buffer);
                    snapshot_args.push_back(geodesic_dT_dt_buffer);
                    snapshot_args.push_back(schwarzs_count_1);
                    snapshot_args.push_back(idx);
                    snapshot_args.push_back(width);
                    snapshot_args.push_back(height);
                    snapshot_args.push_back(scamera);
                    snapshot_args.push_back(camera_quat);
                    snapshot_args.push_back(base_angle);
                    snapshot_args.push_back(dynamic_config);
                    snapshot_args.push_back(geodesic_count_buffer);

                    clctx.cqueue.exec("get_geodesic_path", snapshot_args, {1}, {1});

                    current_geodesic_path = geodesic_trace_buffer.read<cl_float4>(clctx.cqueue);
                    current_geodesic_dT_dt = geodesic_dT_dt_buffer.read<cl_float>(clctx.cqueue);
                    int count = geodesic_count_buffer.read<cl_int>(clctx.cqueue)[0];

                    printf("Found geodesic count %i\n", count);

                    current_geodesic_path.resize(count);
                    current_geodesic_dT_dt.resize(count);
                }*/

                int rays_num = calculate_ray_count(width, height);

                execute_kernel(clctx.cqueue, schwarzs_1, schwarzs_scratch, finished_1, schwarzs_count_1, schwarzs_count_scratch, finished_count_1, rays_num, cfg.use_device_side_enqueue, dynamic_config);


                cl::args texture_args;
                texture_args.push_back(finished_1);
                texture_args.push_back(finished_count_1);
                texture_args.push_back(texture_coordinates);
                texture_args.push_back(width);
                texture_args.push_back(height);
                texture_args.push_back(g_camera_pos_polar);
                texture_args.push_back(g_camera_quat);
                texture_args.push_back(base_angle);

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

                next = clctx.cqueue.exec("render", render_args, {width * height}, {256});
            }

            rtex.unacquire(clctx.cqueue);

            if(taking_screenshot)
            {
                printf("Taking screenie\n");

                clctx.cqueue.block();

                int high_width = current_settings.screenshot_width * current_settings.supersample_factor;
                int high_height = current_settings.screenshot_height * current_settings.supersample_factor;

                printf("Blocked\n");

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

            if(last_event.has_value())
                last_event.value().block();

            last_event = next;
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
    }

    last_event = std::nullopt;

    clctx.cqueue.block();

    return 0;
}
