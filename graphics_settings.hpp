#ifndef GRAPHICS_SETTINGS_HPP_INCLUDED
#define GRAPHICS_SETTINGS_HPP_INCLUDED

#include <networking/serialisable_fwd.hpp>
#include <nlohmann/json.hpp>
#include <toolkit/opencl.hpp>

struct graphics_settings : serialisable, free_function
{
    int width = 1920;
    int height = 1080;

    int screenshot_width = 1920;
    int screenshot_height = 1080;

    bool fullscreen = true;

    int supersample_factor = 2;
    bool supersample = false;

    bool vsync_enabled = false;
    bool time_adjusted_controls = true;

    float mouse_sensitivity = 1;
    float keyboard_sensitivity = 1;

    bool use_steam_screenshots = true;

    int anisotropy = 8;

    bool no_gpu_reads = false;

    ///Returns true if we need to refresh our opencl context
    bool display_video_settings();
    bool display_control_settings();
};

DECLARE_SERIALISE_FUNCTION(graphics_settings);

struct background_images
{
    cl::context& ctx;
    cl::command_queue& cqueue;

    cl::image i1;
    cl::image i2;

    background_images(cl::context& _ctx, cl::command_queue& _cqueue) : ctx(_ctx), cqueue(_cqueue), i1(ctx), i2(ctx){}

    void load(const std::string& n1, const std::string& n2);
};

struct background_settings : serialisable, free_function
{
    std::string path1;
    std::string path2;

    void display(background_images& bi);
};

DECLARE_SERIALISE_FUNCTION(background_settings);

#endif // GRAPHICS_SETTINGS_HPP_INCLUDED
