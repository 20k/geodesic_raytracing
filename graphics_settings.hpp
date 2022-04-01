#ifndef GRAPHICS_SETTINGS_HPP_INCLUDED
#define GRAPHICS_SETTINGS_HPP_INCLUDED

#include <networking/serialisable_fwd.hpp>
#include <nlohmann/json.hpp>

struct graphics_settings : serialisable, free_function
{
    int width = 1920;
    int height = 1080;

    int screenshot_width = 1920;
    int screenshot_height = 1080;

    bool fullscreen = false;

    int supersample_factor = 2;
    bool supersample = false;

    bool vsync_enabled = false;
    bool time_adjusted_controls = true;

    float mouse_sensitivity = 1;
    float keyboard_sensitivity = 1;

    ///Returns true if we need to refresh our opencl context
    bool display();
};

DECLARE_SERIALISE_FUNCTION(graphics_settings);

#endif // GRAPHICS_SETTINGS_HPP_INCLUDED
