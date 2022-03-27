#ifndef FULLSCREEN_WINDOW_MANAGER_HPP_INCLUDED
#define FULLSCREEN_WINDOW_MANAGER_HPP_INCLUDED

#include <vec/vec.hpp>

struct render_window;

struct fullscreen_window_manager
{
    bool title_dragging = false;
    bool resize_dragging = false;
    vec2f title_start_pos;
    vec2f resize_start_pos;
    bool open = true;

    fullscreen_window_manager();

    void start(render_window& win);
    void stop();
};

#endif // FULLSCREEN_WINDOW_MANAGER_HPP_INCLUDED
