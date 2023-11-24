#ifndef FULLSCREEN_WINDOW_MANAGER_HPP_INCLUDED
#define FULLSCREEN_WINDOW_MANAGER_HPP_INCLUDED

#include <vec/vec.hpp>

struct render_window;

struct fullscreen_window_manager
{
    bool title_dragging = false;
    bool resize_dragging = false;
    vec2f title_start_pos_absolute;
    vec2f title_start_mouse_pos_absolute;
    vec2f resize_start_pos;
    vec2f resize_start_mouse_pos_absolute;
    bool open = true;
    std::string title;

    fullscreen_window_manager(const std::string& _title);

    void start(render_window& win);
    void stop();
};

#endif // FULLSCREEN_WINDOW_MANAGER_HPP_INCLUDED
