#ifndef RAW_INPUT_HPP_INCLUDED
#define RAW_INPUT_HPP_INCLUDED

struct render_window;

struct raw_input
{
    bool is_enabled = false;

    void set_enabled(render_window& win, bool enabled);
};

#endif // RAW_INPUT_HPP_INCLUDED
