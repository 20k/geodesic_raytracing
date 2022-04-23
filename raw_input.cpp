#include "raw_input.hpp"
#include <toolkit/render_window.hpp>
#include <toolkit/render_window_glfw.hpp>
#include <GLFW/glfw3.h>

void raw_input::set_enabled(render_window& win, bool enabled)
{
    if(is_enabled == enabled)
        return;

    GLFWwindow* window = ((glfw_backend*)win.backend)->ctx.window;

    is_enabled = enabled;

    if(enabled)
    {
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        glfwSetInputMode(window, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);
    }
    else
    {
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        glfwSetInputMode(window, GLFW_RAW_MOUSE_MOTION, GLFW_FALSE);
    }
}
