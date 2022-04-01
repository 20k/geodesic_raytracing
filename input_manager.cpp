#include "input_manager.hpp"
#include <GLFW/glfw3.h>
#include <imgui/imgui.h>
#include <stdexcept>

input_manager::input_manager()
{
    linear_keys =
    {
        {"forward", GLFW_KEY_W},
        {"back", GLFW_KEY_S},
        {"right", GLFW_KEY_D},
        {"left", GLFW_KEY_A},
        {"up", GLFW_KEY_Q},
        {"down", GLFW_KEY_E},
        {"toggle_mouse", GLFW_KEY_TAB},
        {"hide_ui", GLFW_KEY_F1},
        {"speed_10x", GLFW_KEY_LEFT_SHIFT},
        {"speed_100x", GLFW_KEY_X},
        {"speed_100th", GLFW_KEY_LEFT_ALT},
        {"speed_superslow", GLFW_KEY_LEFT_CONTROL},
        {"camera_centre", GLFW_KEY_C},
        {"camera_reset", GLFW_KEY_N},
        {"camera_turn_right", GLFW_KEY_RIGHT},
        {"camera_turn_left", GLFW_KEY_LEFT},
        {"camera_turn_down", GLFW_KEY_UP},
        {"camera_turn_up", GLFW_KEY_DOWN},
        {"toggle_wormhole_space", GLFW_KEY_1},
    };

    for(auto [k, v] : linear_keys)
    {
        rebind(k, v);
    }
}

bool input_manager::is_key_down(std::string_view view)
{
    auto it = glfw_key_map.find(view);

    if(it == glfw_key_map.end())
        throw std::runtime_error("No such key " + std::string(view));

    return ImGui::IsKeyDown(it->second);
}

bool input_manager::is_key_pressed(std::string_view view)
{
    auto it = glfw_key_map.find(view);

    if(it == glfw_key_map.end())
        throw std::runtime_error("No such key " + std::string(view));

    return ImGui::IsKeyPressed(it->second);
}

void input_manager::rebind(std::string_view view, int key)
{
    glfw_key_map[std::string(view)] = key;
}
