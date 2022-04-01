#include "input_manager.hpp"
#include <toolkit/render_window.hpp>
#include <GLFW/glfw3.h>
#include <imgui/imgui.h>
#include <stdexcept>
#include <nlohmann/json.hpp>
#include <toolkit/fs_helpers.hpp>
#include <iostream>

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

    if(file::exists("input.json"))
    {
        try
        {
            nlohmann::json js = nlohmann::json::parse(file::read("input.json", file::mode::BINARY));

            std::map<std::string, int, std::less<>> extras = js.get<std::map<std::string, int, std::less<>>>();

            for(auto& [k, v] : extras)
            {
                rebind(k, v);
            }
        }
        catch(std::exception& ex)
        {
            std::cout << "Failed to load input.json " << ex.what() << std::endl;
        }
    }
}

void input_manager::display_key_rebindings(render_window& win)
{
    std::vector<std::pair<std::string, int>> to_rebind;

    int max_length = 0;

    for(auto& [purpose, key] : glfw_key_map)
    {
        max_length = std::max(max_length, (int)purpose.size());
    }

    for(auto& [purpose, _] : linear_keys)
    {
        int key = glfw_key_map[purpose];

        std::string name = win.backend->get_key_name(key);
        std::string dummy_buf = "0";

        std::string c_key_name;

        if(name.size() == 0)
        {
            c_key_name = std::to_string(key);
        }
        else
        {
            c_key_name = name;
        }

        std::string resized_purpose = purpose;

        for(int i=resized_purpose.size(); i < max_length; i++)
        {
            resized_purpose.push_back(' ');
        }

        ImGui::Text(resized_purpose.c_str());

        ImGui::SameLine();

        ImGui::InputText(("##purpose" + purpose).c_str(), &c_key_name[0], c_key_name.size(), ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_ReadOnly);

        if(ImGui::IsItemActive())
        {
            int key_count = sizeof(ImGui::GetIO().KeysDown) / sizeof(ImGui::GetIO().KeysDown[0]);

            int which_key = -1;

            for(int i=0; i < key_count; i++)
            {
                if(ImGui::IsKeyDown(i))
                {
                    which_key = i;
                    break;
                }
            }

            if(which_key != -1)
            {
                to_rebind.push_back({purpose, which_key});
            }
        }
    }

    for(const auto& [k, v] : to_rebind)
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
