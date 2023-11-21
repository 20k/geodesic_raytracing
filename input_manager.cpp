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
        {"forward", {GLFW_KEY_W}},
        {"back", {GLFW_KEY_S}},
        {"right", {GLFW_KEY_D}},
        {"left", {GLFW_KEY_A}},
        {"up", {GLFW_KEY_Q}},
        {"down", {GLFW_KEY_E}},
        {"time_forwards", {GLFW_KEY_R}},
        {"time_backwards", {GLFW_KEY_F}},
        {"toggle_mouse", {GLFW_KEY_TAB}},
        {"hide_ui", {GLFW_KEY_F1}},
        {"speed_10x", {GLFW_KEY_LEFT_SHIFT}},
        {"speed_100x", {GLFW_KEY_X}},
        {"speed_100th", {GLFW_KEY_LEFT_ALT}},
        {"speed_superslow", {GLFW_KEY_LEFT_CONTROL}},
        {"camera_centre", {GLFW_KEY_C}},
        {"camera_reset", {GLFW_KEY_N}},
        {"camera_turn_right", {GLFW_KEY_RIGHT}},
        {"camera_turn_left", {GLFW_KEY_LEFT}},
        {"camera_turn_down", {GLFW_KEY_UP}},
        {"camera_turn_up", {GLFW_KEY_DOWN}},
        {"toggle_wormhole_space", {GLFW_KEY_1}},
        {"toggle_geodesic_play", {GLFW_KEY_F2}},
        {"play_speed_minus", {GLFW_KEY_MINUS}},
        {"play_speed_plus", {GLFW_KEY_EQUAL}},
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

            std::map<std::string, std::vector<int>, std::less<>> extras = js.get<std::map<std::string, std::vector<int>, std::less<>>>();

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
    std::vector<std::pair<std::string, std::vector<int>>> to_rebind;

    int max_length = 0;

    for(auto& [purpose, key] : glfw_key_map)
    {
        max_length = std::max(max_length, (int)purpose.size());
    }

    auto name_keys = [&](const std::vector<int>& keys)
    {
        std::string name;

        for(auto& i : keys)
        {
            std::string sname = win.backend->get_key_name(i);

            if(sname.size() == 0)
                sname = std::to_string(i);

            name += sname + "+";
        }

        if(name.ends_with("+"))
            name.pop_back();

        return name;
    };

    for(auto& [purpose, default_key] : linear_keys)
    {
        std::vector<int> keys = glfw_key_map[purpose];

        std::string name = name_keys(keys);

        if(last_rebind_purpose.has_value() && last_rebind_purpose.value() == purpose)
            name = name_keys(rebinding_keys);

        std::string resized_purpose = purpose;

        for(int i=resized_purpose.size(); i < max_length; i++)
        {
            resized_purpose.push_back(' ');
        }

        ImGui::Text(resized_purpose.c_str());

        ImGui::SameLine();

        ImGui::InputText(("##purpose" + purpose).c_str(), &name[0], name.size(), ImGuiInputTextFlags_ReadOnly);

        if(ImGui::IsItemActive())
        {
            last_rebind_purpose = purpose;

            int key_count = sizeof(ImGui::GetIO().KeysDown) / sizeof(ImGui::GetIO().KeysDown[0]);

            for(int i=0; i < key_count; i++)
            {
                if(i == GLFW_KEY_ENTER)
                    continue;

                if(ImGui::IsKeyPressed(i, false))
                {
                    rebinding_keys.push_back(i);
                }
            }
        }

        ImGui::SameLine();

        if(ImGui::Button(("Reset##" + purpose).c_str()))
        {
            to_rebind.push_back({purpose, default_key});

            last_rebind_purpose = std::nullopt;
            rebinding_keys.clear();
        }
    }

    if(last_rebind_purpose.has_value())
    {
        ImGui::SetTooltip("Hold key combo, then hit Enter to save");

        if(ImGui::IsKeyDown(GLFW_KEY_ENTER))
        {
            to_rebind.push_back({last_rebind_purpose.value(), rebinding_keys});

            last_rebind_purpose = std::nullopt;
            rebinding_keys.clear();
        }
    }

    for(const auto& [k, v] : to_rebind)
    {
        rebind(k, v);
    }

    if(to_rebind.size() > 0)
    {
        nlohmann::json js = glfw_key_map;

        file::write_atomic("input.json", js.dump(), file::mode::BINARY);
    }
}

bool input_manager::is_key_down(std::string_view view)
{
    auto it = glfw_key_map.find(view);

    if(it == glfw_key_map.end())
        throw std::runtime_error("No such key " + std::string(view));

    for(auto& i : it->second)
    {
        if(!ImGui::IsKeyDown(i))
            return false;
    }

    return true;
}

bool input_manager::is_key_pressed(std::string_view view)
{
    auto it = glfw_key_map.find(view);

    if(it == glfw_key_map.end())
        throw std::runtime_error("No such key " + std::string(view));

    int pressed_count = 0;

    for(auto& i : it->second)
    {
        bool any = false;

        if(ImGui::IsKeyPressed(i))
        {
            pressed_count++;
            any = true;
        }

        any = any || ImGui::IsKeyDown(i);

        if(!any)
            return false;
    }

    return pressed_count > 0;
}

void input_manager::rebind(std::string_view view, const std::vector<int>& key)
{
    glfw_key_map[std::string(view)] = key;
}
