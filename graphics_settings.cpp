#include "graphics_settings.hpp"
#include <imgui/imgui.h>
#include <networking/serialisable.hpp>

DEFINE_SERIALISE_FUNCTION(graphics_settings)
{
    DO_FSERIALISE(width);
    DO_FSERIALISE(height);
    DO_FSERIALISE(fullscreen);
    DO_FSERIALISE(supersample);
    DO_FSERIALISE(supersample_factor);
    DO_FSERIALISE(vsync_enabled);
    DO_FSERIALISE(screenshot_width);
    DO_FSERIALISE(screenshot_height);
    DO_FSERIALISE(time_adjusted_controls);
    DO_FSERIALISE(mouse_sensitivity);
    DO_FSERIALISE(keyboard_sensitivity);
    DO_FSERIALISE(use_steam_screenshots);
    DO_FSERIALISE(anisotropy);
}

bool graphics_settings::display_video_settings()
{
    ImGui::InputInt("Width", &width);
    ImGui::InputInt("Height", &height);

    ImGui::Checkbox("Fullscreen", &fullscreen);

    ImGui::Checkbox("Vsync", &vsync_enabled);

    ImGui::Checkbox("Supersample", &supersample);
    ImGui::InputInt("Supersample Factor", &supersample_factor);

    ImGui::InputInt("Screenshot Width", &screenshot_width);
    ImGui::InputInt("Screenshot Height", &screenshot_height);

    ImGui::Checkbox("Save screenshots to steam", &use_steam_screenshots);

    ImGui::DragInt("Anisotropic Filtering", &anisotropy, 1, 1, 256);

    ImGui::NewLine();

    return ImGui::Button("Apply");
}

bool graphics_settings::display_control_settings()
{
    ImGui::Checkbox("Time adjusted controls", &time_adjusted_controls);

    if(ImGui::IsItemHovered())
    {
        ImGui::SetTooltip("Setting this to true means that camera moves at a constant amount per second\nSetting this to false means that the camera moves at a constant speed per frame");
    }

    ImGui::SliderFloat("Mouse Sensitivity", &mouse_sensitivity, 0.f, 5.f);
    ImGui::SliderFloat("Keyboard Sensitivity", &keyboard_sensitivity, 0.f, 5.f);

    ImGui::NewLine();

    return ImGui::Button("Apply");
}
