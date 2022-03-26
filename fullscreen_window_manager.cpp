#include "fullscreen_window_manager.hpp"
#include <imgui/imgui.h>
#include <imgui/imgui_internal.h>
#include <toolkit/render_window.hpp>

namespace
{
    bool any_scrollbar_active()
    {
        ImGuiWindow* window = ImGui::GetCurrentWindow();
        ImGuiID active_id = ImGui::GetActiveID();
        return active_id && (active_id == ImGui::GetWindowScrollbarID(window, ImGuiAxis_X) || active_id == ImGui::GetWindowScrollbarID(window, ImGuiAxis_Y));
    }
}

fullscreen_window_manager::fullscreen_window_manager()
{
    ImGuiIO& io = ImGui::GetIO();

    io.MouseDragThreshold = 0;
    io.ConfigWindowsMoveFromTitleBarOnly = true;
}

void fullscreen_window_manager::start(render_window& win)
{
    int flags = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoResize;

    if(win.backend->is_maximised())
    {
        flags |= ImGuiWindowFlags_NoTitleBar;
    }

    vec2i real_window_size = win.backend->get_window_size();

    ImVec2 viewport_pos = ImGui::GetMainViewport()->Pos;

    ImGui::SetNextWindowSize(ImVec2(real_window_size.x(), real_window_size.y()));
    ImGui::SetNextWindowViewport(ImGui::GetMainViewport()->ID);
    ImGui::SetNextWindowPos(viewport_pos);

    win.backend->set_window_position({viewport_pos.x, viewport_pos.y});

    ImVec4 style_col = ImGui::GetStyleColorVec4(ImGuiCol_TitleBgActive);

    ImGui::PushStyleColor(ImGuiCol_TitleBg, style_col);
    ImGui::PushStyleColor(ImGuiCol_TitleBgCollapsed, style_col);

    ImVec4 resize_col = ImGui::GetStyleColorVec4(ImGuiCol_ResizeGrip);
    ImU32 resize_colu32 = ImGui::ColorConvertFloat4ToU32(resize_col);

    bool rendering = ImGui::Begin("Main Window", &open, flags);

    if(ImGui::IsItemHovered() &&
       ImGui::IsMouseDragging(0) && !title_dragging && !resize_dragging)
    {
        if(!title_dragging)
        {
            title_dragging = true;
            title_start_pos = xy_to_vec(ImGui::GetMainViewport()->Pos);
        }
    }

    #ifndef __EMSCRIPTEN__
    if(ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(0))
    {
        win.backend->set_is_maximised(!win.backend->is_maximised());
    }
    #endif // __EMSCRIPTEN__

    if(title_dragging)
    {
        ImVec2 delta = ImGui::GetMouseDragDelta();

        ImVec2 real_pos;
        real_pos.x = delta.x + title_start_pos.x();
        real_pos.y = delta.y + title_start_pos.y();

        win.backend->set_window_position({real_pos.x, real_pos.y});
    }

    if(!win.backend->is_maximised())
    {
        vec2f label_br = (vec2f){viewport_pos.x, viewport_pos.y} + (vec2f){real_window_size.x(), real_window_size.y()};
        vec2f label_tl = label_br - (vec2f){30, 30};

        bool hovering_label = ImGui::IsMouseHoveringRect({label_tl.x(), label_tl.y()}, {label_br.x(), label_br.y()}, true);

        if(hovering_label || resize_dragging)
            resize_colu32 = ImGui::ColorConvertFloat4ToU32(ImGui::GetStyleColorVec4(ImGuiCol_ResizeGripActive));

        if(hovering_label && !any_scrollbar_active())
        {
            if(ImGui::IsMouseDragging(0) && !title_dragging && !resize_dragging)
            {
                if(!resize_dragging)
                {
                    resize_dragging = true;
                    resize_start_pos = {real_window_size.x(), real_window_size.y()};
                }
            }
        }

        if(resize_dragging)
        {
            ImVec2 delta = ImGui::GetMouseDragDelta();

            int width = delta.x + resize_start_pos.x();
            int height = delta.y + resize_start_pos.y();

            if(width >= 50 && height >= 50)
                win.backend->resize({width, height});
        }

        if(!ImGui::IsMouseDown(0))
        {
            title_dragging = false;
            resize_dragging = false;
        }

        ImGui::GetWindowDrawList()->AddTriangleFilled({label_tl.x(), label_br.y()}, {label_br.x(), label_br.y()}, {label_br.x(), label_tl.y()}, resize_colu32);
    }
}

void fullscreen_window_manager::stop()
{
    ImGui::End();
    ImGui::PopStyleColor(2);
}
