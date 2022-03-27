#include "fullscreen_window_manager.hpp"
#include <imgui/imgui.h>
#include <imgui/imgui_internal.h>
#include <toolkit/render_window.hpp>
#include <toolkit/render_window_glfw.hpp>
#include <GLFW/glfw3.h>

namespace
{
    vec2f absolute_mouse_position(render_window& win)
    {
        glfw_backend* bck = dynamic_cast<glfw_backend*>(win.backend);

        assert(bck);

        double mouse_x, mouse_y;
        glfwGetCursorPos(bck->ctx.window, &mouse_x, &mouse_y);

        int window_x, window_y;
        glfwGetWindowPos(bck->ctx.window, &window_x, &window_y);

        mouse_x += window_x;
        mouse_y += window_y;

        return {mouse_x, mouse_y};
    }

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

    ImGui::SetNextWindowSize(ImVec2(real_window_size.x(), real_window_size.y()));
    ImGui::SetNextWindowViewport(ImGui::GetMainViewport()->ID);

    if(ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
    {
        ImVec2 viewport_pos = ImGui::GetMainViewport()->Pos;
        ImGui::SetNextWindowPos(viewport_pos);
        win.backend->set_window_position({viewport_pos.x, viewport_pos.y});
    }
    else
    {
        ImGui::SetNextWindowPos({0,0});
    }

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

            vec2i save_position = win.get_window_position();

            title_start_pos_absolute = {save_position.x(), save_position.y()};
            title_start_mouse_pos_absolute = absolute_mouse_position(win);
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
        vec2f mouse_current_pos = absolute_mouse_position(win);

        vec2f delta = mouse_current_pos - title_start_mouse_pos_absolute;

        vec2f real_pos = delta + title_start_pos_absolute;

        win.backend->set_window_position({real_pos.x(), real_pos.y()});
    }

    if(!win.backend->is_maximised())
    {
        ImVec2 render_tl = {0,0};

        if(ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
        {
            render_tl = ImGui::GetMainViewport()->Pos;
        }

        vec2f label_br = (vec2f){render_tl.x, render_tl.y} + (vec2f){real_window_size.x(), real_window_size.y()};
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
                    resize_start_mouse_pos_absolute = absolute_mouse_position(win);
                }
            }
        }

        if(resize_dragging)
        {
            vec2f delta = absolute_mouse_position(win) - resize_start_mouse_pos_absolute;

            vec2f dim = delta + resize_start_pos;

            if(dim.x() >= 50 && dim.y() >= 50)
                win.backend->resize({dim.x(), dim.y()});
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
