#include <iostream>
#include <toolkit/render_window.hpp>
#include <toolkit/clock.hpp>
#include <imgui/misc/freetype/imgui_freetype.h>
#include <steam/steam_api.h>
#include <steam/steam_api_flat.h>
#include <optional>
#include <functional>
#include <imgui/misc/cpp/imgui_stdlib.h>
#include <bit>
#include <filesystem>
#include <set>
#include <compare>
#include "steam_ugc_manager.hpp"

ugc_visibility visibility_from_int(int v)
{
    if(v < 0)
        v = 0;

    if(v > 3)
        v = 3;

    return ugc_visibility{v};
}

void create_directories(std::vector<ugc_storage>& items)
{
    for(ugc_storage& ustore : items)
    {
        std::filesystem::create_directory("./content/" + std::to_string(ustore.det.id));
        std::filesystem::create_directory("./content/" + std::to_string(ustore.det.id) + "/data");
    }
}

void display(steam_ugc_update_manager& steam, std::vector<ugc_storage>& items, const steam_info& info)
{
    for(ugc_storage& ustore : items)
    {
        std::string directory = "./content/" + std::to_string(ustore.det.id);

        ugc_details& det = ustore.det;

        std::string unique_id = std::to_string(det.id);

        std::string name_placeholder = det.name == "" ? "Untitled" : det.name;

        std::string tree_id = name_placeholder + " | " + std::to_string(det.id) + "###treenode_title" + std::to_string(det.id);

        if(ImGui::TreeNode(tree_id.c_str()))
        {
            //ImGui::Text(("Folder: " + std::to_string(det.id)).c_str());

            ImGui::Text("Name");
            ImGui::SameLine();
            ImGui::InputText(("##name" + unique_id).c_str(), &det.name);

            ImGui::Text("Desc");
            ImGui::SameLine();
            ImGui::InputText(("##desc" + unique_id).c_str(), &det.description);

            ImGui::Text("Tags");
            ImGui::SameLine();
            ImGui::InputText(("##tags" + unique_id).c_str(), &det.tags);

            ImGui::Text("Visibility");

            std::string vis_name;

            if(det.visibility == ugc_visibility::is_public)
                vis_name = "public";

            if(det.visibility == ugc_visibility::is_friends_only)
                vis_name = "friends only";

            if(det.visibility == ugc_visibility::is_private)
                vis_name = "private";

            if(det.visibility == ugc_visibility::is_unlisted)
                vis_name = "unlisted";

            ImGui::SameLine();

            std::array<std::string, 4> labels =
            {
                "public",
                "friends only",
                "private",
                "unlisted",
            };

            if(ImGui::BeginCombo(("##combobox" + unique_id).c_str(), vis_name.c_str()))
            {
                for(int idx = 0; idx < (int)labels.size(); idx++)
                {
                    bool is_selected = idx == (int)det.visibility;

                    if(ImGui::Selectable((labels[idx] + "##" + unique_id).c_str()))
                    {
                        det.visibility = visibility_from_int((int)idx);
                    }

                    if(is_selected)
                    {
                        ImGui::SetItemDefaultFocus();
                    }
                }

                ImGui::EndCombo();
            }

            bool has_preview = std::filesystem::exists(directory + "/preview.png");

            if(!has_preview)
            {
                ImGui::Text("No valid preview.png in folder");
            }

            if(ustore.completed)
            {
                if(ustore.error_message == "")
                    ImGui::Text("Upload Completed Successfully");
                else
                    ImGui::Text("Error in upload %s", ustore.error_message.c_str());
            }
            else
            {
                ImGui::Text("Upload in progress..");
            }

            if(ImGui::Button(("Update Metadata##" + unique_id).c_str()))
            {
                ustore.completed = false;
                ustore.error_message = "";

                ugc_update update;
                update.id = det.id;
                update.name = det.name;
                update.description = det.description;
                update.tags = det.tags;
                update.visibility = det.visibility;

                PublishedFileId_t id = det.id;

                steam.network_update_item(info, update, [id, &steam](auto err_opt)
                {
                    auto store_opt = steam.find_local_item(id);

                    if(store_opt.has_value())
                    {
                        store_opt.value()->completed = true;
                        store_opt.value()->error_message = err_opt.value_or("");
                    }
                });
            }

            if(ImGui::Button(("Update Metadata and Contents##" + unique_id).c_str()))
            {
                if(!has_preview)
                {
                    ustore.completed = true;
                    ustore.error_message = "No valid preview.png in root folder " + std::to_string(ustore.det.id);
                }
                else
                {
                    ustore.completed = false;
                    ustore.error_message = "";

                    ugc_update update;
                    update.id = det.id;
                    update.name = det.name;
                    update.description = det.description;
                    update.tags = det.tags;
                    update.visibility = det.visibility;
                    update.local_content_path = directory + "/data";
                    update.local_preview_path = directory + "/preview.png";

                    PublishedFileId_t id = det.id;

                    steam.network_update_item(info, update, [id, &steam](auto err_opt)
                    {
                        auto store_opt = steam.find_local_item(id);

                        if(store_opt.has_value())
                        {
                            store_opt.value()->completed = true;
                            store_opt.value()->error_message = err_opt.value_or("");
                        }
                    });
                }
            }

            if(ImGui::Button(("Open Directory##" + unique_id).c_str()))
            {
                std::string apath = std::filesystem::absolute(directory).string();

                system(("start " + apath).c_str());
            }

            if(ImGui::Button(("Delete (confirms)##" + unique_id).c_str()))
            {
                ustore.are_you_sure = true;
            }

            if(ustore.are_you_sure)
            {
                ImGui::Text("Type YESPLEASE and hit enter to delete. Anything else cancels");

                if(ImGui::InputText(("##inputtextdelete" + unique_id).c_str(), &ustore.confirm_string, ImGuiInputTextFlags_EnterReturnsTrue))
                {
                    if(ustore.confirm_string == "YESPLEASE")
                    {
                        steam.network_delete_item(info, ustore.det.id);
                    }

                    ustore.are_you_sure = false;
                    ustore.confirm_string = "";
                }
            }

            if(ustore.has_server_changes)
            {
                ImGui::Text("Server content differs from client content");

                ImGui::SameLine();

                if(ImGui::Button(("Pull from server##" + unique_id).c_str()))
                {
                    ustore.pull_from_server = true;

                    steam.network_fetch(info);
                }
            }

            ImGui::TreePop();
        }
    }

    if(ImGui::Button("Create New"))
    {
        steam.network_create_item(info);
    }
}

int main()
{
    render_settings sett;
    sett.width = 800;
    sett.height = 600;
    sett.no_double_buffer = true;
    sett.is_srgb = true;

    render_window win(sett, "Geodesics");

    ImFontAtlas* atlas = ImGui::GetIO().Fonts;
    atlas->FontBuilderFlags = ImGuiFreeTypeBuilderFlags_LCD | ImGuiFreeTypeBuilderFlags_FILTER_DEFAULT | ImGuiFreeTypeBuilderFlags_LoadColor;

    ImFontConfig font_cfg;
    font_cfg.GlyphExtraSpacing = ImVec2(0, 0);
    font_cfg.FontBuilderFlags = ImGuiFreeTypeBuilderFlags_LCD | ImGuiFreeTypeBuilderFlags_FILTER_DEFAULT | ImGuiFreeTypeBuilderFlags_LoadColor;

    ImGuiIO& io = ImGui::GetIO();

    io.Fonts->Clear();
    io.Fonts->AddFontFromFileTTF("VeraMono.ttf", 14, &font_cfg);

    std::filesystem::create_directory("./content");

    steam_info info;

    steam_ugc_update_manager steam;

    while(!win.should_close())
    {
        win.poll();

        steam.poll(info);

        vec2i window_size = win.get_window_size();

        ImGui::SetNextWindowPos(ImVec2(0,0), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(window_size.x(),window_size.y()), ImGuiCond_Always);

        ImGui::Begin("Workshop Editor", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);

        create_directories(steam.items);
        display(steam, steam.items, info);

        ImGui::End();

        win.display();
    }

    return 0;
}
