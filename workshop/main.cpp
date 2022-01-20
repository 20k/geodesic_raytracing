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

std::vector<std::string> split(const std::string& str, const std::string& delim)
{
    std::vector<std::string> tokens;
    size_t prev = 0, pos = 0;
    do
    {
        pos = str.find(delim, prev);
        if (pos == std::string::npos) pos = str.length();
        std::string token = str.substr(prev, pos-prev);
        if (!token.empty()) tokens.push_back(token);
        prev = pos + delim.length();
    }
    while (pos < str.length() && prev < str.length());
    return tokens;
}

struct steam_api_call_base
{
    virtual bool poll() = 0;

    virtual ~steam_api_call_base(){}
};

template<typename T>
struct steam_api_call : steam_api_call_base
{
    SteamAPICall_t call;
    std::function<void(const T&)> callback;

    template<typename U>
    steam_api_call(SteamAPICall_t in, U&& func) : call(in), callback(func) {}

    virtual bool poll() override
    {
        ISteamUtils* utils = SteamAPI_SteamUtils();

        bool failed = false;
        if(SteamAPI_ISteamUtils_IsAPICallCompleted(utils, call, &failed))
        {
            if(!failed)
            {
                T completed;

                SteamAPI_ISteamUtils_GetAPICallResult(utils, call, &completed, sizeof(completed), T::k_iCallback, &failed);

                callback(completed);

                assert(!failed);

                return true;
            }
        }

        return false;
    }
};

struct steam_callback_executor
{
    std::vector<steam_api_call_base*> callbacks;

    template<typename T>
    void add(const steam_api_call<T>& in)
    {
        steam_api_call_base* b = new steam_api_call<T>(in);

        callbacks.push_back(b);
    }

    void poll()
    {
        for(int i=0; i < (int)callbacks.size(); i++)
        {
            steam_api_call_base* val = callbacks[i];

            if(val->poll())
            {
                callbacks.erase(callbacks.begin() + i);
                i--;

                delete val;

                continue;
            }
        }
    }
};

enum class ugc_visibility
{
    is_public = 0,
    is_friends_only = 1,
    is_private = 2,
    is_unlisted = 3,
};

ugc_visibility visibility_from_int(int v)
{
    if(v < 0)
        v = 0;

    if(v > 3)
        v = 3;

    return ugc_visibility{v};
}

std::string from_c_str(const char* ptr)
{
    int size = strlen(ptr);

    return std::string(ptr, ptr + size);
}

struct ugc_update
{
    PublishedFileId_t id;;

    std::optional<std::string> name;
    std::optional<std::string> description;
    std::optional<std::string> tags;
    std::optional<ugc_visibility> visibility;

    std::optional<std::string> local_content_path;
    std::optional<std::string> local_preview_path;

    void modify(UGCUpdateHandle_t handle) const
    {
        ISteamUGC* ugc = SteamAPI_SteamUGC();

        if(name.has_value())
        {
            SteamAPI_ISteamUGC_SetItemTitle(ugc, handle, name.value().c_str());
        }

        if(description.has_value())
        {
            SteamAPI_ISteamUGC_SetItemDescription(ugc, handle, description.value().c_str());
        }

        if(tags.has_value())
        {
            std::vector<std::string> split_tags = split(tags.value(), ",");

            std::vector<const char*> as_cc;

            for(const auto& i : split_tags)
            {
                as_cc.push_back(i.c_str());
            }

            const char** as_pp = nullptr;

            if(as_cc.size() > 0)
            {
                as_pp = &as_cc[0];
            }

            struct SteamParamStringArray_t arr;
            arr.m_nNumStrings = split_tags.size();
            arr.m_ppStrings = as_pp;

            SteamAPI_ISteamUGC_SetItemTags(ugc, handle, &arr);
        }

        if(visibility.has_value())
        {
            ERemoteStoragePublishedFileVisibility vis = std::bit_cast<ERemoteStoragePublishedFileVisibility>(visibility.value());

            SteamAPI_ISteamUGC_SetItemVisibility(ugc, handle, vis);
        }

        if(local_content_path.has_value())
        {
            std::string absolute_path = std::filesystem::absolute(local_content_path.value()).string();

            std::cout << "Absolute Path " << absolute_path << std::endl;

            ISteamUGC* ugc = SteamAPI_SteamUGC();

            assert(SteamAPI_ISteamUGC_SetItemContent(ugc, handle, absolute_path.c_str()));
        }

        if(local_preview_path.has_value())
        {
            std::string absolute_path = std::filesystem::absolute(local_preview_path.value()).string();

            std::cout << "Absolute Preview Path " << absolute_path << std::endl;

            ISteamUGC* ugc = SteamAPI_SteamUGC();

            assert(SteamAPI_ISteamUGC_SetItemPreview(ugc, handle, absolute_path.c_str()));
        }
    }
};

struct ugc_details
{
    PublishedFileId_t id;
    std::string name;
    std::string description;
    std::string tags;
    ugc_visibility visibility;

    ///can be "" indicating not installed
    std::string absolute_content_path;

    void load(SteamUGCDetails_t in)
    {
        id = in.m_nPublishedFileId;
        name = from_c_str(in.m_rgchTitle);
        description = from_c_str(in.m_rgchDescription);
        tags = from_c_str(in.m_rgchTags);
        visibility = ugc_visibility{in.m_eVisibility};

        ISteamUGC* ugc = SteamAPI_SteamUGC();

        std::string folder_storage;
        folder_storage.resize(1024 * 1024);

        uint64_t size_on_disk = 0;
        uint32_t timestamp = 0;
        SteamAPI_ISteamUGC_GetItemInstallInfo(ugc, in.m_nPublishedFileId, &size_on_disk, &folder_storage[0], folder_storage.size(), &timestamp);

        int c_length = strlen(folder_storage.c_str());

        absolute_content_path = std::string(folder_storage.begin(), folder_storage.begin() + c_length);
    }

    auto operator<=>(const ugc_details&) const = default;
};

struct ugc_storage
{
    ugc_details det;

    bool completed = true;
    std::string error_message;
    bool should_delete = false;

    bool are_you_sure = false;
    std::string confirm_string;

    bool has_server_changes = false;
    bool pull_from_server = false;
};

std::vector<ugc_details> query_to_details(const SteamUGCQueryCompleted_t& query)
{
    int num = query.m_unNumResultsReturned;

    std::cout << "Found " << num << " published workshop items for user" << std::endl;

    std::vector<ugc_details> ret;

    ISteamUGC* ugc = SteamAPI_SteamUGC();

    for(int i=0; i < num; i++)
    {
        SteamUGCDetails_t details;

        if(SteamAPI_ISteamUGC_GetQueryUGCResult(ugc, query.m_handle, i, &details))
        {
            ugc_details item;
            item.load(details);

            ret.push_back(item);
        }
    }

    return ret;
}

struct steam_info
{
    uint32_t appid = 0;
    uint32_t account_id = 0;
    uint64_t steam_id = 0;

    steam_info()
    {
        printf("Initialising steam api\n");

        if(!SteamAPI_Init())
            throw std::runtime_error("Could not init api");

        ISteamUtils* util = SteamAPI_SteamUtils();
        ISteamUser* usr = SteamAPI_SteamUser();

        appid = SteamAPI_ISteamUtils_GetAppID(util);

        steam_id = SteamAPI_ISteamUser_GetSteamID(usr);
        ///see: steamclientpublic.h
        account_id = (steam_id & 0xFFFFFFFF);
    }

    ~steam_info()
    {
        printf("Destroying steam api\n");

        SteamAPI_Shutdown();
    }

    steam_info(const steam_info&) = delete;
    steam_info(steam_info&&) = delete;
    steam_info& operator=(const steam_info&) = delete;
    steam_info& operator=(steam_info&&) = delete;
};

struct ugc_view
{
    std::vector<ugc_details> items;

    template<typename T>
    void fetch(const steam_info& info, steam_callback_executor& exec, T&& on_complete)
    {
        ISteamUGC* ugc = SteamAPI_SteamUGC();

        UGCQueryHandle_t raw_ugc_handle = SteamAPI_ISteamUGC_CreateQueryUserUGCRequest(ugc, info.account_id, k_EUserUGCList_Published, k_EUGCMatchingUGCType_All, k_EUserUGCListSortOrder_CreationOrderDesc, info.appid, info.appid, 1);

        SteamAPI_ISteamUGC_SetReturnKeyValueTags(ugc, raw_ugc_handle, true);

        SteamAPICall_t result = SteamAPI_ISteamUGC_SendQueryUGCRequest(ugc, raw_ugc_handle);

        steam_api_call<SteamUGCQueryCompleted_t> call(result, [on_complete = std::move(on_complete), raw_ugc_handle, this](const SteamUGCQueryCompleted_t& result)
        {
            items = query_to_details(result);

            ISteamUGC* ugc = SteamAPI_SteamUGC();

            SteamAPI_ISteamUGC_ReleaseQueryUGCRequest(ugc, raw_ugc_handle);

            printf("Completed and released query\n");

            on_complete();
        });

        exec.add(call);
    }
};

struct steam_api
{
    steady_timer last_poll;

    bool once = false;

    std::vector<ugc_storage> items;
    ugc_view view;

    steam_callback_executor exec;

    std::optional<ugc_storage*> find_local_item(PublishedFileId_t id)
    {
        for(ugc_storage& i : items)
        {
            if(i.det.id == id)
                return &i;
        }

        return std::nullopt;
    }

    void network_fetch(const steam_info& info)
    {
        view.fetch(info, exec, [this]()
        {
            set_network_items(view.items);
        });
    }

    template<typename T>
    void network_update_item(const steam_info& info, const ugc_update& update, T&& on_complete)
    {
        ISteamUGC* ugc = SteamAPI_SteamUGC();

        UGCUpdateHandle_t handle = SteamAPI_ISteamUGC_StartItemUpdate(ugc, info.appid, update.id);

        update.modify(handle);

        SteamAPICall_t raw_api_call = SteamAPI_ISteamUGC_SubmitItemUpdate(ugc, handle, nullptr);

        steam_api_call<SubmitItemUpdateResult_t> api_result(raw_api_call, [on_complete = std::move(on_complete)](const SubmitItemUpdateResult_t& val)
        {
            std::optional<std::string> error_message;

            if(val.m_bUserNeedsToAcceptWorkshopLegalAgreement)
            {
                std::cout << "Need to accept workshop legal agreement" << std::endl;

                error_message = "Need to accept workshop legal agreement";
            }

            if(val.m_eResult != k_EResultOK)
            {
                std::cout << "Error submitting update " << val.m_eResult << std::endl;

                error_message = "Steam API error in submit " + std::to_string(val.m_eResult);
            }

            std::cout << "SubmitItemUpdate callback" << std::endl;

            on_complete(error_message);
        });

        exec.add(api_result);

        std::cout << "Started item update" << std::endl;
    }

    void network_delete_item(const steam_info& info, PublishedFileId_t id)
    {
        ISteamUGC* ugc = SteamAPI_SteamUGC();

        SteamAPICall_t raw_api_call = SteamAPI_ISteamUGC_DeleteItem(ugc, id);

        steam_api_call<DeleteItemResult_t> api_result(raw_api_call, [this, id, &info](const DeleteItemResult_t& result)
        {
            if(result.m_eResult == k_EResultOK)
            {
                ///?
            }
            else
            {
                printf("Error deleting item\n");
            }

            ///re-fetch from valve after a delete
            network_fetch(info);
        });

        exec.add(api_result);
    }

    void network_create_item(const steam_info& info)
    {
        ISteamUGC* ugc = SteamAPI_SteamUGC();

        auto on_created = [&](CreateItemResult_t result)
        {
            if(result.m_bUserNeedsToAcceptWorkshopLegalAgreement)
            {
                std::cout << "Need to accept workshop legal agreement in create item" << std::endl;
            }

            PublishedFileId_t id = result.m_nPublishedFileId;

            std::cout << "Created item with id " << id << std::endl;

            ugc_update upd;
            upd.id = id;

            network_update_item(info, upd, [](auto err_opt){});

            network_fetch(info);
        };

        steam_api_call<CreateItemResult_t> result(SteamAPI_ISteamUGC_CreateItem(ugc, info.appid, k_EWorkshopFileTypeCommunity), on_created);

        exec.add(result);
    }

    void set_network_items(const std::vector<ugc_details>& details)
    {
        std::map<PublishedFileId_t, ugc_storage*> item_map;

        for(ugc_storage& store : items)
        {
            item_map[store.det.id] = &store;
        }

        std::vector<ugc_storage> additional_items;

        std::set<PublishedFileId_t> new_items;

        for(const ugc_details& i : details)
        {
            new_items.insert(i.id);

            ///found an existing item
            if(auto it = item_map.find(i.id); it != item_map.end())
            {
                ///the servers version is different from the clients
                if(it->second->det != i)
                {
                    it->second->has_server_changes = true;

                    ///requested overwriting the local version
                    if(it->second->pull_from_server)
                    {
                        it->second->pull_from_server = false;
                        it->second->has_server_changes = false;

                        it->second->det = i;
                    }
                }
                else
                {
                    it->second->has_server_changes = false;
                    it->second->pull_from_server = false;
                }
            }
            else
            {
                ugc_storage store;
                store.det = i;

                additional_items.push_back(store);
            }
        }

        for(int i=0; i < (int)items.size(); i++)
        {
            if(new_items.count(items[i].det.id) == 0)
            {
                items.erase(items.begin() + i);
                i--;
                continue;
            }
        }

        ///this invalidates item_map, hence why this doesn't directly insert into items in the details loop
        items.insert(items.end(), additional_items.begin(), additional_items.end());
    }

    void poll(const steam_info& info)
    {
        if(last_poll.get_elapsed_time_s() > 20 || !once)
        {
            last_poll.restart();

            network_fetch(info);

            once = true;
        }

        SteamAPI_RunCallbacks();

        exec.poll();
    }
};

void create_directories(std::vector<ugc_storage>& items)
{
    for(ugc_storage& ustore : items)
    {
        std::filesystem::create_directory("./content/" + std::to_string(ustore.det.id));
        std::filesystem::create_directory("./content/" + std::to_string(ustore.det.id) + "/data");
    }
}

void display(steam_api& steam, std::vector<ugc_storage>& items, const steam_info& info)
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

    steam_api steam;

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
