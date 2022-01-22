#include "steam_ugc_manager.hpp"
#include <bit>
#include <filesystem>
#include <iostream>
#include <map>
#include <set>

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

std::string from_c_str(const char* ptr)
{
    int size = strlen(ptr);

    return std::string(ptr, ptr + size);
}

void ugc_update::modify(UGCUpdateHandle_t handle) const
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

void ugc_details::load(SteamUGCDetails_t in)
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

std::optional<ugc_storage*> steam_ugc_update_manager::find_local_item(PublishedFileId_t id)
{
    for(ugc_storage& i : items)
    {
        if(i.det.id == id)
            return &i;
    }

    return std::nullopt;
}

void steam_ugc_update_manager::network_fetch(const steam_info& info)
{
    view.fetch(info, exec, [this]()
    {
        set_network_items(view.items);
    });
}

void steam_ugc_update_manager::network_delete_item(const steam_info& info, PublishedFileId_t id)
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

void steam_ugc_update_manager::network_create_item(const steam_info& info)
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

void steam_ugc_update_manager::set_network_items(const std::vector<ugc_details>& details)
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

void steam_ugc_update_manager::poll(const steam_info& info)
{
    auto now = std::chrono::steady_clock::now();

    double elapsed = std::chrono::duration<double>(now - last_poll_time).count();

    if(elapsed > 20 || !once)
    {
        last_poll_time = now;

        network_fetch(info);

        once = true;
    }

    SteamAPI_RunCallbacks();

    exec.poll();
}
