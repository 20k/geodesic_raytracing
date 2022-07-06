#ifndef STEAM_UGC_MANAGER_HPP_INCLUDED
#define STEAM_UGC_MANAGER_HPP_INCLUDED

#include <steam/steam_api_flat.h>
#include <functional>
#include <optional>
#include <compare>
#include <chrono>
#include <assert.h>
#include <string>
#include <stdexcept>
#include <iostream>

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

struct ugc_update
{
    PublishedFileId_t id;;

    std::optional<std::string> name;
    std::optional<std::string> description;
    std::optional<std::string> tags;
    std::optional<ugc_visibility> visibility;

    std::optional<std::string> local_content_path;
    std::optional<std::string> local_preview_path;

    void modify(UGCUpdateHandle_t handle) const;
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

    void load(SteamUGCDetails_t in);

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

std::vector<ugc_details> query_to_details(const SteamUGCQueryCompleted_t& query);

struct steam_info
{
    uint32_t appid = 0;
    uint32_t account_id = 0;
    uint64_t steam_id = 0;

    ///placeholder functionality
    bool enabled = false;

    steam_info()
    {
        if(!SteamAPI_Init())
            return;

        enabled = true;

        ISteamUtils* util = SteamAPI_SteamUtils();
        ISteamUser* usr = SteamAPI_SteamUser();

        appid = SteamAPI_ISteamUtils_GetAppID(util);

        steam_id = SteamAPI_ISteamUser_GetSteamID(usr);
        ///see: steamclientpublic.h
        account_id = (steam_id & 0xFFFFFFFF);
    }

    ~steam_info()
    {
        if(enabled)
            SteamAPI_Shutdown();
    }

    steam_info(const steam_info&) = delete;
    steam_info(steam_info&&) = delete;
    steam_info& operator=(const steam_info&) = delete;
    steam_info& operator=(steam_info&&) = delete;

    bool is_enabled()
    {
        return enabled;
    }
};

struct ugc_view
{
    std::vector<ugc_details> items;
    EUserUGCList type = k_EUserUGCList_Published;

    void only_get_published();
    void only_get_subscribed();

    template<typename T>
    void fetch(const steam_info& info, steam_callback_executor& exec, T&& on_complete)
    {
        if(!info.enabled)
            return;

        ISteamUGC* ugc = SteamAPI_SteamUGC();

        UGCQueryHandle_t raw_ugc_handle = SteamAPI_ISteamUGC_CreateQueryUserUGCRequest(ugc, info.account_id, type, k_EUGCMatchingUGCType_All, k_EUserUGCListSortOrder_CreationOrderDesc, info.appid, info.appid, 1);

        SteamAPI_ISteamUGC_SetReturnKeyValueTags(ugc, raw_ugc_handle, true);

        SteamAPICall_t result = SteamAPI_ISteamUGC_SendQueryUGCRequest(ugc, raw_ugc_handle);

        steam_api_call<SteamUGCQueryCompleted_t> call(result, [on_complete = std::move(on_complete), raw_ugc_handle, this](const SteamUGCQueryCompleted_t& result)
        {
            items = query_to_details(result);

            ISteamUGC* ugc = SteamAPI_SteamUGC();

            SteamAPI_ISteamUGC_ReleaseQueryUGCRequest(ugc, raw_ugc_handle);

            on_complete();
        });

        exec.add(call);
    }
};

struct steam_ugc_update_manager
{
    std::chrono::time_point<std::chrono::steady_clock> last_poll_time = std::chrono::steady_clock::now();

    bool once = false;

    std::vector<ugc_storage> items;
    ugc_view view;

    steam_callback_executor exec;

    steam_ugc_update_manager();

    std::optional<ugc_storage*> find_local_item(PublishedFileId_t id);

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

    void network_fetch(const steam_info& info);

    void network_delete_item(const steam_info& info, PublishedFileId_t id);
    void network_create_item(const steam_info& info);

    void set_network_items(const std::vector<ugc_details>& details);

    void poll(const steam_info& info);
};

#endif // STEAM_UGC_MANAGER_HPP_INCLUDED
