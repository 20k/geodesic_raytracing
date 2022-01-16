#include <iostream>
#include <toolkit/render_window.hpp>
#include <toolkit/clock.hpp>
#include <imgui/misc/freetype/imgui_freetype.h>
#include <steam/steam_api.h>
#include <steam/steam_api_flat.h>
#include <optional>
#include <functional>

void free_query_handle(UGCQueryHandle_t* in)
{
    ISteamUGC* self = SteamAPI_SteamUGC();

    printf("Freed\n");

    SteamAPI_ISteamUGC_ReleaseQueryUGCRequest(self, *in);

    delete in;
}

struct steam_api_call_base
{
    virtual bool poll() = 0;
};

template<typename T>
struct steam_api_call : steam_api_call_base
{
    bool has_call = false;
    SteamAPICall_t call;
    std::function<void(const T&)> callback;

    steam_api_call(){}

    template<typename U>
    steam_api_call(SteamAPICall_t in, U&& func)
    {
        start(in, std::forward<U>(func));
    }

    template<typename U>
    void start(SteamAPICall_t in, U&& func)
    {
        callback = func;
        call = in;
        has_call = true;
    }

    virtual bool poll() override
    {
        if(!has_call)
            return false;

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
                delete val;

                callbacks.erase(callbacks.begin() + i);
                i--;
                continue;
            }
        }
    }
};

struct ugc_request_handle
{
    std::shared_ptr<UGCQueryHandle_t> handle;
    steam_api_call<SteamUGCQueryCompleted_t> call;

    ugc_request_handle(UGCQueryHandle_t in) : handle(new UGCQueryHandle_t(in), free_query_handle)
    {

    }

    void dispatch()
    {
        ISteamUGC* self = SteamAPI_SteamUGC();

        SteamAPICall_t result = SteamAPI_ISteamUGC_SendQueryUGCRequest(self, *handle);

        printf("Dispatched\n");

        call.start(result, [&](const SteamUGCQueryCompleted_t& result)
        {
            query_details(result.m_unNumResultsReturned);
        });
    }

    void query_details(int num)
    {
        std::cout << "Found " << num << " published workshop items for user" << std::endl;

        ISteamUtils* utils = SteamAPI_SteamUtils();
        ISteamUGC* ugc = SteamAPI_SteamUGC();

        for(int i=0; i < num; i++)
        {
            SteamUGCDetails_t details;

            if(SteamAPI_ISteamUGC_GetQueryUGCResult(ugc, *handle, i, &details))
            {
                std::string name(&details.m_rgchTitle[0], &details.m_rgchTitle[k_cchPublishedDocumentTitleMax - 1]);

                std::cout << "name " << name << std::endl;
            }
        }

    }

    bool poll()
    {
        return call.poll();
    }
};

struct steam_api
{
    steady_timer last_poll;
    std::vector<uint64_t> published_items;
    uint32_t appid = 0;
    uint32_t account_id = 0;
    uint64_t steam_id = 0;

    bool once = false;

    std::optional<ugc_request_handle> current_query;

    steam_callback_executor exec;

    steam_api()
    {
        if(!SteamAPI_Init())
            throw std::runtime_error("Could not initialise the steam api");

        appid = SteamUtils()->GetAppID();

        ISteamUser* usr = SteamAPI_SteamUser();

        steam_id = SteamAPI_ISteamUser_GetSteamID(usr);
        ///see: steamclientpublic.h
        account_id = (steam_id & 0xFFFFFFFF);

        std::cout << "Appid " << appid << std::endl;
        std::cout << "Account " << account_id << std::endl;
    }

    ugc_request_handle request_published_items()
    {
        ISteamUGC* ugc = SteamAPI_SteamUGC();

        UGCQueryHandle_t ugchandle = SteamAPI_ISteamUGC_CreateQueryUserUGCRequest(ugc, account_id, k_EUserUGCList_Published, k_EUGCMatchingUGCType_All, k_EUserUGCListSortOrder_CreationOrderDesc, appid, appid, 1);

        //SteamAPI_ISteamUGC_SetReturnOnlyIDs(ugc, ugchandle, true);
        SteamAPI_ISteamUGC_SetReturnKeyValueTags(ugc, ugchandle, true);
        //SteamAPI_ISteamUGC_SetReturn

        return ugchandle;
    }

    void create_item()
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

            ISteamUGC* ugc = SteamAPI_SteamUGC();

            UGCUpdateHandle_t handle = SteamAPI_ISteamUGC_StartItemUpdate(ugc, appid, id);

            SteamAPI_ISteamUGC_SetItemTitle(ugc, handle, "Hello there");

            SteamAPICall_t raw_api_call = SteamAPI_ISteamUGC_SubmitItemUpdate(ugc, handle, nullptr);

            steam_api_call<SubmitItemUpdateResult_t> api_result(raw_api_call, [](const SubmitItemUpdateResult_t& val)
            {
                if(val.m_bUserNeedsToAcceptWorkshopLegalAgreement)
                {
                    std::cout << "Need to accept workshop legal agreement" << std::endl;
                }

                if(val.m_eResult != k_EResultOK)
                {
                    std::cout << "Error submitting update" << std::endl;
                }

                std::cout << "SubmitItemUpdate callback" << std::endl;
            });

            exec.add(api_result);

            std::cout << "Started item update" << std::endl;
        };

        steam_api_call<CreateItemResult_t> result(SteamAPI_ISteamUGC_CreateItem(ugc, appid, k_EWorkshopFileTypeCommunity), on_created);

        exec.add(result);
    }

    void poll()
    {
        //if(last_poll.get_elapsed_time_s() > 5)
        if(!once)
        {
            last_poll.restart();

            current_query = request_published_items();
            current_query.value().dispatch();

            once = true;
        }

        SteamAPI_RunCallbacks();

        if(current_query.has_value())
        {
            if(current_query->poll())
            {
                current_query = std::nullopt;
            }
        }

        exec.poll();
    }

    ~steam_api()
    {
        SteamAPI_Shutdown();
    }
};

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

    steam_api steam;

    //steam.create_item();

    while(!win.should_close())
    {
        win.poll();

        steam.poll();

        vec2i window_size = win.get_window_size();

        ImGui::SetNextWindowPos(ImVec2(0,0), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(window_size.x(),window_size.y()), ImGuiCond_Always);

        ImGui::Begin("Workshop Editor", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);

        ImGui::End();

        win.display();
    }

    return 0;
}
