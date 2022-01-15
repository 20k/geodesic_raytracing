#include <iostream>
#include <toolkit/render_window.hpp>
#include <toolkit/clock.hpp>
#include <imgui/misc/freetype/imgui_freetype.h>
#include <steam/steam_api.h>
#include <steam/steam_api_flat.h>
#include <optional>

/*struct steam_api_call
{
    SteamAPICall_t call = 0;

    steam_api_call(SteamAPICall_t in)
    {
        call = in;
    }
};*/

void free_query_handle(UGCQueryHandle_t* in)
{
    ISteamUGC* self = SteamAPI_SteamUGC();

    printf("Freed\n");

    SteamAPI_ISteamUGC_ReleaseQueryUGCRequest(self, *in);

    delete in;
}

struct request_handle
{
    std::shared_ptr<UGCQueryHandle_t> handle;
    SteamAPICall_t call = 0;

    request_handle(UGCQueryHandle_t in) : handle(new UGCQueryHandle_t(in), free_query_handle)
    {

    }

    void dispatch()
    {
        ISteamUGC* self = SteamAPI_SteamUGC();

        SteamAPICall_t result = SteamAPI_ISteamUGC_SendQueryUGCRequest(self, *handle);

        printf("Dispatched\n");

        call = result;
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

    std::optional<request_handle> current_query;

    steam_api()
    {
        if(!SteamAPI_Init())
            throw std::runtime_error("Could not initialise the steam api");

        appid = SteamUtils()->GetAppID();

        ISteamUser* usr = SteamAPI_SteamUser();

        steam_id = SteamAPI_ISteamUser_GetSteamID(usr);
        ///see: steamclientpublic.h
        account_id = (steam_id >> 32);

        std::cout << "Appid " << appid << std::endl;
        std::cout << "Account " << account_id << std::endl;
    }

    void poll()
    {
        #if 0
        HSteamPipe hSteamPipe = SteamAPI_GetHSteamPipe(); // See also SteamGameServer_GetHSteamPipe()
        SteamAPI_ManualDispatch_RunFrame(hSteamPipe);
        CallbackMsg_t callback;
        while(SteamAPI_ManualDispatch_GetNextCallback( hSteamPipe, &callback))
        {
            printf("Callback\n");

            // Check for dispatching API call results
            if( callback.m_iCallback == SteamAPICallCompleted_t::k_iCallback )
            {
                SteamAPICallCompleted_t *pCallCompleted = (SteamAPICallCompleted_t*)callback.m_pubParam;
                void *pTmpCallResult = malloc(callback.m_cubParam);
                bool bFailed;

                if(SteamAPI_ManualDispatch_GetAPICallResult(hSteamPipe, pCallCompleted->m_hAsyncCall, pTmpCallResult, callback.m_cubParam, callback.m_iCallback, &bFailed))
                {
                    // Dispatch the call result to the registered handler(s) for the
                    // call identified by pCallCompleted->m_hAsyncCall

                    printf("Got a steam callback result\n");
                }

                free(pTmpCallResult);
            }
            else if(callback.m_iCallback == SteamUGCQueryCompleted_t::k_iCallback)
            {
                SteamUGCQueryCompleted_t* pCallCompleted = (SteamUGCQueryCompleted_t*)callback.m_pubParam;

                void *pTmpCallResult = malloc(callback.m_cubParam);

                bool bFailed;

                /*if(SteamAPI_ManualDispatch_GetAPICallResult(hSteamPipe, pCallCompleted->m_hAsyncCall, pTmpCallResult, callback.m_cubParam, callback.m_iCallback, &bFailed))
                {
                    // Dispatch the call result to the registered handler(s) for the
                    // call identified by pCallCompleted->m_hAsyncCall

                    printf("Got a UGC callback result\n");
                }*/

                printf("Ugc call?\n");

                free(pTmpCallResult);
            }
            else
            {
                // Look at callback.m_iCallback to see what kind of callback it is,
                // and dispatch to appropriate handler(s)
            }

            SteamAPI_ManualDispatch_FreeLastCallback(hSteamPipe);
        }
        #endif // 0

        //if(last_poll.get_elapsed_time_s() > 5)
        if(!once)
        {
            ISteamUGC* ugc = SteamAPI_SteamUGC();

            UGCQueryHandle_t ugchandle = SteamAPI_ISteamUGC_CreateQueryUserUGCRequest(ugc, account_id, k_EUserUGCList_Published, k_EUGCMatchingUGCType_All, k_EUserUGCListSortOrder_CreationOrderDesc, appid, appid, 1);

            SteamAPI_ISteamUGC_SetReturnOnlyIDs(ugc, ugchandle, true);
            SteamAPI_ISteamUGC_SetReturnKeyValueTags(ugc, ugchandle, true);

            request_handle handle(ugchandle);

            last_poll.restart();

            current_query = std::move(handle);
            current_query.value().dispatch();

            once = true;
        }

        SteamAPI_RunCallbacks();

        ISteamUtils* utils = SteamAPI_SteamUtils();

        if(current_query.has_value())
        {
            SteamAPICall_t call = current_query.value().call;

            /*bool failed = false;
            std::cout << "Finished? " << SteamAPI_ISteamUtils_IsAPICallCompleted(utils, call, &failed) << std::endl;

            if(failed)
            {
                std::cout << "Failed\n";
            }*/

            bool failed = false;
            if(SteamAPI_ISteamUtils_IsAPICallCompleted(utils, call, &failed))
            {
                if(!failed)
                {
                    SteamUGCQueryCompleted_t completed;

                    SteamAPI_ISteamUtils_GetAPICallResult(utils, call, &completed, sizeof(completed), SteamUGCQueryCompleted_t::k_iCallback, &failed);

                    assert(!failed);

                    printf("Done\n");
                }
            }
        }
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
