#include "steam.hpp"
#include <steamworks_sdk_153a/sdk/public/steam/steam_api.h>
//#include <steamworks_sdk_153a/sdk/public/steam/steam_api_flat.h>

void steam_api::init()
{
    enabled = SteamAPI_Init();
}

std::vector<uint64_t> steam_api::get_subscribed_items()
{
    if(!enabled)
        return std::vector<uint64_t>();

    ISteamUGC* ptr = SteamUGC();

    uint32_t num = ptr->GetNumSubscribedItems();

    if(num == 0)
        return std::vector<uint64_t>();

    std::vector<uint64_t> result;
    result.resize(num);

    uint32_t actual_num = ptr->GetSubscribedItems(&result[0], num);

    result.resize(actual_num);

    return result;
}

steam_api::~steam_api()
{
    if(enabled)
    {
        SteamAPI_Shutdown();
    }
}
