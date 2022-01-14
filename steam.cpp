#include "steam.hpp"
#include <steamworks_sdk_153a/sdk/public/steam/steam_api_flat.h>

void steam_api::init()
{
    enabled = SteamAPI_Init();
}

steam_api::~steam_api()
{
    if(enabled)
    {
        SteamAPI_Shutdown();
    }
}
