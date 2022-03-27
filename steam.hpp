#ifndef STEAM_HPP_INCLUDED
#define STEAM_HPP_INCLUDED

#include <vector>
#include <stdint.h>

struct steam_api
{
    bool enabled = false;

    void init();

    std::vector<uint64_t> get_subscribed_items();

    ~steam_api();
};

#endif // STEAM_HPP_INCLUDED
