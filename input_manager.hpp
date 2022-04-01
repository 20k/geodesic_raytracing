#ifndef INPUT_MANAGER_HPP_INCLUDED
#define INPUT_MANAGER_HPP_INCLUDED

#include <string_view>
#include <string>
#include <vector>
#include <utility>
#include <map>

struct input_manager
{
    std::vector<std::pair<std::string, int>> linear_keys;
    std::map<std::string, int, std::less<>> glfw_key_map;

    input_manager();

    bool is_key_down(std::string_view view);
    bool is_key_pressed(std::string_view view);

    void rebind(std::string_view name, int glfw_key);
};

#endif // INPUT_MANAGER_HPP_INCLUDED
