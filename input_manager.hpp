#ifndef INPUT_MANAGER_HPP_INCLUDED
#define INPUT_MANAGER_HPP_INCLUDED

#include <string_view>
#include <string>
#include <vector>
#include <utility>
#include <map>
#include <optional>

struct render_window;

struct input_manager
{
    std::vector<std::pair<std::string, std::vector<int>>> linear_keys;
    std::map<std::string, std::vector<int>, std::less<>> glfw_key_map;

    input_manager();

    void display_key_rebindings(render_window& win);

    bool is_key_down(std::string_view view);
    bool is_key_pressed(std::string_view view);

    void rebind(std::string_view name, const std::vector<int>& glfw_key);

private:
    std::vector<int> rebinding_keys;
    std::optional<std::string> last_rebind_purpose;
};

#endif // INPUT_MANAGER_HPP_INCLUDED
