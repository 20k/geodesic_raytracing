#ifndef CONTENT_MANAGER_HPP_INCLUDED
#define CONTENT_MANAGER_HPP_INCLUDED

#include "metric.hpp"
#include <filesystem>
#include <string>
#include <vector>
#include <map>
#include <optional>

struct content_manager;
struct content;

struct metric_cache
{
    metrics::metric* met = nullptr;

    metrics::metric* lazy_fetch(content_manager& manage, content& c, const std::string& friendly_name);
};

struct content
{
    std::filesystem::path folder;

    std::vector<std::string> filename_sorting;

    std::vector<std::filesystem::path> metrics;
    std::vector<std::filesystem::path> configs;
    std::vector<metrics::metric_config> base_configs;
    std::vector<std::filesystem::path> coordinates;
    std::vector<std::filesystem::path> origins;

    std::map<std::string, metric_cache> cache;

    void load(content_manager& all_content, std::filesystem::path path);

    std::optional<std::filesystem::path> lookup_path_to_metric_file(const std::string& name);
    metrics::metric_config* get_config_of_filename(std::filesystem::path filename);

    metrics::metric* lazy_fetch(content_manager& manage, const std::string& friendly_name);
};

struct content_manager
{
    std::vector<content> content_directories;

    void add_content_folder(std::filesystem::path folder);

    std::optional<std::filesystem::path> lookup_path_to_metric_file(const std::string& name);
    std::optional<std::filesystem::path> lookup_path_to_config_file(const std::string& name);
    std::optional<std::filesystem::path> lookup_path_to_coordinates_file(const std::string& name);
    std::optional<std::filesystem::path> lookup_path_to_origins_file(const std::string& name);
};

#endif // CONTENT_MANAGER_HPP_INCLUDED
