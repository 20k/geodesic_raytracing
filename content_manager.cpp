#include "content_manager.hpp"
#include <toolkit/fs_helpers.hpp>
#include "metric.hpp"
#include "js_interop.hpp"
#include <algorithm>

metrics::metric_config load_config(content_manager& all_content, std::filesystem::path filename, bool inherit);

metrics::metric* load_metric_from_script(content_manager& all_content, std::filesystem::path sname)
{
    std::filesystem::path cfg = sname;

    cfg.replace_extension(".json");

    auto met = new metrics::metric;

    met->metric_cfg = load_config(all_content, cfg, true);

    js_metric jfunc(met->sand, file::read(sname.string(), file::mode::TEXT));

    std::optional<std::filesystem::path> to_polar_file = all_content.lookup_path_to_coordinates_file(met->metric_cfg.to_polar);
    std::optional<std::filesystem::path> from_polar_file = all_content.lookup_path_to_coordinates_file(met->metric_cfg.from_polar);
    std::optional<std::filesystem::path> origin_file = all_content.lookup_path_to_origins_file(met->metric_cfg.origin_distance);

    if(!to_polar_file.has_value())
        throw std::runtime_error("No to polar file " + met->metric_cfg.to_polar);

    if(!from_polar_file.has_value())
        throw std::runtime_error("No from polar file " + met->metric_cfg.from_polar);

    if(!origin_file.has_value())
        throw std::runtime_error("No origin coordinate system file " + met->metric_cfg.origin_distance);

    js_function func_to_polar(met->sand, file::read(to_polar_file.value().string(), file::mode::TEXT));
    js_function func_from_polar(met->sand, file::read(from_polar_file.value().string(), file::mode::TEXT));
    js_single_function fun_origin_distance(met->sand, file::read(origin_file.value().string(), file::mode::TEXT));

    std::cout << "loading " << sname << std::endl;

    met->desc.load(jfunc, func_to_polar, func_from_polar, fun_origin_distance);

    printf("Finished loading script\n");

    return met;
}

metrics::metric* load_metric_from_folder(content_manager& all_content, std::filesystem::path folder)
{
    for(const auto& entry : std::filesystem::directory_iterator{folder})
    {
        std::filesystem::path name = entry.path();

        if(name.extension().string() == ".js")
        {
            return load_metric_from_script(all_content, name);
        }
    }

    throw std::runtime_error("No .js files found in folder");
}

metrics::metric_config load_config(content_manager& all_content, std::filesystem::path filename, bool inherit)
{
    metrics::metric_config cfg;

    try
    {
        nlohmann::json js = nlohmann::json::parse(file::read(filename.string(), file::mode::TEXT));

        if(inherit && js.count("inherit_settings"))
        {
            try
            {
                std::string new_filename = js["inherit_settings"];

                std::optional<std::filesystem::path> lookup_file = all_content.lookup_path_to_config_file(new_filename);

                if(!lookup_file.has_value())
                    throw std::runtime_error("Could not lookup " + new_filename);

                nlohmann::json parent_json = nlohmann::json::parse(file::read(lookup_file.value().string(), file::mode::TEXT));

                metrics::metric_config parent;
                parent.load(parent_json);

                cfg = parent;
            }
            catch(std::exception& err)
            {
                std::cout << "Error loading inherited config in " << filename << " with error " << err.what() << std::endl;
            }
        }

        cfg.load(js);
    }
    catch(std::exception& err)
    {
        cfg.name = filename.filename().string() + " (broken)";

        std::cout << "Error loading config " << filename << " with error " << err.what() << std::endl;
    }

    return cfg;
}

metrics::metric* metric_cache::lazy_fetch(content_manager& manage, content& c, const std::string& friendly_name)
{
    if(met == nullptr)
    {
        std::cout << "Serving up " << friendly_name << std::endl;

        std::optional<std::filesystem::path> path = c.lookup_path_to_metric_file(friendly_name);

        if(!path.has_value())
        {
            std::cout << "No metric found for " << friendly_name << std::endl;

            return nullptr;
        }

        std::cout << "Found path " << path.value().string() << std::endl;

        try
        {
            met = load_metric_from_script(manage, path.value());
        }
        catch(std::exception& err)
        {
            std::cout << "Error loading metric " << err.what() << std::endl;

            return nullptr;
        }
    }

    return met;
}

std::vector<std::string> load_sorting_file(std::filesystem::path filename)
{
    nlohmann::json js = nlohmann::json::parse(file::read(filename.string(), file::mode::TEXT));

    return js.get<std::vector<std::string>>();
}

std::vector<std::filesystem::path> get_files_with_extension(std::filesystem::path folder, const std::string& ext)
{
    std::vector<std::filesystem::path> ret;

    try
    {
        for(const auto& entry : std::filesystem::directory_iterator{folder})
        {
            if(entry.path().string().ends_with(ext))
            {
                ret.push_back(std::filesystem::absolute(entry));
            }
        }
    }
    catch(...)
    {
        std::cout << "No content " << folder << " for ext " << ext << std::endl;
    }

    return ret;
}

void content::load(content_manager& all_content, std::filesystem::path path)
{
    folder = path;

    std::filesystem::path coordinate = folder / std::filesystem::path("coordinates");
    std::filesystem::path origin = folder / std::filesystem::path("origins");

    metrics = get_files_with_extension(path, ".js");
    configs = get_files_with_extension(path, ".json");
    coordinates = get_files_with_extension(coordinate.string(), ".js");
    origins = get_files_with_extension(origin.string(), ".js");

    std::optional<std::filesystem::path> sorting;

    for(int i=0; i < (int)configs.size(); i++)
    {
        if(configs[i].filename() == "sorting.json")
        {
            sorting = configs[i];
            configs.erase(configs.begin() + i);
            break;
        }
    }

    ///this section handles the sorting file. It sorts specified metrics according to the input sorting.json
    ///unspecified metrics are sent to the end of the list
    try
    {
        if(sorting.has_value())
        {
            filename_sorting = load_sorting_file(sorting.value());

            std::set<std::string> metric_names(filename_sorting.begin(), filename_sorting.end());

            std::set<std::string> touched_metric_names;

            std::vector<std::filesystem::path> unsorted_elements = metrics;

            std::erase_if(unsorted_elements,
            [&metric_names, &touched_metric_names](const std::filesystem::path& in)
            {
                std::string filename = in.filename().string();

                ///kind of crappy
                touched_metric_names.insert(filename);

                return metric_names.find(filename) != metric_names.end();
            });

            std::vector<std::filesystem::path> reconstructed_metrics;

            for(const std::string& name : filename_sorting)
            {
                reconstructed_metrics.push_back(std::filesystem::absolute(path / name));
            }

            reconstructed_metrics.insert(reconstructed_metrics.end(), unsorted_elements.begin(), unsorted_elements.end());

            metrics = std::move(reconstructed_metrics);

            for(const std::string& s : metric_names)
            {
                if(!touched_metric_names.contains(s))
                {
                    std::cout << "Warning: Specific sorted metric name " << s << " is unused" << std::endl;
                }
            }
        }
    }
    catch(std::exception& e)
    {
        std::cout << "Error loading sorting file " << e.what() << std::endl;
    }

    for(const std::filesystem::path& cfg_name : configs)
    {
        metrics::metric_config cfg = load_config(all_content, cfg_name, false);

        base_configs.push_back(cfg);
    }
}

std::optional<std::filesystem::path> content::lookup_path_to_metric_file(const std::string& name)
{
    for(int i=0; i < (int)base_configs.size(); i++)
    {
        if(base_configs[i].name == name)
        {
            std::filesystem::path cfg_path = configs[i];

            cfg_path.replace_extension(".js");

            return cfg_path;
       }
    }

    return std::nullopt;
}

std::optional<metrics::metric_config*> content::get_config_of_filename(std::filesystem::path filename)
{
    assert(filename.extension().string() == ".js");

    filename.replace_extension(".json");

    for(int i=0; i < (int)configs.size(); i++)
    {
        if(configs[i] == filename)
        {
            return &base_configs[i];
        }
    }

    return std::nullopt;
}

metrics::metric* content::lazy_fetch(content_manager& manage, const std::string& friendly_name)
{
    return cache[friendly_name].lazy_fetch(manage, *this, friendly_name);
}

void content_manager::add_content_folder(std::filesystem::path folder)
{
    for(const content& c : content_directories)
    {
        if(c.folder == folder)
            return;
    }

    content con;
    con.load(*this, folder);

    content_directories.push_back(con);
}

std::optional<std::filesystem::path> content_manager::lookup_path_to_metric_file(const std::string& name)
{
    std::optional<std::filesystem::path> cfg = lookup_path_to_config_file(name);

    if(!cfg.has_value())
        return std::nullopt;

    std::filesystem::path met = cfg.value();
    met.replace_extension(".js");

    return met;
}

std::optional<std::filesystem::path> content_manager::lookup_path_to_config_file(const std::string& name)
{
    for(const content& c : content_directories)
    {
        for(int i=0; i < (int)c.base_configs.size(); i++)
        {
            if(c.base_configs[i].name == name)
                return c.configs[i];
        }
    }

    return std::nullopt;
}

std::optional<std::filesystem::path> content_manager::lookup_path_to_coordinates_file(const std::string& name)
{
    std::string name_as_file = name + ".js";

    for(const content& c : content_directories)
    {
        for(const std::filesystem::path& s : c.coordinates)
        {
            std::string fname = s.filename().string();

            if(fname == name_as_file)
                return s;

        }
    }

    return std::nullopt;
}

std::optional<std::filesystem::path> content_manager::lookup_path_to_origins_file(const std::string& name)
{
    std::string name_as_file = name + ".js";

    for(const content& c : content_directories)
    {
        for(const std::filesystem::path& s : c.origins)
        {
            std::string fname = s.filename().string();

            if(fname == name_as_file)
                return s;

        }
    }

    return std::nullopt;
}
