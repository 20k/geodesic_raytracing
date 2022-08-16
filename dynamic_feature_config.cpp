#include "dynamic_feature_config.hpp"
#include <sstream>
#include <iomanip>

namespace
{
    std::string to_string_s(float v)
    {
        std::ostringstream oss;
        oss << std::setprecision(32) << std::fixed << std::showpoint << v;
        std::string str = oss.str();

        while(str.size() > 0 && str.back() == '0')
            str.pop_back();

        if(str.size() > 0 && str.back() == '.')
            str += "0";

        return str;
    }

    template<typename T>
    std::string to_string_s(T in)
    {
        return std::to_string(in);
    }
}

bool valid_feature(const dynamic_feature_config& cfg, const std::string& feature)
{
    return cfg.features_enabled.find(feature) != cfg.features_enabled.end();
}

void dynamic_feature_config::add_feature_impl(const std::string& feature, const std::type_info& inf)
{
    is_dirty = true;

    if(inf == typeid(bool))
        features_enabled[feature] = bool();

    else if(inf == typeid(float))
        features_enabled[feature] = float();

    else
        assert(false);
}

void dynamic_feature_config::remove_feature(const std::string& feature)
{
    is_dirty = true;

    features_enabled.erase(features_enabled.find(feature));
}

bool dynamic_feature_config::is_enabled(const std::string& feature)
{
    assert(valid_feature(*this, feature));

    return std::get<0>(features_enabled[feature]);
}

void append_features(std::string& accum, const std::string& name, const std::vector<std::string>& names)
{
    if(names.size() == 0)
        return;

    accum += "-D" + name + "=";

    for(const std::string& name : names)
    {
        accum += name + ",";
    }

    if(accum.back() == ',')
        accum.pop_back();

    accum += " ";
}

template<typename T>
void append_feature_values(std::string& accum, const std::vector<std::pair<std::string, T>>& names)
{
    for(auto& [name, val] : names)
    {
        std::string val_as_string = to_string_s(val);

        accum += "-DFEATURE_" + name + "=" + val_as_string + " ";
    }
}

std::string dynamic_feature_config::generate_dynamic_argument_string()const
{
    ///so. On the OpenCL side I need to be able to query something, and have it not be a disaster
    ///OpenCL needs to be able to use one singular token, always
    ///preferably with an API like if(HAS_FEATURE(name, cfg))
    ///and GET_FEATURE_VALUE(name, cfg)
    ///whether we're in the static, or the dynamic kernel is a global flag
    ///if in a static kernel, HAS_FEATURE can return DEFFEATURE_##name
    ///if in a dynamic kernel, HAS_FEATURE can return cfg->name
    ///same as GET_FEATURE_VALUE, so actually no reason to reimpl
    std::string str = "-DKERNEL_IS_DYNAMIC ";

    std::vector<std::string> bool_features;
    std::vector<std::string> float_features;

    for(const auto& [name, val] : features_enabled)
    {
        if(val.index() == 0)
        {
            bool_features.push_back(name);
        }

        if(val.index() == 1)
        {
            float_features.push_back(name);
        }
    }

    append_features(str, "DYNAMIC_FLOAT_FEATURES", float_features);
    append_features(str, "DYNAMIC_BOOL_FEATURES", bool_features);

    return str;
}

std::string dynamic_feature_config::generate_static_argument_string() const
{
    std::string str = "-DKERNEL_IS_STATIC ";

    std::vector<std::pair<std::string, bool>> bool_features;
    std::vector<std::pair<std::string, float>> float_features;

    for(auto& [name, val] : features_enabled)
    {
        if(val.index() == 0)
        {
            bool_features.push_back({name, std::get<0>(val)});
        }

        if(val.index() == 1)
        {
            float_features.push_back({name, std::get<0>(val)});
        }
    }

    append_feature_values(str, float_features);
    append_feature_values(str, bool_features);

    return str;
}

void dynamic_feature_config::alloc_and_write_gpu_buffer(cl::command_queue& cqueue, cl::buffer& inout)
{
    if(!is_dirty)
        return;

    std::vector<std::pair<std::string, bool>> bool_features;
    std::vector<std::pair<std::string, float>> float_features;

    for(const auto& [name, val] : features_enabled)
    {
        if(val.index() == 0)
        {
            bool_features.push_back({name, std::get<0>(val)});
        }

        if(val.index() == 1)
        {
            float_features.push_back({name, std::get<0>(val)});
        }
    }

    int struct_size = sizeof(cl_float) * float_features.size() + sizeof(cl_bool) * bool_features.size();

    if(inout.alloc_size != struct_size)
        inout.alloc(struct_size);

    std::vector<cl_char> buf;
    buf.resize(struct_size);

    cl_char* buf_ptr = buf.data();

    for(int i=0; i < (int)float_features.size(); i++)
    {
        cl_float val = float_features[i].second;

        memcpy(buf_ptr, &val, sizeof(val));

        buf_ptr += sizeof(cl_float);
    }

    for(int i=0; i < (int)bool_features.size(); i++)
    {
        cl_bool val = bool_features[i].second;

        memcpy(buf_ptr, &val, sizeof(val));

        buf_ptr += sizeof(cl_bool);
    }

    inout.write(cqueue, buf);

    is_dirty = false;
}
