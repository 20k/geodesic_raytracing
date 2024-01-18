#ifndef DYNAMIC_FEATURE_CONFIG_HPP_INCLUDED
#define DYNAMIC_FEATURE_CONFIG_HPP_INCLUDED

#include <map>
#include <set>
#include <string>
#include <toolkit/opencl.hpp>
#include <variant>
#include <typeinfo>

///this class provides a gpu compatible struct that's dynamically generated
///but also provides defines so that the struct may be bypassed
///two kernels then get generated: One using the struct, and one using the hardcoded configs
///there are now a *lot* of cases where functionality like this is necessary, and the perf tradeoff is unacceptable
struct dynamic_feature_config
{
    bool is_dirty = false;
    bool is_static_dirty = false;
    std::map<std::string, std::variant<bool, float>> features_enabled;
    std::set<std::string> always_static;

    template<typename T>
    void add_feature(const std::string& feature)
    {
        add_feature_impl(feature, typeid(T));
    }

    void remove_feature(const std::string& feature);

    bool is_enabled(const std::string& feature);

    template<typename T>
    void set_feature(const std::string& feature, const T& val)
    {
        assert(features_enabled.find(feature) != features_enabled.end());

        if(val != std::get<T>(features_enabled[feature]))
        {
            is_dirty = true;

            if(always_static.count(feature) == 1)
                is_static_dirty = true;
        }

        features_enabled[feature] = val;
    }

    void set_always_static(const std::string& feature, bool val);

    template<typename T>
    T get_feature(const std::string& feature)
    {
        assert(features_enabled.find(feature) != features_enabled.end());

        return std::get<T>(features_enabled[feature]);
    }

    std::string generate_dynamic_argument_string() const;
    std::string generate_static_argument_string() const;
    bool alloc_and_write_gpu_buffer(cl::command_queue& cqueue, cl::buffer& inout);

private:
    void add_feature_impl(const std::string& feature, const std::type_info& inf);
};

//void alloc_from_dfg(dynamic_feature_config& dfg, cl::command_queue& cqueue, cl::buffer& inout);

#endif // DYNAMIC_FEATURE_CONFIG_HPP_INCLUDED
