#ifndef DYNAMIC_FEATURE_CONFIG_HPP_INCLUDED
#define DYNAMIC_FEATURE_CONFIG_HPP_INCLUDED

#include <map>
#include <string>
#include <toolkit/opencl.hpp>

///this class provides a gpu compatible struct that's dynamically generated
///but also provides defines so that the struct may be bypassed
///two kernels then get generated: One using the struct, and one using the hardcoded configs
///there are now a *lot* of cases where functionality like this is necessary, and the perf tradeoff is unacceptable
struct dynamic_feature_config
{
    bool is_dirty = false;
    std::map<std::string, bool> features_enabled;

    void add_feature(const std::string& feature);

    bool is_enabled(const std::string& feature);
    void enable(const std::string& feature);
    void disable(const std::string& feature);

    std::string generate_dynamic_argument_string();
    std::string generate_static_argument_string();
    void alloc_and_write_gpu_buffer(cl::buffer& in, cl::command_queue& cqueue);
};

#endif // DYNAMIC_FEATURE_CONFIG_HPP_INCLUDED
