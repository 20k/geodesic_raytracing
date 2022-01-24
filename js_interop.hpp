#ifndef JS_INTEROP_HPP_INCLUDED
#define JS_INTEROP_HPP_INCLUDED

#include <quickjs_cpp/quickjs_cpp.hpp>
#include <string>
#include "dual_value.hpp"
#include <compare>

namespace js = js_quickjs;

struct js_metric
{
    js::value_context vctx;
    js::value func;

    js_metric(const std::string& script);

    std::array<dual, 16> operator()(dual t, dual r, dual theta, dual phi);
};

struct js_function
{
    js::value_context vctx;
    js::value func;

    js_function(const std::string& script);

    std::array<dual, 4> operator()(dual v1, dual v2, dual v3, dual v4);
};

struct js_single_function
{
    js::value_context vctx;
    js::value func;

    js_single_function(const std::string& script);

    dual operator()(dual v1, dual v2, dual v3, dual v4);
};

struct config_variable
{
    std::string name;
    double default_value = 0;

    auto operator<=>(const config_variable&) const = default;
};

std::vector<config_variable> pull_configs(js::value_context& vctx);

#endif // JS_INTEROP_HPP_INCLUDED
