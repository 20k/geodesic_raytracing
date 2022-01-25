#ifndef JS_INTEROP_HPP_INCLUDED
#define JS_INTEROP_HPP_INCLUDED

#include <quickjs_cpp/quickjs_cpp.hpp>
#include <string>
#include "dual_value.hpp"
#include <compare>
#include "equation_context.hpp"

namespace js = js_quickjs;

struct config_variables
{
    std::vector<std::string> names;
    std::vector<float> default_values;
    std::vector<float> current_values;

    void add(const std::string& name, float val);
    void set_default(const std::string& name, float val);
    bool display();
};

struct sandbox
{
    config_variables cfg;
    equation_context ctx;
};

struct js_metric
{
    js::value_context vctx;
    js::value func;

    js_metric(sandbox& sand, const std::string& script);

    std::array<dual, 16> operator()(dual t, dual r, dual theta, dual phi);
};

struct js_function
{
    js::value_context vctx;
    js::value func;

    js_function(sandbox& sand, const std::string& script);

    std::array<dual, 4> operator()(dual v1, dual v2, dual v3, dual v4);
};

struct js_single_function
{
    js::value_context vctx;
    js::value func;

    js_single_function(sandbox& sand, const std::string& script);

    dual operator()(dual v1, dual v2, dual v3, dual v4);
};

#endif // JS_INTEROP_HPP_INCLUDED
