#ifndef JS_INTEROP_HPP_INCLUDED
#define JS_INTEROP_HPP_INCLUDED

#include <quickjs_cpp/quickjs_cpp.hpp>
#include <string>
#include "dual_value.hpp"

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

#endif // JS_INTEROP_HPP_INCLUDED
