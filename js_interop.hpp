#ifndef JS_INTEROP_HPP_INCLUDED
#define JS_INTEROP_HPP_INCLUDED

#include <quickjs_cpp/quickjs_cpp.hpp>
#include <string>
#include "dual_value.hpp"

namespace js = js_quickjs;

struct js_function
{
    js::value_context vctx;
    js::value func;

    js_function(const std::string& script);

    std::array<dual, 16> operator()(dual t, dual r, dual theta, dual phi);
};

#endif // JS_INTEROP_HPP_INCLUDED
