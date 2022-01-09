#include "js_interop.hpp"
#include <toolkit/fs_helpers.hpp>
#include <iostream>

namespace js = js_quickjs;

js::value add(js::value_context* vctx, js::value v1, js::value v2)
{
    return js::value(*vctx);
}

std::string js_argument_string(const std::string& script_data)
{
    std::string wrapper = file::read("./number.js",file::mode::TEXT);

    js::value_context vctx(nullptr, nullptr);

    JS_AddIntrinsicBigFloat(vctx.ctx);
    JS_AddIntrinsicBigDecimal(vctx.ctx);
    JS_AddIntrinsicOperators(vctx.ctx);
    JS_EnableBignumExt(vctx.ctx, true);

    js::value cshim(vctx);

    js::add_key_value(cshim, "add", js::function<add>);

    js::value global = js::get_global(vctx);

    js::add_key_value(global, "CShim", cshim);

    js::value result = js::eval(vctx, wrapper);

    std::cout << (std::string)result << std::endl;

    return "";
}
