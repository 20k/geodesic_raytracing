#include "js_interop.hpp"
#include <toolkit/fs_helpers.hpp>
#include <iostream>

std::string js_argument_string(const std::string& script_data)
{
    namespace js = js_quickjs;

    std::string wrapper = file::read("./number.js",file::mode::TEXT);

    js::value_context vctx(nullptr, nullptr);

    JS_AddIntrinsicBigFloat(vctx.ctx);
    JS_AddIntrinsicBigDecimal(vctx.ctx);
    JS_AddIntrinsicOperators(vctx.ctx);
    JS_EnableBignumExt(vctx.ctx, true);

    js::value result = js::eval(vctx, wrapper);

    std::cout << (std::string)result << std::endl;

    return "";
}
