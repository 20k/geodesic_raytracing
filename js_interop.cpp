#include "js_interop.hpp"
#include <toolkit/fs_helpers.hpp>
#include <iostream>
#include "dual_value.hpp"

namespace js = js_quickjs;

void san(js::value& v)
{
    if(!v.has("v"))
    {
        dual d;

        js::value next(*v.vctx);

        next.allocate_in_heap(d);

        v.add("v", next);
    }
}

dual& get(js::value& v)
{
    san(v);

    return *v.get("v").get_ptr<dual>();
}

js::value to_value(js::value_context& vctx, dual in)
{
    js::value v(vctx);

    js::value as_object(vctx);
    as_object.allocate_in_heap(in);

    v.add("v", as_object);

    return v;
}

js::value add(js::value_context* vctx, js::value v1, js::value v2)
{
    dual pv1 = get(v1);
    dual pv2 = get(v2);

    return to_value(*vctx, pv1 + pv2);
}

js::value mul(js::value_context* vctx, js::value v1, js::value v2)
{
    dual pv1 = get(v1);
    dual pv2 = get(v2);

    return to_value(*vctx, pv1 * pv2);
}

namespace CMath
{
    js::value sin(js::value_context* vctx, js::value in)
    {
        dual v = get(in);

        return to_value(*vctx, sin(v));
    }
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
    js::add_key_value(cshim, "mul", js::function<mul>);

    js::value global = js::get_global(vctx);

    js::add_key_value(global, "CShim", cshim);

    js::value cmath(vctx);

    js::add_key_value(cmath, "sin", js::function<CMath::sin>);

    js::add_key_value(global, "CMath", cmath);

    js::value result = js::eval(vctx, wrapper);

    std::cout << (std::string)result << std::endl;

    js::value schwarz_result = js::eval(vctx, file::read("./schwarzschild.js", file::mode::TEXT));

    std::cout << (std::string)schwarz_result << std::endl;

    return "";
}
