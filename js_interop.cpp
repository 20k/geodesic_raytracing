#include "js_interop.hpp"
#include <toolkit/fs_helpers.hpp>
#include <iostream>
#include "dual_value.hpp"

js::value to_class(js::value_context& vctx, js::value in)
{
    js::value global = js::get_global(vctx);

    const std::string name = "make_class";

    return js::call_prop(global, name, in).second;
}

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

dual get(js::value& v)
{
    if(v.is_number())
    {
        double val = v;

        return val;
    }

    if(v.is_boolean())
    {
        double val = (bool)v;

        return val;
    }

    san(v);

    return *v.get("v").get_ptr<dual>();
}

js::value to_value(js::value_context& vctx, dual in)
{
    js::value v(vctx);

    js::value as_object(vctx);
    as_object.allocate_in_heap(in);

    v.add("v", as_object);

    return to_class(vctx, v);
}

void construct(js::value_context* vctx, js::value js_this, js::value v2)
{
    if(v2.has("v"))
    {
        js_this.add("v", v2.get("v"));
    }
    else
    {
        double value = v2;

        dual as_dual = value;

        js::value val(*vctx);
        val.allocate_in_heap(as_dual);

        js_this.add("v", val);
    }
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

js::value sub(js::value_context* vctx, js::value v1, js::value v2)
{
    dual pv1 = get(v1);
    dual pv2 = get(v2);

    return to_value(*vctx, pv1 - pv2);
}

js::value jdiv(js::value_context* vctx, js::value v1, js::value v2)
{
    dual pv1 = get(v1);
    dual pv2 = get(v2);

    return to_value(*vctx, pv1 / pv2);
}

js::value neg(js::value_context* vctx, js::value v1)
{
    dual pv1 = get(v1);

    return to_value(*vctx, -pv1);
}

js::value lt(js::value_context* vctx, js::value v1, js::value v2)
{
    dual pv1 = get(v1);
    dual pv2 = get(v2);

    return to_value(*vctx, pv1 < pv2);
}

js::value eq(js::value_context* vctx, js::value v1, js::value v2)
{
    dual pv1 = get(v1);
    dual pv2 = get(v2);

    return to_value(*vctx, pv1 == pv2);
}

namespace CMath
{
    #define UNARY_JS(func) js::value func(js::value_context* vctx, js::value in) { \
                            dual v = get(in); \
                            return to_value(*vctx, func(v)); \
                           }

    #define BINARY_JS(func) js::value func(js::value_context* vctx, js::value in, js::value in2) { \
                            dual v = get(in); \
                            dual v2 = get(in); \
                            return to_value(*vctx, func(v, v2)); \
                           }

    UNARY_JS(sin);
    UNARY_JS(cos);
    UNARY_JS(tan);
    UNARY_JS(asin);
    UNARY_JS(acos);
    UNARY_JS(atan);
    BINARY_JS(atan2);

    UNARY_JS(fabs);
    UNARY_JS(log);

    js::value select(js::value_context* vctx, js::value condition, js::value if_true, js::value if_false)
    {
        dual dcondition = get(condition);
        dual dif_true = get(if_true);
        dual dif_false = get(if_false);

        dual selected = dual_if(dcondition.real, [&](){return dif_true;}, [&](){return dif_false;});

        return to_value(*vctx, selected);
    }
}

js::value extract_function(js::value_context& vctx, const std::string& script_data)
{
    std::string wrapper = file::read("./number.js", file::mode::TEXT);

    JS_AddIntrinsicBigFloat(vctx.ctx);
    JS_AddIntrinsicBigDecimal(vctx.ctx);
    JS_AddIntrinsicOperators(vctx.ctx);
    JS_EnableBignumExt(vctx.ctx, true);

    js::value cshim(vctx);

    js::add_key_value(cshim, "add", js::function<add>);
    js::add_key_value(cshim, "mul", js::function<mul>);
    js::add_key_value(cshim, "sub", js::function<sub>);
    js::add_key_value(cshim, "div", js::function<jdiv>);
    js::add_key_value(cshim, "neg", js::function<neg>);
    js::add_key_value(cshim, "lt", js::function<lt>);
    js::add_key_value(cshim, "eq", js::function<eq>);
    js::add_key_value(cshim, "construct", js::function<construct>);

    js::value global = js::get_global(vctx);

    js::add_key_value(global, "CShim", cshim);

    js::value cmath(vctx);

    js::add_key_value(cmath, "sin", js::function<CMath::sin>);
    js::add_key_value(cmath, "cos", js::function<CMath::cos>);
    js::add_key_value(cmath, "tan", js::function<CMath::tan>);
    js::add_key_value(cmath, "asin", js::function<CMath::asin>);
    js::add_key_value(cmath, "acos", js::function<CMath::acos>);
    js::add_key_value(cmath, "atan", js::function<CMath::atan>);
    js::add_key_value(cmath, "atan2", js::function<CMath::atan2>);
    js::add_key_value(cmath, "fabs", js::function<CMath::fabs>);
    js::add_key_value(cmath, "log", js::function<CMath::log>);
    js::add_key_value(cmath, "select", js::function<CMath::select>);

    js::add_key_value(global, "CMath", cmath);

    js::value result = js::eval(vctx, wrapper);

    std::cout << (std::string)result << std::endl;

    return js::eval(vctx, script_data);
}

js_metric::js_metric(const std::string& script_data) : vctx(nullptr, nullptr), func(vctx)
{
    func = extract_function(vctx, script_data);
}

std::array<dual, 16> js_metric::operator()(dual t, dual r, dual theta, dual phi)
{
    js::value v1 = to_value(vctx, t);
    js::value v2 = to_value(vctx, r);
    js::value v3 = to_value(vctx, theta);
    js::value v4 = to_value(vctx, phi);

    std::cout << (std::string)type_to_string(v2.get("v").get_ptr<dual>()->real) << std::endl;

    auto [success, result] = js::call(func, v1, v2, v3, v4);

    std::cout << "Res " << (std::string)result << std::endl;

    if(!success)
        throw std::runtime_error("Error in script exec " + (std::string)result.to_error_message());

    if(!result.is_array())
        throw std::runtime_error("Must return array");

    std::vector<js::value> values = result;

    if(values.size() == 4)
    {
        return {get(values[0]), 0, 0, 0,
                0, get(values[1]), 0, 0,
                0, 0, get(values[2]), 0,
                0, 0, 0, get(values[3])};
    }
    else
    {
        if(values.size() != 16)
            throw std::runtime_error("Must return array length of 4 or 16");

        return {get(values[0]), get(values[1]), get(values[2]), get(values[3]),
                get(values[4]), get(values[5]), get(values[6]), get(values[7]),
                get(values[8]), get(values[9]), get(values[10]),get(values[11]),
                get(values[12]),get(values[13]),get(values[14]),get(values[15])};
    }
}

js_function::js_function(const std::string& script_data) : vctx(nullptr, nullptr), func(vctx)
{
    func = extract_function(vctx, script_data);
}

std::array<dual, 4> js_function::operator()(dual iv1, dual iv2, dual iv3, dual iv4)
{
    js::value v1 = to_value(vctx, iv1);
    js::value v2 = to_value(vctx, iv2);
    js::value v3 = to_value(vctx, iv3);
    js::value v4 = to_value(vctx, iv4);

    auto [success, result] = js::call(func, v1, v2, v3, v4);

    if(!success)
        throw std::runtime_error("Error in script exec " + (std::string)result);

    if(!result.is_array())
        throw std::runtime_error("Must return array");

    std::vector<js::value> values = result;

    if(values.size() != 4)
        throw std::runtime_error("Must return array size of 4");

    return {get(values[0]), get(values[1]), get(values[2]), get(values[3])};
}

js_single_function::js_single_function(const std::string& script_data) : vctx(nullptr, nullptr), func(vctx)
{
    func = extract_function(vctx, script_data);
}

dual js_single_function::operator()(dual iv1, dual iv2, dual iv3, dual iv4)
{
    js::value v1 = to_value(vctx, iv1);
    js::value v2 = to_value(vctx, iv2);
    js::value v3 = to_value(vctx, iv3);
    js::value v4 = to_value(vctx, iv4);

    auto [success, result] = js::call(func, v1, v2, v3, v4);

    if(!success)
        throw std::runtime_error("Error in script exec " + (std::string)result);

    return {get(result)};
}
