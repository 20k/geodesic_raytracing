#include "js_interop.hpp"
#include <toolkit/fs_helpers.hpp>
#include <iostream>
#include "dual_value.hpp"

#define M_E		2.7182818284590452354
#define M_LOG2E		1.4426950408889634074
#define M_LOG10E	0.43429448190325182765
#define M_LN2		0.69314718055994530942
#define M_LN10		2.30258509299404568402
#define M_PI		3.14159265358979323846
#define M_PI_2		1.57079632679489661923
#define M_PI_4		0.78539816339744830962
#define M_1_PI		0.31830988618379067154
#define M_2_PI		0.63661977236758134308
#define M_2_SQRTPI	1.12837916709551257390
#define M_SQRT2		1.41421356237309504880
#define M_SQRT1_2	0.70710678118654752440

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
    if(v.is_undefined())
        return dual();

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
    dual v = get(v2);

    js::value as_object(*vctx);
    as_object.allocate_in_heap(v);

    js_this.add("v", as_object);
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
                            return to_value(*vctx, dual_types::func(v)); \
                           }

    #define BINARY_JS(func) js::value func(js::value_context* vctx, js::value in, js::value in2) { \
                            dual v = get(in); \
                            dual v2 = get(in2); \
                            return to_value(*vctx, dual_types::func(v, v2)); \
                           }

    #define TERNARY_JS(func) js::value func(js::value_context* vctx, js::value in, js::value in2, js::value in3) { \
                             dual v = get(in); \
                             dual v2 = get(in2); \
                             dual v3 = get(in3); \
                             return to_value(*vctx, dual_types::func(v, v2, v3)); \
                            }

    UNARY_JS(sin);
    UNARY_JS(cos);
    UNARY_JS(tan);
    UNARY_JS(asin);
    UNARY_JS(acos);
    UNARY_JS(atan);
    BINARY_JS(atan2);
    BINARY_JS(pow);

    UNARY_JS(fabs);
    UNARY_JS(log);
    UNARY_JS(sqrt);
    UNARY_JS(exp);

    UNARY_JS(sinh);
    UNARY_JS(cosh);
    UNARY_JS(tanh);

    TERNARY_JS(fast_length);

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
    js::add_key_value(cmath, "pow", js::function<CMath::pow>);
    js::add_key_value(cmath, "sqrt", js::function<CMath::sqrt>);
    js::add_key_value(cmath, "exp", js::function<CMath::exp>);
    js::add_key_value(cmath, "fast_length", js::function<CMath::fast_length>);
    js::add_key_value(cmath, "length", js::function<CMath::fast_length>);

    js::add_key_value(cmath, "sinh", js::function<CMath::sinh>);
    js::add_key_value(cmath, "cosh", js::function<CMath::cosh>);
    js::add_key_value(cmath, "tanh", js::function<CMath::tanh>);

    js::add_key_value(cmath, "M_PI", js::make_value(vctx, M_PI));
    js::add_key_value(cmath, "PI", js::make_value(vctx, M_PI));

    js::add_key_value(global, "CMath", cmath);

    js::add_key_value(global, "M_PI", js::make_value(vctx, M_PI));

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
