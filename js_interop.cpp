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

struct storage
{
    int which = 0;

    dual d;
    dual_complex c;

    storage(){}

    storage(const dual& d_in)
    {
        d = d_in;
        which =  0;
    }

    storage(const dual_complex& c_in)
    {
        c = c_in;
        which = 1;
    }
};

template<typename T>
storage func(const storage& s1, T&& functor)
{
    if(s1.which == 0)
    {
        storage s(functor(s1.d));

        return s;
    }
    else if(s1.which == 1)
    {
        storage s(functor(s1.c));

        return s;
    }
    else
    {
        throw std::runtime_error("Invalid type in dual/complex internals");
    }
}
template<typename T>
storage func_real(const storage& s1, T&& functor)
{
    if(s1.which == 0)
    {
        storage s(functor(s1.d));

        return s;
    }
    else
    {
        throw std::runtime_error("Invalid type in dual/complex internals, this is a real-only function");
    }
}

template<typename T>
storage func(const storage& s1, const storage& s2, T&& functor)
{
    if(s1.which == 0 && s2.which == 0)
    {
        storage s(functor(s1.d, s2.d));

        return s;
    }
    else if(s1.which == 0 && s2.which == 1)
    {
        storage s(functor(s1.d, s2.c));

        return s;
    }
    else if(s1.which == 1 && s2.which == 0)
    {
        storage s(functor(s1.c, s2.d));

        return s;
    }
    else if(s1.which == 1 && s2.which == 1)
    {
        storage s(functor(s1.c, s2.c));

        return s;
    }
    else
    {
        throw std::runtime_error("Invalid types in dual/complex internals");
    }
}

storage s_add(const storage& s1, const storage& s2)
{
    return func(s1, s2, [](auto v1, auto v2)
    {
        return v1 + v2;
    });
}

storage s_sub(const storage& s1, const storage& s2)
{
    return func(s1, s2, [](auto v1, auto v2)
    {
        return v1 - v2;
    });
}

storage s_mul(const storage& s1, const storage& s2)
{
    return func(s1, s2, [](auto v1, auto v2)
    {
        return v1 * v2;
    });
}

storage s_div(const storage& s1, const storage& s2)
{
    return func(s1, s2, [](auto v1, auto v2)
    {
        return v1 / v2;
    });
}

storage s_neg(const storage& s1)
{
    return func(s1, [](auto v1)
    {
        return -v1;
    });
}

storage s_lt(const storage& s1, const storage& s2)
{
    if(s1.which == 0 && s2.which == 0)
        return (dual)(s1.d < s2.d);
    else
        throw std::runtime_error("Can only use < on complex values");
}

storage s_eq(const storage& s1, const storage& s2)
{
    if(s1.which == 0 && s2.which == 0)
        return (dual)(s1.d == s2.d);
    else
        throw std::runtime_error("Can only use < on complex values");
}

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
        storage s;

        js::value next(*v.vctx);

        next.allocate_in_heap(s);

        v.add("v", next);
    }
}

storage get(js::value& v)
{
    if(v.is_undefined())
        return storage();

    if(v.is_number())
    {
        double val = v;

        storage s;
        s.d = val;

        return s;
    }

    if(v.is_boolean())
    {
        double val = (bool)v;

        storage s;
        s.d = val;

        return s;
    }

    san(v);

    return *v.get("v").get_ptr<storage>();
}

js::value to_value(js::value_context& vctx, dual in)
{
    storage s;
    s.d = in;

    js::value v(vctx);

    js::value as_object(vctx);
    as_object.allocate_in_heap(s);

    v.add("v", as_object);

    return to_class(vctx, v);
}

js::value to_value(js::value_context& vctx, dual_complex in)
{
    storage s;
    s.which = 1;
    s.c = in;

    js::value v(vctx);

    js::value as_object(vctx);
    as_object.allocate_in_heap(s);

    v.add("v", as_object);

    return to_class(vctx, v);
}

js::value to_value(js::value_context& vctx, storage in)
{
    js::value v(vctx);

    js::value as_object(vctx);
    as_object.allocate_in_heap(in);

    v.add("v", as_object);

    return to_class(vctx, v);
}

void construct(js::value_context* vctx, js::value js_this, js::value v2)
{
    storage v = get(v2);

    js::value as_object(*vctx);
    as_object.allocate_in_heap(v);

    js_this.add("v", as_object);
}

js::value add(js::value_context* vctx, js::value v1, js::value v2)
{
    storage pv1 = get(v1);
    storage pv2 = get(v2);

    return to_value(*vctx, s_add(pv1, pv2));
}

js::value mul(js::value_context* vctx, js::value v1, js::value v2)
{
    storage pv1 = get(v1);
    storage pv2 = get(v2);

    return to_value(*vctx, s_mul(pv1, pv2));
}

js::value sub(js::value_context* vctx, js::value v1, js::value v2)
{
    storage pv1 = get(v1);
    storage pv2 = get(v2);

    return to_value(*vctx, s_sub(pv1, pv2));
}

js::value jdiv(js::value_context* vctx, js::value v1, js::value v2)
{
    storage pv1 = get(v1);
    storage pv2 = get(v2);

    return to_value(*vctx, s_div(pv1, pv2));
}

js::value neg(js::value_context* vctx, js::value v1)
{
    storage pv1 = get(v1);

    return to_value(*vctx, s_neg(pv1));
}

js::value lt(js::value_context* vctx, js::value v1, js::value v2)
{
    storage pv1 = get(v1);
    storage pv2 = get(v2);

    return to_value(*vctx, s_lt(pv1, pv2));
}

js::value eq(js::value_context* vctx, js::value v1, js::value v2)
{
    storage pv1 = get(v1);
    storage pv2 = get(v2);

    return to_value(*vctx, s_eq(pv1, pv2));
}

namespace CMath
{
    #define UNARY_JS(name) js::value name(js::value_context* vctx, js::value in) { \
                            storage v = get(in); \
                            auto result = func(v, [](auto in){return name(in);}); \
                            return to_value(*vctx, result); \
                           }

    #define UNARY_JS_REAL(name) js::value name(js::value_context* vctx, js::value in) { \
                            storage v = get(in); \
                            auto result = func_real(v, [](auto in){return name(in);}); \
                            return to_value(*vctx, result); \
                           }

    #define BINARY_JS(name) js::value name(js::value_context* vctx, js::value in, js::value in2) { \
                            dual v = get(in).d; \
                            dual v2 = get(in2).d; \
                            return to_value(*vctx, dual_types::name(v, v2)); \
                           }

    #define TERNARY_JS(name) js::value name(js::value_context* vctx, js::value in, js::value in2, js::value in3) { \
                             dual v = get(in).d; \
                             dual v2 = get(in2).d; \
                             dual v3 = get(in3).d; \
                             return to_value(*vctx, dual_types::name(v, v2, v3)); \
                            }

    UNARY_JS(sin);
    UNARY_JS(cos);
    UNARY_JS_REAL(tan);
    UNARY_JS_REAL(asin);
    UNARY_JS_REAL(acos);
    UNARY_JS_REAL(atan);
    BINARY_JS(atan2);
    BINARY_JS(pow);

    UNARY_JS(fabs);
    UNARY_JS_REAL(log);
    UNARY_JS(sqrt);
    UNARY_JS(psqrt);
    UNARY_JS_REAL(exp);

    UNARY_JS_REAL(sinh);
    UNARY_JS_REAL(cosh);
    UNARY_JS_REAL(tanh);

    TERNARY_JS(fast_length);

    js::value select(js::value_context* vctx, js::value condition, js::value if_true, js::value if_false)
    {
        dual dcondition = get(condition).d;
        dual dif_true = get(if_true).d;
        dual dif_false = get(if_false).d;

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
    js::add_key_value(cmath, "psqrt", js::function<CMath::psqrt>);
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

    if(func.is_error() || func.is_exception())
    {
        std::cout << "Function object error " << func.to_error_message() << std::endl;
        throw std::runtime_error("Err");
    }

    auto [success, result] = js::call(func, v1, v2, v3, v4);

    if(!success)
        throw std::runtime_error("Error in script exec " + (std::string)result.to_error_message());

    if(!result.is_array())
        throw std::runtime_error("Must return array");

    std::vector<js::value> values = result;

    if(values.size() == 4)
    {
        return {get(values[0]).d, 0, 0, 0,
                0, get(values[1]).d, 0, 0,
                0, 0, get(values[2]).d, 0,
                0, 0, 0, get(values[3]).d};
    }
    else
    {
        if(values.size() != 16)
            throw std::runtime_error("Must return array length of 4 or 16");

        return {get(values[0]).d, get(values[1]).d, get(values[2]).d, get(values[3]).d,
                get(values[4]).d, get(values[5]).d, get(values[6]).d, get(values[7]).d,
                get(values[8]).d, get(values[9]).d, get(values[10]).d,get(values[11]).d,
                get(values[12]).d,get(values[13]).d,get(values[14]).d,get(values[15]).d};
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

    return {get(values[0]).d, get(values[1]).d, get(values[2]).d, get(values[3]).d};
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

    return {get(result).d};
}
