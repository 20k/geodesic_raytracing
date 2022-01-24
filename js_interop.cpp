#include "js_interop.hpp"
#include <toolkit/fs_helpers.hpp>
#include <iostream>
#include "dual_value.hpp"
#include <cctype>
#include <imgui/imgui.h>
#include <toolkit/clock.hpp>

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

struct timer
{
    steady_timer t;
    std::string msg;

    timer(std::string in = "")
    {
        msg = in;
    }

    ~timer()
    {
        double elapsed = t.get_elapsed_time_s();

        printf("Elapsed %s %f\n", msg.c_str(), elapsed);
    }
};

void config_variables::add(const std::string& name, float val)
{
    for(const std::string& existing : names)
    {
        if(name == existing)
            return;
    }

    names.push_back(name);
    default_values.push_back(val);
    current_values.push_back(val);
}

void config_variables::set_default(const std::string& name, float val)
{
    for(int i=0; i < (int)names.size(); i++)
    {
        const std::string& existing = names[i];

        if(name == existing)
        {
            default_values[i] = val;
            current_values[i] = val;
            return;
        }
    }

    names.push_back(name);
    default_values.push_back(val);
    current_values.push_back(val);
}

bool config_variables::display()
{
    bool any_modified = false;

    for(int i=0; i < (int)names.size(); i++)
    {
        any_modified |= ImGui::DragFloat((names[i] + "##df").c_str(), &current_values[i], 0.1f);
    }

    return any_modified;
}

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

    storage(dual&& d_in)
    {
        d = std::move(d_in);
        which = 0;
    }

    storage(const dual_complex& c_in)
    {
        c = c_in;
        which = 1;
    }

    storage(dual_complex&& c_in)
    {
        c = std::move(c_in);
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
        return functor(s1.d, s2.d);
    }
    else if(s1.which == 0 && s2.which == 1)
    {
        return functor(s1.d, s2.c);
    }
    else if(s1.which == 1 && s2.which == 0)
    {
        return functor(s1.c, s2.d);
    }
    else if(s1.which == 1 && s2.which == 1)
    {
        return functor(s1.c, s2.c);
    }
    else
    {
        throw std::runtime_error("Invalid types in dual/complex internals");
    }
}

storage s_add(const storage& s1, const storage& s2)
{
    timer t("add");

    return func(s1, s2, [](const auto& v1, const auto& v2)
    {
        return v1 + v2;
    });
}

storage s_sub(const storage& s1, const storage& s2)
{
    timer t("sub");

    return func(s1, s2, [](const auto& v1, const auto& v2)
    {
        return v1 - v2;
    });
}

storage s_mul(const storage& s1, const storage& s2)
{
    timer t("mul");

    return func(s1, s2, [](const auto& v1, const auto& v2)
    {
        return v1 * v2;
    });
}

storage s_div(const storage& s1, const storage& s2)
{
    timer t("div");

    return func(s1, s2, [](const auto& v1, const auto& v2)
    {
        return v1 / v2;
    });
}

storage s_neg(const storage& s1)
{
    timer t("neg");

    return func(s1, [](const auto& v1)
    {
        return -v1;
    });
}

storage s_lt(const storage& s1, const storage& s2)
{
    timer t("lt");

    if(s1.which == 0 && s2.which == 0)
        return (dual)(s1.d < s2.d);
    else
        throw std::runtime_error("Can only use < on real values");
}

storage s_eq(const storage& s1, const storage& s2)
{
    timer t("eq");

    if(s1.which == 0 && s2.which == 0)
        return (dual)(s1.d == s2.d);
    else
        throw std::runtime_error("Can only use == on real values");
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

dual getr(js::value& v)
{
    storage s = get(v);

    if(s.which == 0)
        return s.d;
    else if(s.which == 1)
        return Real(s.c);
    else
        throw std::runtime_error("getr which");
}

js::value to_value(js::value_context& vctx, const dual& in)
{
    storage s;
    s.d = in;

    js::value v(vctx);

    js::value as_object(vctx);
    as_object.allocate_in_heap(s);

    v.add("v", as_object);

    return to_class(vctx, v);
}

js::value to_value(js::value_context& vctx, const dual_complex& in)
{
    storage s;
    s.which = 1;
    s.c = in;

    js::value v(vctx);

    js::value as_object(vctx);
    as_object.allocate_in_heap(s);

    v.add("v", as_object);

    assert(v.has("v"));

    return to_class(vctx, v);
}

js::value to_value(js::value_context& vctx, const storage& in)
{
    js::value v(vctx);

    storage* ptr = new storage(in);

    js::value as_object(vctx);
    as_object.set_ptr(ptr);

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
    void debug(js::value_context* vctx, js::value v);

    #define UNARY_JS(name) js::value name(js::value_context* vctx, js::value in) { \
                            storage v = get(in); \
                            auto result = func(v, [](const auto& in){return name(in);}); \
                            return to_value(*vctx, result); \
                           }

    #define UNARY_JS_REAL(name) js::value name(js::value_context* vctx, js::value in) { \
                            storage v = get(in); \
                            auto result = func_real(v, [](const auto& in){return name(in);}); \
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
    //BINARY_JS(pow);

    UNARY_JS(fabs);
    UNARY_JS_REAL(log);
    UNARY_JS(sqrt);
    UNARY_JS(psqrt);
    UNARY_JS_REAL(exp);

    UNARY_JS_REAL(sinh);
    UNARY_JS_REAL(cosh);
    UNARY_JS_REAL(tanh);

    UNARY_JS(conjugate);
    UNARY_JS(self_conjugate_multiply);;

    TERNARY_JS(fast_length);

    js::value select(js::value_context* vctx, js::value condition, js::value if_true, js::value if_false)
    {
        dual dcondition = get(condition).d;
        dual dif_true = get(if_true).d;
        dual dif_false = get(if_false).d;

        dual selected = dual_if(dcondition.real, [&](){return dif_true;}, [&](){return dif_false;});

        return to_value(*vctx, selected);
    }

    js::value Real(js::value_context* vctx, js::value v)
    {
        storage s = get(v);

        if(s.which == 0)
            return to_value(*vctx, s.d);
        else if(s.which == 1)
            return to_value(*vctx, Real(s.c));
        else
            throw std::runtime_error("Some kind of weird error in Real");
    }

    js::value Imaginary(js::value_context* vctx, js::value v)
    {
        storage s = get(v);

        if(s.which == 0)
            return to_value(*vctx, (dual)(0.f));
        else if(s.which == 1)
            return to_value(*vctx, Imaginary(s.c));
        else
            throw std::runtime_error("Some kind of weird error in Imaginary");
    }

    js::value csqrt(js::value_context* vctx, js::value v)
    {
        storage s = get(v);

        if(s.which == 0)
            return to_value(*vctx, csqrt(s.d));
        else if(s.which == 1)
            throw std::runtime_error("csqrt must be used with purely real arguments");
        else
            throw std::runtime_error("Some kind of weird error in Imaginary");
    }

    js::value pow(js::value_context* vctx, js::value in1, js::value in2)
    {
        storage s1 = get(in1);
        storage s2 = get(in2);

        if(s2.which != 0)
            throw std::runtime_error("Pow cannot be used with a complex second argument");

        if(!s2.d.real.is_constant())
            throw std::runtime_error("Pow's second argument must be constant");

        double exponent = s2.d.real.get_constant();

        if(s1.which == 0)
        {
            return to_value(*vctx, pow(s1.d, exponent));
        }

        if(s1.which == 1)
        {
            if(exponent != (int)exponent)
                throw std::runtime_error("With a complex first argument, the exponent must be integral");

            return to_value(*vctx, pow(s1.c, (int)exponent));
        }

        throw std::runtime_error("Cannot call pow with two complex arguments");
    }

    void debug(js::value_context* vctx, js::value v)
    {
        if(v.is_string())
        {
            std::cout << (std::string)v << std::endl;
            return;
        }

        storage base = get(v);

        if(base.which == 0)
        {
            std::cout << "base " << type_to_string(base.d.real) << std::endl;
        }
        else
        {
            std::cout << "Basi " << type_to_string(base.c.real.real) << " I " << type_to_string(base.c.real.imaginary) << std::endl;
        }

        std::cout << "well hello there " << std::endl;

        js::value real_part = Real(vctx, v);

        std::cout << "Hello1\n";

        storage s2 = get(real_part);

        std::cout << "Hello2\n";

        js::value imaginary = Imaginary(vctx, v);

        std::cout << "Hello3\n";

        storage s3 = get(imaginary);

        std::cout << "Val " << type_to_string(s2.d.real) << std::endl;
        std::cout << "Vali " << type_to_string(s3.c.real.imaginary) << std::endl;
    }
}

js::value get_unit_i(js::value_context* vctx)
{
    dual_complex i = dual_types::unit_i();

    return to_value(*vctx, i);
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
    js::add_key_value(cmath, "csqrt", js::function<CMath::csqrt>);
    js::add_key_value(cmath, "exp", js::function<CMath::exp>);
    js::add_key_value(cmath, "fast_length", js::function<CMath::fast_length>);
    js::add_key_value(cmath, "length", js::function<CMath::fast_length>);

    js::add_key_value(cmath, "sinh", js::function<CMath::sinh>);
    js::add_key_value(cmath, "cosh", js::function<CMath::cosh>);
    js::add_key_value(cmath, "tanh", js::function<CMath::tanh>);

    js::add_key_value(cmath, "M_PI", js::make_value(vctx, M_PI));
    js::add_key_value(cmath, "PI", js::make_value(vctx, M_PI));

    js::add_key_value(cmath, "conjugate", js::function<CMath::conjugate>);
    js::add_key_value(cmath, "self_conjugate_multiply", js::function<CMath::self_conjugate_multiply>);

    js::add_key_value(cmath, "Real", js::function<CMath::Real>);
    js::add_key_value(cmath, "Imaginary", js::function<CMath::Imaginary>);

    js::add_key_value(cmath, "debug", js::function<CMath::debug>);

    js::add_key_value(cmath, "get_i", js::function<get_unit_i>);

    js::add_key_value(global, "CMath", cmath);

    js::add_key_value(global, "M_PI", js::make_value(vctx, M_PI));

    js::value result = js::eval(vctx, wrapper);

    std::cout << (std::string)result << std::endl;

    return js::eval(vctx, script_data);
}

std::pair<js::value, js::value> get_proxy_handlers(js::value_context& vctx)
{
    js::value dummy_func = js::make_value(vctx, js::function<js::empty_function>);
    js::value dummy_obj(vctx);

    return {dummy_func, dummy_obj};
}

void validate(const std::string& in)
{
    for(char c : in)
    {
        if(!std::isalnum(c))
            throw std::runtime_error("Value must be alphanumeric");
    }
}

js::value setter_set_default(js::value_context* vctx, js::value value)
{
    js::value object = js::get_this(*vctx);

    storage s = get(object);

    if(s.which != 0)
        throw std::runtime_error("Something really weirds happened in setter_set_default");

    if(!s.d.real.value_payload.has_value())
        throw std::runtime_error("Must be pseudoconstant value in $default set");

    std::string name = s.d.real.value_payload.value();

    if(name.starts_with("cfg->"))
    {
        for(int i=0; i < strlen("cfg->"); i++)
            name.erase(name.begin());
    }

    validate(name);

    float valf = (double)value;

    config_variables* sandbox = js::get_sandbox_data<config_variables>(*vctx);

    assert(sandbox);

    sandbox->set_default(name, valf);

    return js::make_value(*vctx, 0.f);
}

js::value cfg_proxy_get(js::value_context* vctx, js::value target, js::value prop, js::value receiver)
{
    std::string key = prop;

    validate(key);

    config_variables* sandbox = js::get_sandbox_data<config_variables>(*vctx);

    assert(sandbox);

    sandbox->add(key, 0.f);

    dual v;
    v.make_constant("cfg->" + key);

    js::value result = to_value(*vctx, v);

    js::add_getter_setter(result, "$default", js::function<js::empty_function>, js::function<setter_set_default>);

    return result;
}

js::value cfg_proxy_set(js::value_context* vctx, js::value target, js::value prop, js::value val, js::value receiver)
{
    std::string key = prop;

    std::cout << "Warning, setting a config from js" << std::endl;

    return js::make_success(*vctx);
}

js::value finish_proxy(js::value& func, js::value& object)
{
    object.get("get") = js::function<cfg_proxy_get>;
    object.get("set") = js::function<cfg_proxy_set>;

    return js::make_proxy(func, object);
}

js::value cfg_getter(js::value_context* vctx)
{
    auto [func, object] = get_proxy_handlers(*vctx);

    return finish_proxy(func, object);
}

void inject_config(js::value_context& vctx)
{
    js::value global = js::get_global(vctx);

    js::add_getter_setter(global, "$cfg", js::function<cfg_getter>, js::function<js::empty_function>);
}

js_metric::js_metric(config_variables& cfg, const std::string& script_data) : vctx(nullptr, &cfg), func(vctx)
{
    func = extract_function(vctx, script_data);

    inject_config(vctx);
}

std::array<dual, 16> js_metric::operator()(dual t, dual r, dual theta, dual phi)
{
    js::value v1 = to_value(vctx, t);
    js::value v2 = to_value(vctx, r);
    js::value v3 = to_value(vctx, theta);
    js::value v4 = to_value(vctx, phi);

    if(func.is_error() || func.is_exception())
    {
        std::cout << "Function object error (16x4) " << func.to_error_message() << std::endl;
        throw std::runtime_error("Err");
    }

    if(!func.is_function())
    {
        std::cout << "Expected function in eval of script" << std::endl;
        throw std::runtime_error("Func eval fail");
    }

    auto [success, result] = js::call(func, v1, v2, v3, v4);

    if(!success)
        throw std::runtime_error("Error in script exec (16x4) " + (std::string)result.to_error_message());

    if(!result.is_array())
        throw std::runtime_error("Must return array");

    std::vector<js::value> values = result;

    if(values.size() == 4)
    {
        return {getr(values[0]), 0, 0, 0,
                0, getr(values[1]), 0, 0,
                0, 0, getr(values[2]), 0,
                0, 0, 0, getr(values[3])};
    }
    else
    {
        if(values.size() != 16)
            throw std::runtime_error("Must return array length of 4 or 16");

        return {getr(values[0]), getr(values[1]), getr(values[2]), getr(values[3]),
                getr(values[4]), getr(values[5]), getr(values[6]), getr(values[7]),
                getr(values[8]), getr(values[9]), getr(values[10]),getr(values[11]),
                getr(values[12]),getr(values[13]),getr(values[14]),getr(values[15])};
    }
}

js_function::js_function(config_variables& cfg, const std::string& script_data) : vctx(nullptr, &cfg), func(vctx)
{
    func = extract_function(vctx, script_data);

    inject_config(vctx);
}

std::array<dual, 4> js_function::operator()(dual iv1, dual iv2, dual iv3, dual iv4)
{
    js::value v1 = to_value(vctx, iv1);
    js::value v2 = to_value(vctx, iv2);
    js::value v3 = to_value(vctx, iv3);
    js::value v4 = to_value(vctx, iv4);

    auto [success, result] = js::call(func, v1, v2, v3, v4);

    if(!success)
        throw std::runtime_error("Error in 4x4 execution " + (std::string)result.to_error_message());

    if(!result.is_array())
        throw std::runtime_error("Must return array");

    std::vector<js::value> values = result;

    if(values.size() != 4)
        throw std::runtime_error("Must return array size of 4");

    return {getr(values[0]), getr(values[1]), getr(values[2]), getr(values[3])};
}

js_single_function::js_single_function(config_variables& cfg, const std::string& script_data) : vctx(nullptr, &cfg), func(vctx)
{
    func = extract_function(vctx, script_data);

    inject_config(vctx);
}

dual js_single_function::operator()(dual iv1, dual iv2, dual iv3, dual iv4)
{
    js::value v1 = to_value(vctx, iv1);
    js::value v2 = to_value(vctx, iv2);
    js::value v3 = to_value(vctx, iv3);
    js::value v4 = to_value(vctx, iv4);

    auto [success, result] = js::call(func, v1, v2, v3, v4);

    if(!success)
        throw std::runtime_error("Error in 1x4 exec " + (std::string)result.to_error_message());

    return {getr(result)};
}
