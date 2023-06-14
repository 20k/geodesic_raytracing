#ifndef DUAL2_HPP_INCLUDED
#define DUAL2_HPP_INCLUDED

#include "dual.hpp"
#include <set>
#include <optional>
#include <cmath>
#include <map>
#include <assert.h>
#include <variant>
#include <vec/tensor.hpp>
#include <CL/cl.h>
#include <stdfloat>
#include <concepts>

namespace dual_types
{
    template<typename T>
    struct value;

    template<typename T>
    inline
    std::string name_type(T tag)
    {
        #define CMAP(x, y) if constexpr(std::is_same_v<T, x>) {return #y;};

        CMAP(float, float);
        //CMAP(cl_float, float);

        CMAP(double, double);
        //CMAP(cl_double, double);

        CMAP(std::float16_t, half);
        //CMAP(cl_half, half);

        //CMAP(cl_int, int);
        CMAP(int, int);

        //CMAP(cl_short, short);
        CMAP(short, short);

        //CMAP(cl_uint, unsigned int);
        CMAP(unsigned int, unsigned int);

        //CMAP(cl_ushort, unsigned short);
        CMAP(unsigned short, unsigned short);

        CMAP(cl_float4, float4);
        CMAP(cl_float3, float3);
        CMAP(cl_float2, float2);

        CMAP(cl_int4, int4);
        CMAP(cl_int3, int3);
        CMAP(cl_int2, int2);

        CMAP(cl_uint4, uint4);
        CMAP(cl_uint3, uint3);
        CMAP(cl_uint2, uint2);

        CMAP(cl_short4, short4);
        CMAP(cl_short3, short3);
        CMAP(cl_short2, short2);

        CMAP(cl_ushort4, ushort4);
        CMAP(cl_ushort3, ushort3);
        CMAP(cl_ushort2, ushort2);

        if constexpr(std::is_same_v<T, tensor<value<float>, 4>>)
            return "float4";

        if constexpr(std::is_same_v<T, tensor<value<float>, 3>>)
            return "float3";

        if constexpr(std::is_same_v<T, tensor<value<float>, 2>>)
            return "float2";

        if constexpr(std::is_same_v<T, tensor<value<int>, 4>>)
            return "int4";

        if constexpr(std::is_same_v<T, tensor<value<int>, 3>>)
            return "int3";

        if constexpr(std::is_same_v<T, tensor<value<int>, 2>>)
            return "int2";

        if constexpr(std::is_same_v<T, tensor<value<unsigned short>, 4>>)
            return "ushort4";

        if constexpr(std::is_same_v<T, tensor<value<unsigned short>, 3>>)
            return "ushort3";

        if constexpr(std::is_same_v<T, tensor<value<unsigned short>, 2>>)
            return "ushort2";

        if constexpr(std::is_same_v<T, tensor<value<std::float16_t>, 4>>)
            return "half4";

        if constexpr(std::is_same_v<T, tensor<value<std::float16_t>, 3>>)
            return "half3";

        if constexpr(std::is_same_v<T, tensor<value<std::float16_t>, 2>>)
            return "half2";

        #undef CMAP

        assert(false);
    }

    template<typename T>
    inline
    std::string name_type(value<T> tag)
    {
        return name_type(typename value<T>::value_type());
    }

    template<typename T>
    concept Arithmetic = std::is_arithmetic_v<T>;

    inline
    std::string to_string_s(float v)
    {
        std::ostringstream oss;
        oss << std::setprecision(32) << std::fixed << std::showpoint << v;
        std::string str = oss.str();

        while(str.size() > 0 && str.back() == '0')
            str.pop_back();

        if(str.size() > 0 && str.back() == '.')
            str += "0";

        str += "f";

        return str;
    }

    inline
    std::string to_string_s(std::integral auto v)
    {
        return std::to_string(v);
    }

    template<typename T>
    inline
    std::optional<T> get_value(std::string_view in)
    {
        if(in.size() == 0)
            throw std::runtime_error("Bad in size, 0");

        while(in.size() > 2 && in.front() == '(' && in.back() == ')')
        {
            in.remove_prefix(1);
            in.remove_suffix(1);
        }

        if(in == "nan")
            throw std::runtime_error("Nan in get_value");

        std::string cstr(in);

        char* ptr = nullptr;
        double val = std::strtod(cstr.c_str(), &ptr);

        if(ptr == cstr.c_str() + cstr.size())
            return val;

        return std::nullopt;
    }

    template<typename T>
    inline
    std::variant<std::string, T> get_value_or_string(std::string_view in)
    {
        auto result_opt = get_value<T>(in);

        if(result_opt.has_value())
        {
            return result_opt.value();
        }

        return std::string(in);
    }

    template<typename T>
    inline
    std::string to_string_either(const std::variant<std::string, T>& in)
    {
        if(in.index() == 0)
            return std::get<0>(in);

        if constexpr(std::is_same_v<T, std::monostate>)
        {
            assert(false);
        }
        else
        {
            return to_string_s(std::get<1>(in));
        }
    }

    struct operation_desc
    {
        bool is_infix = false;
        std::string_view sym;
    };

    namespace ops
    {
        enum type_t
        {
            PLUS,
            UMINUS,
            MINUS,
            MULTIPLY,
            DIVIDE,
            MODULUS, ///c style %
            AND,
            ASSIGN,
            RETURN,
            IF_S,

            LESS,
            LESS_EQUAL,

            GREATER,
            GREATER_EQUAL,

            EQUAL,

            LAND,
            LOR,
            LNOT,

            COMMA,

            IDOT,
            CONVERT,

            SIN,
            COS,
            TAN,

            ASIN,
            ACOS,
            ATAN,
            ATAN2,

            EXP,

            SQRT,

            SINH,
            COSH,
            TANH,

            ASINH,
            ACOSH,
            ATANH,

            LOG,

            ISFINITE,
            SIGNBIT,
            SIGN,
            FABS,
            ABS,

            SELECT,
            POW,
            MAX,
            MIN,

            LAMBERT_W0,

            FMA,
            MAD,

            BRACKET,

            UNKNOWN_FUNCTION,

            VALUE,
            NONE,
        };
    }

    inline
    operation_desc get_description(ops::type_t type)
    {
        using namespace ops;

        operation_desc ret;

        if(type == PLUS || type == MINUS || type == MULTIPLY || type == MODULUS || type == AND || type == ASSIGN ||
           type == LESS || type == LESS_EQUAL || type == GREATER || type == GREATER_EQUAL ||
           type == EQUAL || type == LAND || type == LOR || type == LNOT || type == COMMA || type == IDOT)
        {
            ret.is_infix = true;
        }

        #define NATIVE_DIVIDE
        #ifndef NATIVE_DIVIDE
        if(type == DIVIDE)
        {
            ret.is_infix = true;
        }
        #endif // NATIVE_DIVIDE

        std::array syms = {
            "+",
            "-",
            "-",
            "*",
            #ifdef NATIVE_DIVIDE
            "native_divide",
            #else
            "/",
            #endif // NATIVE_DIVIDE
            "%",
            "&",
            "=",
            "return",
            "if",
            "<",
            "<=",
            ">",
            ">=",
            "==",
            "&&",
            "||",
            "!",
            ",",
            ".",
            "(#err)",
            "native_sin",
            "native_cos",
            "native_tan",
            "asin",
            "acos",
            "atan",
            "atan2",
            "native_exp",
            "native_sqrt",
            "sinh",
            "cosh",
            "tanh",
            "asinh",
            "acosh",
            "atanh",
            "native_log",
            "isfinite",
            "signbit",
            "sign",
            "fabs",
            "abs",
            "select",
            "pow",
            "max",
            "min",
            "lambert_w0",
            "fma",
            "mad",
            "bad#bracket",
            "generated#function#failure",
            "ERROR#"
        };

        static_assert(syms.size() == ops::type_t::NONE);

        ret.sym = syms[(int)type];

        return ret;
    }

    template<typename T>
    inline
    T signbit(T in)
    {
        return in < 0;
    }

    template<typename T>
    inline
    T sign(T in)
    {
        if(in == T(-0.0))
            return T(-0.0);

        if(in == T(0.0))
            return T(0.0);

        if(in > 0)
            return 1;

        if(in < 0)
            return -1;

        if(std::isnan(in))
            return 0;

        throw std::runtime_error("Bad sign function");
    }

    template<typename T>
    requires std::is_arithmetic_v<T>
    inline
    T select(T v1, T v2, T v3)
    {
        if(v3 > 0)
            return v2;

        return v1;
    }

    template<typename T>
    inline
    T mad(T a, T b, T c)
    {
        return a * b + c;
    }

    template<typename T>
    std::string type_to_string(const value<T>& op);

    template<typename U, typename... T>
    value<U> make_op(ops::type_t type, T&&... args);

    template<auto N, typename T>
    inline
    bool is_value_equal(T f)
    {
        return f == (T)N;
    }

    template<typename T>
    bool equivalent(const value<T>& d1, const value<T>& d2);

    template<typename T>
    value<T> fma(const value<T>&, const value<T>&, const value<T>&);
    template<typename T>
    value<T> mad(const value<T>&, const value<T>&, const value<T>&);

    template<typename T>
    struct value
    {
        using value_type = T;
        using is_complex = std::false_type;
        static constexpr bool is_dual = false;

        ops::type_t type = ops::NONE;

        std::optional<std::variant<std::string, T>> value_payload;

        //std::optional<std::string> value_payload;
        std::vector<value<T>> args;

        value(){value_payload = T{}; type = ops::VALUE;}
        //value(T v){value_payload = v; type = ops::VALUE;}
        //value(int v){value_payload = T(v); type = ops::VALUE;}

        template<typename U>
        requires std::is_arithmetic_v<U>
        value(U u)
        {
            value_payload = static_cast<T>(u); type = ops::VALUE;
        }

        value(const std::string& str){value_payload = str; type = ops::VALUE;}
        value(const char* str){assert(str); value_payload = std::string(str); type = ops::VALUE;}

        template<typename U>
        value<U> as() const
        {
            value<U> result;
            result.type = type;
            result.value_payload = std::nullopt;

            if(value_payload.has_value())
            {
                if constexpr(std::is_same_v<T, std::monostate>)
                {
                    assert(value_payload.value().index() == 0);

                    result.value_payload = std::get<0>(value_payload.value());
                }
                else
                {
                    result.value_payload = to_string_either(value_payload.value());
                }
            }

            for(const value& v : args)
            {
                result.args.push_back(v.template as<U>());
            }

            return result;
        }

        template<typename U>
        value<U> convert() const
        {
            return make_op<U>(ops::CONVERT, as<U>(), name_type(U()));
        }

        value<std::monostate> as_generic() const
        {
            if constexpr(std::is_same_v<T, std::monostate>)
                return *this;
            else
            {
                value<std::monostate> result;
                result.type = type;
                result.value_payload = std::nullopt;

                if(value_payload.has_value())
                {
                    result.value_payload = to_string_either(value_payload.value());
                }

                for(const value<T>& v : args)
                {
                    result.args.push_back(v.as_generic());
                }

                return result;
            }
        }

        template<typename U>
        explicit operator value<U>()
        {
            return convert<U>();
        }

        bool is_value() const
        {
            return type == ops::VALUE;
        }

        bool is_constant() const
        {
            return is_value() && value_payload.has_value() && value_payload.value().index() == 1;
        }

        template<typename U>
        bool is_constant_constraint(U&& func) const
        {
            if(!is_constant())
                return false;

            return func(get_constant());
        }

        T get_constant() const
        {
            assert(is_constant());

            return std::get<1>(value_payload.value());
        }

        T get(int idx) const
        {
            return std::get<1>(args[idx].value_payload.value());
        }

        void make_value(const std::string& str)
        {
            value_payload = get_value_or_string<T>(str);
            type = ops::VALUE;
        }

        void set_dual_constant()
        {
            value_payload = T(0.);
            type = ops::VALUE;
        }

        void set_dual_variable()
        {
            value_payload = T(1.);
            type = ops::VALUE;
        }

        #define PROPAGATE1(x, y) if(type == ops::x){return y(get(0));}
        #define PROPAGATE2(x, y) if(type == ops::x){return y(get(0), get(1));}
        #define PROPAGATE3(x, y) if(type == ops::x){return y(get(0), get(1), get(2));}

        value flatten(bool recurse = false) const
        {
            if(type == ops::VALUE)
                return *this;

            bool all_constant = true;

            for(auto& i : args)
            {
                if(!i.is_constant())
                {
                    all_constant = false;
                    break;
                }
            }

            if(all_constant)
            {
                if(type == ops::PLUS)
                    return get(0) + get(1);

                if(type == ops::UMINUS)
                    return -get(0);

                if(type == ops::MINUS)
                    return get(0) - get(1);

                if(type == ops::MULTIPLY)
                    return get(0) * get(1);

                if(type == ops::DIVIDE)
                    return get(0) / get(1);

                //if(type == ops::MODULUS)
                //    return make_op_value(get(0) % get(1));

                ///can't propagate because we only do doubles oops
                //if(type == ops::AND)
                //    return make_op_value(get(0) & get(1));

                if(type == ops::LESS)
                    return get(0) < get(1);

                if(type == ops::LESS_EQUAL)
                    return get(0) <= get(1);

                if(type == ops::GREATER)
                    return get(0) > get(1);

                if(type == ops::GREATER_EQUAL)
                    return get(0) >= get(1);

                if(type == ops::EQUAL)
                    return get(0) == get(1);

                PROPAGATE1(SIN, std::sin);
                PROPAGATE1(COS, std::cos);
                PROPAGATE1(TAN, std::tan);
                PROPAGATE1(ASIN, std::asin);
                PROPAGATE1(ACOS, std::acos);
                PROPAGATE1(ATAN, std::atan);
                PROPAGATE2(ATAN2, std::atan2);
                PROPAGATE1(EXP, std::exp);
                PROPAGATE1(SQRT, std::sqrt);
                PROPAGATE1(SINH, std::sinh);
                PROPAGATE1(COSH, std::cosh);
                PROPAGATE1(TANH, std::tanh);
                PROPAGATE1(ASINH, std::asinh);
                PROPAGATE1(ACOSH, std::acosh);
                PROPAGATE1(ATANH, std::atanh);
                PROPAGATE1(LOG, std::log);
                PROPAGATE1(ISFINITE, std::isfinite);
                PROPAGATE1(SIGNBIT, signbit);
                PROPAGATE1(SIGN, sign);
                PROPAGATE1(FABS, std::fabs);
                PROPAGATE1(ABS, std::abs);
                PROPAGATE3(SELECT, select);
                PROPAGATE2(POW, std::pow);
                PROPAGATE2(MAX, std::max);
                PROPAGATE2(MIN, std::min);
                //PROPAGATE2(LAMBERT_W0, lambert_w0);

                ///FMA is not propagated as we can't actually simulate it? Does that matter?
                PROPAGATE3(MAD, mad);
            }

            if(type == ops::SELECT)
            {
                if(args[2].is_constant())
                    return args[2].get_constant() > 0 ? args[1] : args[0];
            }

            auto is_zero = [](T f){return f == 0;};

            if(type == ops::MULTIPLY)
            {
                if(args[0].is_constant_constraint(is_zero) || args[1].is_constant_constraint(is_zero))
                    return value(0);

                //std::cout << "hello " << type_to_string(args[0]) << " with " << type_to_string(args[1]) << std::endl;

                if(args[0].is_constant_constraint(is_value_equal<1, T>))
                    return args[1].flatten();

                if(args[1].is_constant_constraint(is_value_equal<1, T>))
                    return args[0].flatten();
            }

            if(type == ops::PLUS)
            {
                if(args[0].is_constant_constraint(is_zero))
                    return args[1].flatten();

                if(args[1].is_constant_constraint(is_zero))
                    return args[0].flatten();

                if(equivalent(args[0], args[1]))
                    return 2 * args[0].flatten();
            }

            if(type == ops::MINUS)
            {
                if(args[0].is_constant_constraint(is_zero))
                    return -args[1].flatten();

                if(args[1].is_constant_constraint(is_zero))
                    return args[0].flatten();

                if(equivalent(args[0], args[1]))
                    return 0;
            }

            if(type == ops::DIVIDE)
            {
                if(args[0].is_constant_constraint(is_zero))
                    return 0;

                if(args[1].is_constant_constraint(is_value_equal<1, T>))
                    return args[0].flatten();

                if(equivalent(args[0], args[1]))
                    return 1;
            }

            ///ops::MODULUS

            if(type == ops::POW)
            {
                if(args[1].is_constant_constraint(is_value_equal<1, T>))
                    return args[0].flatten();

                if(args[1].is_constant_constraint(is_zero))
                    return 1;

                /*if(args[1].is_constant())
                {
                    value ret = args[0];

                    for(int i=0; i < args[1].get_constant() - 1; i++)
                    {
                        ret = ret * args[0];
                    }

                    return ret;
                }*/
            }

            if(type == ops::FMA || type == ops::MAD)
            {
                ///a * 0 + c or 0 * b + c
                if(args[0].is_constant_constraint(is_zero) || args[1].is_constant_constraint(is_zero))
                    return args[2].flatten();

                ///1 * b + c
                if(args[0].is_constant_constraint(is_value_equal<1, T>))
                    return (args[1] + args[2]).flatten();

                ///a * 1 + c
                if(args[1].is_constant_constraint(is_value_equal<1, T>))
                    return (args[0] + args[2]).flatten();

                ///a * b + 0
                if(args[2].is_constant_constraint(is_zero))
                {
                    return (args[0] * args[1]).flatten();
                }
            }

            value ret = *this;

            ///much worse than letting the compiler do it, even with mad
            #ifdef FMA_REPLACE
            if(ret.type == ops::PLUS)
            {
                if(ret.args[0].type == ops::MULTIPLY)
                {
                    value c = ret.args[1];
                    value a = ret.args[0].args[0];
                    value b = ret.args[0].args[1];

                    ret = fma(a, b, c);
                }
                else if(ret.args[1].type == ops::MULTIPLY)
                {
                    value c = ret.args[0];
                    value a = ret.args[1].args[0];
                    value b = ret.args[1].args[1];

                    ret = fma(a, b, c);
                }
            }
            #endif

            if(recurse)
            {
                for(auto& i : ret.args)
                    i = i.flatten(true);
            }

            return ret;
        }

        dual_types::dual_v<value<T>> dual(const std::string& sym) const
        {
            #define DUAL_CHECK1(x, y) if(type == x) { return y(args[0].dual(sym)); }
            #define DUAL_CHECK2(x, y) if(type == x) { return y(args[0].dual(sym), args[1].dual(sym)); }
            #define DUAL_CHECK3(x, y) if(type == x) { return y(args[0].dual(sym), args[1].dual(sym), args[2].dual(sym)); }

            using namespace ops;

            if(type == VALUE)
            {
                dual_types::dual_v<value<T>> ret;

                if(value_payload.value().index() == 0 && std::get<0>(value_payload.value()) == sym)
                {
                    ret.make_variable(*this);
                }
                else
                {
                    ret.make_constant(*this);
                }

                return ret;
            }
            if(type == PLUS)
            {
                return args[0].dual(sym) + args[1].dual(sym);
            }
            if(type == UMINUS)
            {
                return -args[0].dual(sym);
            }
            if(type == MINUS)
            {
                return args[0].dual(sym) - args[1].dual(sym);
            }
            if(type == MULTIPLY)
            {
                return args[0].dual(sym) * args[1].dual(sym);
            }
            if(type == DIVIDE)
            {
                return args[0].dual(sym) / args[1].dual(sym);
            }
            /*if(type == MODULUS)
            {

            }*/
            /*if(type == AND)
            {

            }*/
            if(type == LESS)
            {
                return args[0].dual(sym) < args[1].dual(sym);
            }
            if(type == LESS_EQUAL)
            {
                return args[0].dual(sym) <= args[1].dual(sym);
            }

            if(type == FMA || type == MAD)
            {
                return args[0].dual(sym) * args[1].dual(sym) + args[2].dual(sym);
            }

            /*if(type == EQUAL)
            {
                return args[0].dual(sym) == args[1].dual(sym);
            }*/

            /*if(type == GREATER)
            {
                return args[0].dual(sym) > args[1].dual(sym);
            }
            if(type == GREATER_EQUAL)
            {
                return args[0].dual(sym) >= args[1].dual(sym);
            }*/

            DUAL_CHECK1(SIN, sin);
            DUAL_CHECK1(COS, cos);
            DUAL_CHECK1(TAN, tan);
            DUAL_CHECK1(ASIN, asin);
            DUAL_CHECK1(ACOS, acos);
            DUAL_CHECK1(ATAN, atan);
            DUAL_CHECK2(ATAN2, atan2);
            DUAL_CHECK1(EXP, exp);
            DUAL_CHECK1(SQRT, sqrt);
            DUAL_CHECK1(SINH, sinh);
            DUAL_CHECK1(COSH, cosh);
            DUAL_CHECK1(TANH, tanh);
            DUAL_CHECK1(LOG, log);
            //DUAL_CHECK1(ISFINITE, isfinite);
            //DUAL_CHECK1(SIGNBIT, signbit);
            //DUAL_CHECK1(SIGN, sign);
            DUAL_CHECK1(FABS, fabs);
            //DUAL_CHECK1(ABS, abs);

            if(type == SELECT)
            {
                return dual_types::select(args[0].dual(sym), args[1].dual(sym), args[2]);
            }

            DUAL_CHECK2(POW, pow);
            DUAL_CHECK2(MAX, max);
            DUAL_CHECK2(MIN, min);
            DUAL_CHECK1(LAMBERT_W0, lambert_w0);

            assert(false);
        }

        value<T> differentiate(const std::string& sym) const
        {
            return dual(sym).dual;
        }

        void substitute_impl(const std::string& sym, T value)
        {
            if(type == ops::VALUE && value_payload.value().index() == 0 && std::get<0>(value_payload.value()) == sym)
            {
                value_payload = double{value};
                return;
            }

            int start = 0;

            if(type == ops::UNKNOWN_FUNCTION)
                start = 1;

            for(int kk=start; kk < (int)args.size(); kk++)
            {
                args[kk].substitute_impl(sym, value);
                args[kk] = args[kk].flatten();
            }

            *this = flatten();
        }

        value substitute(const std::string& sym, T value) const
        {
            auto cp = *this;

            cp.substitute_impl(sym, value);

            return cp;
        }

        void get_all_variables_impl(std::set<std::string>& v) const
        {
            if(type == ops::VALUE)
            {
                if(is_constant())
                    return;

                v.insert(std::get<0>(value_payload.value()));
                return;
            }

            int start = 0;

            if(type == ops::UNKNOWN_FUNCTION)
                start = 1;

            for(int i=start; i < (int)args.size(); i++)
            {
                args[i].get_all_variables_impl(v);
            }
        }

        std::vector<std::string> get_all_variables() const
        {
            std::set<std::string> v;
            get_all_variables_impl(v);

            std::vector<std::string> ret(v.begin(), v.end());

            return ret;
        }

        template<typename U>
        void recurse_arguments(const U& in) const
        {
            in(*this);

            int start = 0;

            if(type == ops::UNKNOWN_FUNCTION)
                start = 1;

            for(int i=start; i < (int)args.size(); i++)
            {
                args[i].recurse_arguments(in);
            }
        }

        template<typename U>
        void recurse_arguments(const U& in)
        {
            in(*this);

            int start = 0;

            if(type == ops::UNKNOWN_FUNCTION)
                start = 1;

            for(int i=start; i < (int)args.size(); i++)
            {
                args[i].recurse_arguments(in);
            }
        }

        void substitute(const std::map<std::string, std::string>& variables)
        {
            if(type == ops::VALUE)
            {
                if(is_constant())
                    return;

                auto it = variables.find(std::get<0>(value_payload.value()));

                if(it == variables.end())
                    return;

                value_payload = get_value_or_string(it->second);
                return;
            }

            for(auto& val : args)
            {
                val.substitute(variables);
            }
        }

        void substitute(const std::vector<std::pair<value<T>, value<T>>>& variables)
        {
            for(auto& [conc, rep] : variables)
            {
                if(equivalent(*this, conc))
                {
                    *this = rep;
                    break;
                }
            }

            for(auto& val : args)
            {
                val.substitute(variables);
            }
        }

        template<typename U>
        void substitute_generic(U&& func)
        {
            auto sub_opt = func(*this);

            if(sub_opt.has_value())
            {
                *this = *sub_opt;
                return;
            }

            for(auto& val : args)
            {
                val.substitute_generic(func);
            }

            *this = flatten();
        }

        template<typename U>
        value<U> x()
        {
            return make_op<U>(ops::IDOT, as<U>(), "x");
        }

        template<typename U>
        value<U> y()
        {
            return make_op<U>(ops::IDOT, as<U>(), "y");
        }

        template<typename U>
        value<U> z()
        {
            return make_op<U>(ops::IDOT, as<U>(), "z");
        }

        template<typename U>
        value<U> w()
        {
            return make_op<U>(ops::IDOT, as<U>(), "w");
        }

        template<typename U>
        value<U> index(int idx)
        {
            if(idx == 0)
                return x<U>();
            if(idx == 1)
                return y<U>();
            if(idx == 2)
                return z<U>();
            if(idx == 3)
                return w<U>();

            assert(false);
        }

        value<T>& operator+=(const value<T>& d1)
        {
            *this = *this + d1;

            return *this;
        }

        friend
        value<T> operator<(const value<T>& d1, const value<T>& d2)
        {
            return make_op<T>(ops::LESS, d1, d2);
        }

        friend
        value<T> operator<=(const value<T>& d1, const value<T>& d2)
        {
            return make_op<T>(ops::LESS_EQUAL, d1, d2);
        }

        friend
        value<T> operator>(const value<T>& d1, const value<T>& d2)
        {
            return make_op<T>(ops::GREATER, d1, d2);
        }

        friend
        value<T> operator>=(const value<T>& d1, const value<T>& d2)
        {
            return make_op<T>(ops::GREATER_EQUAL, d1, d2);
        }

        friend
        value<T> operator==(const value<T>& d1, const value<T>& d2)
        {
            return make_op<T>(ops::EQUAL, d1, d2);
        }

        friend
        value<T> operator+(const value<T>& d1, const value<T>& d2)
        {
            return make_op<T>(ops::PLUS, d1, d2);
        }

        friend
        value<T> operator-(const value<T>& d1, const value<T>& d2)
        {
            return make_op<T>(ops::MINUS, d1, d2);
            //return make_op<T>(ops::PLUS, d1, make_op<T>(ops::UMINUS, d2));
        }

        friend
        value<T> operator-(const value& d1)
        {
            return make_op<T>(ops::UMINUS, d1);
        }

        friend
        value<T> operator*(const value<T>& d1, const value<T>& d2)
        {
            return make_op<T>(ops::MULTIPLY, d1, d2);
        }

        friend
        value<T> operator*(const T& d1, const value<T>& d2)
        {
            return make_op<T>(ops::MULTIPLY, value<T>(d1), d2);
        }

        friend
        value<T> operator*(const value<T>& d1, const T& d2)
        {
            return make_op<T>(ops::MULTIPLY, d1, value<T>(d2));
        }

        friend
        value<T> operator/(const value<T>& d1, const value<T>& d2)
        {
            return make_op<T>(ops::DIVIDE, d1, d2);
        }

        friend
        value<T> operator%(const value<T>& d1, const value<T>& d2)
        {
            return make_op<T>(ops::MODULUS, d1, d2);
        }

        friend
        value<T> operator&(const value<T>& d1, const value<T>& d2)
        {
            return make_op<T>(ops::AND, d1, d2);
        }

        friend
        value<T> operator||(const value<T>& d1, const value<T>& d2)
        {
            return make_op<T>(ops::LOR, d1, d2);
        }

        friend
        value<T> operator&&(const value<T>& d1, const value<T>& d2)
        {
            return make_op<T>(ops::LAND, d1, d2);
        }

        friend
        value<T> operator!(const value<T>& d1)
        {
            return make_op<T>(ops::LNOT, d1);
        }

        template<typename U>
        friend
        value<std::monostate> operator,(const value<T>& d1, const value<U>& d2)
        {
            return make_op<std::monostate>(ops::COMMA, d1.as_generic(), d2.as_generic());
        }
    };

    template<typename T>
    inline
    std::string get_function_name(const value<T>& v)
    {
        assert(!v.is_value());

        if(v.type == ops::UNKNOWN_FUNCTION)
            return type_to_string(v.args.at(0));
        else
            return std::string(get_description(v.type).sym);
    }

    template<typename T>
    inline
    std::vector<value<T>> get_function_args(const value<T>& v)
    {
        assert(!v.is_value());

        if(v.type == ops::UNKNOWN_FUNCTION)
            return std::vector<value<T>>(v.args.begin() + 1, v.args.end());
        else
            return v.args;
    }

    template<typename T>
    inline
    bool equivalent(const value<T>& d1, const value<T>& d2)
    {
        if(d1.type != d2.type)
            return false;

        if(d1.type == ops::VALUE)
        {
            if(d1.is_constant() != d2.is_constant())
                return false;

            if(d1.is_constant())
            {
                return d1.get_constant() == d2.get_constant();
            }

            return d1.value_payload.value() == d2.value_payload.value();
        }

        if(d1.type == ops::VALUE && d1.get_constant() == d2.get_constant())
            return true;

        if(d1.args.size() == 2 && (d1.type == ops::MULTIPLY || d1.type == ops::PLUS))
        {
            return (equivalent(d1.args[0], d2.args[0]) && equivalent(d1.args[1], d2.args[1])) ||
                   (equivalent(d1.args[1], d2.args[0]) && equivalent(d1.args[0], d2.args[1]));
        }

        for(int i=0; i < (int)d1.args.size(); i++)
        {
            if(!equivalent(d1.args[i], d2.args[i]))
                return false;
        }

        return true;
    }

    template<typename T>
    inline
    std::string type_to_string(const value<T>& op)
    {
        static_assert(std::is_same_v<T, double> || std::is_same_v<T, float> || std::is_same_v<T, int> || std::is_same_v<T, short> || std::is_same_v<T, unsigned short> || std::is_same_v<T, std::monostate>);

        if(op.type == ops::VALUE)
        {
            if constexpr(!std::is_same_v<T, std::monostate>)
            {
                if(op.is_constant())
                {
                    if(op.get_constant() < 0)
                        return "(" + to_string_either(op.value_payload.value()) + ")";
                }
            }

            return to_string_either(op.value_payload.value());
        }

        if(op.type == ops::BRACKET)
        {
            return "(" + type_to_string(op.args[0]) + "[" + type_to_string(op.args[1]) + "])";
        }

        if(op.type == ops::RETURN)
        {
            return "return";
        }

        if(op.type == ops::IF_S)
        {
            return "if(" + type_to_string(op.args[0]) + "){" + type_to_string(op.args[1]) + ";}";
        }

        if(op.type == ops::COMMA)
        {
            return type_to_string(op.args[0]) + ";" + type_to_string(op.args[1]);
        }

        if(op.type == ops::CONVERT)
        {
            return "((" + type_to_string(op.args[1]) + ")" + type_to_string(op.args[0]) + ")";
        }

        if(op.type == ops::LNOT)
        {
            return "(!(" + type_to_string(op.args[0]) + "))";
        }

        const operation_desc desc = get_description(op.type);

        if(desc.is_infix)
        {
            assert(op.args.size() == 2);

            return "(" + type_to_string(op.args[0]) + std::string(desc.sym) + type_to_string(op.args[1]) + ")";
        }
        else
        {
           if(op.type == ops::UMINUS)
                return "(-(" + type_to_string(op.args[0]) + "))";

            std::string build;

            if(op.type == ops::UNKNOWN_FUNCTION)
            {
                assert(op.args[0].is_value());

                std::string name = type_to_string(op.args[0]);

                build = name + "(";

                ///starts at one, ignoring the first argument which is the function name
                for(int i=1; i < (int)op.args.size() - 1; i++)
                {
                    build += type_to_string(op.args[i]) + ",";
                }

                if(op.args.size() > 1)
                    build += type_to_string(op.args.back());

                build += ")";
            }
            else
            {
                build = std::string(desc.sym) + "(";

                for(int i=0; i < (int)op.args.size() - 1; i++)
                {
                    build += type_to_string(op.args[i]) + ",";
                }

                if(op.args.size() > 0)
                    build += type_to_string(op.args.back());

                build += ")";
            }

            return build;
        }
    }

    template<typename U, typename... T>
    inline
    value<U> make_op(ops::type_t type, T&&... args)
    {
        value<U> ret;
        ret.type = type;
        ret.args = {args...};
        ret.value_payload = std::nullopt;

        if constexpr(std::is_same_v<U, std::monostate>)
            return ret;
        else
            return ret.flatten();
    }

    #define UNARY(x, y) template<typename T> inline value<T> x(const value<T>& d1){return make_op<T>(ops::y, d1);}
    #define BINARY(x, y) template<typename T, typename U> inline value<T> x(const value<T>& d1, const U& d2){return make_op<T>(ops::y, d1, d2);}
    #define TRINARY(x, y) template<typename T, typename U, typename V> inline value<T> x(const value<T>& d1, const U& d2, const V& d3){return make_op<T>(ops::y, d1, d2, d3);}

    UNARY(sqrt, SQRT);
    UNARY(psqrt, SQRT);
    UNARY(log, LOG);
    UNARY(fabs, FABS);
    UNARY(abs, ABS);
    UNARY(exp, EXP);
    UNARY(sin, SIN);
    UNARY(cos, COS);
    UNARY(tan, TAN);
    UNARY(sinh, SINH);
    UNARY(cosh, COSH);
    UNARY(tanh, TANH);
    UNARY(asinh, ASINH);
    UNARY(acosh, ACOSH);
    UNARY(atanh, ATANH);
    UNARY(asin, ASIN);
    UNARY(acos, ACOS);
    UNARY(atan, ATAN);
    BINARY(atan2, ATAN2);
    UNARY(isfinite, ISFINITE);
    UNARY(signbit, SIGNBIT);
    UNARY(sign, SIGN);
    //TRINARY(select, SELECT);
    BINARY(pow, POW);
    BINARY(max, MAX);
    BINARY(min, MIN);
    UNARY(lambert_w0, LAMBERT_W0);
    TRINARY(mad, MAD);
    TRINARY(fma, FMA);

    template<typename T, typename U, typename V>
    inline
    value<V> select(const T& v1, const U& v2, const value<V>& v3)
    {
        return make_op<V>(ops::SELECT, value<V>(v1), value<V>(v2), v3);
    }

    template<typename T>
    inline
    T clamp(const T& val, const T& lower, const T& upper)
    {
        return min(max(val, lower), upper);
    }

    ///select
    template<typename T, typename U, typename V>
    inline
    value<T> if_v(const value<T>& condition, const U& if_true, const V& if_false)
    {
        if(condition.is_constant())
        {
            T val = condition.get_constant();

            if(val == 0)
                return if_false;
            else
                return if_true;
        }

        return select<V, U, T>(if_false, if_true, condition);
    }

    inline
    value<std::monostate> make_return_s()
    {
        return make_op<std::monostate>(ops::RETURN);
    }

    const inline value<std::monostate> return_s = make_return_s();

    ///true branch
    template<typename T, typename U>
    inline
    value<std::monostate> if_s(const value<T>& condition, const value<U>& to_execute)
    {
        return make_op<std::monostate>(ops::IF_S, condition.as_generic(), to_execute.as_generic());
    }

    ///select
    template<typename T>
    requires std::is_arithmetic_v<T>
    auto if_v(bool condition, const T& if_true, const T& if_false)
    {
        return condition ? if_true : if_false;
    }

    template<typename T, typename U>
    inline
    T divide_with_limit(const T& top, const T& bottom, const U& limit, float tol = 0.001f)
    {
        if constexpr(std::is_arithmetic_v<T>)
            return dual_types::if_v(std::fabs(bottom) >= tol, top / bottom, limit);
        else
            return dual_types::if_v(fabs(bottom) >= tol, top / bottom, limit);
    }

    template<typename U, typename... T>
    value<U> apply(const value<U>& name, T&&... args)
    {
        return make_op<U>(ops::UNKNOWN_FUNCTION, name, std::forward<T>(args)...);
    }

    template<typename T>
    auto assert_s(const value<T>& is_true)
    {
        value<std::monostate> print = "printf(\"Failed: %s\",\"" + type_to_string(is_true) + "\")";

        return if_s(!is_true, print);
    }

    template<typename T>
    inline
    value<T> conjugate(const value<T>& d1)
    {
        return d1;
    }


    template<typename T>
    inline
    complex<value<T>> psqrt(const complex<value<T>>& d1)
    {
        if(d1.imaginary.is_constant() && d1.imaginary.get_constant() == 0)
        {
            return sqrt(d1.real);
        }

        return sqrt(d1);
    }

    template<typename U>
    static U parse_tensor(const U& tag, value<int> op)
    {
        return op.as<typename U::value_type>();
    }

    template<typename U, int N>
    static tensor<U, N> parse_tensor(const tensor<U, N>& tag, value<int> op)
    {
        tensor<U, N> ret;

        for(int i=0; i < N; i++)
        {
            ret.idx(i) = op.index<typename U::value_type>(i);
        }

        return ret;
    }

    template<typename T>
    struct literal
    {
        using value_type = T;

        std::string name;

        literal(){}
        literal(const std::string& str) : name(str){}
        literal(const char* str) : name(str){}

        T get()
        {
            value<int> op(name);

            return parse_tensor(T(), op);
        }

        operator T()
        {
            return get();
        }
    };

    template<typename T>
    inline
    value<T> assign(const value<T>& location, const value<T>& what)
    {
        return make_op<T>(ops::ASSIGN, location, what);
    }

    template<typename T, int dimensions>
    struct buffer
    {
        using value_type = T;

        std::string name;
        tensor<value<int>, dimensions> size;
        //std::string read_function;
        //std::string write_function;

        /*template<typename... U>
        T read(U&&... u) const
        {
            return dual_types::apply(T(read_function), std::forward<U>(u)...);
        }

        template<typename V, typename... U>
        auto write(V&& what, U&&... u)
        {
            return dual_types::apply(T(write_function), std::forward<V>(what), std::forward<U>(u)...);
        }*/

        T operator[](const value<int>& in) const
        {
            value<int> op = make_op<int>(ops::BRACKET, value<int>(name), in);

            return parse_tensor(T(), op);
        }

        T operator[](const value<int>& ix, const value<int>& iy, const value<int>& iz) const
        {
            value<int> index = iz * size.idx(0) * size.idx(1) + iy * size.idx(0) + ix;

            value<int> op = make_op<int>(ops::BRACKET, value<int>(name), index);

            return parse_tensor(T(), op);
        }

        T assign(const T& location, const T& what)
        {
            return make_op<typename T::value_type>(ops::ASSIGN, location, what);
        }
    };

    inline
    void test_operation()
    {
        value<float> v1 = 1;
        value<float> v2 = 2;

        value<float> sum = v1 + v2;

        value<float> root_3 = sqrt(sum);

        dual_v<value<float>> test_dual = 1234;

        dual_v<value<float>> test_operator = test_dual * 1111;

        value<float> v = "v";

        value<float> test_op = 2 * (v/2);

        assert(type_to_string(test_op) == "v");

        value<float> test_op2 = (v * 2)/2;

        assert(type_to_string(test_op2) == "v");

        value<float> test_op3 = (2 * ((2 * v/2) / 2) * 2) / 2;

        assert(type_to_string(test_op3) == "v");

        value<float> test_op4 = (2 * v) / v;

        assert(type_to_string(test_op4) == "2.0");

        value<float> test_op5 = (2 * sin(v)) / sin(v);

        assert(type_to_string(test_op5) == "2.0");

        //std::cout << type_to_string(root_3) << std::endl;
    }
}

template<typename T, typename U>
inline
T divide_with_callback(const T& top, const T& bottom, U&& if_nonfinite)
{
    using namespace std;

    T result = top / bottom;

    auto is_finite = isfinite(result);

    if constexpr(std::is_same_v<T, float>)
    {
        static_assert(std::is_same_v<decltype(is_finite), bool>);
    }

    return dual_types::if_v(is_finite, result, if_nonfinite(top, bottom));
}

using dual = dual_types::dual_v<dual_types::value<float>>;
using dual_complex = dual_types::dual_v<dual_types::complex<dual_types::value<float>>>;
template<typename T>
using value_base = dual_types::value<T>;
using value = dual_types::value<float>;
using value_i = dual_types::value<int>;
using value_s = dual_types::value<short>;
using value_us = dual_types::value<unsigned short>;
using value_v = dual_types::value<std::monostate>;
using value_h = dual_types::value<std::float16_t>;
template<typename T, int N>
using buffer = dual_types::buffer<T, N>;
template<typename T>
using literal = dual_types::literal<T>;
const inline auto return_s = dual_types::make_return_s();

#endif // DUAL2_HPP_INCLUDED
