#ifndef DUAL2_HPP_INCLUDED
#define DUAL2_HPP_INCLUDED

#include "dual.hpp"
#include <set>
#include <optional>
#include <cmath>
#include <map>
#include <assert.h>
#include <variant>

namespace dual_types
{
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
    std::optional<double> get_value(std::string_view in)
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

    inline
    std::variant<std::string, double> get_value_or_string(std::string_view in)
    {
        auto result_opt = get_value(in);

        if(result_opt.has_value())
        {
            return result_opt.value();
        }

        return std::string(in);
    }

    inline
    std::string to_string_either(const std::variant<std::string, double>& in)
    {
        if(in.index() == 0)
            return std::get<0>(in);

        return to_string_s(std::get<1>(in));
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

            LESS,
            LESS_EQUAL,

            GREATER,
            GREATER_EQUAL,

            EQUAL,

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

            UNKNOWN_FUNCTION,

            VALUE,
            NONE,
        };
    }

    template<typename... T>
    inline
    auto make_array(T&&... args)
    {
        return std::array{args...};
    }

    inline
    operation_desc get_description(ops::type_t type)
    {
        using namespace ops;

        operation_desc ret;

        if(type == PLUS || type == MINUS || type == MULTIPLY || type == MODULUS || type == AND ||
           type == LESS || type == LESS_EQUAL || type == GREATER || type == GREATER_EQUAL ||
           type == EQUAL)
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

        std::array syms = make_array(
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
        "<",
        "<=",
        ">",
        ">=",
        "==",
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
        "generated#function#failure",
        "ERROR"
        );

        static_assert(syms.size() == ops::type_t::NONE);

        ret.sym = syms[(int)type];

        return ret;
    }

    inline
    double signbit(double in)
    {
        return in < 0;
    }

    inline
    double sign(double in)
    {
        if(in == -0.0)
            return -0.0;

        if(in == 0.0)
            return 0.0;

        if(in > 0)
            return 1;

        if(in < 0)
            return -1;

        if(std::isnan(in))
            return 0;

        throw std::runtime_error("Bad sign function");
    }

    inline
    double select(double v1, double v2, double v3)
    {
        if(v3 > 0)
            return v2;

        return v1;
    }

    inline
    double mad(double a, double b, double c)
    {
        return a * b + c;
    }

    struct value;

    std::string type_to_string(const value& op, bool is_int = false);

    value make_op_value(const std::string& str);
    template<Arithmetic T>
    value make_op_value(const T& v);
    template<typename... T>
    value make_op(ops::type_t type, T&&... args);

    template<auto N>
    inline
    bool is_value_equal(double f)
    {
        return f == (double)N;
    }

    value operator<(const value& d1, const value& d2);

    value operator<=(const value& d1, const value& d2);

    value operator>(const value& d1, const value& d2);

    value operator>=(const value& d1, const value& d2);

    value operator+(const value& d1, const value& d2);

    value operator-(const value& d1, const value& d2);

    value operator-(const value& d1);

    value operator*(const value& d1, const value& d2);

    value operator/(const value& d1, const value& d2);

    value operator%(const value& d1, const value& d2);

    value operator&(const value& d1, const value& d2);

    bool equivalent(const value& d1, const value& d2);

    value fma(const value&, const value&, const value&);
    value mad(const value&, const value&, const value&);

    struct value
    {
        using is_complex = std::false_type;
        static constexpr bool is_dual = false;

        ops::type_t type = ops::NONE;

        std::optional<std::variant<std::string, double>> value_payload;

        //std::optional<std::string> value_payload;
        std::vector<value> args;

        //value(){value_payload = to_string_s(0); type = ops::VALUE;}
        value(){value_payload = 0.; type = ops::VALUE;}
        template<Arithmetic T>
        value(T v){value_payload = static_cast<double>(v); type = ops::VALUE;}
        value(const std::string& str){value_payload = str; type = ops::VALUE;}
        value(const char* str){value_payload = std::string(str); type = ops::VALUE;}

        bool is_value() const
        {
            return type == ops::VALUE;
        }

        bool is_constant() const
        {
            return is_value() && value_payload.has_value() && value_payload.value().index() == 1;
        }

        template<typename T>
        bool is_constant_constraint(T&& func) const
        {
            if(!is_constant())
                return false;

            return func(get_constant());
        }

        double get_constant() const
        {
            assert(is_constant());

            return std::get<1>(value_payload.value());
        }

        double get(int idx) const
        {
            return std::get<1>(args[idx].value_payload.value());
        }

        void make_value(const std::string& str)
        {
            value_payload = get_value_or_string(str);
            type = ops::VALUE;
        }

        void set_dual_constant()
        {
            value_payload = 0.;
            type = ops::VALUE;
        }

        void set_dual_variable()
        {
            value_payload = 1.;
            type = ops::VALUE;
        }

        #define PROPAGATE1(x, y) if(type == ops::x){return make_op_value(y(get(0)));}
        #define PROPAGATE2(x, y) if(type == ops::x){return make_op_value(y(get(0), get(1)));}
        #define PROPAGATE3(x, y) if(type == ops::x){return make_op_value(y(get(0), get(1), get(2)));}

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
                    return make_op_value(get(0) + get(1));

                if(type == ops::UMINUS)
                    return make_op_value(-get(0));

                if(type == ops::MINUS)
                    return make_op_value(get(0) - get(1));

                if(type == ops::MULTIPLY)
                    return make_op_value(get(0) * get(1));

                if(type == ops::DIVIDE)
                    return make_op_value(get(0) / get(1));

                //if(type == ops::MODULUS)
                //    return make_op_value(get(0) % get(1));

                ///can't propagate because we only do doubles oops
                //if(type == ops::AND)
                //    return make_op_value(get(0) & get(1));

                if(type == ops::LESS)
                    return make_op_value(get(0) < get(1));

                if(type == ops::LESS_EQUAL)
                    return make_op_value(get(0) <= get(1));

                if(type == ops::GREATER)
                    return make_op_value(get(0) > get(1));

                if(type == ops::GREATER_EQUAL)
                    return make_op_value(get(0) >= get(1));

                if(type == ops::EQUAL)
                    return make_op_value(get(0) == get(1));

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

            auto is_zero = [](double f){return f == 0;};

            if(type == ops::MULTIPLY)
            {
                if(args[0].is_constant_constraint(is_zero) || args[1].is_constant_constraint(is_zero))
                    return value(0);

                //std::cout << "hello " << type_to_string(args[0]) << " with " << type_to_string(args[1]) << std::endl;

                if(args[0].is_constant_constraint(is_value_equal<1>))
                    return args[1].flatten();

                if(args[1].is_constant_constraint(is_value_equal<1>))
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
                    return value(0);
            }

            if(type == ops::DIVIDE)
            {
                if(args[0].is_constant_constraint(is_zero))
                    return value(0);

                if(args[1].is_constant_constraint(is_value_equal<1>))
                    return args[0].flatten();

                if(equivalent(args[0], args[1]))
                    return value(1);
            }

            ///ops::MODULUS

            if(type == ops::POW)
            {
                if(args[1].is_constant_constraint(is_value_equal<1>))
                    return args[0].flatten();

                if(args[1].is_constant_constraint(is_zero))
                    return value(1);

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
                if(args[0].is_constant_constraint(is_value_equal<1>))
                    return (args[1] + args[2]).flatten();

                ///a * 1 + c
                if(args[1].is_constant_constraint(is_value_equal<1>))
                    return (args[0] + args[2]).flatten();

                ///a * b + 0
                if(args[2].is_constant_constraint(is_zero))
                {
                    return (args[0] * args[1]).flatten();
                }
            }

            value ret = *this;

            ///much worse than letting the compiler do it
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

        dual_types::dual_v<value> dual(const std::string& sym) const
        {
            #define DUAL_CHECK1(x, y) if(type == x) { return y(args[0].dual(sym)); }
            #define DUAL_CHECK2(x, y) if(type == x) { return y(args[0].dual(sym), args[1].dual(sym)); }
            #define DUAL_CHECK3(x, y) if(type == x) { return y(args[0].dual(sym), args[1].dual(sym), args[2].dual(sym)); }

            using namespace ops;

            if(type == VALUE)
            {
                dual_types::dual_v<value> ret;

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
                return select(args[0].dual(sym), args[1].dual(sym), args[2]);
            }

            DUAL_CHECK2(POW, pow);
            DUAL_CHECK2(MAX, max);
            DUAL_CHECK2(MIN, min);
            DUAL_CHECK1(LAMBERT_W0, lambert_w0);

            assert(false);
        }

        value differentiate(const std::string& sym) const
        {
            return dual(sym).dual;
        }

        void substitute_impl(const std::string& sym, float value)
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

        value substitute(const std::string& sym, float value) const
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

        template<typename T>
        void recurse_arguments(const T& in) const
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

        template<typename T>
        void recurse_arguments(const T& in)
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

        void substitute(const std::vector<std::pair<value, value>>& variables)
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

        template<typename T>
        void substitute_generic(T&& func)
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

        value& operator+=(const value& d1)
        {
            *this = *this + d1;

            return *this;
        }
    };

    inline
    std::string get_function_name(const value& v)
    {
        assert(!v.is_value());

        if(v.type == ops::UNKNOWN_FUNCTION)
            return type_to_string(v.args.at(0));
        else
            return std::string(get_description(v.type).sym);
    }

    inline
    std::vector<value> get_function_args(const value& v)
    {
        assert(!v.is_value());

        if(v.type == ops::UNKNOWN_FUNCTION)
            return std::vector<value>(v.args.begin() + 1, v.args.end());
        else
            return v.args;
    }

    inline
    bool equivalent(const value& d1, const value& d2)
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

    inline
    std::string type_to_string(const value& op, bool is_int)
    {
        if(op.type == ops::VALUE)
        {
            std::string prefix = "";

            if(is_int)
            {
                prefix = "(int)";
            }

            if(op.is_constant())
            {
                if(op.get_constant() < 0)
                    return "(" + prefix + to_string_either(op.value_payload.value()) + ")";
            }

            return prefix + to_string_either(op.value_payload.value());
        }

        const operation_desc desc = get_description(op.type);

        if(desc.is_infix)
        {
            return "(" + type_to_string(op.args[0], is_int) + std::string(desc.sym) + type_to_string(op.args[1], is_int) + ")";
        }
        else
        {
           if(op.type == ops::UMINUS)
                return "(-(" + type_to_string(op.args[0], is_int) + "))";

            std::string build;

            if(op.type == ops::UNKNOWN_FUNCTION)
            {
                assert(op.args[0].is_value());

                std::string name = type_to_string(op.args[0]);

                build = name + "(";

                ///starts at one, ignoring the first argument which is the function name
                for(int i=1; i < (int)op.args.size() - 1; i++)
                {
                    build += type_to_string(op.args[i], is_int) + ",";
                }

                if(op.args.size() > 1)
                    build += type_to_string(op.args.back(), is_int);

                build += ")";
            }
            else
            {
                build = std::string(desc.sym) + "(";

                for(int i=0; i < (int)op.args.size() - 1; i++)
                {
                    build += type_to_string(op.args[i], is_int) + ",";
                }

                if(op.args.size() > 0)
                    build += type_to_string(op.args.back(), is_int);

                build += ")";
            }

            return build;
        }
    }

    inline
    value make_op_value(const std::string& str)
    {
        return value(str);
    }

    template<Arithmetic T>
    inline
    value make_op_value(const T& v)
    {
        return value(v);
    }

    template<typename... T>
    inline
    value make_op(ops::type_t type, T&&... args)
    {
        value ret;
        ret.type = type;
        ret.args = {args...};
        ret.value_payload = std::nullopt;

        return ret.flatten();
    }

    inline
    value operator<(const value& d1, const value& d2)
    {
        return make_op(ops::LESS, d1, d2);
    }

    inline
    value operator<=(const value& d1, const value& d2)
    {
        return make_op(ops::LESS_EQUAL, d1, d2);
    }

    inline
    value operator>(const value& d1, const value& d2)
    {
        return make_op(ops::GREATER, d1, d2);
    }

    inline
    value operator>=(const value& d1, const value& d2)
    {
        return make_op(ops::GREATER_EQUAL, d1, d2);
    }

    inline
    value operator==(const value& d1, const value& d2)
    {
        return make_op(ops::EQUAL, d1, d2);
    }

    inline
    value operator+(const value& d1, const value& d2)
    {
        return make_op(ops::PLUS, d1, d2);
    }

    inline
    value operator-(const value& d1, const value& d2)
    {
        return make_op(ops::PLUS, d1, make_op(ops::UMINUS, d2));
    }

    inline
    value operator-(const value& d1)
    {
        return make_op(ops::UMINUS, d1);
    }

    inline
    value operator*(const value& d1, const value& d2)
    {
        return make_op(ops::MULTIPLY, d1, d2);
    }

    inline
    value operator/(const value& d1, const value& d2)
    {
        return make_op(ops::DIVIDE, d1, d2);
    }

    inline
    value operator%(const value& d1, const value& d2)
    {
        return make_op(ops::MODULUS, d1, d2);
    }

    inline
    value operator&(const value& d1, const value& d2)
    {
        return make_op(ops::AND, d1, d2);
    }

    #define UNARY(x, y) inline value x(const value& d1){return make_op(ops::y, d1);}
    #define BINARY(x, y) inline value x(const value& d1, const value& d2){return make_op(ops::y, d1, d2);}
    #define TRINARY(x, y) inline value x(const value& d1, const value& d2, const value& d3){return make_op(ops::y, d1, d2, d3);}

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
    TRINARY(select, SELECT);
    BINARY(pow, POW);
    BINARY(max, MAX);
    BINARY(min, MIN);
    UNARY(lambert_w0, LAMBERT_W0);
    TRINARY(mad, MAD);
    TRINARY(fma, FMA);

    template<typename T>
    inline
    T clamp(const T& val, const T& lower, const T& upper)
    {
        return min(max(val, lower), upper);
    }

    inline
    auto if_v(const value& condition, const value& if_true, const value& if_false)
    {
        if(condition.is_constant())
        {
            double val = condition.get_constant();

            if(val == 0)
                return if_false;
            else
                return if_true;
        }

        return select(if_false, if_true, condition);
    }

    template<typename T>
    inline
    auto if_v(bool condition, const T& if_true, const T& if_false)
    {
        if(condition)
            return if_true;
        else
            return if_false;
    }

    template<typename T, typename U>
    inline
    T divide_with_limit(const T& top, const T& bottom, const U& limit, float tol = 0.001f)
    {
        return dual_types::if_v(bottom >= tol, top / bottom, limit);
    }

    template<typename... T>
    value apply(const value& name, T&&... args)
    {
        return make_op(ops::UNKNOWN_FUNCTION, name, std::forward<T>(args)...);
    }

    inline
    value conjugate(const value& d1)
    {
        return d1;
    }

    inline
    complex<value> psqrt(const complex<value>& d1)
    {
        if(d1.imaginary.is_constant() && d1.imaginary.get_constant() == 0)
        {
            return sqrt(d1.real);
        }

        return sqrt(d1);
    }

    inline
    void test_operation()
    {
        value v1 = 1;
        value v2 = 2;

        value sum = v1 + v2;

        value root_3 = sqrt(sum);

        dual_v<value> test_dual = 1234;

        dual_v<value> test_operator = test_dual * 1111;

        value v = std::string("v");

        value test_op = 2 * (v/2);

        assert(type_to_string(test_op) == "v");

        value test_op2 = (v * 2)/2;

        assert(type_to_string(test_op2) == "v");

        value test_op3 = (2 * ((2 * v/2) / 2) * 2) / 2;

        assert(type_to_string(test_op3) == "v");

        value test_op4 = (2 * v) / v;

        assert(type_to_string(test_op4) == "2.0");

        value test_op5 = (2 * sin(v)) / sin(v);

        assert(type_to_string(test_op5) == "2.0");

        //std::cout << type_to_string(root_3) << std::endl;
    }
}

using dual = dual_types::dual_v<dual_types::value>;
using dual_complex = dual_types::dual_v<dual_types::complex<dual_types::value>>;
using value = dual_types::value;

#endif // DUAL2_HPP_INCLUDED
