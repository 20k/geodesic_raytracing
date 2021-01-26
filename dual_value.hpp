#ifndef DUAL2_HPP_INCLUDED
#define DUAL2_HPP_INCLUDED

#include "dual.hpp"
#include <set>

namespace dual_types
{
    inline
    std::string to_string_s(float v)
    {
        std::ostringstream oss;
        oss << std::setprecision(16) << std::showpoint << v;
        std::string str = oss.str();

        while(str.size() > 0 && str.back() == '0')
            str.pop_back();

        if(str.size() > 0 && str.back() == '.')
            str += "0";

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
            throw std::runtime_error("Nan");

        std::string cstr(in);

        char* ptr = nullptr;
        double val = std::strtod(cstr.c_str(), &ptr);

        if(ptr == cstr.c_str() + cstr.size())
            return val;

        return std::nullopt;
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

            LESS,
            LESS_EQUAL,

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

        if(type == PLUS || type == MINUS || type == MULTIPLY || type == DIVIDE || type == MODULUS ||
           type == LESS || type == LESS_EQUAL)
        {
            ret.is_infix = true;
        }

        std::array syms = make_array(
        "+",
        "-",
        "-",
        "*",
        "/",
        "%",
        "<",
        "<=",
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
    float sign(float in)
    {
        if(in == -0.0f)
            return -0.0f;

        if(in == 0.0f)
            return 0.0f;

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

    struct value;

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

    value operator+(const value& d1, const value& d2);

    value operator-(const value& d1, const value& d2);

    value operator-(const value& d1);

    value operator*(const value& d1, const value& d2);

    value operator/(const value& d1, const value& d2);

    value operator%(const value& d1, const value& d2);

    bool equivalent(const value& d1, const value& d2);

    struct value
    {
        using is_complex = std::false_type;
        static constexpr bool is_dual = false;

        ops::type_t type = ops::NONE;

        std::optional<std::string> value_payload;
        std::vector<value> args;

        //value(){value_payload = to_string_s(0); type = ops::VALUE;}
        value(){value_payload = to_string_s(0); type = ops::VALUE;}
        template<Arithmetic T>
        value(T v){value_payload = to_string_s(v); type = ops::VALUE;}
        value(const std::string& str){value_payload = str; type = ops::VALUE;}
        value(const char* str){value_payload = std::string(str); type = ops::VALUE;}

        bool is_value() const
        {
            return type == ops::VALUE;
        }

        bool is_constant() const
        {
            return is_value() && value_payload.has_value() && get_value(value_payload.value()).has_value();
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

            return get_value(value_payload.value()).value();
        }

        double get(int idx) const
        {
            return get_value(args[idx].value_payload.value()).value();
        }

        void make_value(const std::string& str)
        {
            value_payload = str;
            type = ops::VALUE;
        }

        void set_dual_constant()
        {
            value_payload = to_string_s(0);
            type = ops::VALUE;
        }

        void set_dual_variable()
        {
            value_payload = to_string_s(1);
            type = ops::VALUE;
        }

        #define PROPAGATE1(x, y) if(type == ops::x){return make_op_value(y(get(0)));}
        #define PROPAGATE2(x, y) if(type == ops::x){return make_op_value(y(get(0), get(1)));}
        #define PROPAGATE3(x, y) if(type == ops::x){return make_op_value(y(get(0), get(1), get(2)));}

        template<typename T, typename U, typename V>
        inline
        void configurable_recurse_impl(value& op, const std::vector<std::pair<value*, int>>& op_chain, T&& is_valid_candidate, U&& should_terminate, V&& should_recurse, bool& terminated)
        {
            if(terminated)
                return;

            if(!is_valid_candidate(op))
                return;

            if(should_terminate(op, op_chain))
            {
                terminated = true;
                return;
            }

            if(op.type == ops::VALUE)
                return;

            for(int idx = 0; idx < (int)op.args.size(); idx++)
            {
                //if(op.args[idx].type == ops::VALUE)
                //    continue;

                if(!should_recurse(op, idx))
                    continue;

                std::vector<std::pair<value*, int>> next_chain = op_chain;
                next_chain.push_back({&op.args[idx], idx});

                configurable_recurse_impl(op.args[idx], next_chain, std::forward<T>(is_valid_candidate), std::forward<U>(should_terminate), std::forward<V>(should_recurse), terminated);
            }
        }

        template<typename T, typename U, typename V>
        inline
        void configurable_recurse(value& op, T&& is_valid_candidate, U&& should_terminate, V&& should_recurse)
        {
            bool terminated = false;
            std::vector<std::pair<value*, int>> op_chain{{&op, 0}};

            return configurable_recurse_impl(op, op_chain, std::forward<T>(is_valid_candidate), std::forward<U>(should_terminate), std::forward<V>(should_recurse), terminated);
        }

        bool invasive_flatten()
        {
            bool any_change = false;

            #if 0
            if(type == ops::MULTIPLY || type == ops::DIVIDE)
            {
                auto is_mult_node_or_expr = [](const value& op)
                {
                    return op.type == ops::MULTIPLY || op.type == ops::DIVIDE || op.type == ops::UMINUS || op.type == ops::VALUE || !get_description(op.type).is_infix;
                };

                std::vector<value*> constants;
                std::vector<std::vector<std::pair<value*, int>>> op_chains;

                std::vector<value*> variables;
                std::vector<std::vector<std::pair<value*, int>>> vop_chains;

                auto found_constant = [&](value& op, const std::vector<std::pair<value*, int>>& op_chain)
                {
                    if(op.is_constant())
                    {
                        constants.push_back(&op);
                        op_chains.push_back(op_chain);

                        return false;
                    }

                    if(op.type == ops::VALUE && !op.is_constant())
                    {
                        variables.push_back(&op);
                        vop_chains.push_back(op_chain);

                        return false;
                    }

                    if(!get_description(op.type).is_infix && !op.is_constant())
                    {
                        variables.push_back(&op);
                        vop_chains.push_back(op_chain);

                        return false;
                    }

                    return false;
                };

                auto should_recurse = [](value& op, int idx)
                {
                    if(op.type == ops::MULTIPLY)
                        return true;

                    if(op.type == ops::DIVIDE)
                        return true;

                    if(op.type == ops::UMINUS)
                        return true;

                    return false;
                };

                configurable_recurse(args[0], is_mult_node_or_expr, found_constant, should_recurse);
                configurable_recurse(args[1], is_mult_node_or_expr, found_constant, should_recurse);

                auto propagate_constants = [&](value& base_op, int idx)
                {
                    if(base_op.is_constant())
                    {
                        double my_value = base_op.get_constant();

                        for(int i=0; i < (int)constants.size(); i++)
                        {
                            if(&base_op == constants[i])
                                continue;

                            bool tip = false;

                            if(type == ops::DIVIDE && idx == 1)
                            {
                                tip = true;
                            }

                            for(int kk=1; kk < (int)op_chains[i].size(); kk++)
                            {
                                value* parent_op = op_chains[i][kk - 1].first;

                                if(parent_op->type == ops::DIVIDE)
                                {
                                    if(op_chains[i][kk].second == 1)
                                    {
                                        tip = !tip;
                                    }
                                }
                            }

                            value* op = constants[i];

                            if(!tip)
                                my_value *= op->get_constant();
                            else
                                my_value /= op->get_constant();

                            *op = value(1);

                            any_change = true;
                        }

                        base_op = my_value;

                        constants.clear();
                        op_chains.clear();
                    }
                };

                propagate_constants(args[0], 0);
                propagate_constants(args[1], 1);

                auto propagate_variables = [&](value& base_op, int idx)
                {
                    for(int i=0; i < (int)variables.size(); i++)
                    {
                        if(&base_op == variables[i])
                            continue;

                        bool tip = false;

                        if(type == ops::DIVIDE && idx == 1)
                        {
                            tip = true;
                        }

                        for(int kk=1; kk < (int)vop_chains[i].size(); kk++)
                        {
                            value* parent_op = vop_chains[i][kk - 1].first;

                            if(parent_op->type == ops::DIVIDE)
                            {
                                if(vop_chains[i][kk].second == 1)
                                {
                                    tip = !tip;
                                }
                            }
                        }

                        value* op = variables[i];

                        if(tip)
                        {
                            if(equivalent(*op, base_op))
                            {
                                any_change = true;

                                base_op = value(1);
                                *op = value(1);

                                return;
                            }
                        }
                    }
                };

                propagate_variables(args[0], 0);
                propagate_variables(args[1], 1);
            }

            if(type == ops::PLUS)
            {
                auto is_add_node = [](const value& op)
                {
                    return op.type == ops::PLUS || op.type == ops::UMINUS;
                };

                std::vector<value*> constants;
                std::vector<std::vector<std::pair<value*, int>>> op_chains;

                auto found_constant = [&](value& op, const std::vector<std::pair<value*, int>>& op_chain)
                {
                    if(op.args[0].is_constant())
                    {
                        constants.push_back(&op.args[0]);
                        op_chains.push_back(op_chain);
                    }

                    if(op.args[1].is_constant())
                    {
                        constants.push_back(&op.args[1]);
                        op_chains.push_back(op_chain);
                    }

                    return false;
                };

                auto should_recurse = [](value& op, int idx)
                {
                    if(op.type == ops::VALUE)
                        return false;

                    return true;
                };

                configurable_recurse(args[0], is_add_node, found_constant, should_recurse);
                configurable_recurse(args[1], is_add_node, found_constant, should_recurse);

                //any_change = constants.size() > 0;

                auto propagate_constants = [&](value& base_op)
                {
                    if(base_op.is_constant())
                    {
                        double my_value = base_op.get_constant();

                        for(int i=0; i < (int)constants.size(); i++)
                        {
                            value* op = constants[i];

                            double sign = 1;

                            for(auto [kk, idx] : op_chains[i])
                            {
                                if(kk->type == ops::UMINUS)
                                    sign *= -1;
                            }

                            my_value += op->get_constant() * sign;

                            *op = value(0);

                            any_change = true;
                        }

                        base_op = my_value;

                        constants.clear();
                        op_chains.clear();
                    }
                };

                propagate_constants(args[0]);
                propagate_constants(args[1]);
            }
            #endif // 0

            return any_change;
        }

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

                if(type == ops::LESS)
                    return make_op_value(get(0) < get(1));

                if(type == ops::LESS_EQUAL)
                    return make_op_value(get(0) <= get(1));

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

                if(args[0].is_constant_constraint(is_value_equal<1>))
                    return args[1];

                if(args[1].is_constant_constraint(is_value_equal<1>))
                    return args[0];
            }

            if(type == ops::PLUS)
            {
                if(args[0].is_constant_constraint(is_zero))
                    return args[1];

                if(args[1].is_constant_constraint(is_zero))
                    return args[0];

                if(equivalent(args[0], args[1]))
                    return 2 * args[0];
            }

            if(type == ops::MINUS)
            {
                if(args[0].is_constant_constraint(is_zero))
                    return -args[1];

                if(args[1].is_constant_constraint(is_zero))
                    return args[0];

                if(equivalent(args[0], args[1]))
                    return value(0);
            }

            if(type == ops::DIVIDE)
            {
                if(args[0].is_constant_constraint(is_zero))
                    return value(0);

                if(args[1].is_constant_constraint(is_value_equal<1>))
                    return args[0];

                if(equivalent(args[0], args[1]))
                    return value(1);
            }

            ///ops::MODULUS

            if(type == ops::POW)
            {
                if(args[1].is_constant_constraint(is_value_equal<1>))
                    return args[0];

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

            value ret = *this;

            bool any_dirty = false;

            any_dirty = ret.invasive_flatten();

            recurse = recurse || any_dirty;

            if(recurse)
            {
                for(auto& i : ret.args)
                    i = i.flatten(true);

                if(any_dirty)
                {
                    ret = ret.flatten(false);
                }
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

                if(value_payload.value() == sym)
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
            if(type == LESS)
            {
                return args[0].dual(sym) < args[1].dual(sym);
            }
            if(type == LESS_EQUAL)
            {
                return args[0].dual(sym) <= args[1].dual(sym);
            }

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
            if(type == ops::VALUE && value_payload.value() == sym)
            {
                value_payload = to_string_s(value);
                return;
            }

            for(auto& i : args)
            {
                i.substitute_impl(sym, value);
                i = i.flatten();
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

                v.insert(value_payload.value());
                return;
            }

            for(const auto& val : args)
            {
                get_all_variables_impl(v);
            }
        }

        std::vector<std::string> get_all_variables() const
        {
            std::set<std::string> v;
            get_all_variables_impl(v);

            std::vector<std::string> ret(v.begin(), v.end());

            return ret;
        }
    };

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
    std::string type_to_string(const value& op)
    {
        if(op.type == ops::VALUE)
        {
            if(op.is_constant())
            {
                if(op.get_constant() < 0)
                    return "(" + op.value_payload.value() + ")";
            }

            return op.value_payload.value();
        }

        const operation_desc desc = get_description(op.type);

        if(desc.is_infix)
        {
            return "(" + type_to_string(op.args[0]) + std::string(desc.sym) + type_to_string(op.args[1]) + ")";
        }
        else
        {
           if(op.type == ops::UMINUS)
                return "(-(" + type_to_string(op.args[0]) + "))";

            std::string build = std::string(desc.sym) + "(";

            for(int i=0; i < (int)op.args.size() - 1; i++)
            {
                build += type_to_string(op.args[i]) + ",";
            }

            if(op.args.size() > 0)
                build += type_to_string(op.args.back());

            build += ")";

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
