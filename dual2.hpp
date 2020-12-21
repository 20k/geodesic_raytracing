#ifndef DUAL2_HPP_INCLUDED
#define DUAL2_HPP_INCLUDED

#include "dual.hpp"

namespace dual_types
{
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

            SELECT,
            POW,
            MAX,
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

        if(type == PLUS || type == MINUS || type == MULTIPLY || type == DIVIDE ||
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
        "select",
        "pow",
        "max",
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

    struct operation;

    operation make_op_value(const std::string& str);
    template<Arithmetic T>
    operation make_op_value(const T& v);

    struct operation
    {
        ops::type_t type = ops::NONE;

        std::optional<std::string> value_payload;
        std::vector<operation> args;

        operation(){}
        template<Arithmetic T>
        operation(T v){value_payload = to_string_s(v); type = ops::VALUE;}
        operation(const std::string& str){value_payload = str; type = ops::VALUE;}

        bool is_value() const
        {
            return type == ops::VALUE;
        }

        bool is_constant() const
        {
            return is_value() && value_payload.has_value() && get_value(value_payload.value()).has_value();
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

        operation flatten() const
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
                PROPAGATE3(SELECT, select);
                PROPAGATE2(POW, std::pow);
                PROPAGATE2(MAX, std::max);
            }

            if(type == ops::SELECT)
            {
                if(args[2].is_constant())
                    return args[2].get_constant() > 0 ? args[1] : args[0];
            }

            return *this;
        }
    };

    inline
    std::string type_to_string(const operation& op)
    {
        if(op.type == ops::VALUE)
            return op.value_payload.value();

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
    operation make_op_value(const std::string& str)
    {
        return operation(str);
    }

    template<Arithmetic T>
    inline
    operation make_op_value(const T& v)
    {
        return operation(v);
    }

    template<typename... T>
    inline
    operation make_op(ops::type_t type, T&&... args)
    {
        operation ret;
        ret.type = type;
        ret.args = {args...};

        return ret.flatten();
    }

    inline
    operation operator<(const operation& d1, const operation& d2)
    {
        return make_op(ops::LESS, d1, d2);
    }

    inline
    operation operator<=(const operation& d1, const operation& d2)
    {
        return make_op(ops::LESS_EQUAL, d1, d2);
    }

    inline
    operation operator+(const operation& d1, const operation& d2)
    {
        return make_op(ops::PLUS, d1, d2);
    }

    inline
    operation operator-(const operation& d1, const operation& d2)
    {
        return make_op(ops::MINUS, d1, d2);
    }

    inline
    operation operator-(const operation& d1)
    {
        return make_op(ops::UMINUS, d1);
    }

    inline
    operation operator*(const operation& d1, const operation& d2)
    {
        return make_op(ops::MULTIPLY, d1, d2);
    }

    inline
    operation operator/(const operation& d1, const operation& d2)
    {
        return make_op(ops::DIVIDE, d1, d2);
    }

    #define UNARY(x, y) inline operation x(const operation& d1){return make_op(ops::y, d1);}
    #define BINARY(x, y) inline operation x(const operation& d1, const operation& d2){return make_op(ops::y, d1, d2);}
    #define TRINARY(x, y) inline operation x(const operation& d1, const operation& d2, const operation& d3){return make_op(ops::y, d1, d2, d3);}

    UNARY(sqrt, SQRT);
    UNARY(psqrt, SQRT);
    BINARY(pow, POW);
    UNARY(log, LOG);
    UNARY(fabs, FABS);
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
    UNARY(signbit, SIGNBIT);
    TRINARY(select, SELECT);

    inline
    void test_operation()
    {
        operation v1 = 1;
        operation v2 = 2;

        operation sum = v1 + v2;

        operation root_3 = sqrt(sum);

        dual_v<operation> test_dual = 1234;

        dual_v<operation> test_operator = test_dual * 1111;

        std::cout << type_to_string(root_3) << std::endl;
    }
}

#endif // DUAL2_HPP_INCLUDED
