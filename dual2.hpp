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

        bool is_value()
        {
            return type == ops::VALUE;
        }

        bool is_constant()
        {
            return is_value() && value_payload.has_value() && get_value(value_payload.value()).has_value();
        }

        double get_constant()
        {
            assert(is_constant());

            return get_value(value_payload.value()).value();
        }

        double get(int idx)
        {
            return get_value(args[idx].value_payload.value()).value();
        }

        operation flatten()
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
            }
        }
    };

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
    }
}

#endif // DUAL2_HPP_INCLUDED
