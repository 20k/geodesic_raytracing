#ifndef DUAL2_HPP_INCLUDED
#define DUAL2_HPP_INCLUDED

#include "dual.hpp"

namespace dual_types
{
    struct operation
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

        type_t type = NONE;

        std::optional<std::string> value_payload;
        std::vector<operation> args;

        operation(){}
        template<Arithmetic T>
        operation(T v){value_payload = to_string_s(v); type = VALUE;}
        operation(const std::string& str){value_payload = str; type = VALUE;}

        bool is_value()
        {
            return type == VALUE;
        }

        bool is_constant()
        {
            return is_value() && value_payload.has_value() && get_value(value_payload.value()).has_value();
        }
    };

    template<typename... T>
    inline
    operation make_op(operation::type_t type, T&&... args)
    {
        operation ret;
        ret.type = type;

        if(type == operation::type_t::VALUE)
        {
            ret = operation{args...};
        }
        else
        {
            ret.args = {args...};
        }

        return ret;
    }
}

#endif // DUAL2_HPP_INCLUDED
