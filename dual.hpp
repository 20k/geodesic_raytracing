#ifndef DUAL_HPP_INCLUDED
#define DUAL_HPP_INCLUDED

#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <type_traits>

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

namespace dual_types
{
    template<typename T>
    concept Arithmetic = std::is_arithmetic_v<T>;

    struct symbol
    {
        std::string sym = "0.0";

        symbol(){}
        symbol(const std::string& value){sym = value;}

        template<Arithmetic T>
        symbol(T v){sym = to_string_s(v);}

        void set_dual_constant()
        {
            sym = "0.0";
        }

        void set_dual_variable()
        {
            sym = "1.0";
        }
    };

    inline
    std::string type_to_string(const symbol& sym)
    {
        return sym.sym;
    }

    template<typename T>
    struct complex
    {
        using underlying_type = T;

        T real;
        T imaginary;

        complex(){}
        complex(const std::string& v1, const std::string& v2) : real(v1), imaginary(v2) {}
        complex(float v1, float v2) : real(v1), imaginary(v2) {}
        complex(float v1) : real(v1), imaginary(0) {}
        complex(T v1, T v2) : real(v1), imaginary(v2) {}
        complex(T v1) : real(v1) {}

        void set_dual_constant()
        {
            real.set_dual_constant();
            imaginary.set_dual_constant();
        }

        void set_dual_variable()
        {
            real.set_dual_variable();
            imaginary.set_dual_variable();
        }
    };

    template<typename T>
    struct dual_v
    {
        T real = T();
        T dual = T();

        dual_v(){}
        dual_v(const T& _real, const T& _dual)
        {
            real = _real;
            dual = _dual;
        }

        template<typename U>
        requires std::is_constructible_v<T, U>
        dual_v(const U& _real) : real(_real)
        {
            dual = T();
        }

        template<typename U>
        requires std::is_constructible_v<T, U>
        dual_v(const dual_v<U>& other) : real(other.real), dual(other.dual)
        {

        }

        void make_constant(T val)
        {
            real = val;
            dual.set_dual_constant();
        }

        void make_variable(T val)
        {
            real = val;
            dual.set_dual_variable();
        }
    };

    template<typename T, typename U>
    concept DualValue = std::is_constructible_v<dual_v<T>, U>;

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

    std::string unary(const std::string& v1, const std::string& op);

    inline
    std::string infix(const std::string& v1, const std::string& v2, const std::string& op)
    {
        auto c1 = get_value(v1);
        auto c2 = get_value(v2);

        //std::cout << "V " << op << " " << v1 << " " << v2 << std::endl;

        if(op == "*")
        {
            if((c1.has_value() && c1.value() == 0) || (c2.has_value() && c2.value() == 0))
                return to_string_s(0);

            if(c1.has_value() && c2.has_value())
                return to_string_s(c1.value() * c2.value());

            if(c1.has_value() && c1.value() == 1)
                return v2;

            if(c2.has_value() && c2.value() == 1)
                return v1;

            if(c1.has_value() && c1.value() == -1)
                return unary(v2, "-");

            if(c2.has_value() && c2.value() == -1)
                return unary(v1, "-");
        }

        if(op == "+")
        {
            if(c1.has_value() && c1.value() == 0)
                return v2;

            if(c2.has_value() && c2.value() == 0)
                return v1;

            if(c1.has_value() && c2.has_value())
                return to_string_s(c1.value() + c2.value());

            if(v1 == v2)
                return infix("2", v1, "*");
        }

        if(op == "-")
        {
            if(c1.has_value() && c1.value() == 0)
            {
                if(c2.has_value() && c2.value() == 0)
                    return to_string_s(0);

                if(c2.has_value())
                {
                    return "(" + to_string_s(-c2.value()) + ")";
                }

                return "(-(" + v2 + "))";
            }

            if(c2.has_value() && c2.value() == 0)
                return v1;

            if(c1.has_value() && c2.has_value() && c1.value() == c2.value())
                return to_string_s(0);

            if(c1.has_value() && c2.has_value())
                return to_string_s(c1.value() - c2.value());

            if(v1 == v2)
                return to_string_s(0);

            return "(" + v1 + op + "(" + v2 + "))";
        }

        if(op == "/")
        {
            if(c1.has_value() && c1.value() == 0)
                return to_string_s(0);

            if(c2.has_value() && c2.value() == 1)
                return v1;

            if(c1.has_value() && c2.has_value())
                return to_string_s(c1.value() / c2.value());

            #define RECIPROCAL_CONSTANTS
            #ifdef RECIPROCAL_CONSTANTS
            if(!c1.has_value() && c2.has_value())
            {
                if(c2.value() == 0)
                    throw std::runtime_error("c2 cannot be 0 in / expression");

                return infix(v1, to_string_s(1/c2.value()), "*");
            }
            #endif // RECIPROCAL_CONSTANTS

            #define EMIT_NATIVE_RECIP
            #ifdef EMIT_NATIVE_RECIP
            if(c1.has_value() && c1.value() == 1 && !c2.has_value())
                return unary(v2, "native_recip");

            if(c1.has_value() && c1.value() == -1 && !c2.has_value())
                return unary(unary(v2, "native_recip"), "-");
            #endif // EMIT_NATIVE_RECIP

            if(v1 == v2)
                return to_string_s(1);
        }

        if(op == "<")
        {
            if(c1.has_value() && c2.has_value())
                return to_string_s((int)(c1.value() < c2.value()));

            return "((float)(" + v1 + op + v2 + "))";
        }

        if(op == "<=")
        {
            if(c1.has_value() && c2.has_value())
                return to_string_s((int)(c1.value() <= c2.value()));

            return "((float)(" + v1 + op + v2 + "))";
        }

        return "(" + v1 + op + v2 + ")";
    }

    inline
    std::string outer(const std::string& v1, const std::string& v2, const std::string& op)
    {
        auto c1 = get_value(v1);
        auto c2 = get_value(v2);

        if(op == "pow")
        {
            if(c2.has_value() && c2.value() == 0)
                return to_string_s(1);

            if(c1.has_value() && c2.has_value())
                return to_string_s(pow(c1.value(), c2.value()));
        }

        if(op == "max")
        {
            if(c1.has_value() && c2.has_value())
                return to_string_s(max(c1.value(), c2.value()));
        }

        return op + "(" + v1 + "," + v2 + ")";
    }

    inline
    std::string threearg(const std::string& v1, const std::string& v2, const std::string& v3, const std::string& op)
    {
        if(op == "select")
        {
            auto c1 = get_value(v1);
            auto c2 = get_value(v2);
            auto c3 = get_value(v3);

            if(c1 && c2 && c3)
                return c3.value() != 0 ? to_string_s(c2.value()) : to_string_s(c1.value());

            if(c3)
                return c3.value() != 0 ? v2 : v1;

            return op + "(" + v1 + "," + v2 + ",((int)(" + v3 + ")))";
        }

        return op + "(" + v1 + "," + v2 + "," + v3 + ")";
    }

    inline
    std::string unary(const std::string& v1, const std::string& op)
    {
        auto c1 = get_value(v1);

        if(op == "-")
        {
            if(c1.has_value())
                return "(" + to_string_s(-c1.value()) + ")";
        }

        if(op == "sin" || op == "native_sin")
        {
            if(c1.has_value())
                return to_string_s(sin(c1.value()));
        }

        if(op == "cos" || op == "native_cos")
        {
            if(c1.has_value())
                return to_string_s(cos(c1.value()));
        }

        if(op == "tan" || op == "native_tan")
        {
            if(c1.has_value())
                return to_string_s(tan(c1.value()));
        }

        if(op == "native_exp")
        {
            if(c1.has_value())
                return to_string_s(exp(c1.value()));
        }

        if(op == "sqrt" || op == "native_sqrt")
        {
            if(c1.has_value())
                return to_string_s(sqrt(c1.value()));
        }

        if(op == "sinh")
        {
            if(c1.has_value())
                return to_string_s(sinh(c1.value()));
        }

        if(op == "cosh")
        {
            if(c1.has_value())
                return to_string_s(cosh(c1.value()));
        }

        if(op == "tanh")
        {
            if(c1.has_value())
                return to_string_s(tanh(c1.value()));
        }

        if(op == "log" || op == "native_log")
        {
            if(c1.has_value())
                return to_string_s(log(c1.value()));
        }

        if(op == "isfinite")
        {
            if(c1.has_value())
                return to_string_s(isfinite(c1.value()));
        }

        if(op == "signbit")
        {
            if(c1.has_value())
                return to_string_s((int)(c1.value() < 0));
        }

        if(op == "sign")
        {
            if(c1.has_value())
            {
                ///very deliberately a float
                float val = c1.value();

                if(val == -0.0f)
                    return to_string_s(-0.0f);

                if(val == 0.0f)
                    return to_string_s(0.0f);

                if(val > 0)
                    return to_string_s(1);

                if(val < 0)
                    return to_string_s(-1);

                if(std::isnan(val))
                    return to_string_s(0);

                throw std::runtime_error("Bad value for sign");
            }
        }

        if(op == "fabs")
        {
            if(c1.has_value())
                return to_string_s(fabs(c1.value()));
        }

        if(op == "native_recip")
        {
            if(c1.has_value())
                return to_string_s(1/c1.value());
        }

        if(op == "-")
        {
            return "(" + op + "(" + v1 + "))";
        }

        return op + "(" + v1 + ")";
    }

    inline
    dual_types::symbol operator<(const dual_types::symbol& d1, const dual_types::symbol& d2)
    {
        return infix(d1.sym, d2.sym, "<");
    }

    inline
    dual_types::symbol operator<=(const dual_types::symbol& d1, const dual_types::symbol& d2)
    {
        return infix(d1.sym, d2.sym, "<=");
    }

    inline
    dual_types::symbol operator+(const dual_types::symbol& d1, const dual_types::symbol& d2)
    {
        return dual_types::symbol(infix(d1.sym, d2.sym, "+"));
    }

    inline
    dual_types::symbol operator+(const dual_types::symbol& d1, float v)
    {
        return d1 + dual_types::symbol(to_string_s(v));
    }

    inline
    dual_types::symbol operator+(float v, const dual_types::symbol& d1)
    {
        return dual_types::symbol(to_string_s(v)) + d1;
    }

    inline
    dual_types::symbol operator-(const dual_types::symbol& d1, const dual_types::symbol& d2)
    {
        return dual_types::symbol(infix(d1.sym, d2.sym, "-"));
    }

    inline
    dual_types::symbol operator-(const dual_types::symbol& d1, float v)
    {
        return d1 - dual_types::symbol(to_string_s(v));
    }

    inline
    dual_types::symbol operator-(float v, const dual_types::symbol& d1)
    {
        return dual_types::symbol(to_string_s(v)) - d1;
    }

    inline
    dual_types::symbol operator-(const dual_types::symbol& d1)
    {
        return dual_types::symbol(unary(d1.sym, "-"));
    }

    inline
    dual_types::symbol operator*(const dual_types::symbol& d1, const dual_types::symbol& d2)
    {
        return dual_types::symbol(infix(d1.sym, d2.sym, "*"));
    }

    inline
    dual_types::symbol operator*(const dual_types::symbol& d1, float v)
    {
        return d1 * dual_types::symbol(to_string_s(v));
    }

    inline
    dual_types::symbol operator*(float v, const dual_types::symbol& d1)
    {
        return dual_types::symbol(to_string_s(v)) * d1;
    }

    inline
    dual_types::symbol operator/(const dual_types::symbol& d1, const dual_types::symbol& d2)
    {
        return dual_types::symbol(infix(d1.sym, d2.sym, "/"));
    }

    inline
    dual_types::symbol operator/(const dual_types::symbol& d1, float v)
    {
        return d1 / dual_types::symbol(to_string_s(v));
    }

    inline
    dual_types::symbol operator/(float v, const dual_types::symbol& d1)
    {
        return dual_types::symbol(to_string_s(v)) / d1;
    }

    inline
    dual_types::symbol sqrt(const dual_types::symbol& d1)
    {
        return dual_types::symbol(unary(d1.sym, "native_sqrt"));
    }

    inline
    dual_types::symbol psqrt(const dual_types::symbol& d1)
    {
        return sqrt(d1);
    }

    inline
    dual_types::symbol pow(const dual_types::symbol& d1, const dual_types::symbol& d2)
    {
        return dual_types::symbol(outer(d1.sym, d2.sym, "pow"));
    }

    inline
    dual_types::symbol log(const dual_types::symbol& d1)
    {
        return dual_types::symbol(unary(d1.sym, "native_log"));
    }

    inline
    dual_types::symbol fabs(const dual_types::symbol& d1)
    {
        return dual_types::symbol(unary(d1.sym, "fabs"));
    }

    inline
    dual_types::symbol exp(const dual_types::symbol& d1)
    {
        return dual_types::symbol(unary(d1.sym, "native_exp"));
    }

    inline
    dual_types::symbol sin(const dual_types::symbol& d1)
    {
        return dual_types::symbol(unary(d1.sym, "native_sin"));
    }

    inline
    dual_types::symbol cos(const dual_types::symbol& d1)
    {
        return dual_types::symbol(unary(d1.sym, "native_cos"));
    }

    inline
    dual_types::symbol tan(const dual_types::symbol& d1)
    {
        return dual_types::symbol(unary(d1.sym, "native_tan"));
    }

    inline
    dual_types::symbol sinh(const dual_types::symbol& d1)
    {
        return dual_types::symbol(unary(d1.sym, "sinh"));
    }

    inline
    dual_types::symbol cosh(const dual_types::symbol& d1)
    {
        return dual_types::symbol(unary(d1.sym, "cosh"));
    }

    inline
    dual_types::symbol tanh(const dual_types::symbol& d1)
    {
        return dual_types::symbol(unary(d1.sym, "tanh"));
    }

    inline
    dual_types::symbol asin(const dual_types::symbol& d1)
    {
        return dual_types::symbol(unary(d1.sym, "asin"));
    }

    inline
    dual_types::symbol acos(const dual_types::symbol& d1)
    {
        return dual_types::symbol(unary(d1.sym, "acos"));
    }

    inline
    dual_types::symbol atan(const dual_types::symbol& d1)
    {
        return dual_types::symbol(unary(d1.sym, "atan"));
    }

    inline
    dual_types::symbol atan2(const dual_types::symbol& d1, const dual_types::symbol& d2)
    {
        return dual_types::symbol(outer(d1.sym, d2.sym, "atan2"));
    }

    inline
    dual_types::symbol signbit(const dual_types::symbol& d1)
    {
        return unary(d1.sym, "signbit");
    }

    inline
    dual_types::symbol select(const dual_types::symbol& d1, const dual_types::symbol& d2, const dual_types::symbol& d3)
    {
        return threearg(d1.sym, d2.sym, d3.sym, "select");
    }

    template<typename T>
    inline
    dual_types::complex<T> csqrt(const T& d1)
    {
        T is_negative = signbit(d1);

        T positive_sqrt = sqrt(fabs(d1));

        return dual_types::complex<T>(select(positive_sqrt, 0, is_negative), select(0, positive_sqrt, is_negative));
    }

    inline
    dual_types::symbol lambert_w0(const dual_types::symbol& d1)
    {
        return dual_types::symbol(unary(d1.sym, "lambert_w0"));
    }

    inline
    dual_types::symbol conjugate(const dual_types::symbol& d1)
    {
        return d1;
    }

    inline
    dual_types::symbol makefinite(const dual_types::symbol& d1)
    {
        return dual_types::symbol(threearg("0.f", d1.sym, unary(d1.sym, "isfinite"), "select"));
    }

    inline
    dual_types::symbol length(const dual_types::symbol& d1, const dual_types::symbol& d2, const dual_types::symbol& d3)
    {
        return dual_types::symbol("length((float3){" + d1.sym + "," + d2.sym + "," + d3.sym + "})");
    }

    inline
    dual_types::symbol fast_length(const dual_types::symbol& d1, const dual_types::symbol& d2, const dual_types::symbol& d3)
    {
        return dual_types::symbol("fast_length((float3){" + d1.sym + "," + d2.sym + "," + d3.sym + "})");
    }

    inline
    dual_types::symbol sign(const dual_types::symbol& d1)
    {
        return unary(d1.sym, "sign");
    }

    using complex_v = dual_types::complex<dual_types::symbol>;

    template<typename T>
    inline
    complex<T> operator+(const complex<T>& c1, const complex<T>& c2)
    {
        return complex<T>(c1.real + c2.real, c1.imaginary + c2.imaginary);
    }

    template<typename T>
    inline
    complex<T> operator-(const complex<T>& c1, const complex<T>& c2)
    {
        return complex<T>(c1.real - c2.real, c1.imaginary - c2.imaginary);
    }

    template<typename T>
    inline
    complex<T> operator-(const complex<T>& c1)
    {
        return complex<T>(-c1.real, -c1.imaginary);
    }

    template<typename T>
    inline
    complex<T> operator*(const complex<T>& c1, const complex<T>& c2)
    {
        return complex<T>(c1.real * c2.real - c1.imaginary * c2.imaginary, c1.imaginary * c2.real + c1.real * c2.imaginary);
    }

    template<typename T>
    inline
    complex<T> operator/(const complex<T>& c1, const complex<T>& c2)
    {
        T divisor = c2.real * c2.real + c2.imaginary * c2.imaginary;

        return complex<T>((c1.real * c2.real + c1.imaginary * c2.imaginary) / divisor, (c1.imaginary * c2.real - c1.real * c2.imaginary) / divisor);
    }

    template<typename T>
    inline
    complex<T> sin(const complex<T>& c1)
    {
        return complex<T>(sin(c1.real) * cosh(c1.imaginary), cos(c1.real) * sinh(c1.imaginary));
    }

    template<typename T>
    inline
    complex<T> cos(const complex<T>& c1)
    {
        return complex<T>(cos(c1.real) * cosh(c1.imaginary), -sin(c1.real) * sinh(c1.imaginary));
    }

    template<typename T>
    inline
    complex<T> conjugate(const complex<T>& c1)
    {
        return complex<T>(c1.real, -c1.imaginary);
    }

    template<typename T>
    inline
    complex<T> makefinite(const complex<T>& c1)
    {
        return complex<T>(makefinite(c1.real), makefinite(c1.imaginary));
    }

    template<typename T>
    inline
    T fabs(const complex<T>& c1)
    {
        return sqrt(c1.real * c1.real + c1.imaginary * c1.imaginary);
    }

    template<typename T>
    inline
    T Imaginary(const complex<T>& c1)
    {
        return c1.imaginary;
    }

    template<typename T>
    inline
    T Real(const complex<T>& c1)
    {
        return c1.real;
    }

    inline
    dual_types::symbol max(const dual_types::symbol& d1, const dual_types::symbol& d2)
    {
        return outer(d1.sym, d2.sym, "max");
    }

    template<typename T>
    inline
    complex<T> sqrt(const complex<T>& d1)
    {
        /*auto i_cst_opt = get_value(d1.imaginary.sym);

        if(i_cst_opt.has_value() && i_cst_opt.value() == 0)
            return csqrt(d1.real);*/

        dual_types::symbol r_part = sqrt(max((d1.real + sqrt(d1.real * d1.real + d1.imaginary * d1.imaginary))/2, 0));
        dual_types::symbol i_part = sign(d1.imaginary) * sqrt(max((-d1.real + sqrt(d1.real * d1.real + d1.imaginary * d1.imaginary))/2, 0));

        return complex<T>(r_part, i_part);
    }

    ///if this is known to have no imaginary components, the real component is guaranteed to be >= 0
    ///otherwise, it calls regular complex square root
    inline
    dual_types::complex<dual_types::symbol> psqrt(const dual_types::complex<dual_types::symbol>& d1)
    {
        auto v = get_value(d1.imaginary.sym);

        if(v.has_value() && v.value() == 0)
            return sqrt(d1.real);

        return sqrt(d1);
    }

    template<typename T>
    inline
    complex<T> pow(const complex<T>& d1, int exponent)
    {
        complex<T> ret = d1;

        for(int i=0; i < exponent - 1; i++)
        {
            ret = ret * d1;
        }

        return ret;
    }

    template<typename T>
    inline
    dual_types::dual_v<T> operator+(const dual_types::dual_v<T>& d1, const dual_types::dual_v<T>& d2)
    {
        return dual_types::dual_v<T>(d1.real + d2.real, d1.dual + d2.dual);
    }

    template<typename T, typename U>
    requires DualValue<T, U>
    inline
    dual_types::dual_v<T> operator+(const dual_types::dual_v<T>& d1, const U& v)
    {
        return dual_types::dual_v<T>(d1.real + T(v), d1.dual);
    }

    template<typename T, typename U>
    requires DualValue<T, U>
    inline
    dual_types::dual_v<T> operator+(const U& v, const dual_types::dual_v<T>& d1)
    {
        return dual_types::dual_v<T>(T(v) + d1.real, d1.dual);
    }

    template<typename T, typename U>
    requires DualValue<T, U>
    inline
    dual_types::dual_v<T> operator+(const dual_types::dual_v<T>& d1, const dual_types::dual_v<U>& d2)
    {
        return d1 + dual_types::dual_v<T>(d2.real, d2.dual);
    }

    template<typename T, typename U>
    requires DualValue<T, U>
    inline
    dual_types::dual_v<T> operator+(const dual_types::dual_v<U>& d1, const dual_types::dual_v<T>& d2)
    {
        return dual_types::dual_v<T>(d1.real, d1.dual) + d2;
    }

    template<typename T>
    inline
    void operator+=(dual_types::dual_v<T>& d1, const dual_types::dual_v<T>& d2)
    {
        d1 = d1 + d2;
    }

    template<typename T>
    inline
    dual_types::dual_v<T> operator-(const dual_types::dual_v<T>& d1, const dual_types::dual_v<T>& d2)
    {
        return dual_types::dual_v<T>(d1.real - d2.real, d1.dual - d2.dual);
    }

    template<typename T, typename U>
    requires DualValue<T, U>
    inline
    dual_types::dual_v<T> operator-(const dual_types::dual_v<T>& d1, const U& v)
    {
        return dual_types::dual_v<T>(d1.real - T(v), d1.dual);
    }

    template<typename T, typename U>
    requires DualValue<T, U>
    inline
    dual_types::dual_v<T> operator-(const U& v, const dual_types::dual_v<T>& d1)
    {
        return dual_types::dual_v<T>(T(v) - d1.real, -d1.dual);
    }

    template<typename T, typename U>
    requires DualValue<T, U>
    inline
    dual_types::dual_v<T> operator-(const dual_types::dual_v<T>& d1, const dual_types::dual_v<U>& d2)
    {
        return d1 - dual_types::dual_v<T>(d2.real, d2.dual);
    }

    template<typename T, typename U>
    requires DualValue<T, U>
    inline
    dual_types::dual_v<T> operator-(const dual_types::dual_v<U>& d1, const dual_types::dual_v<T>& d2)
    {
        return dual_types::dual_v<T>(d1.real, d1.dual) - d2;
    }

    template<typename T>
    inline
    dual_types::dual_v<T> operator-(const dual_types::dual_v<T>& d1)
    {
        return dual_types::dual_v<T>(-d1.real, -d1.dual);
    }

    template<typename T>
    inline
    dual_types::dual_v<T> operator*(const dual_types::dual_v<T>& d1, const dual_types::dual_v<T>& d2)
    {
        return dual_types::dual_v<T>(d1.real * d2.real, d1.real * d2.dual + d2.real * d1.dual);
    }

    template<typename T, typename U>
    requires DualValue<T, U>
    inline
    dual_types::dual_v<T> operator*(const dual_types::dual_v<T>& d1, const U& v)
    {
        return d1 * dual_types::dual_v<T>(T(v), T());
    }

    template<typename T, typename U>
    requires DualValue<T, U>
    inline
    dual_types::dual_v<T> operator*(const U& v, const dual_types::dual_v<T>& d1)
    {
        return dual_types::dual_v<T>(T(v), T()) * d1;
    }

    template<typename T, typename U>
    requires DualValue<T, U>
    inline
    dual_types::dual_v<T> operator*(const dual_types::dual_v<T>& d1, const dual_types::dual_v<U>& d2)
    {
        return d1 * dual_types::dual_v<T>(d2.real, d2.dual);
    }

    template<typename T, typename U>
    requires DualValue<T, U>
    inline
    dual_types::dual_v<T> operator*(const dual_types::dual_v<U>& d1, const dual_types::dual_v<T>& d2)
    {
        return dual_types::dual_v<T>(d1.real, d1.dual) * d2;
    }

    template<typename T>
    inline
    dual_types::dual_v<T> operator/(const dual_types::dual_v<T>& d1, const dual_types::dual_v<T>& d2)
    {
        return dual_types::dual_v<T>(d1.real / d2.real, ((d1.dual * d2.real - d1.real * d2.dual) / (d2.real * d2.real)));
    }

    template<typename T, typename U>
    requires DualValue<T, U>
    inline
    dual_types::dual_v<T> operator/(const dual_types::dual_v<T>& d1, const U& v)
    {
        return d1 / dual_types::dual_v<T>(T(v), T());
    }

    template<typename T, typename U>
    requires DualValue<T, U>
    inline
    dual_types::dual_v<T> operator/(const U& v, const dual_types::dual_v<T>& d1)
    {
        return dual_types::dual_v<T>(T(v), T()) / d1;
    }

    template<typename T, typename U>
    requires DualValue<T, U>
    inline
    dual_types::dual_v<T> operator/(const dual_types::dual_v<T>& d1, const dual_types::dual_v<U>& d2)
    {
        return d1 / dual_types::dual_v<T>(d2.real, d2.dual);
    }

    template<typename T, typename U>
    requires DualValue<T, U>
    inline
    dual_types::dual_v<T> operator/(const dual_types::dual_v<U>& d1, const dual_types::dual_v<T>& d2)
    {
        return dual_types::dual_v<T>(d1.real, d1.dual) / d2;
    }

    template<typename T>
    inline
    dual_types::dual_v<T> sqrt(const dual_types::dual_v<T>& d1)
    {
        return dual_types::dual_v<T>(sqrt(d1.real), T(0.5f) * d1.dual / sqrt(d1.real));
    }

    ///if this has no imaginary components, its guaranteed to be >= 0
    ///if it has imaginary components, all bets are off
    template<typename T>
    inline
    dual_types::dual_v<T> psqrt(const dual_types::dual_v<T>& d1)
    {
        return dual_types::dual_v<T>(psqrt(d1.real), T(0.5f) * d1.dual / psqrt(d1.real));
    }

    template<typename T>
    inline
    dual_types::dual_v<dual_types::complex<T>> csqrt(const dual_types::dual_v<T>& d1)
    {
        return dual_types::dual_v<dual_types::complex<T>>(csqrt(d1.real), complex<T>(0.5f * d1.dual, 0) / csqrt(d1.real));
    }

    template<typename T>
    inline
    dual_types::dual_v<T> pow(const dual_types::dual_v<T>& d1, const dual_types::dual_v<T>& d2)
    {
        return dual_types::dual_v<T>(pow(d1.real, d2.real), pow(d1.real, d2.real) * (d1.dual * (d2.real / d1.real) + d2.dual * log(d1.real)));
    }

    template<typename T, typename U>
    requires DualValue<T, U>
    inline
    dual_types::dual_v<T> pow(const dual_types::dual_v<T>& d1, const U& d2)
    {
        static_assert(!std::is_same_v<U, complex_v> && !std::is_same_v<U, dual_types::dual_v<complex_v>>);

        if constexpr(std::is_same_v<T, complex_v>)
        {
            static_assert(std::is_same_v<U, int>);

            return dual_types::dual_v<T>(pow(d1.real, d2), pow(d1.real, d2 - 1) * T(d2) * d1.dual);
        }
        else
        {
            return dual_types::dual_v<T>(pow(d1.real, T(d2)), pow(d1.real, T(d2 - 1)) * T(d2) * d1.dual);
        }
    }

    template<typename T>
    inline
    dual_types::dual_v<T> log(const dual_types::dual_v<T>& d1)
    {
        return dual_types::dual_v<T>(log(d1.real), d1.dual / d1.real);
    }

    template<typename T>
    inline
    dual_types::dual_v<T> fabs(const dual_types::dual_v<T>& d1)
    {
        return dual_types::dual_v<T>(fabs(d1.real), d1.real * d1.dual / fabs(d1.real));
    }

    ///https://math.stackexchange.com/questions/2352341/the-derivative-of-absolute-value-of-complex-function-fx-z-where-x-in-math
    template<typename T>
    inline
    dual_types::dual_v<T> fabs(const dual_types::dual_v<dual_types::complex<T>>& d1)
    {
        return dual_types::dual_v<T>(fabs(d1.real), Real(d1.real * conjugate(d1.dual)) / fabs(d1.real));
    }

    template<typename T>
    inline
    dual_types::dual_v<T> exp(const dual_types::dual_v<T>& d1)
    {
        return dual_types::dual_v<T>(exp(d1.real), d1.dual * exp(d1.real));
    }

    template<typename T>
    inline
    dual_types::dual_v<T> sin(const dual_types::dual_v<T>& d1)
    {
        return dual_types::dual_v<T>(sin(d1.real), d1.dual * cos(d1.real));
    }

    template<typename T>
    inline
    dual_types::dual_v<T> cos(const dual_types::dual_v<T>& d1)
    {
        return dual_types::dual_v<T>(cos(d1.real), -d1.dual * sin(d1.real));
    }

    template<typename T>
    inline
    dual_types::dual_v<T> sec(const dual_types::dual_v<T>& d1)
    {
        return 1/cos(d1);
    }

    template<typename T>
    inline
    dual_types::dual_v<T> tan(const dual_types::dual_v<T>& d1)
    {
        return dual_types::dual_v<T>(tan(d1.real), d1.dual / (cos(d1.real) * cos(d1.real)));
    }

    template<typename T>
    inline
    dual_types::dual_v<T> sinh(const dual_types::dual_v<T>& d1)
    {
        return dual_types::dual_v<T>(sinh(d1.real), d1.dual * cosh(d1.real));
    }

    template<typename T>
    inline
    dual_types::dual_v<T> cosh(const dual_types::dual_v<T>& d1)
    {
        return dual_types::dual_v<T>(cosh(d1.real), d1.dual * sinh(d1.real));
    }

    template<typename T>
    inline
    dual_types::dual_v<T> tanh(const dual_types::dual_v<T>& d1)
    {
        return dual_types::dual_v<T>(tanh(d1.real), d1.dual * (1 - tanh(d1.real) * tanh(d1.real)));
    }

    template<typename T>
    inline
    dual_types::dual_v<T> asin(const dual_types::dual_v<T>& d1)
    {
        return dual_types::dual_v<T>(asin(d1.real), d1.dual / sqrt(1 - d1.real * d1.real));
    }

    template<typename T>
    inline
    dual_types::dual_v<T> acos(const dual_types::dual_v<T>& d1)
    {
        return dual_types::dual_v<T>(acos(d1.real), -d1.dual / sqrt(1 - d1.real * d1.real));
    }

    template<typename T>
    inline
    dual_types::dual_v<T> atan(const dual_types::dual_v<T>& d1)
    {
        return dual_types::dual_v<T>(atan(d1.real), d1.dual / (1 + d1.real * d1.real));
    }

    template<typename T>
    inline
    dual_types::dual_v<T> atan2(const dual_types::dual_v<T>& d1, const dual_types::dual_v<T>& d2)
    {
        return dual_types::dual_v<T>(atan2(d1.real, d2.real), (-d1.real * d2.dual / (d2.real * d2.real + d1.real * d1.real)) + d1.dual * d2.real / (d2.real * d2.real + d1.real * d1.real));
    }

    template<typename T>
    inline
    dual_types::dual_v<T> lambert_w0(const dual_types::dual_v<T>& d1)
    {
        return dual_types::dual_v<T>(lambert_w0(d1.real), d1.dual * lambert_w0(d1.real) / (d1.real * lambert_w0(d1.real) + d1.real));
    }

    ///https://math.stackexchange.com/questions/1052500/what-is-the-general-definition-of-the-conjugate-of-a-multiple-component-number
    template<typename T>
    inline
    dual_types::dual_v<T> conjugate(const dual_types::dual_v<T>& d1)
    {
        return dual_types::dual_v<T>(conjugate(d1.real), conjugate(d1.dual));
    }

    template<typename T>
    inline
    dual_types::dual_v<T> length(const dual_types::dual_v<T>& d1, const dual_types::dual_v<T>& d2, const dual_types::dual_v<T>& d3)
    {
        T bottom = 2 * length(d1.real, d2.real, d3.real);

        return dual_types::dual_v<T>(length(d1.real, d2.real, d3.real), (2 * d1.real * d1.dual + 2 * d2.real * d2.dual + 2 * d3.real * d3.dual) / bottom);
    }

    template<typename T>
    inline
    dual_types::dual_v<T> fast_length(const dual_types::dual_v<T>& d1, const dual_types::dual_v<T>& d2, const dual_types::dual_v<T>& d3)
    {
        T bottom = 2 * fast_length(d1.real, d2.real, d3.real);

        return dual_types::dual_v<T>(fast_length(d1.real, d2.real, d3.real), (2 * d1.real * d1.dual + 2 * d2.real * d2.dual + 2 * d3.real * d3.dual) / bottom);
    }

    template<typename T>
    inline
    dual_types::dual_v<T> Real(const dual_types::dual_v<dual_types::complex<T>>& c1)
    {
        return dual_types::dual_v<dual_types::symbol>(Real(c1.real), Real(c1.dual));
    }

    template<typename T>
    inline
    dual_types::dual_v<T> Imaginary(const dual_types::dual_v<dual_types::complex<T>>& c1)
    {
        return dual_types::dual_v<dual_types::symbol>(Imaginary(c1.real), Imaginary(c1.dual));
    }

    ///(a + bi) (a - bi) = a^2 + b^2
    template<typename T>
    inline
    dual_types::dual_v<T> self_conjugate_multiply(const dual_types::dual_v<dual_types::complex<T>>& c1)
    {
        return Real(c1 * conjugate(c1));
    }

    inline
    dual_types::dual_v<dual_types::symbol> self_conjugate_multiply(const dual_types::dual_v<dual_types::symbol>& c1)
    {
        return c1 * c1;
    }

    inline
    dual_types::dual_v<dual_types::complex<dual_types::symbol>> unit_i()
    {
        return complex<dual_types::symbol>(0, 1);
    }

    template<typename T>
    inline
    dual_types::dual_v<T> select(const dual_types::dual_v<T>& d1, const dual_types::dual_v<T>& d2, const T& d3)
    {
        return dual_types::dual_v<T>(select(d1.real, d2.real, d3), select(d1.dual, d2.dual, d3));
    }

    template<typename T, typename U, typename V>
    inline
    auto dual_if(const T& condition, U&& if_true, V&& if_false)
    {
        return select(if_false(), if_true(), condition);
    }

    template<typename T>
    inline
    T operator<(const dual_types::dual_v<T>& d1, const dual_types::dual_v<T>& d2)
    {
        return d1.real < d2.real;
    }

    template<typename T>
    inline
    T operator<=(const dual_types::dual_v<T>& d1, const dual_types::dual_v<T>& d2)
    {
        return d1.real <= d2.real;
    }
};

inline
std::string pad(std::string in, int len)
{
    in.resize(len, ' ');

    return in;
}

using dual = dual_types::dual_v<dual_types::symbol>;
using dual_complex = dual_types::dual_v<dual_types::complex<dual_types::symbol>>;

inline
std::array<dual, 4> schwarzschild_metric(dual t, dual r, dual theta, dual phi)
{
    dual rs("rs");
    dual c("c");

    dual dt = -(1 - rs / r) * c * c;
    dual dr = 1/(1 - rs / r);
    dual dtheta = r * r;
    dual dphi = r * r * sin(theta) * sin(theta);

    return {dt, dr, dtheta, dphi};
}

inline
std::array<dual, 3> schwarzschild_reduced(dual t, dual r, dual omega)
{
    dual rs("rs");
    dual c("c");

    dual dt = -(1 - rs / r) * c * c;
    dual dr = 1/(1 - rs / r);
    dual domega = r * r;

    return {dt, dr, domega};
}

template<typename T, size_t N, size_t... Is>
inline
auto array_apply(T&& func, const std::array<dual, N>& arr, std::index_sequence<Is...>)
{
    return func(arr[Is]...);
}

template <typename T, size_t N>
inline
auto array_apply(T&& func, const std::array<dual, N>& arr)
{
    return array_apply(std::forward<T>(func), arr, std::make_index_sequence<N>{});
}

template<typename R, typename... T>
inline
auto get_function_args_array(R(T...))
{
    return std::array{T()...};
}

template<typename Func, typename... T>
inline
std::pair<std::vector<std::string>, std::vector<std::string>> evaluate_metric(Func&& f, T... raw_variables)
{
    std::array<std::string, sizeof...(T)> variable_names{raw_variables...};
    constexpr int N = sizeof...(T);

    std::vector<std::string> all_equations;
    std::vector<std::string> all_derivatives = {" "};

    for(auto& i : variable_names)
    {
        all_derivatives.push_back(i);
    }

    std::vector<std::string> raw_eq;
    std::vector<std::string> raw_derivatives;

    for(int i=0; i < (int)variable_names.size(); i++)
    {
        auto variables = get_function_args_array(f);

        for(int j=0; j < (int)variable_names.size(); j++)
        {
            if(i == j)
            {
                variables[j].make_variable(variable_names[j]);
            }
            else
            {
                variables[j].make_constant(variable_names[j]);
            }
        }

        std::array eqs = array_apply(std::forward<Func>(f), variables);

        if(i == 0)
        {
            for(auto& kk : eqs)
            {
                raw_eq.push_back(kk.real.sym);
            }
        }

        for(auto& kk : eqs)
        {
            raw_derivatives.push_back(kk.dual.sym);
        }

        for(auto& kk : eqs)
        {
            all_equations.push_back(kk.real.sym);
        }

        all_derivatives.push_back("d" + variable_names[i]);

        for(auto& kk : eqs)
        {
            all_derivatives.push_back(kk.dual.sym);
        }

        //std::cout << "var " << dphi.real << std::endl;
    }

    std::array<int, N+1> column_width = {0};

    for(int i=0; i < N+1; i++)
    {
        for(int j=0; j < N+1; j++)
        {
            column_width[i] = std::max((int)column_width[i], (int)all_derivatives[j * (N+1) + i].size());
        }
    }

    for(int j=0; j < (N+1); j++)
    {
        for(int i=0; i < (N+1); i++)
        {
            std::string real = all_derivatives[j * (N+1) + i];

            real = pad(real, column_width[i] + 1);

            printf("%s | ", real.c_str());
        }

        printf("\n");
    }

    return {raw_eq, raw_derivatives};
}

template<typename Func, typename... T>
inline
std::pair<std::vector<std::string>, std::vector<std::string>> evaluate_metric2D(Func&& f, T... raw_variables)
{
    std::array<std::string, sizeof...(T)> variable_names{raw_variables...};
    constexpr int N = sizeof...(T);

    std::vector<std::string> raw_eq;
    std::vector<std::string> raw_derivatives;

    for(int i=0; i < (int)variable_names.size(); i++)
    {
        auto variables = get_function_args_array(f);

        for(int j=0; j < (int)variable_names.size(); j++)
        {
            if(i == j)
            {
                variables[j].make_variable(variable_names[j]);
            }
            else
            {
                variables[j].make_constant(variable_names[j]);
            }

            //variables[j] = make_variable(variable_names[j], i == j);
        }

        std::array eqs = array_apply(std::forward<Func>(f), variables);

        //static_assert(eqs.size() == N * N);

        if(i == 0)
        {
            for(auto& kk : eqs)
            {
                raw_eq.push_back(type_to_string(kk.real));
            }
        }

        for(auto& kk : eqs)
        {
            raw_derivatives.push_back(type_to_string(kk.dual));
        }
    }

    return {raw_eq, raw_derivatives};
}

template<typename Func, typename... T>
inline
std::pair<std::vector<std::string>, std::vector<std::string>> total_diff(Func&& f, T... raw_variables)
{
    std::array<std::string, sizeof...(T)> variable_names{raw_variables...};

    auto [full_eqs, partial_differentials] = evaluate_metric2D(f, raw_variables...);

    constexpr int N = sizeof...(T);

    std::vector<std::string> total_differentials;

    for(int i=0; i < N; i++)
    {
        std::string accum;

        for(int j=0; j < N; j++)
        {
            accum += "(" + partial_differentials[j * N + i] + ")*d" + variable_names[j];

            if(j != N-1)
                accum += "+";
        }

        total_differentials.push_back(accum);
    }

    return {full_eqs, total_differentials};
}

template<typename Func, typename... T>
inline
std::string get_function(Func&& f, T... raw_variables)
{
    constexpr int N = sizeof...(T);
    std::array<std::string, N> variable_names{raw_variables...};

    auto variables = get_function_args_array(f);

    for(int i=0; i < N; i++)
    {
        variables[i].make_variable(variable_names[i]);
    }

    auto result = array_apply(std::forward<Func>(f), variables);

    return type_to_string(result.real);
}

template<typename T, size_t N, size_t... Is>
inline
auto array_apply(T&& func, const std::array<dual_complex, N>& arr, std::index_sequence<Is...>)
{
    return func(arr[Is]...);
}

template <typename T, size_t N>
inline
auto array_apply(T&& func, const std::array<dual_complex, N>& arr)
{
    return array_apply(std::forward<T>(func), arr, std::make_index_sequence<N>{});
}

#endif // DUAL_HPP_INCLUDED
