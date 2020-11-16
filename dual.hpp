#ifndef DUAL_HPP_INCLUDED
#define DUAL_HPP_INCLUDED

#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>

inline
std::string to_string_s(float v)
{
    std::ostringstream oss;
    oss << std::setprecision(16) << std::noshowpoint << v;
    std::string str = oss.str();

    return str;
}

namespace dual_types
{
    struct symbol
    {
        std::string sym = "0";

        symbol(){}
        symbol(const std::string& value){sym = value;}
        symbol(float v){sym = to_string_s(v);}

        void set_dual_constant()
        {
            sym = "0";
        }

        void set_dual_variable()
        {
            sym = "1";
        }
    };

    struct symbol_complex
    {
        symbol real;
        symbol imaginary;

        symbol_complex(){}
        symbol_complex(const std::string& v1, const std::string& v2) : real(v1), imaginary(v2) {}
        symbol_complex(float v1, float v2) : real(v1), imaginary(v2) {}
        symbol_complex(float v1) : real(v1), imaginary(0) {}
        symbol_complex(symbol v1, symbol v2) : real(v1), imaginary(v2) {}

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
        dual_v(const U& _real) : real(_real)
        {
            dual = T();
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
};

inline
std::optional<float> get_value(std::string in)
{
    if(in.size() == 0)
        throw std::runtime_error("Bad in size, 0");

    if(in.size() > 2)
    {
        if(in.front() == '(' && in.back() == ')')
        {
            in.erase(in.begin());
            in.pop_back();
        }
    }

    char* ptr = nullptr;
    float val = std::strtof(in.c_str(), &ptr);

    if(ptr == in.c_str() + in.size())
        return val;

    return std::nullopt;
}

inline
std::string infix(std::string v1, std::string v2, std::string op)
{
    auto c1 = get_value(v1);
    auto c2 = get_value(v2);

    //std::cout << "V " << op << " " << v1 << " " << v2 << std::endl;

    if(op == "*")
    {
        if((c1.has_value() && c1.value() == 0) || (c2.has_value() && c2.value() == 0))
            return "0";

        if(c1.has_value() && c2.has_value())
            return to_string_s(c1.value() * c2.value());

        if(c1.has_value() && c1.value() == 1)
            return v2;

        if(c2.has_value() && c2.value() == 1)
            return v1;
    }

    if(op == "+")
    {
        if(c1.has_value() && c1.value() == 0)
            return v2;

        if(c2.has_value() && c2.value() == 0)
            return v1;

        if(c1.has_value() && c2.has_value())
            return to_string_s(c1.value() + c2.value());
    }

    if(op == "-")
    {
        if(c1.has_value() && c1.value() == 0)
        {
            if(c2.has_value() && c2.value() == 0)
                return "0";

            return "(-(" + v2 + "))";
        }

        if(c2.has_value() && c2.value() == 0)
            return v1;

        if(c1.has_value() && c2.has_value() && c1.value() == c2.value())
            return "0";

        if(c1.has_value() && c2.has_value())
            return to_string_s(c1.value() - c2.value());

        return "(" + v1 + op + "(" + v2 + "))";
    }

    if(op == "/")
    {
        if(c1.has_value() && c1.value() == 0)
            return "0";

        if(c2.has_value() && c2.value() == 1)
            return v1;

        if(c1.has_value() && c2.has_value())
            return to_string_s(c1.value() / c2.value());

        #ifdef RECIPROCAL_CONSTANTS
        if(!c1.has_value() && c2.has_value())
        {
            if(c2.value() == 0)
                throw std::runtime_error("c2 cannot be 0 in / expression");

            return infix(v1, to_string_s(1/c2.value()), "*");
        }
        #endif // RECIPROCAL_CONSTANTS
    }

    return "(" + v1 + "" + op + "" + v2 + ")";
}

inline
std::string outer(std::string v1, std::string v2, std::string op)
{
    if(op == "pow")
    {
        if(v2 == "0")
            return "1";

        auto c1 = get_value(v1);
        auto c2 = get_value(v2);

        if(c1.has_value() && c2.has_value())
            return to_string_s(pow(c1.value(), c2.value()));
    }

    return op + "(" + v1 + "," + v2 + ")";
}

inline
std::string threearg(std::string v1, std::string v2, std::string v3, std::string op)
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
    }

    return op + "(" + v1 + "," + v2 + "," + v3 + ")";
}

inline
std::string unary(std::string v1, std::string op)
{
    auto c1 = get_value(v1);

    if(op == "-" && c1.has_value() && c1.value() == 0)
        return "0";

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

    if(op == "native_sqrt")
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

    if(op == "-")
    {
        return "(" + op + "(" + v1 + "))";
    }

    return op + "(" + v1 + ")";
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
    //return d1.sym;
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

using complex_v = dual_types::symbol_complex;

inline
complex_v operator+(const complex_v& c1, const complex_v& c2)
{
    return complex_v(c1.real + c2.real, c1.imaginary + c2.imaginary);
}

inline
complex_v operator-(const complex_v& c1, const complex_v& c2)
{
    return complex_v(c1.real - c2.real, c1.imaginary - c2.imaginary);
}

inline
complex_v operator-(const complex_v& c1)
{
    return complex_v(-c1.real, -c1.imaginary);
}

inline
complex_v operator*(const complex_v& c1, const complex_v& c2)
{
    return complex_v(c1.real * c2.real - c1.imaginary * c2.imaginary, c1.imaginary * c2.real + c1.real * c2.imaginary);
}

inline
complex_v operator/(const complex_v& c1, const complex_v& c2)
{
    dual_types::symbol divisor = c2.real * c2.real + c2.imaginary * c2.imaginary;

    return complex_v((c1.real * c2.real + c1.imaginary * c2.imaginary) / divisor, (c1.imaginary * c2.real - c1.real * c2.imaginary));
}

inline
complex_v sin(const complex_v& c1)
{
    return complex_v(sin(c1.real) * cosh(c1.imaginary), cos(c1.real) * sinh(c1.imaginary));
}

inline
complex_v cos(const complex_v& c1)
{
    return complex_v(cos(c1.real) * cosh(c1.imaginary), -sin(c1.real) * sinh(c1.imaginary));
}

inline
complex_v conjugate(const complex_v& c1)
{
    return complex_v(c1.real, -c1.imaginary);
}

template<typename T>
inline
dual_types::dual_v<T> operator+(const dual_types::dual_v<T>& d1, const dual_types::dual_v<T>& d2)
{
    return dual_types::dual_v<T>(d1.real + d2.real, d1.dual + d2.dual);
}

template<typename T, typename U>
inline
dual_types::dual_v<T> operator+(const dual_types::dual_v<T>& d1, const U& v)
{
    return dual_types::dual_v<T>(d1.real + T(v), d1.dual);
}

template<typename T, typename U>
inline
dual_types::dual_v<T> operator+(const U& v, const dual_types::dual_v<T>& d1)
{
    return dual_types::dual_v<T>(T(v) + d1.real, d1.dual);
}

template<typename T>
inline
dual_types::dual_v<T> operator-(const dual_types::dual_v<T>& d1, const dual_types::dual_v<T>& d2)
{
    return dual_types::dual_v<T>(d1.real - d2.real, d1.dual - d2.dual);
}

template<typename T, typename U>
inline
dual_types::dual_v<T> operator-(const dual_types::dual_v<T>& d1, const U& v)
{
    return dual_types::dual_v<T>(d1.real - T(v), d1.dual);
}

template<typename T, typename U>
inline
dual_types::dual_v<T> operator-(const U& v, const dual_types::dual_v<T>& d1)
{
    return dual_types::dual_v<T>(T(v) - d1.real, -d1.dual);
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
inline
dual_types::dual_v<T> operator*(const dual_types::dual_v<T>& d1, const U& v)
{
    return d1 * dual_types::dual_v<T>(T(v), T());
}

template<typename T, typename U>
inline
dual_types::dual_v<T> operator*(const U& v, const dual_types::dual_v<T>& d1)
{
    return dual_types::dual_v<T>(T(v), T()) * d1;
}

template<typename T>
inline
dual_types::dual_v<T> operator/(const dual_types::dual_v<T>& d1, const dual_types::dual_v<T>& d2)
{
    return dual_types::dual_v<T>(d1.real / d2.real, makefinite((d1.dual * d2.real - d1.real * d2.dual) / (d2.real * d2.real)));
}

template<typename T, typename U>
inline
dual_types::dual_v<T> operator/(const dual_types::dual_v<T>& d1, const U& v)
{
    return d1 / dual_types::dual_v<T>(T(v), T());
}

template<typename T, typename U>
inline
dual_types::dual_v<T> operator/(const U& v, const dual_types::dual_v<T>& d1)
{
    return dual_types::dual_v<T>(T(v), T()) / d1;
}

template<typename T>
inline
dual_types::dual_v<T> sqrt(const dual_types::dual_v<T>& d1)
{
    return dual_types::dual_v<T>(sqrt(d1.real), 0.5f * d1.dual / sqrt(d1.real));
}

template<typename T>
inline
dual_types::dual_v<T> pow(const dual_types::dual_v<T>& d1, const dual_types::dual_v<T>& d2)
{
    return dual_types::dual_v<T>(pow(d1.real, d2.real), pow(d1.real, d2.real) * (d1.dual * (d2.real / d1.real) + d2.dual * log(d1.real)));
}

template<typename T, typename U>
inline
dual_types::dual_v<T> pow(const dual_types::dual_v<T>& d1, const U& d2)
{
    return pow(d1, dual_types::dual_v<T>(T(d2), T()));
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

inline
std::string pad(std::string in, int len)
{
    in.resize(len, ' ');

    return in;
}

using dual = dual_types::dual_v<dual_types::symbol>;
using dual_complex = dual_types::dual_v<dual_types::symbol_complex>;

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
        std::array<dual, variable_names.size()> variables;

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
        std::array<dual, variable_names.size()> variables;

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
                raw_eq.push_back(kk.real.sym);
            }
        }

        for(auto& kk : eqs)
        {
            raw_derivatives.push_back(kk.dual.sym);
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
            accum += partial_differentials[j * N + i] + "*d" + variable_names[j];

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

    std::array<dual, N> variables;

    for(int i=0; i < N; i++)
    {
        variables[i].make_variable(variable_names[i]);
    }

    dual result = array_apply(std::forward<Func>(f), variables);

    return result.real.sym;
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

template<typename Func, typename... T>
inline
std::pair<std::vector<std::string>, std::vector<std::string>> evaluate_metric2D_DC(Func&& f, T... raw_variables)
{
    std::array<std::string, sizeof...(T)> variable_names{raw_variables...};
    constexpr int N = sizeof...(T);

    std::vector<std::string> raw_eq;
    std::vector<std::string> raw_derivatives;

    for(int i=0; i < (int)variable_names.size(); i++)
    {
        std::array<dual_complex, variable_names.size()> variables;

        for(int j=0; j < (int)variable_names.size(); j++)
        {
            if(i == j)
            {
                variables[j].make_variable(complex_v(variable_names[j], "0"));
            }
            else
            {
                variables[j].make_constant(complex_v(variable_names[j], "0"));
            }
        }

        std::array eqs = array_apply(std::forward<Func>(f), variables);

        static_assert(eqs.size() == N * N);

        if(i == 0)
        {
            for(auto& kk : eqs)
            {
                raw_eq.push_back(kk.real.real.sym);
            }
        }

        for(auto& kk : eqs)
        {
            raw_derivatives.push_back(kk.dual.real.sym);
        }
    }

    return {raw_eq, raw_derivatives};
}

#endif // DUAL_HPP_INCLUDED
