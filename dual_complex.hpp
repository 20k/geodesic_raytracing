#ifndef DUAL_COMPLEX_HPP_INCLUDED
#define DUAL_COMPLEX_HPP_INCLUDED

#include "dual.hpp"

namespace dual_complex
{
    struct complex_value
    {
        std::string real = "0";
        std::string imaginary = "0";
    };

    complex_value make_complex_value(std::string real, std::string imaginary)
    {
        complex_value ret;
        ret.real = real;
        ret.imaginary = imaginary;

        return ret;
    }

    struct dual_complex_value
    {
        complex_value real;
        complex_value dual;
    };

    dual_complex_value make_constant(complex_value v)
    {
        dual_complex_value ret;
        ret.real = v;
        ret.dual.real = "0";
        ret.dual.imaginary = "0";

        return ret;
    }

    dual_complex_value make_real_constant(std::string v)
    {
        dual_complex_value ret;
        ret.real.real = v;
        ret.real.imaginary = "0";
        ret.dual.real = "0";
        ret.dual.imaginary = "0";

        return ret;
    }

    dual_complex_value make_variable(complex_value v, bool is_variable)
    {
        dual_complex_value ret;
        ret.real = v;

        if(is_variable)
        {
            ret.dual.real = "1";
            ret.dual.imaginary = "1";
        }
        else
        {
            ret.dual.real = "0";
            ret.dual.imaginary = "0";
        }

        return ret;
    }

    dual_complex_value make_value(complex_value real, complex_value dual)
    {
        dual_complex_value ret;
        ret.real = real;
        ret.dual = dual;

        return ret;
    }
}

using dual_complex_v = dual_complex::dual_complex_value;
using complex_v = dual_complex::complex_value;

inline
complex_v operator+(const complex_v& c1, const complex_v& c2)
{
    return dual_complex::make_complex_value(infix(c1.real, c2.real, "+"), infix(c1.imaginary, c2.imaginary, "+"));
}

inline
complex_v operator-(const complex_v& c1, const complex_v& c2)
{
    return dual_complex::make_complex_value(infix(c1.real, c2.real, "-"), infix(c1.imaginary, c2.imaginary, "-"));
}

inline
complex_v operator-(const complex_v& c1)
{
    return dual_complex::make_complex_value(unary(c1.real, "-"), unary(c1.imaginary, "-"));
}

///(a + bi) * (c + di)
///ac - bd + bci + adi
inline
complex_v operator*(const complex_v& c1, const complex_v& c2)
{
    std::string real = infix(infix(c1.real, c2.real, "*"), infix(c1.imaginary, c2.imaginary, "*"), "-");
    std::string imaginary = infix(infix(c1.imaginary, c2.real, "*"), infix(c1.real, c2.imaginary, "*"), "+");

    return dual_complex::make_complex_value(real, imaginary);
}

inline
complex_v operator/(const complex_v& c1, const complex_v& c2)
{
    std::string divisor = infix(infix(c2.real, c2.real, "*"), infix(c2.imaginary, c2.imaginary, "*"), "+");

    std::string real_top = infix(infix(c1.real, c1.real, "*"), infix(c1.imaginary, c1.imaginary, "*"), "+");
    std::string imaginary_top = infix(infix(c1.imaginary, c1.real, "*"), infix(c1.real, c1.imaginary, "*"), "-");

    return dual_complex::make_complex_value(infix(real_top, divisor, "/"), infix(imaginary_top, divisor, "/"));
}

inline
complex_v sin(const complex_v& c1)
{
    return dual_complex::make_complex_value(infix(unary(c1.real, "native_sin"), unary(c1.imaginary, "cosh"), "*"), infix(unary(c1.real, "native_cos"), unary(c1.imaginary, "sinh"), "*"));
}

inline
complex_v cos(const complex_v& c1)
{
    return dual_complex::make_complex_value(infix(unary(c1.real, "native_cos"), unary(c1.imaginary, "cosh"), "*"), unary(infix(unary(c1.real, "native_sin"), unary(c1.imaginary, "sinh"), "*"), "-"));
}

inline
complex_v conjugate(const complex_v& c1)
{
    return dual_complex::make_complex_value(c1.real, unary(c1.imaginary, "-"));
}

inline
dual_complex_v operator+(const dual_complex_v& d1, const dual_complex_v& d2)
{
    return dual_complex::make_value(d1.real + d2.real, d1.dual + d2.dual);
}


inline
dual_complex_v operator-(const dual_complex_v& d1, const dual_complex_v& d2)
{
    return dual_complex::make_value(d1.real - d2.real, d1.dual - d2.dual);
}

inline
dual_complex_v operator-(const dual_complex_v& d1)
{
    return dual_complex::make_value(-d1.real, -d1.dual);
}

inline
dual_complex_v operator*(const dual_complex_v& d1, const dual_complex_v& d2)
{
    return dual_complex::make_value(d1.real * d2.real, d1.real * d2.dual + d2.real * d1.dual);
}

inline
dual_complex_v operator/(const dual_complex_v& d1, const dual_complex_v& d2)
{
    return dual_complex::make_value(d1.real / d2.real, (d1.dual * d2.real - d1.real * d2.dual) / (d2.real * d2.real));
}

inline
dual_complex_v conjugate(const dual_complex_v& d1)
{
    return dual_complex::make_value(conjugate(d1.real), conjugate(d1.dual));
}

inline
dual_complex_v sin(const dual_complex_v& d1)
{
    return dual_complex::make_value(sin(d1.real), d1.dual * cos(d1.real));
}

template<typename T, size_t N, size_t... Is>
inline
auto array_apply(T&& func, const std::array<dual_complex_v, N>& arr, std::index_sequence<Is...>)
{
    return func(arr[Is]...);
}

template <typename T, size_t N>
inline
auto array_apply(T&& func, const std::array<dual_complex_v, N>& arr)
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
        std::array<dual_complex_v, variable_names.size()> variables;

        for(int j=0; j < (int)variable_names.size(); j++)
        {
            variables[j] = dual_complex::make_variable(dual_complex::make_complex_value(variable_names[j], "0"), i == j);
        }

        std::array eqs = array_apply(std::forward<Func>(f), variables);

        static_assert(eqs.size() == N * N);

        if(i == 0)
        {
            for(auto& kk : eqs)
            {
                raw_eq.push_back(kk.real.real);
            }
        }

        for(auto& kk : eqs)
        {
            raw_derivatives.push_back(kk.dual.real);
        }
    }

    return {raw_eq, raw_derivatives};
}

#endif // DUAL_COMPLEX_HPP_INCLUDED
