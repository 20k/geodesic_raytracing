#ifndef DUAL_HPP_INCLUDED
#define DUAL_HPP_INCLUDED

#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <type_traits>

namespace dual_types
{
    struct unit_i_t{};

    template<typename T>
    struct complex
    {
        using is_complex = std::true_type;
        using underlying_type = T;

        T real = T();
        T imaginary = T();

        complex(){}
        complex(const std::string& v1, const std::string& v2) : real(v1), imaginary(v2) {}
        template<typename U, typename V>
        requires std::is_constructible_v<T, U> && std::is_constructible_v<T, V>
        complex(U v1, V v2) : real(v1), imaginary(v2) {}
        template<typename U>
        requires std::is_constructible_v<T, U>
        complex(U v1) : real(v1), imaginary(0) {}
        complex(unit_i_t) : real(0), imaginary(1){}

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
        static constexpr bool is_dual = true;

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

    template<typename T>
    inline
    dual_types::complex<T> csqrt(const T& d1)
    {
        T is_negative = signbit(d1);

        T positive_sqrt = sqrt(fabs(d1));

        return dual_types::complex<T>(select(positive_sqrt, 0, is_negative), select(0, positive_sqrt, is_negative));
    }

    template<typename T>
    inline
    complex<T> operator+(const complex<T>& c1, const complex<T>& c2)
    {
        return complex<T>(c1.real + c2.real, c1.imaginary + c2.imaginary);
    }

    template<typename T, typename U>
    inline
    complex<T> operator+(const complex<T>& c1, const U& c2)
    {
        return c1 + complex<T>(c2, 0.f);
    }

    template<typename T, typename U>
    inline
    complex<T> operator+(const U& c1, const complex<T>& c2)
    {
        return complex<T>(c1, 0.f) + c2;
    }

    template<typename T>
    inline
    void operator+=(complex<T>& d1, const complex<T>& d2)
    {
        d1 = d1 + d2;
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

    template<typename T, typename U>
    inline
    complex<T> operator*(const complex<T>& c1, const U& c2)
    {
        return c1 * complex<T>(c2, 0.f);
    }

    template<typename T, typename U>
    inline
    complex<T> operator*(const U& c1, const complex<T>& c2)
    {
        return complex<T>(c1, 0.f) * c2;
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

    template<typename T>
    inline
    complex<T> sqrt(const complex<T>& d1)
    {
        /*auto i_cst_opt = get_value(d1.imaginary.sym);

        if(i_cst_opt.has_value() && i_cst_opt.value() == 0)
            return csqrt(d1.real);*/

        T r_part = sqrt(max((d1.real + sqrt(d1.real * d1.real + d1.imaginary * d1.imaginary))/2, 0));
        T i_part = sign(d1.imaginary) * sqrt(max((-d1.real + sqrt(d1.real * d1.real + d1.imaginary * d1.imaginary))/2, 0));

        return complex<T>(r_part, i_part);
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
    complex<T> expi(const T& d1)
    {
        return complex<T>(cos(d1), sin(d1));
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
        if constexpr(std::is_same_v<std::true_type, typename T::is_complex>)
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

    /*template<typename T>
    inline
    dual_types::dual_v<T> fast_length(const dual_types::dual_v<T>& d1, const dual_types::dual_v<T>& d2, const dual_types::dual_v<T>& d3)
    {
        T bottom = 2 * fast_length(d1.real, d2.real, d3.real);

        return dual_types::dual_v<T>(fast_length(d1.real, d2.real, d3.real), (2 * d1.real * d1.dual + 2 * d2.real * d2.dual + 2 * d3.real * d3.dual) / bottom);
    }*/

    template<typename T>
    inline
    dual_types::dual_v<T> fast_length(const dual_types::dual_v<T>& d1, const dual_types::dual_v<T>& d2, const dual_types::dual_v<T>& d3)
    {
        return sqrt(d1 * d1 + d2 * d2 + d3 * d3);
    }

    template<typename T>
    inline
    dual_types::dual_v<T> Real(const dual_types::dual_v<dual_types::complex<T>>& c1)
    {
        return dual_types::dual_v<T>(Real(c1.real), Real(c1.dual));
    }

    template<typename T>
    inline
    dual_types::dual_v<T> Imaginary(const dual_types::dual_v<dual_types::complex<T>>& c1)
    {
        return dual_types::dual_v<T>(Imaginary(c1.real), Imaginary(c1.dual));
    }

    ///(a + bi) (a - bi) = a^2 + b^2
    template<typename T>
    inline
    dual_types::dual_v<T> self_conjugate_multiply(const dual_types::dual_v<dual_types::complex<T>>& c1)
    {
        return Real(c1 * conjugate(c1));
    }

    template<typename T>
    inline
    dual_types::dual_v<T> self_conjugate_multiply(const dual_types::dual_v<T>& c1)
    {
        return c1 * c1;
    }

    inline
    unit_i_t unit_i()
    {
        return unit_i_t();
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

    template<typename T>
    inline
    auto max(const dual_types::dual_v<T>& d1, const dual_types::dual_v<T>& d2)
    {
        return dual_if(d1 < d2, [&](){return d2;}, [&](){return d1;});
    }

    template<typename T>
    inline
    auto min(const dual_types::dual_v<T>& d1, const dual_types::dual_v<T>& d2)
    {
        return dual_if(d1 < d2, [&](){return d1;}, [&](){return d2;});
    }
};

inline
std::string pad(std::string in, int len)
{
    in.resize(len, ' ');

    return in;
}

//using dual = dual_types::dual_v<dual_types::symbol>;
//using dual_complex = dual_types::dual_v<dual_types::complex<dual_types::symbol>>;

template<typename T, typename U, size_t N, size_t... Is>
inline
auto array_apply(T&& func, const std::array<U, N>& arr, std::index_sequence<Is...>)
{
    return func(arr[Is]...);
}

template<typename T, typename U, size_t N>
inline
auto array_apply(T&& func, const std::array<U, N>& arr)
{
    return array_apply(std::forward<T>(func), arr, std::make_index_sequence<N>{});
}

template<typename R, typename... T>
inline
auto get_function_args_array(R(T...))
{
    return std::array{T()...};
}


template<typename R, typename... T>
constexpr
bool is_dual_impl(R(T...))
{
    return (T::is_dual && ...);
}

template<typename F>
constexpr
bool is_dual()
{
    constexpr std::decay_t<F> f = std::decay_t<F>();

    return is_dual_impl(f);
}

template<typename Func, typename... T>
inline
std::pair<std::vector<std::string>, std::vector<std::string>> evaluate_metric2D(Func&& f, T... raw_variables)
{
    std::array<std::string, sizeof...(T)> variable_names{raw_variables...};

    std::vector<std::string> raw_eq;
    std::vector<std::string> raw_derivatives;

    for(int i=0; i < (int)variable_names.size(); i++)
    {
        auto variables = get_function_args_array(f);

        if constexpr(is_dual<Func>())
        {
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
                    raw_eq.push_back(type_to_string(kk.real));
                }
            }

            for(auto& kk : eqs)
            {
                raw_derivatives.push_back(type_to_string(kk.dual));
            }
        }
        else
        {
            for(int j=0; j < (int)variable_names.size(); j++)
            {
                variables[j].make_value(variable_names[j]);
            }

            std::array eqs = array_apply(std::forward<Func>(f), variables);

            if(i == 0)
            {
                for(auto& kk : eqs)
                {
                    raw_eq.push_back(type_to_string(kk));
                }
            }

            for(auto& kk : eqs)
            {
                auto differentiated = kk.differentiate(variable_names[i]);

                raw_derivatives.push_back(type_to_string(differentiated));
            }
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

#endif // DUAL_HPP_INCLUDED
