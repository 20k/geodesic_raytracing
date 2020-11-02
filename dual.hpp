#ifndef DUAL_HPP_INCLUDED
#define DUAL_HPP_INCLUDED

#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>

struct dual
{
    std::string real = "0";
    std::string dual = "0";
};

inline
dual make_constant(std::string val)
{
    dual ret;
    ret.dual = "0";
    ret.real = val;

    return ret;
}

inline
dual make_value(std::string val)
{
    dual ret;
    ret.real = val;
    ret.dual = "d" + val;

    return ret;
}

inline
dual make_value(std::string val, std::string diff)
{
    dual ret;
    ret.real = val;
    ret.dual = diff;

    return ret;
}

inline
dual make_variable(std::string val, bool is_variable)
{
    dual ret;
    ret.real = val;

    if(is_variable)
        ret.dual = "1";
    else
        ret.dual = "0";

    return ret;
}

inline
std::string to_string_s(float v)
{
    std::ostringstream oss;
    oss << std::setprecision(8) << std::noshowpoint << v;
    std::string str = oss.str();

    return str;
}

inline
std::string infix(std::string v1, std::string v2, std::string op)
{
    if(op == "*")
    {
        if(v1 == "0" || v2 == "0")
            return "0";

        if(v1 == "1")
            return v2;

        if(v2 == "1")
            return v1;
    }

    if(op == "+")
    {
        if(v1 == "0")
            return v2;

        if(v2 == "0")
            return v1;
    }

    if(op == "-")
    {
        if(v1 == "0")
        {
            if(v2 == "0")
                return "0";

            return "(-" + v2 + ")";
        }

        if(v2 == "0")
            return v1;
    }

    if(op == "/")
    {
        if(v1 == "0")
            return "0";
    }

    return "(" + v1 + "" + op + "" + v2 + ")";
}

inline
std::string outer(std::string v1, std::string v2, std::string op)
{
    return op + "(" + v1 + "," + v2 + ")";
}

inline
std::string unary(std::string v1, std::string op)
{
    if(op == "-" && v1 == "0")
        return "0";

    return op + "(" + v1 + ")";
}

inline
dual operator+(const dual& d1, const dual& d2)
{
    return make_value(infix(d1.real, d2.real, "+"), infix(d1.dual, d2.dual, "+"));
}

inline
dual operator+(const dual& d1, float v)
{
    return d1 + make_constant(to_string_s(v));
}

inline
dual operator+(float v, const dual& d1)
{
    return make_constant(to_string_s(v)) + d1;
}

inline
dual operator-(const dual& d1, const dual& d2)
{
    return make_value(infix(d1.real, d2.real, "-"), infix(d1.dual, d2.dual, "-"));
}

inline
dual operator-(const dual& d1, float v)
{
    return d1 - make_constant(to_string_s(v));
}

inline
dual operator-(float v, const dual& d1)
{
    return make_constant(to_string_s(v)) - d1;
}

inline
dual operator-(const dual& d1)
{
    return make_value(unary(d1.real, "-"), unary(d1.dual, "-"));
}

inline
dual operator*(const dual& d1, const dual& d2)
{
    std::string dual_str = infix(infix(d1.real, d2.dual, "*"), infix(d1.dual, d2.real, "*"), "+");

    return make_value(infix(d1.real, d2.real, "*"), dual_str);
}

inline
dual operator*(const dual& d1, float v)
{
    return d1 * make_constant(to_string_s(v));
}

inline
dual operator*(float v, const dual& d1)
{
    return make_constant(to_string_s(v)) * d1;
}

inline
dual operator/(const dual& d1, const dual& d2)
{
    std::string dual_str = infix(infix(infix(d1.dual, d2.real, "*"), infix(d1.real, d2.dual, "*"), "-"), infix(d2.real, d2.real, "*"), "/");

    return make_value(infix(d1.real, d2.real, "/"), dual_str);
}

inline
dual operator/(const dual& d1, float v)
{
    return d1 / make_constant(to_string_s(v));
}

inline
dual operator/(float v, const dual& d1)
{
    return make_constant(to_string_s(v)) / d1;
}

inline
dual sqrt(const dual& d1)
{
    return make_value(unary(d1.real, "sqrt"), infix(infix("0.5f", d1.dual, "*"), unary(d1.real, "sqrt"), "/"));
}

inline
dual pow(const dual& d1, float v)
{
    return make_value(outer(d1.real, to_string_s(v), "pow"), infix(to_string_s(v), infix(d1.dual, outer(d1.real, to_string_s(v - 1), "pow"), "*"), "*"));
    //return make_value(outer(d1.real, to_string_s(v), "pow"), "(" + to_string_s(v) + "*" + d1.dual + "*" + outer(d1.real, infix(to_string_s(v), "1", "-"), "pow"));
}

inline
dual sin(const dual& d1)
{
    return make_value(unary(d1.real, "native_sin"), infix(d1.dual, unary(d1.real, "native_cos"), "*"));
}

inline
dual cos(const dual& d1)
{
    return make_value(unary(d1.real, "native_cos"), unary(infix(d1.dual, unary(d1.real, "native_sin"), "*"), "-"));
}

inline
dual tan(const dual& d1)
{
    std::string cos_real = unary(d1.real, "cos");

    return make_value(unary(d1.real, "tan"), infix(d1.dual, infix(cos_real, cos_real, "*"), "/"));
}

inline
dual atan(const dual& d1)
{
    return make_value(unary(d1.real, "atan"), infix(d1.dual, "(1+" + infix(d1.real, d1.real, "*") + ")", "/"));
}

inline
dual smoothstep(dual x)
{
    return x * x * (3.f - 2.f * x);
}

inline
std::string pad(std::string in, int len)
{
    in.resize(len, ' ');

    return in;
}

inline
std::array<dual, 4> schwarzschild_metric(dual t, dual r, dual theta, dual phi)
{
    dual rs = make_constant("rs");
    dual c = make_constant("c");

    dual dt = -(1 - rs / r) * c * c;
    dual dr = 1/(1 - rs / r);
    dual dtheta = r * r;
    dual dphi = r * r * sin(theta) * sin(theta);

    return {dt, dr, dtheta, dphi};
}

inline
std::array<dual, 3> schwarzschild_reduced(dual t, dual r, dual omega)
{
    dual rs = make_constant("rs");
    dual c = make_constant("c");

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
            variables[j] = make_variable(variable_names[j], i == j);
        }

        std::array eqs = array_apply(std::forward<Func>(f), variables);

        if(i == 0)
        {
            for(auto& kk : eqs)
            {
                raw_eq.push_back(kk.real);
            }
        }

        for(auto& kk : eqs)
        {
            raw_derivatives.push_back(kk.dual);
        }

        for(auto& kk : eqs)
        {
            all_equations.push_back(kk.real);
        }

        all_derivatives.push_back("d" + variable_names[i]);

        for(auto& kk : eqs)
        {
            all_derivatives.push_back(kk.dual);
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
            variables[j] = make_variable(variable_names[j], i == j);
        }

        std::array eqs = array_apply(std::forward<Func>(f), variables);

        static_assert(eqs.size() == N * N);

        if(i == 0)
        {
            for(auto& kk : eqs)
            {
                raw_eq.push_back(kk.real);
            }
        }

        for(auto& kk : eqs)
        {
            raw_derivatives.push_back(kk.dual);
        }
    }

    return {raw_eq, raw_derivatives};
}

#endif // DUAL_HPP_INCLUDED
