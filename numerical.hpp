#ifndef NUMERICAL_HPP_INCLUDED
#define NUMERICAL_HPP_INCLUDED

#include "dual_value.hpp"

///https://physics.stackexchange.com/questions/51915/can-one-raise-indices-on-covariant-derivative-and-products-thereof raising indices on covariant derivatives

///http://web.mit.edu/edbert/GR/gr11.pdf ???
template<typename T, typename U, int N>
inline
tensor<T, N, N, N> christoffel_symbols_2(const tensor<T, N, N>& metric, const std::array<U, N>& variables)
{
    tensor<T, N, N, N> christoff;
    tensor<T, N, N> inverted = metric.invert();

    for(int i=0; i < N; i++)
    {
        for(int k=0; k < N; k++)
        {
            for(int l=0; l < N; l++)
            {
                T sum = 0;

                for(int m=0; m < N; m++)
                {
                    sum += inverted.idx(i, m) * metric.idx(m, k).differentiate(variables[l]);
                    sum += inverted.idx(i, m) * metric.idx(m, l).differentiate(variables[k]);
                    sum -= inverted.idx(i, m) * metric.idx(k, l).differentiate(variables[m]);
                }

                christoff.idx(i, k, l) = 0.5 * sum;
            }
        }
    }

    return christoff;
}

///https://scholarworks.rit.edu/cgi/viewcontent.cgi?article=11286&context=theses
///Du v^v
template<typename T, typename U, int N>
inline
tensor<T, N, N> covariant_derivative_high_vec(const vec<N, T>& v_in, const tensor<T, N, N>& metric, const std::array<U, N>& variables)
{
    tensor<T, N, N, N> christoff = christoffel_symbols_2(metric, variables);
    tensor<T, N, N> duvv;

    for(int v=0; v < N; v++)
    {
        for(int mu=0; mu < N; mu++)
        {
            T sum = 0;

            for(int p=0; p < N; p++)
            {
                sum += christoff.idx(v, p, mu) * v_in.v[p];
            }

            duvv.idx(mu, v) = v_in.v[v].differentiate(variables[mu]) + sum;
        }
    }

    return duvv;
}

///https://en.wikipedia.org/wiki/Covariant_derivative#Covariant_derivative_by_field_type
template<typename T, typename U, int N>
inline
tensor<T, N, N> covariant_derivative_low_vec(const vec<N, T>& v_in, const tensor<T, N, N>& metric, const std::array<U, N>& variables)
{
    tensor<T, N, N, N> christoff = christoffel_symbols_2(metric, variables);
    tensor<T, N, N> lac;

    for(int a=0; a < N; a++)
    {
        for(int c=0; c < N; c++)
        {
            T sum = 0;

            for(int b=0; b < N; b++)
            {
                sum += christoff.idx(b, c, a) * v_in.v[b];
            }

            lac.idx(a, c) = v_in.v[a].differentiate(variables[c]) - sum;
        }
    }

    return lac;
}

template<typename T, typename U, int N>
inline
tensor<T, N> covariant_derivative_scalar(const T& in, const tensor<T, N, N>& metric, const std::array<U, N>& variables)
{
    tensor<T, N> ret;

    for(int i=0; i < N; i++)
    {
        ret.idx(i) = in.differentiate(variables[i]);
    }

    return ret;
}

template<typename T, typename U, int N>
inline
tensor<T, N, N> high_covariant_derivative_scalar(const T& in, const tensor<T, N, N>& metric, const std::array<U, N>& variables)
{
    tensor<T, N, N> iv_metric = metric.invert();

    tensor<T, N> deriv_low = covariant_derivative_scalar(in, metric, variables);

    tensor<T, N> ret;

    for(int i=0; i < N; i++)
    {
        T sum = 0;

        for(int p=0; p < N; p++)
        {
            sum += iv_metric.idx(i, p) * deriv_low.idx(p);
        }

        ret.idx(i) = sum;
    }

    return ret;
}

template<typename T, typename U, int N>
inline
tensor<T, N, N, N> covariant_derivative_mixed_tensor(const tensor<T, N, N>& mT, const tensor<T, N, N>& metric, const std::array<U, N>& variables)
{
    tensor<T, N, N, N> christoff = christoffel_symbols_2(metric, variables);
    tensor<T, N, N, N> tabc;

    for(int a = 0; a < N; a++)
    {
        for(int b=0; b < N; b++)
        {
            for(int c=0; c < N; c++)
            {
                T sum = 0;

                for(int d=0; d < N; d++)
                {
                    sum += christoff.idx(a, c, d) * mT.idx(d, b) - christoff.idx(d, c, b) * mT.idx(a, d);
                }

                tabc.idx(a, b, c) = mT.idx(a, b).differentiate(variables[c]) + sum;
            }
        }
    }

    return tabc;
}


///https://scholarworks.rit.edu/cgi/viewcontent.cgi?article=11286&context=theses 2.24
template<typename T, typename U, int N>
inline
tensor<T, N, N> metric_lie_derivative(const vec<N, T>& u, const tensor<T, N, N>& metric, const std::array<U, N>& variables)
{
    tensor<T, N, N> lie;

    tensor<T, N, N> cov = covariant_derivative(u, metric, variables);

    for(int a=0; a < N; a++)
    {
        for(int b=0; b < N; b++)
        {
            lie.idx(a, b) = cov.idx(a, b) + cov.idx(b, a);
        }
    }

    return lie;
}

/*
///mixed indices for T
template<typename T, typename U, int N>
inline
tensor<T, N, N> lie_derivative_mixed(const vec<N, T>& u, const tensor<T, N, N>& mT, const tensor<T, N, N>& metric, const std::array<U, N>& variables)
{
    tensor<T, N, N> lie;

    for(int a=0; a < N; a++)
    {
        for(int b=0; b < N; b++)
        {
            T sum = 0;

            for(int g=0; g < N; g++)
            {
                sum += u.v[g] * covariant_derivative_mixed(mT, metric, variables).idx(b, a, g) +
                       mT.idx(g, a) * covariant_derivative_raised(u, metric, variables).idx(b, g) -
                       mT.idx(b, g) * covariant_derivative_raised(u, metric, variables).idx(g, a);
            }

            lie.idx(b, a) = sum;
        }
    }

    return lie;
}*/

///https://arxiv.org/pdf/gr-qc/9810065.pdf
///This paper claims that we can't use the regular lie derivative
template<typename T, typename U, int N>
inline
tensor<T, N, N> lie_derivative_weight(const vec<N, T>& B, const tensor<T, N, N>& mT, const tensor<T, N, N>& metric, const std::array<U, N>& variables)
{
    tensor<T, N, N> lie;

    for(int i=0; i < N; i++)
    {
        for(int j=0; j < N; j++)
        {
            T sum = 0;

            for(int k=0; k < N; k++)
            {
                sum += B.v[k] * mT.idx(i, j).differentiate(variables[k]) +
                       mT.idx(i, k) * B.v[k].differentiate(variables[j]) +
                       mT.idx(k, j) * B.v[k].differentiate(variables[i]) -
                       (2.f/3.f) * mT.idx(i, j) * B.v[k].differentiate(variables[k]);
            }

            lie.idx(i, j) = sum;
        }
    }

    return lie;
}


///https://arxiv.org/pdf/gr-qc/9810065.pdf
template<typename T, int N>
inline
T trace(const tensor<T, N, N>& mT, const tensor<T, N, N>& metric)
{
    tensor<T, N, N> inverse = metric.invert();

    T ret = 0;

    for(int i=0; i < N; i++)
    {
        for(int j=0; j < N; j++)
        {
            ret += inverse.idx(i, j) * mT.idx(i, j);
        }
    }

    return ret;
}

template<typename T, int N>
inline
tensor<T, N, N> trace_free(const tensor<T, N, N>& mT, const tensor<T, N, N>& metric)
{
    tensor<T, N, N> inverse = metric.invert();

    tensor<T, N, N> TF;
    T t = trace(mT, metric);

    for(int i=0; i < N; i++)
    {
        for(int j=0; j < N; j++)
        {
            TF.idx(i, j) = mT.idx(i, j) - (1/3.f) * metric.idx(i, j) * t;
        }
    }

    return TF;
}

#endif // NUMERICAL_HPP_INCLUDED
