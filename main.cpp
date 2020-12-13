#include <iostream>
#include <toolkit/render_window.hpp>
#include <toolkit/texture.hpp>
#include <vec/vec.hpp>
#include <GLFW/glfw3.h>
#include <SFML/Graphics.hpp>
#include <CL/cl_ext.h>
#include "dual.hpp"
#include "metric.hpp"
#include "chromaticity.hpp"
//#include "dual_complex.hpp"

/**
Big list of general relativity references so i can shut some browser tabs
https://arxiv.org/pdf/1601.02063.pdf - GPU renderer
https://arxiv.org/pdf/1511.06025.pdf - CPU renderer - primary source from which this raytracer is derived
https://en.wikipedia.org/wiki/Frame_fields_in_general_relativity#Example:_Static_observers_in_Schwarzschild_vacuum - frame fields
https://www.spacetimetravel.org/wurmlochflug/wurmlochflug.html - renderings of wormholes
https://www.damtp.cam.ac.uk/user/hsr1000/lecturenotes_2012.pdf - lecture notes for relativity, misc everything
https://arxiv.org/pdf/0904.4184.pdf - spacetime catalogue, contains a bunch of metrics and tetrads
https://arxiv.org/pdf/1104.4829.pdf - gram schmidt orthonormalisation in a relativistic context
https://arxiv.org/pdf/1702.05802.pdf - double kerr (massless strut)
https://arxiv.org/ftp/arxiv/papers/1008/1008.3244.pdf - double kerr (massles strut)
https://arxiv.org/pdf/1702.02209.pdf - rotating double kerr with a massless strut
https://arxiv.org/pdf/1905.05273.pdf - janis-newman-winicour rendering + accretion disk
https://arxiv.org/pdf/1408.6041.pdf - alternative formulation of janis-newman-winicour line element that's less singularity inducing
http://www.roma1.infn.it/teongrav/VALERIA/TEACHING/ONDE_GRAV_STELLE_BUCHINERI/AA2012_13/Kerr.pdf - kerr info
http://cloud.yukterez.net/relativistic.raytracer/kerr.90.1720.png - kerr reference picture
https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods - runge kutta
https://physics.stackexchange.com/questions/409106/finding-the-metric-tensor-from-a-line-element - line element / metric tensor
https://arxiv.org/pdf/0706.0622.pdf - kerr spacetime coordinate systems
http://www.roma1.infn.it/teongrav/leonardo/bh/bhcap3.pdf - kerr, again
https://www.wolframalpha.com/input/?i=%28x%5E2+%2B+y%5E2%29+%2F+%28r%5E2+%2B+a%5E2%29+%2B+z%5E2%2Fr%5E2+%3D+1%2C+solve+for+r - solving for r in kerr-schild
https://arxiv.org/pdf/0807.0734.pdf - symplectic integrators
https://iopscience.iop.org/article/10.1088/0067-0049/184/2/387/pdf - radiative transport
https://javierrubioblog.files.wordpress.com/2015/12/chapter4.pdf - coordinate transforms

https://www.reed.edu/physics/courses/Physics411/html/page2/page2.html - some useful info
https://www.uio.no/studier/emner/matnat/astro/nedlagte-emner/AST1100/h11/undervisningsmateriale/lecture15.pdf - useful basic info
https://theconfused.me/blog/numerical-integration-of-light-paths-in-a-schwarzschild-metric/ - simple schwarzschild raytracer

https://github.com/stranger80/GraviRayTraceSharp/blob/master/GraviRayTraceSharp/ - raytracer with runge kutta integrator
https://en.wikipedia.org/wiki/Interior_Schwarzschild_metric - metric for the inside of a body
https://en.wikipedia.org/wiki/Vaidya_metric - radiating metric
https://en.wikipedia.org/wiki/Category:Exact_solutions_in_general_relativity - more exact solutions

https://en.wikipedia.org/wiki/Tetrad_formalism - tetrads/coordinate basis
https://arxiv.org/abs/gr-qc/0507014v1 - numerical relativity
https://arxiv.org/pdf/gr-qc/0104063.pdf - numerical relativity

https://en.wikipedia.org/wiki/Two-body_problem_in_general_relativity#Schwarzschild_solution - useful references on numerical relativity
https://en.wikipedia.org/wiki/Kerr%E2%80%93Newman_metric - kerr-newman (charged + rotating)

https://en.wikipedia.org/wiki/Cash%E2%80%93Karp_method - runge-kutta with adaptive error
https://drum.lib.umd.edu/bitstream/handle/1903/2202/2004-berry-healy-jas.pdf;jsessionid=B20F478B9DB479C86B9DD179A24331F3?sequence=7 - integration

https://www.pp.rhul.ac.uk/~cowan/ph2150/kepler_xy.pdf - good explanation of integration

https://core.ac.uk/download/pdf/1321518.pdf - numerical relativity phd
https://arxiv.org/pdf/gr-qc/9509020.pdf - numerical relativity

https://www.cec.uchile.cl/cinetica/pcordero/MC_libros/NumericalRecipesinC.pdf - 710

https://arxiv.org/pdf/0712.4333.pdf - coordinate system choices for schwarzschild and kerr (hyperboloidal)
https://iopscience.iop.org/article/10.1088/1361-6382/ab6e3e/pdf - another

http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.1002.1336&rep=rep1&type=pdf

http://yukterez.net/ - loads of good stuff
https://scholarworks.rit.edu/cgi/viewcontent.cgi?article=11286&context=theses - lots more good stuff, numerical relativity
http://ccom.ucsd.edu/~lindblom/Talks/Milwaukee_14October2011.pdf - simple introduction to numerical relativity
http://ccom.ucsd.edu/~lindblom/Talks/NRBeijing1.pdf - seems to be more up to date

https://www.aanda.org/articles/aa/pdf/2012/09/aa19599-12.pdf - radiative transfer
https://arxiv.org/pdf/0704.0986.pdf - tetrad info
https://www.researchgate.net/figure/View-of-a-static-observer-located-at-x-0-y-4-in-the-positive-y-direction-for-t_fig2_225428633 - alcubierre. Successfully managed to replicate this https://imgur.com/a/48SONjV. This paper is an absolute goldmine of useful information

https://arxiv.org/pdf/astro-ph/9707230.pdf - neutron star numerical relativity
https://www.aanda.org/articles/aa/full_html/2012/07/aa19209-12/aa19209-12.html - a* with a thin disk
https://gyoto.obspm.fr/GyotoManual.pdf - gyoto, general relativity tracer
https://core.ac.uk/download/pdf/25279526.pdf - binary black hole approximation?

"how do i convert rgb to wavelengths"
https://github.com/colour-science/smits1999
https://github.com/appleseedhq/appleseed/blob/54ce23fc940087180511cb5659d8a7aac33712fb/src/appleseed/foundation/image/colorspace.h#L956
https://github.com/wip-/RgbToSpectrum/blob/master/Spectra/SimpleSpectrum.cs
https://en.wikipedia.org/wiki/Dominant_wavelength
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.40.9608&rep=rep1&type=pdf
https://www.fourmilab.ch/documents/specrend/specrend.c
https://www.researchgate.net/publication/308305862_Relationship_between_peak_wavelength_and_dominant_wavelength_of_light_sources_based_on_vector-based_dominant_wavelength_calculation_method
https://www.semrock.com/how-to-calculate-luminosity-dominant-wavelength-and-excitation-purity.aspx
*/

///perfectly fine
vec4f cartesian_to_schwarz(vec4f position)
{
    vec3f polar = cartesian_to_polar((vec3f){position.y(), position.z(), position.w()});

    return (vec4f){position.x(), polar.x(), polar.y(), polar.z()};
}

struct lightray
{
    cl_float4 position;
    cl_float4 velocity;
    cl_float4 acceleration;
    cl_uint sx, sy;
    cl_float ku_uobsu;
    cl_float original_theta;
};

inline
std::array<dual, 4> schwarzschild_blackhole(dual t, dual r, dual theta, dual phi)
{
    dual rs("rs");
    dual c("c");

    //theta = "M_PI/2";

    dual dt = -(1 - rs / r) * c * c;

    dual dr = 1/(1 - rs / r);
    dual dtheta = r * r;
    dual dphi = r * r * sin(theta) * sin(theta);

    return {dt, dr, dtheta, dphi};
}

#define BIG
inline
#ifdef BIG
std::array<dual, 16> schwarzschild_blackhole_lemaitre(dual T, dual p, dual theta, dual phi)
#else
std::array<dual, 4> schwarzschild_blackhole_lemaitre(dual T, dual p, dual theta, dual phi)
#endif // BIG
{
    dual rs = 1;

    theta = M_PI/2;

    dual r = pow(((3/2.f) * (p - T)), 2.f/3.f) * pow(rs, 1.f/3.f);

    dual dT = -1;
    dual dp = (rs / r);
    dual dtheta = r * r;
    dual dphi = r * r * sin(theta) * sin(theta);

    #ifdef BIG
    std::array<dual, 16> ret;
    ret[0 * 4 + 0] = dT;
    ret[1 * 4 + 1] = dp;
    ret[2 * 4 + 2] = dtheta;
    ret[3 * 4 + 3] = dphi;

    return ret;
    #else
    return {dT, dp, dtheta, dphi};
    #endif // BIG
}

inline
std::array<dual, 16> schwarzschild_eddington_finkelstein_outgoing(dual u, dual r, dual theta, dual phi)
{
    std::array<dual, 16> ret;

    dual rs = 1;

    dual du_dr = -2;

    ret[0 * 4 + 0] = -(1 - rs/r);
    ret[0 * 4 + 1] = 0.5 * du_dr;
    ret[1 * 4 + 0] = 0.5 * du_dr;

    ret[2 * 4 + 2] = r * r;
    ret[3 * 4 + 3] = r * r * sin(theta) * sin(theta);

    return ret;
}

////https://arxiv.org/pdf/0904.4184.pdf
inline
std::array<dual, 4> traversible_wormhole(dual t, dual p, dual theta, dual phi)
{
    dual c = "c";
    dual n = 1;

    dual dt = -1 * c * c;
    dual dr = 1;
    dual dtheta = (p * p + n * n);
    dual dphi = (p * p + n * n) * (sin(theta) * sin(theta));

    /*std::array<dual, 16> ret;
    ret[0 * 4 + 0] = dt;
    ret[1 * 4 + 1] = dr;
    ret[2 * 4 + 2] = dtheta;
    ret[3 * 4 + 3] = dphi;

    return ret;*/

    return {dt, dr, dtheta, dphi};
}

/*
///suffers from event horizonitus
inline
std::array<dual, 4> schwarzschild_wormhole(dual t, dual r, dual theta, dual phi)
{
    dual c = make_constant("c");
    dual rs = make_constant("1");

    dual dt = - c * c * (1 - rs/(r * c * c));
    dual dr = 1/(1 - (rs / (r * c * c)));
    dual dtheta = r * r;
    dual dphi = r * r * sin(theta) * sin(theta);

    return {dt, dr, dtheta, dphi};
}*/

inline
std::array<dual, 4> cosmic_string(dual t, dual r, dual theta, dual phi)
{
    dual c = "c";
    dual rs = 1;

    dual dt = -(1 - rs/r) * c * c;
    dual dr = 1 / (1 - rs/r);
    dual dtheta = r * r;

    dual B = 0.3;
    dual dphi = r * r * B * B * sin(theta) * sin(theta);

    return {dt, dr, dtheta, dphi};
}

inline
std::array<dual, 4> ernst_metric(dual t, dual r, dual theta, dual phi)
{
    dual B = 0.05;

    dual lambda_sq = 1 + B * B * r * r * sin(theta) * sin(theta);

    dual rs = 1;
    dual c = 1;

    dual dt = -lambda_sq * (1 - rs/r);
    dual dr = lambda_sq * 1/(1 - rs/r);
    dual dtheta = lambda_sq * r * r;
    dual dphi = r * r * sin(theta) * sin(theta) / (lambda_sq);

    return {dt, dr, dtheta, dphi};
}

///https://arxiv.org/pdf/1408.6041.pdf is where this formulation comes from
std::array<dual, 4> janis_newman_winicour(dual t, dual r, dual theta, dual phi)
{
    dual r0 = 1;
    ///mu = [1, +inf]
    dual mu = 4;

    dual Ar = pow((2 * r - r0 * (mu - 1)) / (2 * r + r0 * (mu + 1)), 1/mu);
    dual Br = (1/4.f) * pow(2 * r + r0 * (mu + 1), (1/mu) + 1) / pow(2 * r - r0 * (mu - 1), (1/mu) - 1);

    dual dt = -Ar;
    dual dr = 1/Ar;
    dual dtheta = Br;
    dual dphi = Br * sin(theta) * sin(theta);

    return {dt, dr, dtheta, dphi};

    ///this formulation has coordinate singularities coming out of its butt
    /*dual q = sqrt(3) * 1.1;
    dual M = 1;
    dual b = 2 * sqrt(M * M + q * q);

    dual gamma = 2*M/b;

    dual dt = -pow(1 - b/r, gamma);
    dual dr = pow(1 - b/r, -gamma);
    dual dtheta = pow(1 - b/r, 1-gamma) * r * r;
    dual dphi = pow(1 - b/r, 1-gamma) * r * r * sin(theta) * sin(theta);

    return {dt, dr, dtheta, dphi};*/
}

inline
std::array<dual, 4> de_sitter(dual t, dual r, dual theta, dual phi)
{
    float cosmo = 0.01;

    dual c = 1;

    dual dt = -(1 - cosmo * r * r/3) * c * c;
    dual dr = 1/(1 - cosmo * r * r / 3);
    dual dtheta = r * r;
    dual dphi = r * r * sin(theta) * sin(theta);

    return {dt, dr, dtheta, dphi};
}

inline auto test_metric = traversible_wormhole;

inline
std::array<dual, 16> ellis_drainhole(dual t, dual r, dual theta, dual phi)
{
    dual c = 1;

    dual m = 0.5;
    dual n = 1;

    dual alpha = sqrt(n * n - m * m);

    dual pseudophi = (n / alpha) * (M_PI/2 - atan((r - m) / alpha));

    dual Fp = -sqrt(1 - exp(-(2 * m/n) * pseudophi));

    dual Rp = sqrt(((r - m) * (r - m) + alpha * alpha) / (1 - Fp * Fp));

    dual dt1 = - c * c;
    dual dp = 1 * 1;
    dual dt = dt1 - Fp * Fp * c * c;
    dual dpdt = -2 * Fp * c;

    dual dtheta = Rp * Rp;
    dual dphi = Rp * Rp * sin(theta) * sin(theta);

    std::array<dual, 16> ret;
    ret[0] = dt;
    ret[1 * 4 + 1] = dp;
    ret[2 * 4 + 2] = dtheta;
    ret[3 * 4 + 3] = dphi;
    ret[0 * 4 + 1] = dpdt * 0.5;
    ret[1 * 4 + 0] = dpdt * 0.5;

    return ret;
}

inline
std::array<dual, 16> big_metric_test(dual t, dual p, dual theta, dual phi)
{
    dual c = "c";
    dual n = 1;

    dual dt = -1 * c * c;
    dual dr = 1;
    dual dtheta = (p * p + n * n);
    dual dphi = (p * p + n * n) * (sin(theta) * sin(theta));

    std::array<dual, 16> ret;
    ret[0] = dt;
    ret[1 * 4 + 1] = dr;
    ret[2 * 4 + 2] = dtheta;
    ret[3 * 4 + 3] = dphi;

    return ret;
}

///nanning derivatives: 21, 31, 16, 37, 44 47 35 19 28 5 15 21
///((((v2+v2)*(((v2*v2)-v2)+4))-(((v2*v2)+((4*native_cos(v3))*native_cos(v3)))*((v2+v2)-1)))/((((v2*v2)-v2)+4)*(((v2*v2)-v2)+4)))
inline
std::array<dual, 16> kerr_metric(dual t, dual r, dual theta, dual phi)
{
    dual rs = 1;

    dual a = -2;
    dual E = r * r + a * a * cos(theta) * cos(theta);
    dual D = r * r  - rs * r + a * a;

    dual c = 1;

    std::array<dual, 16> ret;

    ret[0] = -(1 - rs * r / E) * c * c;
    ret[1 * 4 + 1] = E / D;
    ret[2 * 4 + 2] = E;
    ret[3 * 4 + 3] = (r * r + a * a + (rs * r * a * a / E) * sin(theta) * sin(theta)) * sin(theta) * sin(theta);
    ret[0 * 4 + 3] = 0.5 * -2 * rs * r * a * sin(theta) * sin(theta) * c / E;
    ret[3 * 4 + 0] = ret[0 * 4 + 3];

    return ret;
}

///https://arxiv.org/pdf/0706.0622.pdf
inline
std::array<dual, 16> kerr_schild_metric(dual t, dual x, dual y, dual z)
{
    dual a = -0.5;

    dual R2 = x * x + y * y + z * z;
    dual Rm2 = x * x + y * y - z * z;

    //dual r2 = (R2 - a*a + sqrt((R2 - a*a) * (R2 - a*a) + 4 * a*a * z*z))/2;

    dual r2 = (-a*a + sqrt(a*a*a*a - 2*a*a * Rm2 + R2*R2) + R2) / 2;

    dual r = sqrt(r2);

    std::array<dual, 16> minkowski = {-1, 0, 0, 0,
                                       0, 1, 0, 0,
                                       0, 0, 1, 0,
                                       0, 0, 0, 1};

    std::array<dual, 4> lv = {1, (r*x + a*y) / (r2 + a*a), (r*y - a*x) / (r2 + a*a), z/r};

    dual rs = 1;

    dual f = rs * r2 * r / (r2 * r2 + a*a * z*z);
    //dual f = rs * r*r*r / (r*r*r*r + a*a * z*z);

    std::array<dual, 16> g;

    for(int a=0; a < 4; a++)
    {
        for(int b=0; b < 4; b++)
        {
            g[a * 4 + b] = minkowski[a * 4 + b] + f * lv[a] * lv[b];
        }
    }

    return g;
}

inline
std::array<dual, 16> kerr_rational_polynomial(dual t, dual r, dual X, dual phi)
{
    dual m = 0.5;
    dual a = -2;

    dual dt = -(1 - 2 * m * r / (r * r + a * a * X * X));
    dual dphidt = - (4 * a * m * r * (1 - X * X))/(r * r + a * a * X * X);
    dual dr = (r * r + a * a * X * X) / (r * r - 2 * m * r + a * a);
    dual dX = (r * r + a * a * X * X) / (1 - X * X);
    dual dphi = (1 - X * X) * (r * r + a * a + (2 * m * a * a * r * (1 - X * X)) / (r * r + a * a * X * X));

    std::array<dual, 16> ret;
    ret[0 * 4 + 0] = dt;
    ret[1 * 4 + 1] = dr;
    ret[2 * 4 + 2] = dX;
    ret[3 * 4 + 3] = dphi;
    ret[0 * 4 + 3] = dphidt * 0.5;
    ret[3 * 4 + 0] = dphidt * 0.5;

    return ret;
}

inline
std::array<dual, 16> kerr_newman(dual t, dual r, dual theta, dual phi)
{
    dual c = 1;
    dual rs = 1;
    dual r2q = 0.51;
    //dual r2q = 0.5;
    //dual a = 0.51;
    dual a = -0.51;

    dual p2 = r * r + a * a * cos(theta) * cos(theta);
    dual D = r * r - rs * r + a * a + r2q * r2q;

    dual dr = -p2 / D;
    dual dtheta = -p2;

    dual dt_1 = c * c * D / p2;
    dual dtdphi_1 = -2 * c * a * sin(theta) * sin(theta) * D/p2;
    dual dphi_1 = pow(a * sin(theta) * sin(theta), 2) * D/p2;

    dual dphi_2 = -pow(r * r + a * a, 2) * sin(theta) * sin(theta) / p2;
    dual dtdphi_2 = 2 * a * c * (r * r + a * a) * sin(theta) * sin(theta) / p2;
    dual dt_2 = -a * a * c * c * sin(theta) * sin(theta) / p2;

    dual dtdphi = dtdphi_1 + dtdphi_2;

    std::array<dual, 16> ret;
    ret[0 * 4 + 0] = -(dt_1 + dt_2);
    ret[1 * 4 + 1] = -dr;
    ret[2 * 4 + 2] = -dtheta;
    ret[3 * 4 + 3] = -(dphi_1 + dphi_2);
    ret[0 * 4 + 3] = -dtdphi * 0.5;
    ret[3 * 4 + 0] = -dtdphi * 0.5;

    return ret;
}

inline
std::array<dual, 16> double_kerr(dual t, dual p, dual phi, dual z)
{
    dual_complex i = dual_types::unit_i();

    ///distance between black holes
    dual R = 3;

    dual M = 0.3;
    dual a = 0.27;

    dual d = 2 * M * a * (R * R - 4 * M * M + 4 * a * a) / (R * R + 2 * M * R + 4 * a * a);

    dual sigma_sq = M * M - a * a + (4 * M * M * a * a * (R * R - 4 * M * M + 4 * a * a)) / pow(R * R + 2 * M * R + 4 * a * a, 2);

    dual sigmap = sqrt(sigma_sq);

    dual sigman = -sigmap;

    dual_complex ia = i * a;
    dual_complex id = i * d;

    dual_complex Rp = ((-M * (2 * sigmap + R) + id) / (2 * M * M + (R + 2 * ia) * (sigmap + ia))) * sqrt(p * p + pow((z + 0.5 * R + sigmap), 2));
    dual_complex Rn = ((-M * (2 * sigman + R) + id) / (2 * M * M + (R + 2 * ia) * (sigman + ia))) * sqrt(p * p + pow((z + 0.5 * R + sigman), 2));

    dual_complex rp = ((-M * (2 * sigmap - R) + id) / (2 * M * M - (R - 2 * ia) * (sigmap + ia))) * sqrt(p * p + pow((z - 0.5 * R + sigmap), 2));
    dual_complex rn = ((-M * (2 * sigman - R) + id) / (2 * M * M - (R - 2 * ia) * (sigman + ia))) * sqrt(p * p + pow((z - 0.5 * R + sigman), 2));

    //dual K0 = (4 * R * R * sigmap * sigmap * (R * R - 4 * sigmap * sigmap) * ((R * R + 4 * a * a) * (sigmap * sigmap + a * a) - 4 * M * (M * M * M + a * d))) / ((M * M * pow(R + 2 * sigmap, 2) + d * d) * (M * M * pow(R - 2 * sigmap, 2) + d * d));

    dual K0 = 4 * sigma_sq * (pow(R * R + 2 * M * R + 4 * a * a, 2) - 16 * M * M * a * a) / (M * M * (pow(R + 2 * M, 2) + 4 * a * a));

    dual_complex A = R * R * (Rp - Rn) * (rp - rn) - 4 * sigma_sq * (Rp - rp) * (Rn - rn);
    dual_complex B = 2 * R * sigmap * ((R + 2 * sigmap) * (Rn - rp) - (R - 2 * sigmap) * (Rp - rn));

    dual_complex G = -z * B + R * sigmap * (2 * R * (Rn * rn - Rp * rp) + 4 * sigmap * (Rp * Rn - rp * rn) - (R * R - 4 * sigma_sq) * (Rp - Rn - rp + rn));

    dual w = 4 * a - (2 * Imaginary(G * (conjugate(A) + conjugate(B))) / (self_conjugate_multiply(A) - self_conjugate_multiply(B)));

    ///the denominator only has real components
    dual f = (self_conjugate_multiply(A) - self_conjugate_multiply(B)) / Real((A + B) * (conjugate(A) + conjugate(B)));
    dual i_f = Real((A + B) * (conjugate(A) + conjugate(B))) / (self_conjugate_multiply(A) - self_conjugate_multiply(B));

    dual i_f_e2g = Real((A + B) * (conjugate(A) + conjugate(B))) / Real(K0 * K0 * Rp * Rn * rp * rn);

    //dual i_f = 1/f;

    ///I'm not sure if the denominator is real... but I guess it must be?
    dual e2g = (self_conjugate_multiply(A) - self_conjugate_multiply(B)) / Real(K0 * K0 * Rp * Rn * rp * rn);

    //dual i_f_e2g = i_f * e2g;

    dual dphi2 = w * w * -f;
    dual dphi1 = i_f * p * p;

    dual dt_dphi = f * w * 2;

    dual dp = i_f_e2g;
    dual dz = i_f_e2g;

    std::array<dual, 16> ret;
    ret[0 * 4 + 0] = -f;
    ret[2 * 4 + 2] = dphi1 + dphi2;
    ret[0 * 4 + 2] = dt_dphi * 0.5;
    ret[2 * 4 + 0] = dt_dphi * 0.5;

    ret[1 * 4 + 1] = dp;
    ret[3 * 4 + 3] = dz;

    return ret;
}

void debugp(dual_complex p)
{
    std::cout << p.real.real.sym << std::endl;
}

///https://www.sciencedirect.com/science/article/pii/S0370269319303375
inline
std::array<dual, 16> unequal_double_kerr(dual t, dual p, dual phi, dual z)
{
    dual_complex i = dual_types::unit_i();

    /*dual a1 = -0.09;
    dual a2 = 0.091;*/

    dual m1 = 0.01;
    dual m2 = 1;

    dual fa1 = 0.4;
    dual fa2 = 0.34;

    dual a1 = fa1 * m1;
    dual a2 = fa2 * m2;

    /*dual a1 = 0.3;
    dual a2 = 0.1;

    dual m1 = 0.4;
    dual m2 = 0.4;*/

    dual R = 8;

    dual J = m1 * a1 + m2 * a2;
    dual M = m1 + m2;

    ///https://www.wolframalpha.com/input/?i=%28%28a_1+%2B+a_2+-+x%29+*+%28R%5E2+-+M%5E2+%2B+x%5E2%29+%2F+%282+*+%28R+%2B+M%29%29%29+-+M+*+x+%2B+J+%3D+0+Solve+for+x
    ///https://www.wolframalpha.com/input/?i=%28%28k+-+a%29+*+%28B+%2B+a%5E2%29+%2F+C%29+-+M+*+a+%2B+J+solve+for+a

    dual a = 0;

    {
        dual k = a1 + a2;
        dual B = R*R - M*M;
        dual C = 2 * (R + M);

        dual inner_val = pow(sqrt(pow(18 * B * k + 27 * C * J - 9 * C * k * M + 2 * k*k*k, 2) + 4 * pow(3 * B + 3 * C * M - k*k, 3)) + 18 * B * k + 27 * C * J - 9 * C * k * M + 2 *k*k*k, 1.f/3.f);

        dual third_root_2 = pow(2.f, 1.f/3.f);

        a = (1.f / (3 * third_root_2)) * inner_val - ((third_root_2 * (3 * B + 3 * C * M - k*k)) / (3 * inner_val)) + k/3;
    }

    dual d1 = ((m1 * (a1 - a2 + a) + R * a) * (pow(R + M, 2) + a * a) + m2 * a1 * a*a) / pow(pow(R + M, 2) + a*a, 2);
    dual d2 = ((m2 * (a2 - a1 + a) + R * a) * (pow(R + M, 2) + a * a) + m1 * a2 * a*a) / pow(pow(R + M, 2) + a*a, 2);

    ///todo: need complex sqrt
    dual s1 = sqrt(m1 * m1 - a1 * a1 + 4 * m2 * a1 * d1);
    dual s2 = sqrt(m2 * m2 - a2 * a2 + 4 * m1 * a2 * d2);

    ///R+ with a squiggle on
    dual Rsp = sqrt(p * p + pow(z + 0.5 * R + s2, 2));
    dual Rsn = sqrt(p * p + pow(z + 0.5 * R - s2, 2));

    dual rsp = sqrt(p * p + pow(z - 0.5 * R + s1, 2));
    dual rsn = sqrt(p * p + pow(z - 0.5 * R - s1, 2));

    //std::cout << "S1 " << d1.real.sym << " S2 " << d2.real.sym << std::endl;

    //assert(false);

    //std::cout << "S1 " << Rsp.real.sym << " S2 " << Rsn.real.sym << " S3 " << rsp.real.sym << " S4 " << rsn.real.sym << std::endl;

    //throw std::runtime_error("Err");

    dual_complex mu0 = (R + M - i * a) / (R + M + i * a);


    dual_complex rp = (1/mu0) *  (((s1 - m1 - i * a1) * (pow(R + M, 2) + a*a) + 2 * a1 * (m1 * a + i * M * (R + M))) /
                                  ((s1 - m1 + i * a1) * (pow(R + M, 2) + a*a) + 2 * a1 * (m1 * a - i * M * (R + M))))
                                  * rsp;

    dual_complex rn = (1/mu0) * (((-s1 - m1 - i * a1) * (pow(R + M, 2) + a*a) + 2 * a1 * (m1 * a + i * M * (R + M))) /
                                 ((-s1 - m1 + i * a1) * (pow(R + M, 2) + a*a) + 2 * a1 * (m1 * a - i * M * (R + M))))
                                 * rsn;

    dual_complex Rp = -mu0    *  (((s2 + m2 - i * a2) * (pow(R + M, 2) + a*a) - 2 * a2 * (m2 * a - i * M * (R + M))) /
                                  ((s2 + m2 + i * a2) * (pow(R + M, 2) + a*a) - 2 * a2 * (m2 * a + i * M * (R + M))))
                                 * Rsp;

    dual_complex Rn = -mu0    * (((-s2 + m2 - i * a2) * (pow(R + M, 2) + a*a) - 2 * a2 * (m2 * a - i * M * (R + M))) /
                                 ((-s2 + m2 + i * a2) * (pow(R + M, 2) + a*a) - 2 * a2 * (m2 * a + i * M * (R + M))))
                                 * Rsn;

    //std::cout << "S1 " << rp.real.real.sym << " S2 " << rn.real.real.sym << " S3 " << Rp.real.real.sym << " S4 " << Rn.real.real.sym << std::endl;
    //std::cout << "S1I " << rp.real.imaginary.sym << " S2I " << rn.real.imaginary.sym << " S3I " << Rp.real.imaginary.sym << " S4I " << Rn.real.imaginary.sym << std::endl;

    //throw std::runtime_error("Err");

    /*dual_complex A = R * R * (Rp - Rn) * (rp - rn) - 4 * sigma_sq * (Rp - rp) * (Rn - rn);
    dual_complex B = 2 * R * sigmap * ((R + 2 * sigmap) * (Rn - rp) - (R - 2 * sigmap) * (Rp - rn));*/

    dual_complex A = (R*R - pow(s1 + s2, 2)) * (Rp - Rn) * (rp - rn) - 4 * s1 * s2 * (Rp - rn) * (Rn - rp);
    dual_complex B = 2 * s1 * (R*R - s1*s1 + s2*s2) * (Rn - Rp) + 2 * s2 * (R * R + s1 * s1 - s2 * s2) * (rn - rp) + 4 * R * s1 * s2 * (Rp + Rn - rp - rn);

    //std::cout << "A " << A.real.real.sym << " AI " << A.real.imaginary.sym << std::endl;

    //throw std::runtime_error("Err");

    dual_complex G = -z * B + s1 * (R * R - s1 * s1 + s2 * s2) * (Rn - Rp) * (rp + rn + R) + s2 * (R * R + s1 * s1 - s2*s2) * (rn - rp) * (Rp + Rn - R)
                     -2 * s1 * s2 * (2 * R * (rp * rn - Rp * Rn - s1 * (rn - rp) + s2 * (Rn - Rp)) + (s1 * s1 - s2 * s2) * (rp + rn - Rp - Rn));

    dual K0 = ((pow(R + M, 2) + a*a) * (R*R - pow(m1 - m2, 2) + a*a) - 4 * m1*m1 * m2*m2 * a*a) / (m1 * m2 * (pow(R + M, 2) + a*a));

    dual w = 2 * a - (2 * Imaginary(G * (conjugate(A) + conjugate(B))) / (self_conjugate_multiply(A) - self_conjugate_multiply(B)));

    dual f = (self_conjugate_multiply(A) - self_conjugate_multiply(B)) / Real((A + B) * (conjugate(A) + conjugate(B)));
    dual e2g = (self_conjugate_multiply(A) - self_conjugate_multiply(B)) / (16 * pow(fabs(s1), 2) * pow(fabs(s2), 2) * K0*K0 * Rsp * Rsn * rsp * rsn);

    dual dphi2 = w * w * -f;
    dual dphi1 = (1/f) * p * p;

    dual dt_dphi = f * w * 2;

    dual dp = (1/f) * e2g;
    dual dz = (1/f) * e2g;

    std::array<dual, 16> ret;
    ret[0 * 4 + 0] = -f;
    ret[2 * 4 + 2] = dphi1 + dphi2;
    ret[0 * 4 + 2] = dt_dphi * 0.5;
    ret[2 * 4 + 0] = dt_dphi * 0.5;

    ret[1 * 4 + 1] = dp;
    ret[3 * 4 + 3] = dz;

    return ret;
}

inline
std::array<dual_complex, 16> big_imaginary_metric_test(dual_complex t, dual_complex p, dual_complex theta, dual_complex phi)
{
    dual_complex c = 1;
    dual_complex n = 1;

    dual_complex dt = -c * c;
    dual_complex dr = 1;
    dual_complex dtheta = (p * p + n * n);
    dual_complex dphi = (p * p + n * n) * (sin(theta) * sin(theta));

    std::array<dual_complex, 16> ret_fat;
    ret_fat[0] = dt;
    ret_fat[1 * 4 + 1] = dr;
    ret_fat[2 * 4 + 2] = dtheta;
    ret_fat[3 * 4 + 3] = dphi;

    /*if(phi.dual.real == "1")
    {
        std::cout << "PHI? " << dphi.dual.real << std::endl;
        std::cout << "Theta? " << n.dual.real << std::endl;
    }*/

    return ret_fat;
}

inline
std::array<dual_complex, 16> minkowski_space(dual_complex t, dual_complex x, dual_complex y, dual_complex z)
{
    std::array<dual_complex, 16> ret_fat;
    ret_fat[0] = -1;
    ret_fat[1 * 4 + 1] = 1;
    ret_fat[2 * 4 + 2] = 1;
    ret_fat[3 * 4 + 3] = 1;

    return ret_fat;
}

inline
std::array<dual, 16> minkowski_polar(dual t, dual r, dual theta, dual phi)
{
    std::array<dual, 16> ret_fat;
    ret_fat[0] = -1;
    ret_fat[1 * 4 + 1] = 1;
    ret_fat[2 * 4 + 2] = r * r;
    ret_fat[3 * 4 + 3] = r * r * sin(theta) * sin(theta);

    return ret_fat;
}

///we're in inclination
inline
std::array<dual_complex, 16> minkowski_cylindrical(dual_complex t, dual_complex r, dual_complex phi, dual_complex z)
{
    std::array<dual_complex, 16> ret_fat;
    ret_fat[0] = -1;
    ret_fat[1 * 4 + 1] = 1;
    ret_fat[2 * 4 + 2] = r * r;
    ret_fat[3 * 4 + 3] = 1;

    return ret_fat;
}

///krasnikov tubes: https://core.ac.uk/download/pdf/25208925.pdf
dual krasnikov_thetae(dual v, dual e)
{
    return 0.5f * (tanh(2 * ((2 * v / e) - 1)) + 1);
}

///they call x, z
///krasnikov is extremely terrible, because its situated down the z axis here which is super incredibly bad for performance
std::array<dual, 16> krasnikov_tube_metric(dual t, dual p, dual phi, dual x)
{
    dual e = 0.1; ///width o the tunnel
    dual D = 2; ///length of the tube
    dual pmax = 1; ///size of the mouth

    ///[0, 2], approx= 0?
    dual little_d = 0.01; ///unsure, <1 required for superluminosity

    auto k_t_x_p = [e, pmax, D, little_d](dual t, dual x, dual p)
    {
        return 1 - (2 - little_d) * krasnikov_thetae(pmax - p, e) * krasnikov_thetae(t - x - p, e) * (krasnikov_thetae(x, e) - krasnikov_thetae(x + e - D, e));
    };

    dual dxdt = (1 - k_t_x_p(t, x, p));

    std::array<dual, 16> ret;
    ret[0 * 4 + 0] = -1;
    ret[1 * 4 + 1] = 1;
    ret[2 * 4 + 2] = p * p;
    ret[3 * 4 + 3] = k_t_x_p(t, x, p);
    ret[0 * 4 + 3] = 0.5 * dxdt;
    ret[3 * 4 + 0] = 0.5 * dxdt;

    return ret;
}

///values here are picked for numerical stability, in particular D should be < the precision bounding box, and its more numerically stable the higher e is
std::array<dual, 16> krasnikov_tube_metric_cart(dual t, dual x, dual y, dual z)
{
    dual e = 0.75; ///width o the tunnel
    dual D = 5; ///length of the tube
    dual pmax = 2; ///size of the mouth

    ///[0, 2], approx= 0?
    dual little_d = 0.01; ///unsure, <1 required for superluminosity

    dual p = sqrt(y * y + z * z);

    auto k_t_x_p = [e, pmax, D, little_d](dual t, dual x, dual p)
    {
        return 1 - (2 - little_d) * krasnikov_thetae(pmax - p, e) * krasnikov_thetae(t - x - p, e) * (krasnikov_thetae(x, e) - krasnikov_thetae(x + e - D, e));
    };

    dual dxdt = (1 - k_t_x_p(t, x, p));

    std::array<dual, 16> ret;
    ret[0 * 4 + 0] = -1;
    ret[1 * 4 + 1] = k_t_x_p(t, x, p);
    ret[2 * 4 + 2] = 1;
    ret[3 * 4 + 3] = 1;
    ret[0 * 4 + 1] = 0.5 * dxdt;
    ret[1 * 4 + 0] = 0.5 * dxdt;

    return ret;
}

inline
std::array<dual, 16> natario_warp_drive_metric(dual t, dual rs_t, dual theta, dual phi)
{
    dual sigma = 1;
    dual R = 2;

    std::array<dual, 16> ret;

    dual f_rs = (tanh(sigma * (rs_t + R)) - tanh(sigma * (rs_t - R))) / (2 * tanh(sigma * R));
}

///rendering alcubierre nicely is very hard: the shell is extremely thin, and flat on both sides
///this means that a naive timestepping method results in a lot of distortion
///need to crank down subambient_precision, and crank up new_max to about 20 * rs
///performance and quality is made significantly better with a dynamic timestep based on an error estimate, and then unstepping if it steps too far
inline
std::array<dual, 16> alcubierre_metric(dual t, dual x, dual y, dual z)
{
    dual dxs_t = 2;
    dual xs_t = dxs_t * t;
    dual vs_t = dxs_t;

    dual rs_t = fast_length(x - xs_t, y, z);

    dual sigma = 1;
    dual R = 2;

    dual f_rs = (tanh(sigma * (rs_t + R)) - tanh(sigma * (rs_t - R))) / (2 * tanh(sigma * R));

    dual dt = (vs_t * vs_t * f_rs * f_rs - 1);
    dual dxdt = -2 * vs_t * f_rs;
    dual dx = 1;
    dual dy = 1;
    dual dz = 1;

    std::array<dual, 16> ret;
    ret[0 * 4 + 0] = dt;
    ret[1 * 4 + 0] = dxdt * 0.5;
    ret[0 * 4 + 1] = dxdt * 0.5;
    ret[1 * 4 + 1] = dx;
    ret[2 * 4 + 2] = dy;
    ret[3 * 4 + 3] = dz;

    return ret;
}


/*std::array<dual, 16> ret;
ret[0 * 4 + 0] = dt;
ret[1 * 4 + 1] = yrr;
ret[2 * 4 + 2] = ythetatheta;
ret[3 * 4 + 3] = yphiphi;

return ret;*/


///https://arxiv.org/pdf/2010.11031.pdf
inline
std::array<dual, 4> symmetric_warp_drive(dual t, dual r, dual theta, dual phi)
{
    theta = M_PI/2;

    dual rg = 1;
    dual rk = rg;

    dual a20 = (1 - rg / r);

    dual a0 = sqrt(a20);

    dual a1 = t / theta;

    dual a2 = a20 + a1;

    dual yrr0 = 1 / (1 - (rg / r));
    dual ythetatheta0 = r * r;
    dual yphiphi0 = r * r * sin(theta) * sin(theta);

    dual gamma_0 = pow(r, 4) * sin(theta) * sin(theta) / (1 - rg/r);

    dual littlea = rk * theta * pow(a0, -1);
    dual littleb = rk * theta - sqrt(gamma_0);

    dual Urt = (littlea * pow(a20 + t/theta, 3/2.f) - littleb) / (littlea * a0*a0*a0 - littleb);

    dual yrr = Urt * yrr0;
    dual ythetatheta = Urt * ythetatheta0;
    dual yphiphi = Urt * yphiphi0;

    dual dt = -a2;

    return {dt, yrr, ythetatheta, yphiphi};
}

inline
std::array<dual, 4> polar_to_cartesian_dual(dual t, dual r, dual theta, dual phi)
{
    dual x = r * sin(theta) * cos(phi);
    dual y = r * sin(theta) * sin(phi);
    dual z = r * cos(theta);

    return {t, x, y, z};
}

inline
std::array<dual, 4> cartesian_to_polar_dual(dual t, dual x, dual y, dual z)
{
    dual r = sqrt(x * x + y * y + z * z);
    dual theta = atan2(sqrt(x * x + y * y), z);
    dual phi = atan2(y, x);

    return {t, r, theta, phi};
}

inline
dual alcubierre_distance(dual t, dual r, dual theta, dual phi)
{
    std::array<dual, 4> cart = polar_to_cartesian_dual(t, r, theta, phi);

    dual dxs_t = 2;

    dual x_pos = cart[1] - dxs_t * t;

    return fast_length(x_pos, cart[2], cart[3]);
}

inline
std::array<dual, 4> lemaitre_to_polar(dual T, dual p, dual theta, dual phi)
{
    dual rs = 1;

    dual r = pow((3/2.f) * (p - T), 2.f/3.f) * pow(rs, 1/3.f);

    ///T is incorrect here, but i can't find the correct version, and we need T anyway for polar to lemaitre
    ///because... i can't find the correct equations. Ah well
    return {T, r, theta, phi};
}

inline
std::array<dual, 4> polar_to_lemaitre(dual T, dual r, dual theta, dual phi)
{
    dual rs = 1;

    dual p = (pow((r / pow(rs, 1/3.f)), 3/2.f) * 2/3.f) + T;

    return {T, p, theta, phi};
}

inline
std::array<dual, 4> cylindrical_to_polar(dual t, dual p, dual phi, dual z)
{
    dual rr = sqrt(p * p + z * z);
    dual rtheta = atan2(p, z);
    //dual rtheta = atan(p / z);
    dual rphi = phi;

    return {t, rr, rtheta, rphi};
}

inline
std::array<dual, 4> polar_to_cylindrical(dual t, dual r, dual theta, dual phi)
{
    dual rp = r * sin(theta);
    dual rphi = phi;
    dual rz = r * cos(theta);

    return {t, rp, rphi, rz};
}

inline
std::array<dual, 4> rotated_cylindrical_to_polar(dual t, dual p, dual phi, dual x)
{
    std::array<dual, 4> as_polar = cylindrical_to_polar(t, p, phi, x);

    std::array<dual, 4> as_cart = polar_to_cartesian_dual(as_polar[0], as_polar[1], as_polar[2], as_polar[3]);

    quaternion_base<dual> dual_quat;
    dual_quat.load_from_axis_angle({1, 0, 0, -M_PI/2});

    auto rotated = rot_quat({as_cart[1], as_cart[2], as_cart[3]}, dual_quat);

    return cartesian_to_polar_dual(t, rotated.x(), rotated.y(), rotated.z());
}

inline
std::array<dual, 4> polar_to_cylindrical_rotated(dual t, dual r, dual theta, dual phi)
{
    quaternion_base<dual> dual_quat;
    dual_quat.load_from_axis_angle({1, 0, 0, M_PI/2});

    std::array<dual, 4> as_cart = polar_to_cartesian_dual(t, r, theta, phi);

    auto rotated = rot_quat({as_cart[1], as_cart[2], as_cart[3]}, dual_quat);

    std::array<dual, 4> as_polar = cartesian_to_polar_dual(t, rotated.x(), rotated.y(), rotated.z());

    return polar_to_cylindrical(as_polar[0], as_polar[1], as_polar[2], as_polar[3]);
}

inline
std::array<dual, 4> polar_to_polar(dual t, dual r, dual theta, dual phi)
{
    return {t, r, theta, phi};
}

inline
std::array<dual, 4> rational_to_polar(dual t, dual r, dual X, dual phi)
{
    return {t, r, acos(X), phi};
}

inline
std::array<dual, 4> polar_to_rational(dual t, dual r, dual theta, dual phi)
{
    return {t, r, cos(theta), phi};
}

inline
std::array<dual, 4> oblate_to_polar(dual t, dual r, dual theta, dual phi)
{
    dual a = 2;

    dual cx = sqrt(r * r + a * a) * sin(theta) * cos(phi);
    dual cy = sqrt(r * r + a * a) * sin(theta) * sin(phi);
    dual cz = r * cos(theta);

    return cartesian_to_polar_dual(t, cx, cy, cz);
}

inline
std::array<dual, 4> polar_to_oblate(dual t, dual in_r, dual in_theta, dual in_phi)
{
    dual a = 2;

    std::array<dual, 4> as_cart = polar_to_cartesian_dual(t, in_r, in_theta, in_phi);

    dual x = as_cart[1];
    dual y = as_cart[2];
    dual z = as_cart[3];

    dual tphi = in_phi;
    dual secp = sec(tphi);

    dual tr = sqrt(-a*a - pow(secp, 2) * -sqrt(a*a*a*a * pow(cos(tphi), 4) - 2*a*a*x*x*pow(cos(tphi), 2) + 2*a*a*z*z*pow(cos(tphi),4) + x*x*x*x + 2*x*x*z*z*pow(cos(tphi), 2) + z*z*z*z*pow(cos(tphi), 4)) + x*x*pow(secp, 2) + z*z) / sqrt(2);

    dual ttheta = asin(x * sec(tphi) / sqrt(a * a + tr * tr));

    return {t, tr, ttheta, tphi};
}

inline
std::array<dual, 4> polar_to_tortoise(dual t, dual r, dual theta, dual phi)
{
    dual rs = 1;

    dual tr = r + rs * log(fabs(r / rs - 1));

    return {t, tr, theta, phi};
}

inline
std::array<dual, 4> tortoise_to_polar(dual t, dual tort, dual theta, dual phi)
{
    dual rs = 1;
    ///r* = r + rs * ln(fabs(r/rs - 1))

    ///where m = rs
    ///W = lambert_w0
    ///k = r*
    ///x = r
    ///https://www.wolframalpha.com/input/?i=e%5Ek+%3D+e%5E%28x+%2B+m+*+ln%28abs%28x%2Fm+-+1%29%29%29
    dual r = rs * (lambert_w0(pow(exp(tort - rs), 1/rs)) + 1);

    return {t, r, theta, phi};
}

inline
std::array<dual, 4> polar_to_outgoing_eddington_finkelstein(dual t, dual r, dual theta, dual phi)
{
    dual rstar = polar_to_tortoise(t, r, theta, phi)[1];

    dual u = t - rstar;

    return {u, r, theta, phi};
}

inline
std::array<dual, 4> outgoing_eddington_finkelstein_to_polar(dual u, dual r, dual theta, dual phi)
{
    ///u = t - r*
    ///t = u + r*

    ///ignore t component, its unused in tortoise
    ///the reason to recalculate is to avoid calculating lambert_w0, which is very imprecise
    dual rstar = polar_to_tortoise(0.f, r, theta, phi)[1];

    dual t = u + rstar;

    return {t, r, theta, phi};
}

inline
dual at_origin(dual t, dual r, dual theta, dual phi)
{
    return r;
}

/*inline
std::array<dual, 4> test_metric(dual t, dual p, dual theta, dual phi)
{
    dual c = make_constant("c");
    dual n = make_constant("1");

    dual dt = -make_constant("1");
    dual dr = make_constant("1");
    dual dtheta = p * p;
    dual dphi = p * p * sin(theta) * sin(theta);

    return {dt, dr, dtheta, dphi};
}*/

#define GENERIC_METRIC

vec4f interpolate_geodesic(const std::vector<cl_float4>& geodesic, float coordinate_time)
{
    for(int i=0; i < (int)geodesic.size() - 2; i++)
    {
        vec4f cur = {geodesic[i].s[0], geodesic[i].s[1], geodesic[i].s[2], geodesic[i].s[3]};
        vec4f next = {geodesic[i + 1].s[0], geodesic[i + 1].s[1], geodesic[i + 1].s[2], geodesic[i + 1].s[3]};

        if(geodesic[i + 2].s[0] == 0 && geodesic[i + 1].s[0] == 0)
            break;

        if(next.x() < cur.x())
            std::swap(next, cur);

        if(coordinate_time >= cur.x() && coordinate_time < next.x())
        {
            vec3f as_cart1 = polar_to_cartesian<float>({fabs(cur.y()), cur.z(), cur.w()});
            vec3f as_cart2 = polar_to_cartesian<float>({fabs(next.y()), next.z(), next.w()});

            float r1 = cur.y();
            float r2 = next.y();

            float dx = (coordinate_time - cur.x()) / (next.x() - cur.x());

            vec3f next_cart = cartesian_to_polar(mix(as_cart1, as_cart2, dx));
            float next_r = mix(r1, r2, dx);

            next_cart.x() = next_r;

            return {coordinate_time, next_cart.x(), next_cart.y(), next_cart.z()};
        }
    }

    if(geodesic.size() == 0)
        return {0,0,0,0};

    return {geodesic[0].s[0], geodesic[0].s[1], geodesic[0].s[2], geodesic[0].s[3]};
}

vec2f get_geodesic_intersection(const std::vector<cl_float4>& geodesic)
{
    for(int i=0; i < (int)geodesic.size() - 2; i++)
    {
        if(geodesic[i + 2].s[0] == 0 && geodesic[i + 1].s[0] == 0)
            break;

        vec4f cur = {geodesic[i].s[0], geodesic[i].s[1], geodesic[i].s[2], geodesic[i].s[3]};
        vec4f next = {geodesic[i + 1].s[0], geodesic[i + 1].s[1], geodesic[i + 1].s[2], geodesic[i + 1].s[3]};

        if(signum(geodesic[i].s[1]) != signum(geodesic[i + 1].s[1]))
        {
            float total_r = fabs(geodesic[i].s[1]) + fabs(geodesic[i + 1].s[1]);

            float dx = fabs(geodesic[i].s[1]) / total_r;

            vec3f as_cart1 = polar_to_cartesian<float>({fabs(cur.y()), cur.z(), cur.w()});
            vec3f as_cart2 = polar_to_cartesian<float>({fabs(next.y()), next.z(), next.w()});

            vec3f next_cart = cartesian_to_polar(mix(as_cart1, as_cart2, dx));

            return {next_cart.y(), next_cart.z()};
        }
    }

    return {0, 0};
}

///i need the ability to have dynamic parameters
int main()
{
    render_settings sett;
    sett.width = 1422/1;
    sett.height = 800/1;
    sett.opencl = true;
    sett.no_double_buffer = true;

    render_window win(sett, "Geodesics");

    assert(win.clctx);

    opencl_context& clctx = *win.clctx;

    std::string argument_string = "-O5 -cl-std=CL2.2 ";

    #if 1
    #ifdef GENERIC_METRIC

    metric::metric<schwarzschild_blackhole, polar_to_polar, polar_to_polar, at_origin> schwarzs_polar;
    schwarzs_polar.name = "schwarzschild";
    schwarzs_polar.singular = true;
    //schwarzs_polar.adaptive_precision = false;

    metric::metric<schwarzschild_blackhole_lemaitre, lemaitre_to_polar, polar_to_lemaitre, at_origin> schwarzs_lemaitre;
    schwarzs_lemaitre.name = "schwarzs_lemaitre";
    //schwarzs_lemaitre.singular = true;
    //schwarzs_lemaitre.traversible_event_horizon = true;
    //schwarzs_lemaitre.adaptive_precision = true;
    schwarzs_lemaitre.adaptive_precision = true;
    schwarzs_lemaitre.singular = true;
    schwarzs_lemaitre.singular_terminator = 0.5;
    schwarzs_lemaitre.traversable_event_horizon = true;
    schwarzs_lemaitre.follow_geodesics_forward = true;
    //schwarzs_lemaitre.system = metric::coordinate_system::OTHER;
    //schwarzs_lemaitre.detect_singularities = true;

    metric::metric<schwarzschild_eddington_finkelstein_outgoing, outgoing_eddington_finkelstein_to_polar, polar_to_outgoing_eddington_finkelstein, at_origin> schwarzschild_ef_outgoing;
    schwarzschild_ef_outgoing.name = "schwarzs_ef_out";
    schwarzschild_ef_outgoing.adaptive_precision = true;
    schwarzschild_ef_outgoing.singular = true;
    schwarzschild_ef_outgoing.singular_terminator = 0.5;
    schwarzschild_ef_outgoing.traversable_event_horizon = true;

    metric::metric<traversible_wormhole, polar_to_polar, polar_to_polar, at_origin> simple_wormhole;
    simple_wormhole.name = "wormhole";
    simple_wormhole.adaptive_precision = false;

    metric::metric<cosmic_string, polar_to_polar, polar_to_polar, at_origin> cosmic_string_obj;
    cosmic_string_obj.name = "cosmic_string";
    cosmic_string_obj.adaptive_precision = true;
    cosmic_string_obj.detect_singularities = true;

    ///todo: i forgot what this is and what parameters it might need
    metric::metric<ernst_metric, polar_to_polar, polar_to_polar, at_origin> ernst_metric_obj;
    ernst_metric_obj.name = "ernst";
    ernst_metric_obj.adaptive_precision = true;
    ernst_metric_obj.detect_singularities = true;

    metric::metric<janis_newman_winicour, polar_to_polar, polar_to_polar, at_origin> janis_newman_winicour_obj;
    janis_newman_winicour_obj.name = "janis_newman_winicour";
    janis_newman_winicour_obj.detect_singularities = false;

    metric::metric<ellis_drainhole, polar_to_polar, polar_to_polar, at_origin> ellis_drainhole_obj;
    ellis_drainhole_obj.name = "ellis_drainhole";
    ellis_drainhole_obj.adaptive_precision = false;

    ///kerr family
    metric::metric<kerr_metric, polar_to_polar, polar_to_polar, at_origin> kerr_obj;
    kerr_obj.name = "kerr_boyer";
    kerr_obj.adaptive_precision = true;
    //kerr_obj.detect_singularities = true;

    metric::metric<kerr_newman, polar_to_polar, polar_to_polar, at_origin> kerr_newman_obj;
    kerr_newman_obj.name = "kerrnewman_boyer";
    kerr_newman_obj.adaptive_precision = true;
    //kerr_newman_obj.detect_singularities = true;

    metric::metric<kerr_schild_metric, cartesian_to_polar_dual, polar_to_cartesian_dual, at_origin> kerr_schild_obj;
    kerr_schild_obj.name = "kerr_schild";
    kerr_schild_obj.adaptive_precision = true;
    kerr_schild_obj.detect_singularities = true;
    kerr_schild_obj.system = metric::coordinate_system::CARTESIAN;

    metric::metric<kerr_rational_polynomial, rational_to_polar, polar_to_rational, at_origin> kerr_rational_polynomial_obj;
    kerr_rational_polynomial_obj.name = "kerr_rational_poly";
    kerr_rational_polynomial_obj.adaptive_precision = true;
    kerr_rational_polynomial_obj.detect_singularities = true;

    metric::metric<de_sitter, polar_to_polar, polar_to_polar, at_origin> de_sitter_obj;
    de_sitter_obj.name = "desitter";
    de_sitter_obj.adaptive_precision = false;

    metric::metric<minkowski_space, cartesian_to_polar_dual, polar_to_cartesian_dual, at_origin> minkowski_space_obj;
    minkowski_space_obj.name = "minkowski";
    minkowski_space_obj.adaptive_precision = false;
    minkowski_space_obj.system = metric::coordinate_system::CARTESIAN;

    metric::metric<minkowski_polar, polar_to_polar, polar_to_polar, at_origin> minkowski_polar_obj;
    minkowski_space_obj.name = "minkowski_polar";
    minkowski_space_obj.adaptive_precision = false;

    metric::metric<minkowski_cylindrical, cylindrical_to_polar, polar_to_cylindrical, at_origin> minkowski_cylindrical_obj;
    minkowski_cylindrical_obj.name = "minkowski_cylindrical";
    minkowski_cylindrical_obj.adaptive_precision = false;
    minkowski_cylindrical_obj.system = metric::coordinate_system::OTHER;

    metric::metric<alcubierre_metric, cartesian_to_polar_dual, polar_to_cartesian_dual, alcubierre_distance> alcubierre_metric_obj;
    alcubierre_metric_obj.name = "alcubierre";
    alcubierre_metric_obj.system = metric::coordinate_system::CARTESIAN;

    metric::metric<symmetric_warp_drive, polar_to_polar, polar_to_polar, at_origin> symmetric_warp_obj;
    symmetric_warp_obj.name = "symmetric_warp";
    symmetric_warp_obj.detect_singularities = true;
    symmetric_warp_obj.singular = true;
    symmetric_warp_obj.singular_terminator = 1.001f;
    //symmetric_warp_obj.adaptive_precision = false;

    metric::metric<krasnikov_tube_metric, cylindrical_to_polar, polar_to_cylindrical, at_origin> krasnikov_tube_obj;
    krasnikov_tube_obj.name = "krasnikov_tube";
    krasnikov_tube_obj.adaptive_precision = true;
    krasnikov_tube_obj.system = metric::coordinate_system::OTHER;

    metric::metric<krasnikov_tube_metric_cart, cartesian_to_polar_dual, polar_to_cartesian_dual, at_origin> krasnikov_tube_cart_obj;
    krasnikov_tube_cart_obj.name = "krasnikov_tube_cart";
    krasnikov_tube_cart_obj.adaptive_precision = true;
    krasnikov_tube_cart_obj.system = metric::coordinate_system::CARTESIAN;

    metric::metric<double_kerr, cylindrical_to_polar, polar_to_cylindrical, at_origin> double_kerr_obj;
    double_kerr_obj.name = "double_kerr";
    double_kerr_obj.adaptive_precision = true;
    double_kerr_obj.detect_singularities = true;
    double_kerr_obj.system = metric::coordinate_system::CYLINDRICAL;

    metric::metric<unequal_double_kerr, cylindrical_to_polar, polar_to_cylindrical, at_origin> unequal_double_kerr_obj;
    unequal_double_kerr_obj.name = "unequal_double_kerr";
    unequal_double_kerr_obj.adaptive_precision = true;
    unequal_double_kerr_obj.detect_singularities = true;
    unequal_double_kerr_obj.system = metric::coordinate_system::CYLINDRICAL;

    metric::config cfg;
    cfg.universe_size = 10000;
    //cfg.error_override = 100.f;
    //cfg.error_override = 0.000001f;
    //cfg.error_override = 0.00001f;
    //cfg.error_override = 0.0001f;
    cfg.redshift = true;

    //auto current_metric = symmetric_warp_obj;
    //auto current_metric = kerr_obj;
    //auto current_metric = alcubierre_metric_obj;
    //auto current_metric = kerr_newman_obj;
    //auto current_metric = kerr_schild_obj;
    //auto current_metric = simple_wormhole;
    //auto current_metric = schwarzs_polar;
    //auto current_metric = minkowski_polar_obj;
    //auto current_metric = krasnikov_tube_cart_obj;
    auto current_metric = double_kerr_obj;
    //auto current_metric = unequal_double_kerr_obj;

    argument_string += build_argument_string(current_metric, cfg);
    #endif // GENERIC_METRIC

    std::cout << "ASTRING " << argument_string << std::endl;

    #endif // GENERIC_METRIC

    printf("WLs %f %f %f\n", chromaticity::srgb_to_wavelength({1, 0, 0}), chromaticity::srgb_to_wavelength({0, 1, 0}), chromaticity::srgb_to_wavelength({0, 0, 1}));

    cl::program prog(clctx.ctx, "cl.cl");
    prog.build(clctx.ctx, argument_string);

    clctx.ctx.register_program(prog);

    int supersample_mult = 2;

    int supersample_width = sett.width * supersample_mult;
    int supersample_height = sett.height * supersample_mult;

    texture_settings tsett;
    tsett.width = supersample_width;
    tsett.height = supersample_height;
    tsett.is_srgb = false;

    std::array<texture, 2> tex;
    tex[0].load_from_memory(tsett, nullptr);
    tex[1].load_from_memory(tsett, nullptr);

    std::array<cl::gl_rendertexture, 2> rtex{clctx.ctx, clctx.ctx};
    rtex[0].create_from_texture(tex[0].handle);
    rtex[1].create_from_texture(tex[1].handle);

    int which_buffer = 0;

    sf::Image img;
    img.loadFromFile("background.png");

    cl::image clbackground(clctx.ctx);

    std::vector<vec4f> as_float;
    std::vector<uint8_t> as_uint8;

    for(int y=0; y < img.getSize().y; y++)
    {
        for(int x=0; x < img.getSize().x; x++)
        {
            auto col = img.getPixel(x, y);

            vec4f val = {col.r / 255.f, col.g / 255.f, col.b / 255.f, col.a / 255.f};

            as_float.push_back(val);
            as_uint8.push_back(col.r);
            as_uint8.push_back(col.g);
            as_uint8.push_back(col.b);
            as_uint8.push_back(col.a);
        }
    }

    clbackground.alloc({img.getSize().x, img.getSize().y}, {CL_RGBA, CL_FLOAT});

    vec<2, size_t> origin = {0,0};
    vec<2, size_t> region = {img.getSize().x, img.getSize().y};

    clbackground.write(clctx.cqueue, (const char*)&as_float[0], origin, region);

    texture_settings bsett;
    bsett.width = img.getSize().x;
    bsett.height = img.getSize().y;
    bsett.is_srgb = false;

    texture background_with_mips;
    background_with_mips.load_from_memory(bsett, &as_uint8[0]);

    #define MIP_LEVELS 11

    cl::image_with_mipmaps background_mipped(clctx.ctx);
    background_mipped.alloc((vec2i){img.getSize().x, img.getSize().y}, MIP_LEVELS, {CL_RGBA, CL_FLOAT});

    int swidth = img.getSize().x;
    int sheight = img.getSize().y;

    for(int i=0; i < MIP_LEVELS; i++)
    {
        printf("I is %i\n", i);

        int cwidth = swidth;
        int cheight = sheight;

        swidth /= 2;
        sheight /= 2;

        cl::gl_rendertexture temp(clctx.ctx);
        temp.create_from_texture_with_mipmaps(background_with_mips.handle, i);
        temp.acquire(clctx.cqueue);

        std::vector<cl_uchar4> res = temp.read<2, cl_uchar4>(clctx.cqueue, (vec<2, size_t>){0,0}, (vec<2, size_t>){cwidth, cheight});

        temp.unacquire(clctx.cqueue);

        std::vector<cl_float4> converted;
        converted.reserve(res.size());

        for(auto& i : res)
        {
            converted.push_back({i.s[0] / 255.f, i.s[1] / 255.f, i.s[2] / 255.f, i.s[3] / 255.f});
        }

        background_mipped.write(clctx.cqueue, (char*)&converted[0], vec<2, size_t>{0, 0}, vec<2, size_t>{cwidth, cheight}, i);
    }

    cl::device_command_queue dqueue(clctx.ctx);

    ///t, x, y, z
    //vec4f camera = {0, -2, -2, 0};
    //vec4f camera = {0, -2, -8, 0};
    vec4f camera = {0, 0, -4, 0};
    //vec4f camera = {0, 0.01, -0.024, -5.5};
    //vec4f camera = {0, 0, -4, 0};
    quat camera_quat;

    quat q;
    q.load_from_axis_angle({1, 0, 0, -M_PI/2});

    camera_quat = q * camera_quat;

    //camera_quat.load_from_matrix(axis_angle_to_mat({0, 0, 0}, 0));

    vec3f forward_axis = {0, 0, 1};
    vec3f up_axis = {0, 1, 0};

    sf::Clock clk;

    int ray_count = supersample_width * supersample_height;

    cl::buffer schwarzs_1(clctx.ctx);
    cl::buffer schwarzs_2(clctx.ctx);
    #ifndef GENERIC_METRIC
    cl::buffer kruskal_1(clctx.ctx);
    cl::buffer kruskal_2(clctx.ctx);
    #endif // GENERIC_METRIC
    cl::buffer finished_1(clctx.ctx);

    cl::buffer schwarzs_count_1(clctx.ctx);
    cl::buffer schwarzs_count_2(clctx.ctx);
    cl::buffer kruskal_count_1(clctx.ctx);
    cl::buffer kruskal_count_2(clctx.ctx);
    cl::buffer finished_count_1(clctx.ctx);

    cl::buffer geodesic_trace_buffer(clctx.ctx);
    geodesic_trace_buffer.alloc(64000 * sizeof(cl_float4));

    std::vector<cl_float4> current_geodesic_path;

    cl::buffer texture_coordinates[2] = {clctx.ctx, clctx.ctx};

    for(int i=0; i < 2; i++)
    {
        texture_coordinates[i].alloc(supersample_width * supersample_height * sizeof(float) * 2);
        texture_coordinates[i].set_to_zero(clctx.cqueue);
    }

    schwarzs_1.alloc(sizeof(lightray) * ray_count);
    schwarzs_2.alloc(sizeof(lightray) * ray_count);
    #ifndef GENERIC_METRIC
    kruskal_1.alloc(sizeof(lightray) * ray_count);
    kruskal_2.alloc(sizeof(lightray) * ray_count);
    #endif // GENERIC_METRIC
    finished_1.alloc(sizeof(lightray) * ray_count);

    schwarzs_count_1.alloc(sizeof(int));
    schwarzs_count_2.alloc(sizeof(int));
    kruskal_count_1.alloc(sizeof(int));
    kruskal_count_2.alloc(sizeof(int));
    finished_count_1.alloc(sizeof(int));

    cl_sampler_properties sampler_props[] = {
    CL_SAMPLER_NORMALIZED_COORDS, CL_TRUE,
    CL_SAMPLER_ADDRESSING_MODE, CL_ADDRESS_REPEAT,
    CL_SAMPLER_FILTER_MODE, CL_FILTER_LINEAR,
    CL_SAMPLER_MIP_FILTER_MODE_KHR, CL_FILTER_LINEAR,
    //CL_SAMPLER_LOD_MIN_KHR, 0.0f,
    //CL_SAMPLER_LOD_MAX_KHR, FLT_MAX,
    0
    };

    cl_sampler sam = clCreateSamplerWithProperties(clctx.ctx.native_context.data, sampler_props, nullptr);

    std::optional<cl::event> last_event;

    std::cout << "Supports shared events? " << cl::supports_extension(clctx.ctx, "cl_khr_gl_event") << std::endl;

    bool supersample = false;
    bool should_take_screenshot = false;

    int screenshot_w = 1920;
    int screenshot_h = 1080;
    bool time_progresses = false;
    bool flip_sign = false;
    float current_geodesic_time = 0;
    bool camera_on_geodesic = false;
    bool camera_time_progresses = false;
    bool camera_geodesics_go_foward = true;
    //vec2f base_angle = {M_PI/2, 0};
    vec2f base_angle = {M_PI/2, 0};

    while(!win.should_close())
    {
        win.poll();

        glFinish();

        auto buffer_size = rtex[which_buffer].size<2>();

        bool taking_screenshot = should_take_screenshot;
        should_take_screenshot = false;

        bool should_snapshot_geodesic = false;

        if((vec2i){buffer_size.x() / supersample_mult, buffer_size.y() / supersample_mult} != win.get_window_size() || taking_screenshot)
        {
            if(last_event.has_value())
                last_event.value().block();

            last_event = std::nullopt;

            if(!taking_screenshot)
            {
                supersample_width = win.get_window_size().x() * supersample_mult;
                supersample_height = win.get_window_size().y() * supersample_mult;
            }
            else
            {
                supersample_width = screenshot_w * supersample_mult;
                supersample_height = screenshot_h * supersample_mult;
            }

            ray_count = supersample_width * supersample_height;

            texture_settings new_sett;
            new_sett.width = supersample_width;
            new_sett.height = supersample_height;
            new_sett.is_srgb = false;

            tex[0].load_from_memory(new_sett, nullptr);
            tex[1].load_from_memory(new_sett, nullptr);

            rtex[0].create_from_texture(tex[0].handle);
            rtex[1].create_from_texture(tex[1].handle);

            schwarzs_1.alloc(sizeof(lightray) * ray_count);
            schwarzs_2.alloc(sizeof(lightray) * ray_count);
            #ifndef GENERIC_METRIC
            kruskal_1.alloc(sizeof(lightray) * ray_count);
            kruskal_2.alloc(sizeof(lightray) * ray_count);
            #endif // GENERIC_METRIC
            finished_1.alloc(sizeof(lightray) * ray_count);

            for(int i=0; i < 2; i++)
            {
                texture_coordinates[i].alloc(supersample_width * supersample_height * sizeof(float) * 2);
                texture_coordinates[i].set_to_zero(clctx.cqueue);
            }
        }

        rtex[which_buffer].acquire(clctx.cqueue);

        float ds = 0.01;

        float speed = 0.001;

        if(ImGui::IsKeyDown(GLFW_KEY_LEFT_SHIFT))
            speed = 0.1;

        if(ImGui::IsKeyDown(GLFW_KEY_LEFT_CONTROL))
            speed = 0.00001;

        if(ImGui::IsKeyDown(GLFW_KEY_LEFT_ALT))
            speed /= 1000;

        if(ImGui::IsKeyDown(GLFW_KEY_Z))
            speed *= 100;

        if(ImGui::IsKeyDown(GLFW_KEY_X))
            speed *= 100;

        if(ImGui::IsKeyPressed(GLFW_KEY_B))
        {
            camera = {0, 0, 0, -100};
        }

        if(ImGui::IsKeyPressed(GLFW_KEY_N))
        {
            camera = {0, 0, 0, -1.16};
        }

        if(ImGui::IsKeyPressed(GLFW_KEY_M))
        {
            camera = {0, 0, 0, 1.16};
        }

        if(ImGui::IsKeyPressed(GLFW_KEY_J))
        {
            camera = {0, -1.16, 0, 0};
        }

        if(ImGui::IsKeyPressed(GLFW_KEY_K))
        {
            camera = {0, 1.16, 0, 0};
        }

        if(ImGui::IsKeyPressed(GLFW_KEY_V))
        {
            camera = {0, 0, 0, 1.03};
        }

        if(ImGui::IsKeyPressed(GLFW_KEY_C))
        {
            camera = {0, 0, 0, 0};
        }

        if(ImGui::IsKeyPressed(GLFW_KEY_R))
        {
            camera = {0, 0, 22, 0};
        }

        if(ImGui::IsKeyDown(GLFW_KEY_RIGHT))
        {
            mat3f m = mat3f().ZRot(M_PI/128);

            quat q;
            q.load_from_matrix(m);

            camera_quat = q * camera_quat;
        }

        if(ImGui::IsKeyDown(GLFW_KEY_LEFT))
        {
            mat3f m = mat3f().ZRot(-M_PI/128);

            quat q;
            q.load_from_matrix(m);

            camera_quat = q * camera_quat;
        }

        if(ImGui::IsKeyPressed(GLFW_KEY_1))
        {
            flip_sign = !flip_sign;
        }

        vec3f up = {0, 0, -1};
        vec3f right = rot_quat({1, 0, 0}, camera_quat);
        vec3f forward_axis = rot_quat({0, 0, 1}, camera_quat);

        if(ImGui::IsKeyDown(GLFW_KEY_DOWN))
        {
            quat q;
            q.load_from_axis_angle({right.x(), right.y(), right.z(), M_PI/128});

            camera_quat = q * camera_quat;
        }

        if(ImGui::IsKeyDown(GLFW_KEY_UP))
        {
            quat q;
            q.load_from_axis_angle({right.x(), right.y(), right.z(), -M_PI/128});

            camera_quat = q * camera_quat;
        }

        vec3f offset = {0,0,0};

        offset += forward_axis * ((ImGui::IsKeyDown(GLFW_KEY_W) - ImGui::IsKeyDown(GLFW_KEY_S)) * speed);
        offset += right * (ImGui::IsKeyDown(GLFW_KEY_D) - ImGui::IsKeyDown(GLFW_KEY_A)) * speed;
        offset += up * (ImGui::IsKeyDown(GLFW_KEY_E) - ImGui::IsKeyDown(GLFW_KEY_Q)) * speed;

        camera.y() += offset.x();
        camera.z() += offset.y();
        camera.w() += offset.z();

        vec4f scamera = cartesian_to_schwarz(camera);

        if(flip_sign)
            scamera.y() = -scamera.y();

        float time = clk.restart().asMicroseconds() / 1000.;

        if(camera_on_geodesic)
        {
            scamera = interpolate_geodesic(current_geodesic_path, current_geodesic_time);
            base_angle = get_geodesic_intersection(current_geodesic_path);
        }

        if(!taking_screenshot)
        {
            ImGui::Begin("DBG", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

            ImGui::DragFloat3("Polar Pos", &scamera.v[1]);
            ImGui::DragFloat3("Cart Pos", &camera.v[1]);

            ImGui::DragFloat("Time", &time);

            ImGui::Checkbox("Supersample", &supersample);

            ImGui::SliderFloat("CTime", &camera.v[0], 0.f, 100.f);

            ImGui::Checkbox("Time Progresses", &time_progresses);

            if(time_progresses)
                camera.v[0] += time / 1000.f;

            if(ImGui::Button("Screenshot"))
                should_take_screenshot = true;

            ImGui::DragFloat("Geodesic Camera Time", &current_geodesic_time, 0.1, -100.f, 100.f);
            //ImGui::SliderFloat("Geodesic Camera Time", &current_geodesic_time, -100.f, 100.f);

            ImGui::Checkbox("Use Camera Geodesic", &camera_on_geodesic);

            ImGui::Checkbox("Camera Time Progresses", &camera_time_progresses);

            if(camera_time_progresses)
                current_geodesic_time += time / 1000.f;

            if(ImGui::Button("Snapshot Camera Geodesic"))
            {
                should_snapshot_geodesic = true;
            }

            ImGui::Checkbox("Camera Snapshot Geodesic goes forward", &camera_geodesics_go_foward);

            ImGui::End();
        }

        int width = win.get_window_size().x();
        int height = win.get_window_size().y();

        if(supersample)
        {
            width *= supersample_mult;
            height *= supersample_mult;
        }

        if(taking_screenshot)
        {
            width = rtex[which_buffer].size<2>().x();
            height = rtex[which_buffer].size<2>().y();
        }

        cl::args clr;
        clr.push_back(rtex[which_buffer]);

        clctx.cqueue.exec("clear", clr, {width, height}, {16, 16});

        #if 0
        cl::args args;
        args.push_back(rtex[0]);
        args.push_back(ds);
        args.push_back(camera);
        args.push_back(camera_quat);
        args.push_back(clbackground);

        clctx.cqueue.exec("do_raytracing_multicoordinate", args, {win.get_window_size().x(), win.get_window_size().y()}, {16, 16});

        rtex[0].unacquire(clctx.cqueue);

        glFinish();
        clctx.cqueue.block();
        glFinish();
        #endif // OLD_AND_GOOD

        #if 1
        schwarzs_count_1.set_to_zero(clctx.cqueue);
        schwarzs_count_2.set_to_zero(clctx.cqueue);
        kruskal_count_1.set_to_zero(clctx.cqueue);
        kruskal_count_2.set_to_zero(clctx.cqueue);
        finished_count_1.set_to_zero(clctx.cqueue);

        int fallback = 0;

        cl::buffer* b1 = &schwarzs_1;
        cl::buffer* b2 = &schwarzs_2;
        cl::buffer* c1 = &schwarzs_count_1;
        cl::buffer* c2 = &schwarzs_count_2;

        cl::event next;

        {
            #ifndef GENERIC_METRIC
            cl::args init_args;
            init_args.push_back(camera);
            init_args.push_back(camera_quat);
            init_args.push_back(*b1);
            init_args.push_back(kruskal_1); ///temp
            init_args.push_back(*c1);
            init_args.push_back(kruskal_count_1); ///temp
            init_args.push_back(width);
            init_args.push_back(height);

            clctx.cqueue.exec("init_rays", init_args, {width, height}, {16, 16});

            cl::args run_args;
            run_args.push_back(*b1);
            run_args.push_back(*b2);
            run_args.push_back(kruskal_1);
            run_args.push_back(kruskal_2);
            run_args.push_back(finished_1);
            run_args.push_back(*c1);
            run_args.push_back(*c2);
            run_args.push_back(kruskal_count_1);
            run_args.push_back(kruskal_count_2);
            run_args.push_back(finished_count_1);
            run_args.push_back(width);
            run_args.push_back(height);
            run_args.push_back(fallback);

            clctx.cqueue.exec("relauncher", run_args, {1}, {1});
            #else

            int isnap = should_snapshot_geodesic;

            if(should_snapshot_geodesic)
            {
                if(camera_geodesics_go_foward)
                {
                    isnap = 1;
                }
                else
                {
                    isnap = 0;
                }
            }

            cl::args init_args;
            init_args.push_back(scamera);
            init_args.push_back(camera_quat);
            init_args.push_back(*b1);
            init_args.push_back(*c1);
            init_args.push_back(width);
            init_args.push_back(height);
            init_args.push_back(isnap);
            init_args.push_back(base_angle);

            clctx.cqueue.exec("init_rays_generic", init_args, {width, height}, {16, 16});

            if(should_snapshot_geodesic)
            {
                int idx = (height/2) * width + width/2;

                geodesic_trace_buffer.set_to_zero(clctx.cqueue);

                cl::args snapshot_args;
                snapshot_args.push_back(*b1);
                snapshot_args.push_back(geodesic_trace_buffer);
                snapshot_args.push_back(*c1);
                snapshot_args.push_back(idx);
                snapshot_args.push_back(width);
                snapshot_args.push_back(height);
                snapshot_args.push_back(scamera);
                snapshot_args.push_back(camera_quat);
                snapshot_args.push_back(base_angle);

                clctx.cqueue.exec("get_geodesic_path", snapshot_args, {1}, {1});

                current_geodesic_path = geodesic_trace_buffer.read<cl_float4>(clctx.cqueue);
            }

            cl::args run_args;
            run_args.push_back(*b1);
            run_args.push_back(*b2);
            run_args.push_back(finished_1);
            run_args.push_back(*c1);
            run_args.push_back(*c2);
            run_args.push_back(finished_count_1);
            run_args.push_back(width);
            run_args.push_back(height);
            run_args.push_back(fallback);

            clctx.cqueue.exec("relauncher_generic", run_args, {1}, {1});

            #endif // GENERIC_METRIC

            cl::args texture_args;
            texture_args.push_back(finished_1);
            texture_args.push_back(finished_count_1);
            texture_args.push_back(texture_coordinates[which_buffer]);
            texture_args.push_back(width);
            texture_args.push_back(height);
            texture_args.push_back(scamera);
            texture_args.push_back(camera_quat);
            texture_args.push_back(base_angle);

            clctx.cqueue.exec("calculate_texture_coordinates", texture_args, {width * height}, {256});

            cl::args render_args;
            render_args.push_back(finished_1);
            render_args.push_back(finished_count_1);
            render_args.push_back(rtex[which_buffer]);
            render_args.push_back(background_mipped);
            render_args.push_back(width);
            render_args.push_back(height);
            render_args.push_back(texture_coordinates[which_buffer]);
            render_args.push_back(sam);

            next = clctx.cqueue.exec("render", render_args, {width * height}, {256});
        }

        clctx.cqueue.flush();

        if(taking_screenshot)
        {
            printf("Taking screenie\n");

            clctx.cqueue.block();

            printf("Blocked\n");

            std::cout << "WIDTH " << (screenshot_w * supersample_mult) << " HEIGHT "<< (screenshot_h * supersample_mult) << std::endl;

            std::vector<cl_uchar4> pixels = rtex[which_buffer].read<2, cl_uchar4>(clctx.cqueue, {0,0}, {screenshot_w * supersample_mult, screenshot_h * supersample_mult});

            printf("Readback\n");

            std::cout << "pixels size " << pixels.size() << std::endl;

            sf::Image img;
            img.create(screenshot_w * supersample_mult, screenshot_h * supersample_mult, (const sf::Uint8*)&pixels[0]);

            std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();
            auto duration = now.time_since_epoch();
            auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

            std::string fname = "./screenshots/" + current_metric.name + "_" + std::to_string(millis) + ".png";

            img.saveToFile(fname);
        }

        rtex[which_buffer].unacquire(clctx.cqueue);

        which_buffer = (which_buffer + 1) % 2;

        if(last_event.has_value())
            last_event.value().block();

        last_event = next;
        #endif

        {
            ImDrawList* lst = ImGui::GetBackgroundDrawList();

            ImVec2 screen_pos = ImGui::GetMainViewport()->Pos;

            ImVec2 tl = {0,0};
            ImVec2 br = {win.get_window_size().x(),win.get_window_size().y()};

            if(win.get_render_settings().viewports)
            {
                tl.x += screen_pos.x;
                tl.y += screen_pos.y;

                br.x += screen_pos.x;
                br.y += screen_pos.y;
            }

            if(!supersample)
                lst->AddImage((void*)rtex[which_buffer].texture_id, tl, br, ImVec2(0, 0), ImVec2(1.f/supersample_mult, 1.f/supersample_mult));
            else
                lst->AddImage((void*)rtex[which_buffer].texture_id, tl, br, ImVec2(0, 0), ImVec2(1, 1));
        }

        win.display();
    }

    last_event = std::nullopt;

    clctx.cqueue.block();

    return 0;
}
