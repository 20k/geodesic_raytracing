#include <iostream>
#include <toolkit/render_window.hpp>
#include <toolkit/texture.hpp>
#include <vec/vec.hpp>
#include <GLFW/glfw3.h>
#include <SFML/Graphics.hpp>
#include <CL/cl_ext.h>
#include "dual.hpp"
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
*/

///perfectly fine
vec4f cartesian_to_schwarz(vec4f position)
{
    vec3f polar = cartesian_to_polar((vec3f){position.y(), position.z(), position.w()});

    return (vec4f){position.x(), polar.x(), polar.y(), polar.z()};
}


struct lightray
{
    vec4f position;
    vec4f velocity;
    vec4f acceleration;
    int sx, sy;
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

inline
std::array<dual, 4> schwarzschild_blackhole_lemaitre(dual T, dual p, dual theta, dual phi)
{
    dual rs = 1;

    dual r = pow(((3/2.f) * (p - T)), 2.f/3.f) * pow(rs, 1.f/3.f);

    dual dT = -1;
    dual dp = (rs / r);
    dual dtheta = r * r;
    dual dphi = r * r * sin(theta) * sin(theta);

    return {dT, dp, dtheta, dphi};
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

    dual m = 0;
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

    dual a = 2;
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
    dual a = 2;

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
    dual a = 2;

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

///we're in inclination
inline
std::array<dual_complex, 16> cylinder_test(dual_complex t, dual_complex r, dual_complex phi, dual_complex z)
{
    std::array<dual_complex, 16> ret_fat;
    ret_fat[0] = -1;
    ret_fat[1 * 4 + 1] = 1;
    ret_fat[2 * 4 + 2] = r * r;
    ret_fat[3 * 4 + 3] = 1;

    return ret_fat;
}

///rendering alcubierre nicely is very hard: the shell is extremely thin, and flat on both sides
///this means that a naive timestepping method results in a lot of distortion
///need to crank down subambient_precision, and crank up new_max to about 20 * rs
///performance and quality could be made significantly better with a dynamic timestep that only reduces the timestep around
///the border shell
inline
std::array<dual, 16> alcubierre_metric(dual t, dual x, dual y, dual z)
{
    dual dxs_t = 0.9;
    dual xs_t = dxs_t * t;
    dual vs_t = dxs_t;

    dual rs_t = sqrt((x - xs_t) * (x - xs_t) + y * y + z * z);

    dual sigma = 20;
    dual R = 1;

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

//inline auto coordinate_transform_to = cylindrical_to_polar;
//inline auto coordinate_transform_from = polar_to_cylindrical;

inline auto coordinate_transform_to = polar_to_polar;
inline auto coordinate_transform_from = polar_to_polar;

//inline auto coordinate_transform_to = oblate_to_polar;
//inline auto coordinate_transform_from = polar_to_oblate;

//inline auto coordinate_transform_to = cartesian_to_polar_dual;
//inline auto coordinate_transform_from = polar_to_cartesian_dual;

//inline auto coordinate_transform_to = lemaitre_to_polar;
//inline auto coordinate_transform_from = polar_to_lemaitre;

//inline auto coordinate_transform_to = rational_to_polar;
//inline auto coordinate_transform_from = polar_to_rational;

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

int main()
{
    render_settings sett;
    sett.width = 1000;
    sett.height = 800;
    sett.opencl = true;
    sett.no_double_buffer = true;

    render_window win(sett, "Geodesics");

    assert(win.clctx);

    opencl_context& clctx = *win.clctx;

    std::string argument_string = "-O5 -cl-std=CL2.2 ";

    #ifdef GENERIC_METRIC
    //auto [real_eq, derivatives] = evaluate_metric(test_metric, "v1", "v2", "v3", "v4");
    auto [real_eq, derivatives] = evaluate_metric2D(kerr_metric, "v1", "v2", "v3", "v4");
    //auto [real_eq, derivatives] = evaluate_metric2D(kerr_rational_polynomial, "v1", "v2", "v3", "v4");
    //auto [real_eq, derivatives] = evaluate_metric2D(kerr_schild_metric, "v1", "v2", "v3", "v4");
    //auto [real_eq, derivatives] = evaluate_metric(schwarzschild_blackhole, "v1", "v2", "v3", "v4");
    //auto [real_eq, derivatives] = evaluate_metric(schwarzschild_blackhole_lemaitre, "v1", "v2", "v3", "v4");

    //auto [real_eq, derivatives] = evaluate_metric2D_DC(cylinder_test, "v1", "v2", "v3", "v4");
    //auto [real_eq, derivatives] = evaluate_metric2D_DC(big_imaginary_metric_test, "v1", "v2", "v3", "v4");
    //auto [real_eq, derivatives] = evaluate_metric2D_DC(minkowski_space, "v1", "v2", "v3", "v4");

    //auto [real_eq, derivatives] = evaluate_metric2D(kerr_metric, "v1", "v2", "v3", "v4");
    //auto [real_eq, derivatives] = evaluate_metric2D(ellis_drainhole, "v1", "v2", "v3", "v4");
    //auto [real_eq, derivatives] = evaluate_metric2D(janis_newman_winicour, "v1", "v2", "v3", "v4");
    //auto [real_eq, derivatives] = evaluate_metric2D(alcubierre_metric, "v1", "v2", "v3", "v4");
    //auto [real_eq, derivatives] = evaluate_metric2D(big_metric_test, "v1", "v2", "v3", "v4");

    /*{
        auto [spherical_eq, totals] = total_diff(spherical_to_cartesian, "v1", "v2", "v3");

        for(auto& i : spherical_eq)
        {
            std::cout << "IB " << i << std::endl;
        }

        for(auto& i : totals)
        {
            std::cout << "TOTAL " << i << std::endl;
        }
    }*/

    argument_string += "-DRS_IMPL=1 -DC_IMPL=1 ";

    for(int i=0; i < (int)real_eq.size(); i++)
    {
        argument_string += "-DF" + std::to_string(i + 1) + "_I=" + real_eq[i] + " ";
    }

    if(derivatives.size() == 16)
    {
        for(int j=0; j < 4; j++)
        {
            for(int i=0; i < 4; i++)
            {
                int script_idx = j * 4 + i + 1;
                int my_idx = i * 4 + j;

                argument_string += "-DF" + std::to_string(script_idx) + "_P=" + derivatives[my_idx] + " ";
            }
        }
    }

    if(derivatives.size() == 64)
    {
        for(int i=0; i < 64; i++)
            argument_string += "-DF" + std::to_string(i + 1) + "_P=" + derivatives[i] + " ";

        argument_string += " -DGENERIC_BIG_METRIC ";
    }

    {
        auto [to_polar, dt_to_spherical] = total_diff(coordinate_transform_to, "v1", "v2", "v3", "v4");
        auto [from_polar, dt_from_spherical] = total_diff(coordinate_transform_from, "v1", "v2", "v3", "v4");

        for(int i=0; i < to_polar.size(); i++)
        {
            argument_string += "-DTO_COORD" + std::to_string(i + 1) + "=" + to_polar[i] + " ";
        }

        for(int i=0; i < dt_to_spherical.size(); i++)
        {
            argument_string += "-DTO_DCOORD" + std::to_string(i + 1) + "=" + dt_to_spherical[i] + " ";
        }

        for(int i=0; i < from_polar.size(); i++)
        {
            argument_string += "-DFROM_COORD" + std::to_string(i + 1) + "=" + from_polar[i] + " ";
        }

        for(int i=0; i < dt_from_spherical.size(); i++)
        {
            argument_string += "-DFROM_DCOORD" + std::to_string(i + 1) + "=" + dt_from_spherical[i] + " ";
        }
    }

    argument_string += " -DGENERIC_METRIC";
    //argument_string += " -DEULER_INTEGRATION_GENERIC";
    //argument_string += " -DRK4_GENERIC";
    argument_string += " -DVERLET_INTEGRATION_GENERIC";

    //argument_string += " -DGENERIC_CONSTANT_THETA";
    //argument_string += " -DPOLE_SINGULAIRTY";
    //argument_string += " -DSINGULAR";
    //argument_string += " -DTRAVERSABLE_EVENT_HORIZON";
    //argument_string += " -DSINGULAR_TERMINATOR=0.75";
    argument_string += " -DUNIVERSE_SIZE=200000";
    //argument_string += " -DSINGULAR_TERMINATOR=1.000001";

    argument_string += " -DSINGULARITY_DETECTION";

    argument_string += " -DADAPTIVE_PRECISION";
    argument_string += " -DMAX_ACCELERATION_CHANGE=0.0000001f";

    ///coordinate weights
    ///singular polar
    argument_string += " -DW_V1=1 -DW_V2=1 -DW_V3=8 -DW_V4=32";
    ///non singular polar
    //argument_string += " -DW_V1=1 -DW_V2=1 -DW_V3=8 -DW_V4=8";
    ///cartesian
    //argument_string += " -DW_V1=1 -DW_V2=1 -DW_V3=1 -DW_V4=1";

    std::cout << "ASTRING " << argument_string << std::endl;

    #endif // GENERIC_METRIC

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
    vec4f camera = {0, -2, -8, 0};
    //vec4f camera = {0, 0, -8, 0};
    //vec4f camera = {0, 0.01, -0.024, -5.5};
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
    cl::buffer kruskal_1(clctx.ctx);
    cl::buffer kruskal_2(clctx.ctx);
    cl::buffer finished_1(clctx.ctx);

    cl::buffer schwarzs_count_1(clctx.ctx);
    cl::buffer schwarzs_count_2(clctx.ctx);
    cl::buffer kruskal_count_1(clctx.ctx);
    cl::buffer kruskal_count_2(clctx.ctx);
    cl::buffer finished_count_1(clctx.ctx);

    cl::buffer texture_coordinates[2] = {clctx.ctx, clctx.ctx};

    for(int i=0; i < 2; i++)
    {
        texture_coordinates[i].alloc(supersample_width * supersample_height * sizeof(float) * 2);
        texture_coordinates[i].set_to_zero(clctx.cqueue);
    }

    schwarzs_1.alloc(sizeof(lightray) * ray_count * 3);
    schwarzs_2.alloc(sizeof(lightray) * ray_count * 3);
    kruskal_1.alloc(sizeof(lightray) * ray_count * 3);
    kruskal_2.alloc(sizeof(lightray) * ray_count * 3);
    finished_1.alloc(sizeof(lightray) * ray_count * 3);

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

    while(!win.should_close())
    {
        win.poll();

        glFinish();
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

        float time = clk.restart().asMicroseconds() / 1000.;

        ImGui::Begin("DBG", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

        ImGui::DragFloat3("Pos", &scamera.v[1]);

        ImGui::DragFloat("Time", &time);

        ImGui::Checkbox("Supersample", &supersample);

        ImGui::SliderFloat("CTime", &camera.v[0], 0.f, 100.f);

        ImGui::End();

        int width = win.get_window_size().x();
        int height = win.get_window_size().y();

        if(supersample)
        {
            width *= supersample_mult;
            height *= supersample_mult;
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

            cl::args init_args;
            init_args.push_back(camera);
            init_args.push_back(camera_quat);
            init_args.push_back(*b1);
            init_args.push_back(*c1);
            init_args.push_back(width);
            init_args.push_back(height);

            clctx.cqueue.exec("init_rays_generic", init_args, {width, height}, {16, 16});

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
            texture_args.push_back(camera);
            texture_args.push_back(camera_quat);

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
