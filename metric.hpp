#ifndef METRIC_HPP_INCLUDED
#define METRIC_HPP_INCLUDED

#include "dual.hpp"
#include "dual_value.hpp"
#include <nlohmann/json.hpp>
#include "js_interop.hpp"
#include <vec/tensor.hpp>
#include <span>

namespace metrics
{
    inline
    std::array<dual, 4> cartesian_to_polar_v(const dual& t, const dual& x, const dual& y, const dual& z)
    {
        dual r = sqrt(x * x + y * y + z * z);
        dual theta = atan2(sqrt(x*x + y*y), z);
        dual phi = atan2(y, x);

        return {t, r, theta, phi};
    }

    template<typename Func, typename... T>
    inline
    std::pair<std::vector<value>, std::vector<value>> evaluate_metric2D(Func&& f, T... raw_variables)
    {
        std::array<std::string, sizeof...(T)> variable_names{raw_variables...};

        std::vector<value> raw_eq;
        std::vector<value> raw_derivatives;

        for(int i=0; i < (int)variable_names.size(); i++)
        {
            std::array<dual, sizeof...(T)> variables;

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

    struct metric_info
    {
        metric<value, 4, 4> met;
        tensor<value, 4, 4, 4> partials;

        bool is_diagonal()
        {
            for(int i=0; i < 4; i++)
            {
                for(int j=0; j < 4; j++)
                {
                    if(i == j)
                        continue;

                    if(type_to_string(met.idx(i, j)) != "0" && type_to_string(met.idx(i, j)) != "0.0")
                        return false;
                }
            }

            return true;
        }

        template<typename Func>
        metric_info(Func&& f)
        {
            auto [v_met, v_partials] = evaluate_metric2D(f, "v1", "v2", "v3", "v4");

            assert(v_met.size() == 16);
            assert(v_partials.size() == 64);

            for(int i=0; i < 4; i++)
            {
                for(int j=0; j < 4; j++)
                {
                    met.idx(i,j) = v_met[i * 4 + j];
                }
            }

            for(int i=0; i < 4; i++)
            {
                for(int j=0; j < 4; j++)
                {
                    for(int k=0; k < 4; k++)
                    {
                        partials.idx(i, j, k) = v_partials[i * 16 + j * 4 + k];
                    }
                }
            }
        }
    };

    inline
    tensor<value, 4> fix_light_velocity(metric_info& inf, const tensor<value, 4>& v)
    {
        tensor<value, 3> spatial_v = {v.y(), v.z(), v.w()};
        tensor<value, 3> spatial_m = {inf.met.idx(1, 1), inf.met.idx(2, 2), inf.met.idx(3, 3)};

        value tvl_2 = sum_multiply(spatial_m, spatial_v * spatial_v) / -inf.met.idx(0, 0);

        value sign = dual_types::if_v(v.x() < 0, -1.f, 1.f);

        tensor<value, 4> ret = v;

        ret.x() = sign * sqrt(tvl_2);

        return ret;
    }

    inline
    tensor<value, 4> calculate_acceleration(metric_info& inf)
    {
        tensor<value, 4> velocity = {"iv1", "iv2", "iv3", "iv4"};

        if(inf.is_diagonal())
        {
            value is_lightlike = "always_lightlike";

            tensor<value, 4> fixed = fix_light_velocity(inf, velocity);

            for(int i=0; i < 4; i++)
            {
                velocity.idx(i) = dual_types::if_v(is_lightlike, fixed.idx(i), velocity.idx(i));
            }
        }

        inverse_metric<value, 4, 4> inv = inf.met.invert();

        tensor<value, 4, 4, 4> christoff2;

        for(int i=0; i < 4; i++)
        {
            for(int k=0; k < 4; k++)
            {
                for(int l=0; l < 4; l++)
                {
                    value sum = 0;

                    for(int m=0; m < 4; m++)
                    {
                        sum += inv.idx(i, m) * inf.partials.idx(l, m, k);
                        sum += inv.idx(i, m) * inf.partials.idx(k, m, l);
                        sum += -inv.idx(i, m) * inf.partials.idx(m, k, l);
                    }

                    christoff2.idx(i, k, l) = 0.5f * sum;
                }
            }
        }

        tensor<value, 4> accel;

        for(int uu=0; uu < 4; uu++)
        {
            value sum = 0;

            for(int aa = 0; aa < 4; aa++)
            {
                for(int bb = 0; bb < 4; bb++)
                {
                    sum += velocity[aa] * velocity[bb] * christoff2.idx(uu, aa, bb);
                }
            }

            accel.idx(uu) = -sum;
        }

        return accel;
    }


    template<typename Func, typename... T>
    inline
    std::pair<std::vector<value>, std::vector<value>> total_diff(Func&& f, T... raw_variables)
    {
        std::array<std::string, sizeof...(T)> variable_names{raw_variables...};

        auto [full_eqs, partial_differentials] = evaluate_metric2D(f, raw_variables...);

        constexpr int N = sizeof...(T);

        std::vector<value> total_differentials;

        for(int i=0; i < N; i++)
        {
            value accum = 0;

            for(int j=0; j < N; j++)
            {
                value diff_variable = "d" + variable_names[j];

                accum += partial_differentials[j * N + i] * diff_variable;
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

        std::array<dual, sizeof...(raw_variables)> variables;

        for(int i=0; i < N; i++)
        {
            variables[i].make_variable(variable_names[i]);
        }

        auto result = array_apply(std::forward<Func>(f), variables);

        return type_to_string(result.real);
    }

    template<typename Func, typename... T>
    inline
    std::vector<dual> get_functionv(Func&& f, T... raw_variables)
    {
        constexpr int N = sizeof...(T);
        std::array<std::string, N> variable_names{raw_variables...};

        std::array<dual, sizeof...(raw_variables)> variables;

        for(int i=0; i < N; i++)
        {
            variables[i].make_variable(variable_names[i]);
        }

        auto arr = array_apply(std::forward<Func>(f), variables);

        std::vector<dual> ret;

        for(auto& i : arr)
        {
            ret.push_back(i);
        }

        return ret;
    }

    enum coordinate_system
    {
        //ANGULAR,
        X_Y_THETA_PHI,
        CARTESIAN,
        CYLINDRICAL,
        OTHER
    };

    struct config;

    struct metric_config
    {
        std::string name;
        std::string description;
        bool use_prepass = false;
        float max_acceleration_change = 0.0000001f;

        bool singular = false;
        bool traversable_event_horizon = false;
        float singular_terminator = 1;

        bool adaptive_precision = true;
        bool detect_singularities = false;
        bool follow_geodesics_forward = false;

        bool has_cylindrical_singularity = false;
        float cylindrical_terminator = 0.005;

        coordinate_system system = coordinate_system::X_Y_THETA_PHI;

        std::string to_polar;
        std::string from_polar;
        std::string origin_distance;
        std::string coordinate_periodicity;

        std::string inherit_settings;

        bool unconditionally_nonsingular = false;

        void load(nlohmann::json& js)
        {
            for(auto& [key, value] : js.items())
            {
                if(key == "name")
                    name = value;

                else if(key == "description")
                    description = value;

                else if(key == "use_prepass")
                    use_prepass = value;

                else if(key == "max_acceleration_change")
                    max_acceleration_change = value;

                else if(key == "singular")
                    singular = value;

                else if(key == "traversable_event_horizon")
                    traversable_event_horizon = value;

                else if(key == "singular_terminator")
                    singular_terminator = value;

                else if(key == "adaptive_precision")
                    adaptive_precision = value;

                else if(key == "detect_singularities")
                    detect_singularities = value;

                else if(key == "follow_geodesics_forward")
                    follow_geodesics_forward = value;

                else if(key == "coordinate_system")
                {
                    std::string ssystem = value;

                    if(ssystem == "X_Y_THETA_PHI")
                        system = coordinate_system::X_Y_THETA_PHI;
                    else if(ssystem == "CARTESIAN")
                        system = coordinate_system::CARTESIAN;
                    else if(ssystem == "CYLINDRICAL")
                        system = coordinate_system::CYLINDRICAL;
                    else
                        system = coordinate_system::OTHER;
                }

                else if(key == "to_polar")
                    to_polar = value;

                else if(key == "from_polar")
                    from_polar = value;

                else if(key == "origin_distance")
                    origin_distance = value;

                else if(key == "coordinate_periodicity")
                    coordinate_periodicity = value;

                else if(key == "inherit_settings")
                    inherit_settings = value;

                else if(key == "has_cylindrical_singularity")
                    has_cylindrical_singularity = value;

                else if(key == "cylindrical_terminator")
                    cylindrical_terminator = value;

                else if(key == "unconditionally_nonsingular")
                    unconditionally_nonsingular = value;

                else
                    std::cout << "Warning, unknown key name " << key << std::endl;
            }
        }
    };

    template<typename T>
    struct metric_impl
    {
        std::vector<T> accel;

        std::vector<T> real_eq;
        std::vector<T> derivatives;

        std::vector<T> to_polar;
        std::vector<T> dt_to_spherical;

        std::vector<T> from_polar;
        std::vector<T> dt_from_spherical;

        T distance_function;

        std::vector<T> coordinate_periodicity;
    };

    inline
    std::vector<std::string> stringify_range(const std::vector<value>& in)
    {
        std::vector<std::string> ret;

        for(const value& i : in)
        {
            ret.push_back(type_to_string(i));
        }

        return ret;
    }

    inline
    metric_impl<std::string> stringify(const metric_impl<value>& raw)
    {
        metric_impl<std::string> ret;

        ret.accel = stringify_range(raw.accel);

        ret.real_eq = stringify_range(raw.real_eq);
        ret.derivatives = stringify_range(raw.derivatives);

        ret.to_polar = stringify_range(raw.to_polar);
        ret.dt_to_spherical = stringify_range(raw.dt_to_spherical);

        ret.from_polar = stringify_range(raw.from_polar);
        ret.dt_from_spherical = stringify_range(raw.dt_from_spherical);

        ret.distance_function = type_to_string(raw.distance_function);

        ret.coordinate_periodicity = stringify_range(raw.coordinate_periodicity);

        return ret;
    }

    inline
    metric_impl<std::string> build_concrete(const std::map<std::string, std::string>& mapping, const metric_impl<value>& raw)
    {
        metric_impl<value> raw_copy = raw;

        for(value& v : raw_copy.accel)
        {
            v.substitute(mapping);
        }

        for(value& v : raw_copy.real_eq)
        {
            v.substitute(mapping);
        }

        for(value& v : raw_copy.derivatives)
        {
            v.substitute(mapping);
        }

        for(value& v : raw_copy.to_polar)
        {
            v.substitute(mapping);
        }

        for(value& v : raw_copy.dt_to_spherical)
        {
            v.substitute(mapping);
        }

        for(value& v : raw_copy.from_polar)
        {
            v.substitute(mapping);
        }

        for(value& v : raw_copy.dt_from_spherical)
        {
            v.substitute(mapping);
        }

        raw_copy.distance_function.substitute(mapping);

        for(value& v : raw_copy.coordinate_periodicity)
        {
            v.substitute(mapping);
        }

        return stringify(raw_copy);
    }

    struct metric_descriptor
    {
        metric_impl<value> raw;
        metric_impl<std::string> abstract;
        metric_impl<std::string> concrete;

        bool is_spherical = false;

        bool is_polar_spherically_symmetric(const std::vector<value>& met)
        {
            int sin_count = 0;

            ///theta
            value var = "v3";
            value sin_theta = sin(var);

            auto substitute = [&](const value& in) -> std::optional<value>
            {
                if(sin_count >= 2)
                    return std::nullopt;

                if(equivalent(sin_theta, in))
                {
                    sin_count++;

                    return value{1};
                }

                return std::nullopt;
            };

            value theta;
            value phi;

            if(met.size() == 4)
            {
                theta = met[2];
                phi = met[3];
            }
            else if(met.size() == 16)
            {
                ///so, if there are dtdr components that's fine, but otherwise its borked
                ///that means the metric must be in the following form

                /**
                [dt, dtdr, 0,           0,
                     dr,   0,           0,
                           X*r*r*dtheta,
                                        X*r*r*sin(t)*sin(t)*dphi
                */

                std::vector<int> bad_offdiagonals{0 * 4 + 2, 0 * 4 + 3, 1 * 4 + 2, 1 * 4 + 3};

                for(int idx : bad_offdiagonals)
                {
                    if(!equivalent(met[idx], value{0}))
                    {
                        return false;
                    }
                }

                theta = met[2 * 4 + 2];
                phi = met[3 * 4 + 3];
            }

            phi.substitute_generic(substitute);

            if(sin_count < 2)
                return false;

            phi = phi.flatten(true);

            return equivalent(theta, phi);
        }

        template<typename Met, typename ToPolar, typename FromPolar, typename Distance, typename CoordinatePeriodicity>
        void load(Met& func, ToPolar& func1, FromPolar& func2, Distance& func3, std::optional<CoordinatePeriodicity>& func4)
        {
            metric_info inf(func);

            tensor<value, 4> accel_as_tensor = calculate_acceleration(inf);

            raw.accel = {accel_as_tensor.x(), accel_as_tensor.y(), accel_as_tensor.z(), accel_as_tensor.w()};

            std::tie(raw.real_eq, raw.derivatives) = evaluate_metric2D(func, "v1", "v2", "v3", "v4");

            std::tie(raw.to_polar, raw.dt_to_spherical) = total_diff(func1, "v1", "v2", "v3", "v4");
            std::tie(raw.from_polar, raw.dt_from_spherical) = total_diff(func2, "v1", "v2", "v3", "v4");

            raw.distance_function = get_function(func3, "v1", "v2", "v3", "v4");

            if(func4.has_value())
            {
                std::vector<dual> duals = get_functionv(func4.value(), "v1", "v2", "v3", "v4");

                std::vector<value> linear;

                for(const dual& d : duals)
                {
                    linear.push_back(d.real);
                }

                raw.coordinate_periodicity = linear;
            }

            debiggen();

            abstract = stringify(raw);

            is_spherical = is_polar_spherically_symmetric(raw.real_eq);
        }

        void debiggen()
        {
            if(raw.real_eq.size() == 4)
                return;

            if(raw.real_eq.size() != 16)
                throw std::runtime_error("Something terrible has happened in the metric");

            ///check for offdiagonal components
            for(int i=0; i < 4; i++)
            {
                for(int j=0; j < 4; j++)
                {
                    bool is_zero = type_to_string(raw.real_eq[j * 4 + i]) == "0" || type_to_string(raw.real_eq[j * 4 + i]) == "0.0";

                    if(!is_zero && abs(i) != abs(j))
                        return;
                }
            }

            printj("Offdiagonal reduction");

            std::vector<value> diagonal_equations;
            std::vector<value> diagonal_derivatives;

            for(int i=0; i < 4; i++)
            {
                diagonal_equations.push_back(raw.real_eq[i * 4 + i]);
            }

            for(int k=0; k < 4; k++)
            {
                for(int i=0; i < 4; i++)
                {
                    ///so the structure of the derivatives is that we rows are the derivating variable
                    ///so like [dtdt by dt, dtdr by dt, dtdtheta by dt, dtdphi by dt,
                    ///         dtdt by dr, etc]
                    diagonal_derivatives.push_back(raw.derivatives[k * 16 + i * 4 + i]);
                }
            }

            raw.real_eq = diagonal_equations;
            raw.derivatives = diagonal_derivatives;
        }
    };

    struct metric
    {
        metric_descriptor desc;
        metric_config metric_cfg;
        sandbox sand;
    };

    inline
    std::string build_argument_string(const metric& in, const config& cfg);

    enum integration_type
    {
        EULER,
        VERLET
    };

    struct config
    {
        float universe_size = 200000;
        integration_type type = integration_type::VERLET;
        float error_override = 0;
        bool redshift = false;
        bool use_device_side_enqueue = true;
        float max_precision_radius = 10;
    };

    inline
    std::string build_argument_string(const metric& in, const metric_impl<std::string>& impl, const config& cfg, const std::map<std::string, std::string>& substitution_map)
    {
        std::string argument_string = " -DRS_IMPL=1 -DC_IMPL=1 ";

        auto real_eq = impl.real_eq;
        auto derivatives = impl.derivatives;

        printj("REAL EQ SIZE ", real_eq.size());

        for(int i=0; i < (int)real_eq.size(); i++)
        {
            argument_string += "-DF" + std::to_string(i + 1) + "_I=" + real_eq[i] + " ";
        }

        bool is_polar_spherically_symmetric = in.metric_cfg.system == X_Y_THETA_PHI && in.desc.is_spherical;

        if(is_polar_spherically_symmetric)
        {
            print("Metric is spherically symmetric\n");
        }

        //is_polar_spherically_symmetric = false;

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
            auto to_polar = impl.to_polar;
            auto dt_to_spherical = impl.dt_to_spherical;

            auto from_polar = impl.from_polar;
            auto dt_from_spherical = impl.dt_from_spherical;

            for(int i=0; i < (int)to_polar.size(); i++)
            {
                argument_string += "-DTO_COORD" + std::to_string(i + 1) + "=" + to_polar[i] + " ";
            }

            for(int i=0; i < (int)dt_to_spherical.size(); i++)
            {
                argument_string += "-DTO_DCOORD" + std::to_string(i + 1) + "=" + dt_to_spherical[i] + " ";
            }

            for(int i=0; i < (int)from_polar.size(); i++)
            {
                argument_string += "-DFROM_COORD" + std::to_string(i + 1) + "=" + from_polar[i] + " ";
            }

            for(int i=0; i < (int)dt_from_spherical.size(); i++)
            {
                argument_string += "-DFROM_DCOORD" + std::to_string(i + 1) + "=" + dt_from_spherical[i] + " ";
            }

            auto coordinate_periodicity = impl.coordinate_periodicity;

            for(int i=0; i < (int)coordinate_periodicity.size(); i++)
            {
                argument_string += "-DCOORDINATE_PERIODICITY" + std::to_string(i + 1) + "=" + coordinate_periodicity[i] + " ";
            }

            if(coordinate_periodicity.size() > 0)
            {
                argument_string += "-DHAS_COORDINATE_PERIODICITY ";
            }
        }

        argument_string += " -DGENERIC_METRIC";

        if(cfg.type == integration_type::EULER)
        {
            argument_string += " -DEULER_INTEGRATION_GENERIC";
        }

        if(cfg.type == integration_type::VERLET)
        {
            argument_string += " -DVERLET_INTEGRATION_GENERIC";
        }

        if(is_polar_spherically_symmetric)
        {
            argument_string += " -DGENERIC_CONSTANT_THETA";
        }

        if(in.metric_cfg.singular)
        {
            argument_string += " -DSINGULAR";
            argument_string += " -DSINGULAR_TERMINATOR=" + dual_types::to_string_s(in.metric_cfg.singular_terminator);

            if(in.metric_cfg.traversable_event_horizon)
            {
                argument_string += " -DTRAVERSABLE_EVENT_HORIZON";
            }
        }

        if(in.metric_cfg.adaptive_precision)
        {
            argument_string += " -DADAPTIVE_PRECISION";

            if(cfg.error_override == 0)
                argument_string += " -DMAX_ACCELERATION_CHANGE=" + dual_types::to_string_s(in.metric_cfg.max_acceleration_change);
            else
                argument_string += " -DMAX_ACCELERATION_CHANGE=" + dual_types::to_string_s(cfg.error_override);

            if(in.metric_cfg.detect_singularities)
            {
                argument_string += " -DSINGULARITY_DETECTION";
            }
        }

        if(in.metric_cfg.system == X_Y_THETA_PHI)
        {
            if(is_polar_spherically_symmetric)
            {
                argument_string += " -DW_V1=1 -DW_V2=1 -DW_V3=8 -DW_V4=8";
            }
            else
            {
                argument_string += " -DW_V1=1 -DW_V2=1 -DW_V3=8 -DW_V4=32";
            }
        }
        else if(in.metric_cfg.system == CYLINDRICAL)
        {
            ///t, p, phi, z
            argument_string += " -DW_V1=1 -DW_V2=1 -DW_V3=8 -DW_V4=1";
        }
        else
        {
            ///covers cartesian, and 'other'
            argument_string += " -DW_V1=1 -DW_V2=1 -DW_V3=1 -DW_V4=1";
        }

        if(in.metric_cfg.follow_geodesics_forward)
        {
            argument_string += " -DFORWARD_GEODESIC_PATH";
        }

        if(in.metric_cfg.has_cylindrical_singularity)
        {
            argument_string += " -DHAS_CYLINDRICAL_SINGULARITY";
            argument_string += " -DCYLINDRICAL_TERMINATOR=" + std::to_string(in.metric_cfg.cylindrical_terminator);
        }

        if(in.metric_cfg.unconditionally_nonsingular)
        {
            argument_string += " -DUNCONDITIONALLY_NONSINGULAR";
        }

        if(cfg.redshift)
        {
            argument_string += " -DREDSHIFT";
        }

        argument_string += " -DDISTANCE_FUNC=" + impl.distance_function;

        argument_string += " -DUNIVERSE_SIZE=" + std::to_string(cfg.universe_size);

        argument_string += " -DMAX_PRECISION_RADIUS=" + std::to_string(cfg.max_precision_radius);

        const config_variables& dynamic_vars = in.sand.cfg;

        if(dynamic_vars.names.size() > 0)
        {
            std::string vars = "";

            for(int i=0; i < (int)dynamic_vars.names.size() - 1; i++)
            {
                vars += dynamic_vars.names[i] + ",";
            }

            vars += dynamic_vars.names.back();

            printj("Dynamic variables ", vars);

            argument_string += " -DDYNVARS=" + vars;
        }

        std::vector<value> cart_to_polar;
        std::vector<value> cart_to_polar_derivs;

        std::tie(cart_to_polar, cart_to_polar_derivs) = total_diff(cartesian_to_polar_v, "v1", "v2", "v3", "v4");

        assert(cart_to_polar.size());
        assert(cart_to_polar_derivs.size());

        for(int i=0; i < 4; i++)
        {
            argument_string += " -DCART_TO_POL" + std::to_string(i) + "=" + type_to_string(cart_to_polar[i]);
        }

        for(int i=0; i < 4; i++)
        {
            argument_string += " -DCART_TO_POL_D" + std::to_string(i) + "=" + type_to_string(cart_to_polar_derivs[i]);
        }

        for(int i=0; i < 4; i++)
        {
            argument_string += " -DGEO_ACCEL" + std::to_string(i) + "=" + impl.accel[i];
        }

        argument_string += " -DMETRIC_TIME_G00=" + real_eq[0];

        {
            std::string extra_string;

            in.sand.ctx.build(extra_string, 0, substitution_map);

            //std::cout << "ADDED " << extra_string << std::endl;

            argument_string += " " + extra_string;
        }

        return argument_string;
    }
}

#endif // METRIC_HPP_INCLUDED
