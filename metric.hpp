#ifndef METRIC_HPP_INCLUDED
#define METRIC_HPP_INCLUDED

#include "dual.hpp"
#include "dual_value.hpp"

namespace metrics
{
    template<typename Func, typename... T>
    inline
    std::pair<std::vector<std::string>, std::vector<std::string>> evaluate_metric2D(Func&& f, T... raw_variables)
    {
        std::array<std::string, sizeof...(T)> variable_names{raw_variables...};

        std::vector<std::string> raw_eq;
        std::vector<std::string> raw_derivatives;

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

    enum coordinate_system
    {
        //ANGULAR,
        X_Y_THETA_PHI,
        CARTESIAN,
        CYLINDRICAL,
        OTHER
    };

    struct config;

    struct metric_descriptor
    {
        std::vector<std::string> real_eq;
        std::vector<std::string> derivatives;

        std::vector<std::string> to_polar;
        std::vector<std::string> dt_to_spherical;

        std::vector<std::string> from_polar;
        std::vector<std::string> dt_from_spherical;

        std::string distance_function;

        template<auto T, auto U, auto V, auto distance_func>
        void load()
        {
            std::tie(real_eq, derivatives) = evaluate_metric2D(T, "v1", "v2", "v3", "v4");

            std::tie(to_polar, dt_to_spherical) = total_diff(U, "v1", "v2", "v3", "v4");
            std::tie(from_polar, dt_from_spherical) = total_diff(V, "v1", "v2", "v3", "v4");

            distance_function = get_function(distance_func, "v1", "v2", "v3", "v4");
        }

        template<typename T, typename U, typename V, auto distance_func>
        void load(T& func, U& func1, V& func2)
        {
            std::tie(real_eq, derivatives) = evaluate_metric2D(func, "v1", "v2", "v3", "v4");

            std::tie(to_polar, dt_to_spherical) = total_diff(func1, "v1", "v2", "v3", "v4");
            std::tie(from_polar, dt_from_spherical) = total_diff(func2, "v1", "v2", "v3", "v4");

            distance_function = get_function(distance_func, "v1", "v2", "v3", "v4");
        }
    };

    struct metric_config
    {
        std::string name;
        bool use_prepass = false;
        float max_acceleration_change = 0.0000001f;

        bool singular = false;
        bool traversable_event_horizon = false;
        float singular_terminator = 1;

        bool adaptive_precision = true;
        bool detect_singularities = false;
        bool follow_geodesics_forward = false;

        coordinate_system system = coordinate_system::X_Y_THETA_PHI;

        void load(nlohmann::json& js)
        {
            if(js.count("name"))
                name = js["name"];

            if(js.count("use_prepass"))
                use_prepass = js["use_prepass"];

            if(js.count("max_acceleration_change"))
                max_acceleration_change = js["max_acceleration_change"];

            if(js.count("singular"))
                singular = js["singular"];

            if(js.count("traversable_event_horizon"))
                traversable_event_horizon = js["traversable_event_horizon"];

            if(js.count("singular_terminator"))
                singular_terminator = js["singular_terminator"];

            if(js.count("adaptive_precision"))
                adaptive_precision = js["adaptive_precision"];

            if(js.count("detect_singularities"))
                detect_singularities = js["detect_singularities"];

            if(js.count("follow_geodesics_forward"))
                follow_geodesics_forward = js["follow_geodesics_forward"];

            if(js.count("coordinate_system"))
            {
                std::string ssystem = js["coordinate_system"];

                if(ssystem == "X_Y_THETA_PHI")
                    system = coordinate_system::X_Y_THETA_PHI;
                else if(ssystem == "CARTESIAN")
                    system = coordinate_system::CARTESIAN;
                else if(ssystem == "CYLINDRICAL")
                    system = coordinate_system::CYLINDRICAL;
                else
                    system = coordinate_system::OTHER;
            }
        }
    };

    struct metric_base
    {
        metric_descriptor desc;
        metric_config metric_cfg;

        virtual std::string build(const config& cfg){return std::string();}
    };

    struct metric;

    inline
    std::string build_argument_string(const metric& in, const config& cfg);

    struct metric : metric_base
    {
        virtual std::string build(const config& cfg) override
        {
            return build_argument_string(*this, cfg);
        }
    };

    enum integration_type
    {
        EULER,
        VERLET
    };

    struct config
    {
        float universe_size = 200000;
        integration_type type = integration_type::VERLET;
        std::optional<float> error_override = std::nullopt;
        bool redshift = false;
        bool use_device_side_enqueue = true;
    };

    inline
    std::string build_argument_string(const metric& in, const config& cfg)
    {
        std::string argument_string = " -DRS_IMPL=1 -DC_IMPL=1 ";

        auto real_eq = in.desc.real_eq;
        auto derivatives = in.desc.derivatives;

        for(int i=0; i < (int)real_eq.size(); i++)
        {
            argument_string += "-DF" + std::to_string(i + 1) + "_I=" + real_eq[i] + " ";
        }

        ///only polar
        bool is_polar_spherically_symmetric = false;

        if(real_eq.size() == 4)
            is_polar_spherically_symmetric = in.metric_cfg.system == X_Y_THETA_PHI;

        if(real_eq.size() == 16)
        {
            bool no_offdiagonal_phi_components = true;

            for(int i=0; i < 4; i++)
            {
                ///phi
                int j = 3;

                if(j == i)
                    continue;

                if(real_eq[j * 4 + i] != "0")
                    no_offdiagonal_phi_components = false;
            }

            is_polar_spherically_symmetric = no_offdiagonal_phi_components && in.metric_cfg.system == X_Y_THETA_PHI;
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
            auto to_polar = in.desc.to_polar;
            auto dt_to_spherical = in.desc.dt_to_spherical;

            auto from_polar = in.desc.from_polar;
            auto dt_from_spherical = in.desc.dt_from_spherical;

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

            if(!cfg.error_override)
                argument_string += " -DMAX_ACCELERATION_CHANGE=" + dual_types::to_string_s(in.metric_cfg.max_acceleration_change);
            else
                argument_string += " -DMAX_ACCELERATION_CHANGE=" + dual_types::to_string_s(cfg.error_override.value());

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

        if(cfg.redshift)
        {
            argument_string += " -DREDSHIFT";
        }

        argument_string += " -DDISTANCE_FUNC=" + in.desc.distance_function;

        argument_string += " -DUNIVERSE_SIZE=" + std::to_string(cfg.universe_size);

        return argument_string;
    }
}

#endif // METRIC_HPP_INCLUDED
