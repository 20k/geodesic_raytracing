#ifndef METRIC_HPP_INCLUDED
#define METRIC_HPP_INCLUDED

#include "dual.hpp"
#include "dual_value.hpp"

namespace metric
{
    enum coordinate_system
    {
        //ANGULAR,
        X_Y_THETA_PHI,
        CARTESIAN,
        CYLINDRICAL,
        OTHER
    };

    struct config;

    struct metric_base
    {
        std::string name;
        bool use_prepass = false;

        virtual std::string build(const config& cfg);
    };

    template<auto T, auto U, auto V, auto distance_function>
    struct metric;

    template<auto T, auto U, auto V, auto distance_function>
    inline
    std::string build_argument_string(const metric<T, U, V, distance_function>& in, const config& cfg);

    template<auto T, auto U, auto V, auto distance_function>
    struct metric : metric_base
    {
        bool singular = false;
        bool traversable_event_horizon = false;
        float singular_terminator = 1;

        bool adaptive_precision = true;
        bool detect_singularities = false;
        float max_acceleration_change = 0.0000001f;
        bool follow_geodesics_forward = false;

        coordinate_system system = coordinate_system::X_Y_THETA_PHI;

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
    };

    template<auto T, auto U, auto V, auto distance_function>
    inline
    std::string build_argument_string(const metric<T, U, V, distance_function>& in, const config& cfg)
    {
        std::string argument_string = " -DRS_IMPL=1 -DC_IMPL=1 ";

        auto [real_eq, derivatives] = evaluate_metric2D(T, "v1", "v2", "v3", "v4");

        for(int i=0; i < (int)real_eq.size(); i++)
        {
            argument_string += "-DF" + std::to_string(i + 1) + "_I=" + real_eq[i] + " ";
        }

        ///only polar
        bool is_polar_spherically_symmetric = false;

        if(real_eq.size() == 4)
            is_polar_spherically_symmetric = in.system == X_Y_THETA_PHI;

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

            is_polar_spherically_symmetric = no_offdiagonal_phi_components && in.system == X_Y_THETA_PHI;
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
            auto [to_polar, dt_to_spherical] = total_diff(U, "v1", "v2", "v3", "v4");
            auto [from_polar, dt_from_spherical] = total_diff(V, "v1", "v2", "v3", "v4");

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

        if(in.singular)
        {
            argument_string += " -DSINGULAR";
            argument_string += " -DSINGULAR_TERMINATOR=" + dual_types::to_string_s(in.singular_terminator);

            if(in.traversable_event_horizon)
            {
                argument_string += " -DTRAVERSABLE_EVENT_HORIZON";
            }
        }

        if(in.adaptive_precision)
        {
            argument_string += " -DADAPTIVE_PRECISION";

            if(!cfg.error_override)
                argument_string += " -DMAX_ACCELERATION_CHANGE=" + dual_types::to_string_s(in.max_acceleration_change);
            else
                argument_string += " -DMAX_ACCELERATION_CHANGE=" + dual_types::to_string_s(cfg.error_override.value());

            if(in.detect_singularities)
            {
                argument_string += " -DSINGULARITY_DETECTION";
            }
        }

        if(in.system == X_Y_THETA_PHI)
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
        else if(in.system == CYLINDRICAL)
        {
            ///t, p, phi, z
            argument_string += " -DW_V1=1 -DW_V2=1 -DW_V3=8 -DW_V4=1";
        }
        else
        {
            ///covers cartesian, and 'other'
            argument_string += " -DW_V1=1 -DW_V2=1 -DW_V3=1 -DW_V4=1";
        }

        if(in.follow_geodesics_forward)
        {
            argument_string += " -DFORWARD_GEODESIC_PATH";
        }

        if(cfg.redshift)
        {
            argument_string += " -DREDSHIFT";
        }

        argument_string += " -DDISTANCE_FUNC=" + get_function(distance_function, "v1", "v2", "v3", "v4");

        argument_string += " -DUNIVERSE_SIZE=" + std::to_string(cfg.universe_size);

        return argument_string;
    }
}

#endif // METRIC_HPP_INCLUDED
