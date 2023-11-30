#ifndef METRIC_MANAGER_HPP_INCLUDED
#define METRIC_MANAGER_HPP_INCLUDED

#include "dynamic_feature_config.hpp"
#include <mutex>

struct metric_manager
{
    int current_idx = -1;
    int selected_idx = -1;
    metrics::metric* current_metric = nullptr;

    bool using_swapped = false;
    std::optional<cl::program> substituted_program_opt;
    std::optional<cl::program> pending_dynamic_program_opt;
    bool has_built = false;

    ///this is a bit of a giant mess
    bool check_recompile(bool should_recompile, bool should_soft_recompile,
                         const std::vector<content*>& parent_directories, content_manager& all_content, std::vector<std::string>& metric_names,
                         cl::buffer& dynamic_config, cl::command_queue& cqueue, dynamic_feature_config& dfg, render_settings& sett, cl::context& context, cl::buffer& termination_buffer)
    {
        if(!(should_recompile || current_idx == -1 || should_soft_recompile))
            return false;

        bool should_hard_recompile = should_recompile || current_idx == -1;

        if(selected_idx == -1)
            selected_idx = 0;

        bool should_block = false;

        if(selected_idx != current_idx)
        {
            should_block = true;

            metrics::metric* next = parent_directories[selected_idx]->lazy_fetch(all_content, metric_names[selected_idx]);

            if(next == nullptr)
            {
                printj("Broken metric ", metric_names[selected_idx]);
            }
            else
            {
                current_metric = next;
            }

            assert(current_metric);

            dfg.set_feature("max_acceleration_change", current_metric->metric_cfg.max_acceleration_change);

            printj("ALLOCATING DYNCONFIG ", current_metric->sand.cfg.default_values.size());

            int dyn_config_bytes = current_metric->sand.cfg.default_values.size() * sizeof(cl_float);

            if(dyn_config_bytes < 4)
                dyn_config_bytes = 4;

            dynamic_config.alloc(dyn_config_bytes);

            std::vector<float> vars = current_metric->sand.cfg.default_values;

            if(vars.size() == 0)
                vars.resize(1);

            dynamic_config.write(cqueue, vars);
        }

        current_idx = selected_idx;
        std::string argument_string_prefix = "-cl-std=CL1.2 -cl-unsafe-math-optimizations ";

        ///use device side enqueue
        if(false)
        {
            argument_string_prefix += "-DDEVICE_SIDE_ENQUEUE ";
        }

        if(sett.is_srgb)
        {
            argument_string_prefix += "-DLINEAR_FRAMEBUFFER ";
        }

        if(should_hard_recompile)
        {
            using_swapped = false;

            printj("Building");
            std::string dynamic_argument_string = argument_string_prefix + build_argument_string(*current_metric, current_metric->desc.abstract, false, dfg, {});

            file::write("./argument_string.txt", dynamic_argument_string, file::mode::TEXT);

            if(substituted_program_opt.has_value())
            {
                substituted_program_opt->cancel();
                substituted_program_opt = std::nullopt;
            }

            if(pending_dynamic_program_opt.has_value())
            {
                pending_dynamic_program_opt->cancel();
                pending_dynamic_program_opt = std::nullopt;
            }

            pending_dynamic_program_opt = std::nullopt;

            cl::program pending = cl::build_program_with_cache(context, {"cl.cl"}, true, dynamic_argument_string);

            pending_dynamic_program_opt = pending;

            if(should_block)
            {
                bool has_program = false;

                {
                    std::scoped_lock guard(context.shared->mut);
                    has_program = context.shared->kernels.size() > 0;
                }

                if(has_program)
                    context.deregister_program(0);

                context.register_program(*pending_dynamic_program_opt);
                has_built = true;
            }
            else
            {
                has_built = false;
            }
        }

        if(should_soft_recompile || should_hard_recompile)
        {
            if(using_swapped)
            {
                assert(pending_dynamic_program_opt.has_value());

                bool has_program = false;

                {
                    std::scoped_lock guard(context.shared->mut);
                    has_program = context.shared->kernels.size() > 0;
                }

                if(has_program)
                    context.deregister_program(0);

                ///Reregister the dynamic program again, static is invalid
                context.register_program(*pending_dynamic_program_opt);
            }

            using_swapped = false;

            auto substitution_map = current_metric->sand.cfg.as_substitution_map();

            metrics::metric_impl<std::string> substituted_impl = metrics::build_concrete(substitution_map, current_metric->desc.raw);

            std::string substituted_argument_string = argument_string_prefix + build_argument_string(*current_metric, substituted_impl, true, dfg, substitution_map);

            if(substituted_program_opt.has_value())
            {
                substituted_program_opt->cancel();
                substituted_program_opt = std::nullopt;
            }

            substituted_program_opt.emplace(context, "cl.cl");
            substituted_program_opt->build(context, substituted_argument_string);
        }

        ///Is this necessary?
        termination_buffer.set_to_zero(cqueue);

        return true;
    }

    void check_substitution(cl::context& ctx)
    {
        if(pending_dynamic_program_opt.has_value())
        {
            cl::program& pending = pending_dynamic_program_opt.value();

            if(pending.is_built() && !using_swapped && !has_built)
            {
                bool has_program = false;

                {
                    std::scoped_lock guard(ctx.shared->mut);
                    has_program = ctx.shared->kernels.size() > 0;
                }

                if(has_program)
                    ctx.deregister_program(0);

                ctx.register_program(pending);
                has_built = true;
            }
        }

        if(substituted_program_opt.has_value())
        {
            cl::program& pending = substituted_program_opt.value();

            if(pending.is_built())
            {
                printj("Swapped\n");

                bool has_program = false;

                {
                    std::scoped_lock guard(ctx.shared->mut);
                    has_program = ctx.shared->kernels.size() > 0;
                }

                if(has_program)
                    ctx.deregister_program(0);

                ctx.register_program(pending);

                substituted_program_opt = std::nullopt;
                using_swapped = true;
            }
        }
    }
};

#endif // METRIC_MANAGER_HPP_INCLUDED
