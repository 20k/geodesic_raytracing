#ifndef ASYNC_RENDERING_HPP_INCLUDED
#define ASYNC_RENDERING_HPP_INCLUDED

#include "camera.hpp"
#include <vec/vec.hpp>
#include <optional>
#include <mutex>

#include "triangle_manager.hpp"
#include "physics.hpp"

void execute_kernel(cl::command_queue& cqueue, cl::buffer& rays_in, cl::buffer& rays_out,
                                               cl::buffer& rays_finished,
                                               cl::buffer& count_in, cl::buffer& count_out,
                                               cl::buffer& count_finished,
                                               cl::buffer& ray_time_min, cl::buffer& ray_time_max,
                                               //cl::buffer& visual_path, cl::buffer& visual_ray_counts,
                                               triangle_rendering::manager& manage, cl::buffer& intersections, cl::buffer& intersections_count,
                                               triangle_rendering::acceleration& accel,
                                               physics& phys,
                                               int num_rays,
                                               bool use_device_side_enqueue,
                                               bool use_triangle_rendering,
                                               cl::buffer& dynamic_config,
                                               cl::buffer& dynamic_feature_config)
{
    if(use_device_side_enqueue)
    {
        int fallback = 0;

        cl::args run_args;
        run_args.push_back(rays_in);
        run_args.push_back(rays_out);
        run_args.push_back(rays_finished);
        run_args.push_back(count_in);
        run_args.push_back(count_out);
        run_args.push_back(count_finished);
        run_args.push_back(fallback);
        run_args.push_back(dynamic_config);

        cqueue.exec("relauncher_generic", run_args, {1}, {1});
    }
    else
    {
        count_in.write_async(cqueue, (const char*)&num_rays, sizeof(int));
        count_out.set_to_zero(cqueue);
        count_finished.set_to_zero(cqueue);
        intersections_count.set_to_zero(cqueue);

        if(use_triangle_rendering)
        {
            assert(accel.is_allocated);

            cl_int my_min = INT_MAX;
            cl_int my_max = INT_MIN;

            accel.ray_time_min.write(cqueue, std::vector<cl_int>{my_min});
            accel.ray_time_max.write(cqueue, std::vector<cl_int>{my_max});

            if(accel.use_cell_based_culling)
            {
                accel.cell_time_min.fill(cqueue, my_min);
                accel.cell_time_max.fill(cqueue, my_max);
            }
        }

        int mouse_x = ImGui::GetMousePos().x;
        int mouse_y = ImGui::GetMousePos().y;

        cl::args run_args;
        run_args.push_back(rays_in);
        run_args.push_back(rays_out);
        run_args.push_back(rays_finished);
        run_args.push_back(count_in);
        run_args.push_back(count_out);
        run_args.push_back(count_finished);
        //run_args.push_back(visual_path);
        //run_args.push_back(visual_ray_counts);
        run_args.push_back(manage.tris);
        run_args.push_back(manage.tri_count);
        run_args.push_back(intersections);
        run_args.push_back(intersections_count);
        run_args.push_back(accel.counts);
        run_args.push_back(accel.offsets);
        run_args.push_back(accel.memory);
        run_args.push_back(accel.memory_count);
        run_args.push_back(accel.start_times_memory);
        run_args.push_back(accel.delta_times_memory);
        run_args.push_back(accel.linear_object_positions);
        run_args.push_back(accel.unculled_counts);
        run_args.push_back(accel.any_visible);
        run_args.push_back(accel.offset_width);
        run_args.push_back(accel.time_width);
        run_args.push_back(accel.offset_size.x());
        //run_args.push_back(accel.cell_time_min);
        //run_args.push_back(accel.cell_time_max);
        run_args.push_back(manage.objects);
        run_args.push_back(ray_time_min);
        run_args.push_back(ray_time_max);

        run_args.push_back(phys.subsampled_paths);
        run_args.push_back(phys.subsampled_velocities);

        for(int i=0; i < 4; i++)
        {
            run_args.push_back(phys.subsampled_inverted_tetrads[i]);
        }

        run_args.push_back(dynamic_config);
        run_args.push_back(dynamic_feature_config);
        run_args.push_back(mouse_x);
        run_args.push_back(mouse_y);

        cqueue.exec("do_generic_rays", run_args, {num_rays}, {256});

        ///todo: no idea if this is correct
        accel.ray_time_min = ray_time_min;
        accel.ray_time_max = ray_time_max;
    }
}


template<typename T>
struct mt_queue
{
    std::mutex mut;
    std::vector<T> dat;

    void push(T&& in)
    {
        std::scoped_lock lck(mut);
        dat.push_back(std::move(in));
    }

    void push(const T& in)
    {
        std::scoped_lock lck(mut);
        dat.push_back(in);
    }

    int peek_size()
    {
        std::scoped_lock lck(mut);
        return dat.size();
    }

    std::optional<T> pop()
    {
        std::scoped_lock lck(mut);

        if(dat.size() == 0)
            return std::nullopt;

        T val = std::move(dat.front());
        dat.erase(dat.begin());
        return val;
    }
};

struct event_data
{
    camera cam;
};

struct resize_data
{
    vec2i size;
    bool is_screenshot = false;
};

struct recompile_data
{
    //bool should_soft_recompile = false;
    //bool should_hard_recompile = false;
};

struct settings_data
{
    float anisotropy = 0;
};

struct shared_data
{
    mt_queue<event_data> event_q;
    mt_queue<resize_data> resize_q;
    mt_queue<std::vector<float>> dynamic_config_q;
    mt_queue<int> selected_metric_q;
    mt_queue<recompile_data> recompile_data_q;
    mt_queue<cl::buffer> dfg_q;
    mt_queue<settings_data> settings_q;

    mt_queue<cl::image> finished_textures;

    std::atomic_bool is_open{true};

    std::mutex data_lock;
    float universe_size = 0;
};

void render_thread(cl::context& ctx, shared_data& shared, vec2i start_size, metric_manager& manage, background_images& back_images)
{
    cl::command_queue mqueue(ctx, 1<<9);

    std::vector<render_state> states;

    for(int i=0; i < 3; i++)
    {
        states.emplace_back(ctx, mqueue);
        states[i].realloc(start_size.x(), start_size.y());
    }

    camera cam;

    vec2i window_size = start_size;

    cl::buffer dynamic_config(ctx);
    dynamic_config.alloc(sizeof(cl_float));

    int which_state = 0;

    cl::buffer dynamic_feature_buffer(ctx);
    dynamic_feature_buffer.alloc(sizeof(cl_float) * 2);
    dynamic_feature_buffer.set_to_zero(mqueue);

    bool camera_on_geodesic = false;

    float linear_basis_speed = 0;
    bool flip_sign = false;

    physics phys(ctx);
    triangle_rendering::acceleration accel(ctx);
    triangle_rendering::manager tris(ctx);

    float anisotropy = 16;

    while(shared.is_open)
    {
        render_state& st = states[which_state];
        which_state = (which_state + 1) % states.size();

        while(shared.finished_textures.peek_size() >= 3)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        while(auto opt = shared.resize_q.pop())
        {
            resize_data& next_size = opt.value();

            for(auto& i : states)
            {
                i.realloc(next_size.size.x(), next_size.size.y());
                window_size = next_size.size;
            }
        }

        while(auto opt = shared.event_q.pop())
        {
            /*float universe_size = 0;

            {
                std::scoped_lock lck(shared.data_lock);
                universe_size = shared.universe_size;
            }

            cam.handle_input(opt.value().mouse, opt.value().keyboard, universe_size);*/

            cam = opt.value().cam;
        }

        while(auto opt = shared.dfg_q.pop())
        {
            dynamic_feature_buffer = opt.value();
        }

        while(auto opt = shared.dynamic_config_q.pop())
        {
            std::vector<float> vals = opt.value();

            dynamic_config.alloc(vals.size() * sizeof(cl_float));
            dynamic_config.write(mqueue, vals);
        }

        while(auto opt = shared.settings_q.pop())
        {
            anisotropy = opt.value().anisotropy;
        }

        manage.check_substitution(ctx);

        cl_image_format fmt;
        fmt.image_channel_order = CL_RGBA;
        fmt.image_channel_data_type = CL_UNSIGNED_INT8;

        cl::image img(ctx);
        img.alloc({window_size.x(), window_size.y()}, fmt);

        {
            /*if(dfg.is_static_dirty)
            {
                should_recompile = true;
                dfg.is_static_dirty = false;
            }

            if(metric_manage.check_recompile(should_recompile, should_soft_recompile, parent_directories,
                                          all_content, metric_names, dynamic_config, mqueue, dfg,
                                          sett, clctx.ctx, selected_metric))
            {
                //phys.needs_trace = true;

                alloc_from_dfg(dfg, mqueue, dynamic_feature_buffer);
            }*/

            //dfg.alloc_and_write_gpu_buffer(mqueue, dynamic_feature_buffer);
        }


        cl::buffer g_camera_pos_cart(ctx);
        g_camera_pos_cart.alloc(sizeof(cl_float4));
        g_camera_pos_cart.write(mqueue, std::span{&cam.pos.v[0], 4});

        cl::buffer g_camera_quat(ctx);
        g_camera_quat.alloc(sizeof(cl_float4));
        g_camera_quat.write(mqueue, std::span{&cam.rot.q.v[0], 4});

        cl::buffer g_geodesic_basis_speed(ctx);
        g_geodesic_basis_speed.alloc(sizeof(cl_float4));

        //if(should_set_observer_velocity)
        {
            vec3f base = {0, 0, 1};

            vec3f rotated = rot_quat(base, cam.rot).norm() * linear_basis_speed;

            vec4f geodesic_basis_speed = (vec4f){rotated.x(), rotated.y(), rotated.z(), 0.f};

            g_geodesic_basis_speed.write(mqueue, std::span{&geodesic_basis_speed.v[0], 4});
        }

        if(!camera_on_geodesic)
        {
            {
                cl_float clflip = flip_sign;

                cl::args args;
                args.push_back(g_camera_pos_cart,
                               st.g_camera_pos_generic,
                               (cl_int)1,
                               clflip,
                               dynamic_config);

                mqueue.exec("cart_to_generic_kernel", args, {1}, {1});
            }

            {
                cl_float4 cartesian_basis_speed = {0,0,0,0};

                int count = 1;

                cl::args tetrad_args;
                tetrad_args.push_back(st.g_camera_pos_generic);
                tetrad_args.push_back(count);
                tetrad_args.push_back(cartesian_basis_speed);

                for(int i=0; i < 4; i++)
                {
                    tetrad_args.push_back(st.tetrad[i]);
                }

                tetrad_args.push_back(dynamic_config);

                mqueue.exec("init_basis_vectors", tetrad_args, {1}, {1});
            }
        }

        int width = img.size<2>().x();
        int height = img.size<2>().y();

        cl::args clr;
        clr.push_back(img);

        mqueue.exec("clear", clr, {width, height}, {16, 16});

        {
            int isnap = 0;

            cl_int prepass_width = width/16;
            cl_int prepass_height = height/16;

            #ifdef UNIMPLEMENTED
            if(metric_manage.current_metric->metric_cfg.use_prepass && tris.cpu_objects.size() > 0)
            {
                st.termination_buffer.set_to_zero(mqueue);
            }

            if(metric_manage.current_metric->metric_cfg.use_prepass && tris.cpu_objects.size() == 0)
            {
                cl::args clear_args;
                clear_args.push_back(st.termination_buffer);
                clear_args.push_back(prepass_width);
                clear_args.push_back(prepass_height);

                mqueue.exec("clear_termination_buffer", clear_args, {prepass_width*prepass_height}, {256});

                cl::args init_args_prepass;

                init_args_prepass.push_back(st.g_camera_pos_generic);
                init_args_prepass.push_back(g_camera_quat);
                init_args_prepass.push_back(st.rays_prepass);
                init_args_prepass.push_back(st.rays_count_prepass);
                init_args_prepass.push_back(prepass_width);
                init_args_prepass.push_back(prepass_height);
                init_args_prepass.push_back(st.termination_buffer);
                init_args_prepass.push_back(prepass_width);
                init_args_prepass.push_back(prepass_height);
                init_args_prepass.push_back(isnap);

                for(auto& i : st.tetrad)
                {
                    init_args_prepass.push_back(i);
                }

                init_args_prepass.push_back(dynamic_config);

                mqueue.exec("init_rays_generic", init_args_prepass, {prepass_width*prepass_height}, {256});

                int rays_num = calculate_ray_count(prepass_width, prepass_height);

                execute_kernel(mqueue, st.rays_prepass, st.rays_out, st.rays_finished, st.rays_count_prepass, st.rays_count_out, st.rays_count_finished, st.accel_ray_time_min, st.accel_ray_time_max, tris, st.tri_intersections, st.tri_intersections_count, accel, phys, rays_num, false, dfg, dynamic_config, dynamic_feature_buffer);

                cl::args singular_args;
                singular_args.push_back(st.rays_finished);
                singular_args.push_back(st.rays_count_finished);
                singular_args.push_back(st.termination_buffer);
                singular_args.push_back(prepass_width);
                singular_args.push_back(prepass_height);

                mqueue.exec("calculate_singularities", singular_args, {prepass_width*prepass_height}, {256});
            }
            #endif

            cl::args init_args;
            init_args.push_back(st.g_camera_pos_generic);
            init_args.push_back(g_camera_quat);
            init_args.push_back(st.rays_in);
            init_args.push_back(st.rays_count_in);
            init_args.push_back(width);
            init_args.push_back(height);
            init_args.push_back(st.termination_buffer);
            init_args.push_back(prepass_width);
            init_args.push_back(prepass_height);
            init_args.push_back(isnap);

            for(auto& i : st.tetrad)
            {
                init_args.push_back(i);
            }

            init_args.push_back(dynamic_config);

            mqueue.exec("init_rays_generic", init_args, {width*height}, {16*16});

            int rays_num = width * height;

            execute_kernel(mqueue, st.rays_in, st.rays_out, st.rays_finished, st.rays_count_in, st.rays_count_out, st.rays_count_finished, st.accel_ray_time_min, st.accel_ray_time_max, tris, st.tri_intersections, st.tri_intersections_count, accel, phys, rays_num, false, false, dynamic_config, dynamic_feature_buffer);

            cl::args texture_args;
            texture_args.push_back(st.rays_finished);
            texture_args.push_back(st.rays_count_finished);
            texture_args.push_back(st.texture_coordinates);
            texture_args.push_back(width);
            texture_args.push_back(height);
            texture_args.push_back(dynamic_config);
            texture_args.push_back(dynamic_feature_buffer);

            mqueue.exec("calculate_texture_coordinates", texture_args, {width * height}, {256});

            cl::args render_args;
            render_args.push_back(st.rays_finished);
            render_args.push_back(st.rays_count_finished);
            render_args.push_back(st.rtex);
            render_args.push_back(back_images.i1);
            render_args.push_back(back_images.i2);
            render_args.push_back(width);
            render_args.push_back(height);
            render_args.push_back(st.texture_coordinates);
            render_args.push_back(anisotropy);
            render_args.push_back(dynamic_config);
            render_args.push_back(dynamic_feature_buffer);

            mqueue.exec("render", render_args, {width * height}, {256});
        }

        shared.finished_textures.push(std::move(img));
    }
}


#endif // ASYNC_RENDERING_HPP_INCLUDED
