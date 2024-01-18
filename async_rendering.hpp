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

struct settings_data
{
    float anisotropy = 0;
};

struct dynamic_config_data
{
    std::vector<float> vars;
    int for_whomst = -1;
};

struct shared_data
{
    mt_queue<event_data> event_q;
    mt_queue<resize_data> resize_q;
    mt_queue<dynamic_config_data> dynamic_config_q;
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

    for(int i=0; i < 6; i++)
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

    int live_metric = -1;

    std::vector<cl::image> pending_image_queue;
    std::vector<cl::event> pending_event_queue;

    steady_timer frame_time;

    while(shared.is_open)
    {
        //steady_timer t;

        render_state& st = states[which_state];
        which_state = (which_state + 1) % states.size();

        while(shared.finished_textures.peek_size() >= 8)
        {
            printf("Clogged\n");

            //std::this_thread::sleep_for(std::chrono::milliseconds(1));
            sf::sleep(sf::milliseconds(1));
            continue;
        }

        while(pending_event_queue.size() > 0 && pending_event_queue.front().is_finished())
        {
            std::cout << "Ftime " << frame_time.restart() * 1000. << std::endl;

            shared.finished_textures.push(pending_image_queue.front());

            pending_event_queue.erase(pending_event_queue.begin());
            pending_image_queue.erase(pending_image_queue.begin());
        }

        if(pending_event_queue.size() >= 6)
        {
            printf("Clogged 2\n");

            sf::sleep(sf::milliseconds(1));
            //std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        while(auto opt = shared.resize_q.pop())
        {
            resize_data& next_size = opt.value();

            for(auto& i : states)
            {
                i.realloc(next_size.size.x(), next_size.size.y());
                window_size = next_size.size;

                printf("Realloc?\n");
            }
        }

        while(auto opt = shared.event_q.pop())
        {
            cam = opt.value().cam;
        }

        while(auto opt = shared.dfg_q.pop())
        {
            dynamic_feature_buffer = opt.value();
        }

        while(auto opt = shared.dynamic_config_q.pop())
        {
            dynamic_config_data dcd = opt.value();

            if(dcd.for_whomst != live_metric)
                continue;

            dynamic_config.alloc(dcd.vars.size() * sizeof(cl_float));
            dynamic_config.write(mqueue, dcd.vars);
        }

        while(auto opt = shared.settings_q.pop())
        {
            anisotropy = opt.value().anisotropy;
        }

        live_metric = manage.check_substitution(ctx);

        if(live_metric == -1)
        {
            sf::sleep(sf::milliseconds(1));
            //std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        cl_image_format fmt;
        fmt.image_channel_order = CL_RGBA;
        fmt.image_channel_data_type = CL_FLOAT;

        auto& img = pending_image_queue.emplace_back(ctx);
        auto& next_image_event = pending_event_queue.emplace_back();

        img.alloc({window_size.x(), window_size.y()}, fmt);

        cl::buffer& g_camera_pos_cart = st.g_camera_pos_cart;
        g_camera_pos_cart.write(mqueue, std::span{&cam.pos.v[0], 4});

        cl::buffer& g_camera_quat = st.g_camera_quat;
        g_camera_quat.write(mqueue, std::span{&cam.rot.q.v[0], 4});

        cl::buffer& g_geodesic_basis_speed = st.g_geodesic_basis_speed;

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
            render_args.push_back(img);
            render_args.push_back(back_images.i1);
            render_args.push_back(back_images.i2);
            render_args.push_back(width);
            render_args.push_back(height);
            render_args.push_back(st.texture_coordinates);
            render_args.push_back(anisotropy);
            render_args.push_back(dynamic_config);
            render_args.push_back(dynamic_feature_buffer);

            next_image_event = mqueue.exec("render", render_args, {width * height}, {256});
        }

        std::cout << "GPUT " << gpu_submit_time.get_elapsed_time_s() * 1000. << std::endl;

        //mqueue.flush();

        //std::cout << "Time " << t.get_elapsed_time_s() * 1000. << std::endl;
    }
}


#endif // ASYNC_RENDERING_HPP_INCLUDED
