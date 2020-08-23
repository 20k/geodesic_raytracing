#include <iostream>
#include <toolkit/render_window.hpp>
#include <toolkit/texture.hpp>

int main()
{
    render_settings sett;
    sett.width = 800;
    sett.height = 600;
    sett.opencl = true;
    sett.no_double_buffer = true;

    render_window win(sett, "Geodesics");

    assert(win.clctx);

    opencl_context& clctx = *win.clctx;

    cl::program prog(clctx.ctx, "cl.cl");
    prog.build(clctx.ctx, "");

    clctx.ctx.register_program(prog);

    texture_settings tsett;
    tsett.width = sett.width;
    tsett.height = sett.height;
    tsett.is_srgb = false;

    texture tex;
    tex.load_from_memory(tsett, nullptr);

    cl::gl_rendertexture rtex(clctx.ctx);
    rtex.create_from_texture(tex.handle);

    while(!win.should_close())
    {
        win.poll();

        rtex.acquire(clctx.cqueue);

        cl::args args;
        args.push_back(rtex);

        clctx.cqueue.exec("do_raytracing", args, {win.get_window_size().x(), win.get_window_size().y()}, {16, 16});

        rtex.unacquire(clctx.cqueue);

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

            lst->AddImage((void*)rtex.texture_id, tl, br);
        }

        win.display();

    }

    return 0;
}
