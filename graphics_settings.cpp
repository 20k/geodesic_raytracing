#include "graphics_settings.hpp"
#include <imgui/imgui.h>
#include <networking/serialisable.hpp>
#include <SFML/Graphics.hpp>
#include <toolkit/texture.hpp>
#include <toolkit/clock.hpp>
#include <iostream>
#include <toolkit/fs_helpers.hpp>

DEFINE_SERIALISE_FUNCTION(graphics_settings)
{
    DO_FSERIALISE(pos_x);
    DO_FSERIALISE(pos_y);
    DO_FSERIALISE(width);
    DO_FSERIALISE(height);
    DO_FSERIALISE(fullscreen);
    DO_FSERIALISE(supersample);
    DO_FSERIALISE(supersample_factor);
    DO_FSERIALISE(vsync_enabled);
    DO_FSERIALISE(screenshot_width);
    DO_FSERIALISE(screenshot_height);
    DO_FSERIALISE(time_adjusted_controls);
    DO_FSERIALISE(mouse_sensitivity);
    DO_FSERIALISE(keyboard_sensitivity);
    DO_FSERIALISE(use_steam_screenshots);
    DO_FSERIALISE(anisotropy);
    DO_FSERIALISE(no_gpu_reads);
    DO_FSERIALISE(max_frames_ahead);
}

bool graphics_settings::display_video_settings()
{
    ImGui::InputInt("Position (x)", &pos_x);
    ImGui::InputInt("Position (y)", &pos_y);

    ImGui::InputInt("Width", &width);
    ImGui::InputInt("Height", &height);

    ImGui::Checkbox("Fullscreen", &fullscreen);

    ImGui::Checkbox("Vsync", &vsync_enabled);

    ImGui::Checkbox("Supersample", &supersample);
    ImGui::InputInt("Supersample Factor", &supersample_factor);

    ImGui::InputInt("Screenshot Width", &screenshot_width);
    ImGui::InputInt("Screenshot Height", &screenshot_height);

    ImGui::Checkbox("Save screenshots to steam", &use_steam_screenshots);

    ImGui::DragInt("Anisotropic Filtering", &anisotropy, 1, 1, 256);

    ImGui::Checkbox("No GPU reads", &no_gpu_reads);

    if(ImGui::IsItemHovered())
    {
        ImGui::SetTooltip("May improve performance");
    }

    ImGui::InputInt("Max Frames Ahead", &max_frames_ahead, 1);

    max_frames_ahead = clamp(max_frames_ahead, 0, 6);

    if(ImGui::IsItemHovered())
    {
        ImGui::SetTooltip("Improves performance at the expense of input latency");
    }

    ImGui::NewLine();

    return ImGui::Button("Apply");
}

bool graphics_settings::display_control_settings()
{
    ImGui::Checkbox("Time adjusted controls", &time_adjusted_controls);

    if(ImGui::IsItemHovered())
    {
        ImGui::SetTooltip("Setting this to true means that camera moves at a constant amount per second\nSetting this to false means that the camera moves at a constant speed per frame");
    }

    ImGui::SliderFloat("Mouse Sensitivity", &mouse_sensitivity, 0.f, 5.f);
    ImGui::SliderFloat("Keyboard Sensitivity", &keyboard_sensitivity, 0.f, 5.f);

    ImGui::NewLine();

    return ImGui::Button("Apply");
}

DEFINE_SERIALISE_FUNCTION(background_settings)
{
    DO_FSERIALISE(path1);
    DO_FSERIALISE(path2);
}

sf::Image load_image(const std::string& fname)
{
    sf::Image img;
    img.loadFromFile(fname);

    return img;
}

cl::image load_mipped_image(sf::Image& img, cl::context& ctx, cl::command_queue& cqueue)
{
    const uint8_t* as_uint8 = reinterpret_cast<const uint8_t*>(img.getPixelsPtr());

    texture_settings bsett;
    bsett.width = img.getSize().x;
    bsett.height = img.getSize().y;
    bsett.is_srgb = false;

    texture opengl_tex;
    opengl_tex.load_from_memory(bsett, &as_uint8[0]);

    #define MIP_LEVELS 10

    int max_mips = floor(log2(std::min(img.getSize().x, img.getSize().y))) + 1;

    max_mips = std::min(max_mips, MIP_LEVELS);

    cl::image image_mipped(ctx);
    image_mipped.alloc((vec3i){img.getSize().x, img.getSize().y, max_mips}, {CL_RGBA, CL_UNORM_INT8}, cl::image_flags::ARRAY);

    ///and all of THIS is to work around a bug in AMDs drivers, where you cant write to a specific array level!
    int swidth = img.getSize().x;
    int sheight = img.getSize().y;

    std::vector<vec<4, cl_uchar>> as_uniform;
    as_uniform.reserve(max_mips * sheight * swidth);

    for(int i=0; i < max_mips; i++)
    {
        std::vector<vec4f> mip = opengl_tex.read(i);

        int cwidth = swidth / pow(2, i);
        int cheight = sheight / pow(2, i);

        assert(cwidth * cheight == mip.size());

        for(int y = 0; y < sheight; y++)
        {
            for(int x=0; x < swidth; x++)
            {
                ///clamp to border
                int lx = std::min(x, cwidth - 1);
                int ly = std::min(y, cheight - 1);

                vec4f in = mip[ly * cwidth + lx];

                in = clamp(in, 0.f, 1.f);

                as_uniform.push_back({in.x() * 255, in.y() * 255, in.z() * 255, in.w() * 255});
            }
        }
    }

    vec<3, size_t> origin = {0, 0, 0};
    vec<3, size_t> region = {swidth, sheight, max_mips};

    image_mipped.write(cqueue, (char*)as_uniform.data(), origin, region);

    return image_mipped;
}

void background_images::load(const std::string& n1, const std::string& n2)
{
    sf::Image img_1 = load_image(n1);

    i1 = load_mipped_image(img_1, ctx, cqueue);

    sf::Image img_2;

    if(n1 == n2)
        img_2 = img_1;
    else
        img_2 = load_image(n2);

    bool is_eq = false;

    if(img_1.getSize().x == img_2.getSize().x && img_1.getSize().y == img_2.getSize().y)
    {
        size_t len = size_t{img_1.getSize().x} * size_t{img_1.getSize().y} * 4;

        if(memcmp(img_1.getPixelsPtr(), img_2.getPixelsPtr(), len) == 0)
        {
            i2 = i1;

            is_eq = true;
        }
    }

    if(!is_eq)
        i2 = load_mipped_image(img_2, ctx, cqueue);
}

void background_settings::load()
{
    try
    {
        nlohmann::json js = nlohmann::json::parse(file::read("backgrounds.json", file::mode::BINARY));

        deserialise<background_settings>(js, *this, serialise_mode::DISK);
    }
    catch(...){}
}

void background_settings::display(background_images& bi)
{
    std::string ext = ".png";

    std::vector<std::string> credits;
    std::vector<std::string> name;
    std::vector<std::filesystem::path> paths;
    std::vector<std::string> license_data;

    for(const auto& entry : std::filesystem::directory_iterator{"backgrounds"})
    {
        if(entry.path().string().ends_with(ext))
        {
            name.push_back(entry.path().filename().string());
            paths.push_back(entry.path());

            std::filesystem::path path = entry.path();

            std::string license_path = path.replace_extension("info").string();

            std::string license = file::read(license_path, file::mode::TEXT);
            license_data.push_back(license);
        }
    }

    bool save = false;

    for(int i=0; i < (int)name.size(); i++)
    {
        if(ImGui::TreeNode(name[i].c_str()))
        {
            if(ImGui::Button(("Set##" + std::to_string(i)).c_str()))
            {
                path1 = paths[i].string();
                path2 = paths[i].string();

                bi.load(path1, path2);
                save = true;
            }

            ImGui::SameLine();

            if(ImGui::Button(("Set Main Background##" + std::to_string(i)).c_str()))
            {
                path1 = paths[i].string();
                bi.load(path1, path2);
                save = true;
            }

            ImGui::SameLine();

            if(ImGui::Button(("Set Second Background##" + std::to_string(i)).c_str()))
            {
                path2 = paths[i].string();
                bi.load(path1, path2);
                save = true;
            }

            if(license_data[i].size() > 0)
            {
                ImGui::TextWrapped("%s", license_data[i].c_str());
            }

            ImGui::TreePop();
        }
    }

    if(save)
    {
        file::write_atomic("backgrounds.json", serialise(*this, serialise_mode::DISK).dump(), file::mode::BINARY);
    }
}
