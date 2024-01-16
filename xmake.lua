---note to self: give up on static linking, just do it fully dynamic for xmake, its not worth fighting the build system
add_rules("mode.debug", "mode.release")
add_requireconfs("*", {configs = {shared = false}, system=false})
add_requires("zlib")
add_requires("freetype")
add_requires("openssl", "opengl", "glfw", "glew", "sfml")
add_requires("boost", {})

add_files("workshop/steam_ugc_manager.cpp")
add_files("deps/imgui/backends/imgui_impl_glfw.cpp")
add_files("deps/imgui/backends/imgui_impl_opengl3.cpp")
add_files("deps/imgui/misc/freetype/imgui_freetype.cpp")
add_files("deps/imgui/misc/cpp/imgui_stdlib.cpp")
add_files("deps/imgui/imgui.cpp")
add_files("deps/imgui/imgui_draw.cpp")
add_files("deps/imgui/imgui_tables.cpp")
add_files("deps/imgui/imgui_widgets.cpp")
add_files("deps/libfastcl/fastcl/cl.cpp")
add_files("deps/quickjs/cutils.c")
add_files("deps/quickjs/libbf.c")
add_files("deps/quickjs/libregexp.c")
add_files("deps/quickjs/libunicode.c")
add_files("deps/quickjs/quickjs.c")
add_files("deps/quickjs_cpp/quickjs_cpp.cpp")
add_files("deps/tinyobjloader/tiny_obj_loader.cc")
add_files("deps/toolkit/base_serialisables.cpp")
add_files("deps/toolkit/clipboard.cpp")
add_files("deps/toolkit/clock.cpp")
add_files("deps/toolkit/fs_helpers.cpp")
add_files("deps/toolkit/opencl.cpp")
add_files("deps/toolkit/render_window.cpp")
add_files("deps/toolkit/render_window_glfw.cpp")
add_files("deps/toolkit/texture.cpp")
add_includedirs("./deps")
add_includedirs("./deps/imgui")
add_includedirs("./deps/steamworks_sdk_153a/sdk/public")
add_includedirs("./deps/steamworks_sdk_153a/sdk/public/steam")
set_languages("c99", "cxx23")
add_defines("IMGUI_IMPL_OPENGL_LOADER_GLEW",
"SUBPIXEL_FONT_RENDERING",
"SFML_STATIC",
"GLEW_STATIC",
"GRAPHITE2_STATIC",
"CL_TARGET_OPENCL_VERSION=220",
"IMGUI_ENABLE_FREETYPE",
"CONFIG_VERSION=\"\"",
"CONFIG_BIGNUM",
"NO_SERIALISE_RATELIMIT",
"FAST_CL",
"REDIRECT_STDOUT",
"REMEMBER_SIZE")

add_packages("openssl", "opengl", "glfw", "glew", "freetype", "sfml")

set_optimize("fastest")

add_linkdirs("deps/steamworks_sdk_153a/sdk/redistributable_bin/win64/")
add_links("steam_api64")
add_links("imm32")

if is_plat("mingw") then
    add_ldflags("-static -static-libstdc++")
    add_cxflags("-mwindows")
    add_ldflags("-mwindows")
end

target("swt")
    set_kind("binary")
    add_files("workshop/*.cpp")
    add_defines("NO_OPENCL")
    
target("RelativityWorkshop")
    set_kind("binary")
    add_files("main.cpp")
    add_files("*.cpp")
    
--
-- If you want to known more usage about xmake, please see https://xmake.io
--
-- ## FAQ
--
-- You can enter the project directory firstly before building project.
--
--   $ cd projectdir
--
-- 1. How to build project?
--
--   $ xmake
--
-- 2. How to configure project?
--
--   $ xmake f -p [macosx|linux|iphoneos ..] -a [x86_64|i386|arm64 ..] -m [debug|release]
--
-- 3. Where is the build output directory?
--
--   The default output directory is `./build` and you can configure the output directory.
--
--   $ xmake f -o outputdir
--   $ xmake
--
-- 4. How to run and debug target after building project?
--
--   $ xmake run [targetname]
--   $ xmake run -d [targetname]
--
-- 5. How to install target to the system directory or other output directory?
--
--   $ xmake install
--   $ xmake install -o installdir
--
-- 6. Add some frequently-used compilation flags in xmake.lua
--
-- @code
--    -- add debug and release modes
--    add_rules("mode.debug", "mode.release")
--
--    -- add macro definition
--    add_defines("NDEBUG", "_GNU_SOURCE=1")
--
--    -- set warning all as error
--    set_warnings("all", "error")
--
--    -- set language: c99, c++11
--    set_languages("c99", "c++11")
--
--    -- set optimization: none, faster, fastest, smallest
--    set_optimize("fastest")
--
--    -- add include search directories
--    add_includedirs("/usr/include", "/usr/local/include")
--
--    -- add link libraries and search directories
--    add_links("tbox")
--    add_linkdirs("/usr/local/lib", "/usr/lib")
--
--    -- add system link libraries
--    add_syslinks("z", "pthread")
--
--    -- add compilation and link flags
--    add_cxflags("-stdnolib", "-fno-strict-aliasing")
--    add_ldflags("-L/usr/local/lib", "-lpthread", {force = true})
--
-- @endcode
--

