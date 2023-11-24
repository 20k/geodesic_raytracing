rm -rf ./build_public
mkdir build_public

sh xmake

cd ./deps/steamworks_sdk_153a/sdk/tools/ContentBuilder/
sh ./run_build.bat
cd ..
cd ..
cd ..
cd ..
cd ..

cp number.js ./build_public
cp acknowledgements.txt ./build_public
cp *.cl ./build_public
cp steam_api64.dll ./build_public
cp steam_appid.txt ./build_public
cp VeraMono.ttf ./build_public
cp -r ./scripts ./build_public/scripts
cp build/mingw/x86_64/release/RelativityWorkshop.exe ./build_public
cp build/mingw/x86_64/release/swt.exe ./build_public
cp -r ./backgrounds ./build_public/backgrounds