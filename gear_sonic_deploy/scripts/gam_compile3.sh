cd ~/gam/gear_sonic_deploy

source scripts/setup_env.sh
export HAS_ROS2=0

rm -rf build
mkdir -p build
cd build

CC=gcc-10 CXX=g++-10 cmake -S .. -B . \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

cmake --build . --target g1_deploy_onnx_ref -j$(nproc)
