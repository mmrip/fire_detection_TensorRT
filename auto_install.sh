#!/bin/bash
echo "========== install protobuf =========="

sudo apt-get install protobuf-compiler
cd 3rdparty/protobuf-3.11.4
./configure --prefix=/usr/local/protobuf
make -j4
sudo make install

cd ../..


echo "========== compile project =========="
build_dir=build
if [ ! -d "$filecache_dir" ]; then
    mkdir -p $build_dir
fi


cd build
cmake ..
make -j4