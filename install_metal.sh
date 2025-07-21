#!/bin/bash

# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.

set -eu

echo "Starting installation script..."

# Step 1: Install Python requirements
echo "Installing Python requirements from requirements.txt..."
pip install -r requirements-build.txt
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Error: Failed to install Python requirements."
    exit 1
else
    echo "Python requirements installed successfully."
fi

# Step 2: Define LLVM version and architecture

# Step 9: Clone and build TVM
echo "Cloning TVM repository and initializing submodules..."
# clone and build tvm
git submodule update --init --recursive

if [ -d build ]; then
    rm -rf build
fi

mkdir build
cp 3rdparty/tvm/cmake/config.cmake build
cd build

echo "Configuring TVM build with CUDA paths..."
echo "set(USE_LLVM \"/opt/homebrew/opt/llvm@14/bin/llvm-config\")" >> config.cmake
echo "set(USE_METAL ON)" >> config.cmake

echo "Running CMake for TileLang..."
cmake ..
if [ $? -ne 0 ]; then
    echo "Error: CMake configuration failed."
    exit 1
fi

echo "Building TileLang with make..."

# Calculate 75% of available CPU cores
# Other wise, make will use all available cores
# and it may cause the system to be unresponsive
CORES=$(sysctl -n hw.logicalcpu)
MAKE_JOBS=$(( CORES / 2 ))
make -j${MAKE_JOBS}

if [ $? -ne 0 ]; then
    echo "Error: TileLang build failed."
    exit 1
else
    echo "TileLang build completed successfully."
fi
