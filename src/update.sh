#!/usr/bin/bash

echo "Updating library file..."
cd ..
cmake .
cp build/lib.linux-x86_64-cpython-39/_retro.cpython-39-x86_64-linux-gnu.so scripts/retro/_retro.cpython-39-x86_64-linux-gnu.so
cd src