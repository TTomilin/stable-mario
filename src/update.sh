#!/usr/bin/bash

echo "Updating library file..."
cd ..
make
cp build/lib.linux-x86_64-cpython-39/retro/_retro.cpython-39-x86_64-linux-gnu.so scripts/retro/_retro.cpython-39-x86_64-linux-gnu.so
cd src