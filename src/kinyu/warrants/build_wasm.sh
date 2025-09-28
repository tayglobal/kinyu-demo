#!/bin/bash

# Build script for WebAssembly version of warrants module

echo "Building WebAssembly version of warrants module..."

# Check if wasm-pack is installed
if ! command -v wasm-pack &> /dev/null; then
    echo "wasm-pack not found. Installing..."
    curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
fi

# Build the WebAssembly module using the simplified version
cd wasm-build
wasm-pack build --target web --out-dir pkg
cd ..

echo "WebAssembly build complete!"
echo "Generated files are in the pkg/ directory"
