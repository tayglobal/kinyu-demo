#!/bin/bash

# Build script for WebAssembly version of warrants module

echo "Building WebAssembly version of warrants module..."

# Change to the script's directory to ensure correct paths
cd "$(dirname "$0")"

# Check if wasm-pack is installed
if ! command -v wasm-pack &> /dev/null; then
    echo "wasm-pack not found. Installing..."
    curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
fi

# wasm-pack requires the manifest to be named 'Cargo.toml'.
# We'll rename the files to build the wasm package and then revert them.
mv Cargo.toml Cargo.toml.original
mv Cargo-wasm.toml Cargo.toml

# Build the WebAssembly module.
wasm-pack build --target web --out-dir pkg

# Revert the file names
mv Cargo.toml Cargo-wasm.toml
mv Cargo.toml.original Cargo.toml

echo "WebAssembly build complete!"
echo "Generated files are in the pkg/ directory"