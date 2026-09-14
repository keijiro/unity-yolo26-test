#!/bin/zsh

set -euo pipefail

project_dir="${0:A:h:h}"
source_file="$project_dir/Assets/Plugins/YOLODepth/Native~/YOLODepthPlugin.mm"
plist_file="$project_dir/Assets/Plugins/YOLODepth/Native~/Info.plist"
bundle_dir="$project_dir/Assets/Plugins/macOS/YOLODepthPlugin.bundle"
binary_dir="$bundle_dir/Contents/MacOS"

mkdir -p "$binary_dir"
cp "$plist_file" "$bundle_dir/Contents/Info.plist"

xcrun --sdk macosx clang++ \
    -std=c++17 \
    -fobjc-arc \
    -fvisibility=hidden \
    -bundle \
    -arch arm64 \
    -arch x86_64 \
    -mmacosx-version-min=12.0 \
    -framework Foundation \
    -framework CoreML \
    -framework CoreVideo \
    "$source_file" \
    -o "$binary_dir/YOLODepthPlugin"

codesign --force --sign - "$bundle_dir"
echo "Built $bundle_dir"
