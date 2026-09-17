#!/bin/zsh

set -euo pipefail

project_dir="${0:A:h:h}"
model_dir="$project_dir/Assets/StreamingAssets/Models"
model_name="yolo26n-seg"
destination="$model_dir/$model_name.mlpackage"
sibling_model="$project_dir/../yolo26-coreml-test/Models/$model_name.mlpackage"

mkdir -p "$model_dir"

if [[ -d "$sibling_model" ]]; then
    rm -rf "$destination"
    cp -R "$sibling_model" "$destination"
    echo "Installed model from $sibling_model"
    exit 0
fi

if ! command -v uv >/dev/null; then
    echo "uv is required to download and export $model_name." >&2
    exit 1
fi

work_dir=$(mktemp -d)
trap 'rm -rf "$work_dir"' EXIT
cd "$work_dir"

uv run --python 3.11 \
    --with ultralytics \
    --with coremltools \
    --with 'numpy<=2.3.5' \
    yolo export model="$model_name.pt" format=coreml imgsz=640

package=$(find "$work_dir" -maxdepth 2 -name "$model_name.mlpackage" -print -quit)
if [[ -z "$package" ]]; then
    echo "Core ML export did not produce $model_name.mlpackage" >&2
    exit 1
fi

rm -rf "$destination"
mv "$package" "$destination"
echo "Installed model at $destination"
