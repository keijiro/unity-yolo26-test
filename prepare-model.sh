#!/bin/zsh

set -euo pipefail

project_dir="${0:A:h}"
model_dir="$project_dir/Assets/StreamingAssets/Models"
model_name="yolo26n-seg"
destination="$model_dir/$model_name.mlpackage"

mkdir -p "$model_dir"

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
    --with scikit-learn \
    --with 'numpy<=2.3.5' \
    yolo export model="$model_name.pt" format=coreml imgsz=640 quantize=8

package=$(find "$work_dir" -maxdepth 2 -name "$model_name.mlpackage" -print -quit)
if [[ -z "$package" ]]; then
    echo "Core ML export did not produce $model_name.mlpackage" >&2
    exit 1
fi

rm -rf "$destination"
mv "$package" "$destination"
echo "Installed model at $destination"
