# YOLO26 Depth Camera for Unity

A macOS Unity demo that runs the Core ML export of `yolo26n-depth` on a live
`WebCamTexture`. The camera image is center-cropped to the model's square input
without distortion, and the estimated depth is visualized over an adjustable
range.

## Requirements

- macOS 14 or later
- Unity 6000.6.0f1
- Xcode command-line tools
- `uv` (only when the validated sibling model is unavailable)

The Core ML model and compiled native plug-in are generated dependencies and
are intentionally excluded from Git.

## Setup

Prepare the model, then build the universal macOS plug-in:

```sh
./scripts/prepare-model.sh
./scripts/build-native-plugin.sh
```

The model script first reuses the validated package from
`../yolo26-coreml-test`. When it is unavailable, it downloads the Ultralytics
weights and exports them to Core ML using `uv`.

## Running the demo

1. Open the project in Unity.
2. Open `Assets/Main.unity`.
3. Enter Play mode and grant camera access when prompted.
4. Adjust the Near and Far sliders to change the displayed depth range.

The left pane shows the exact 640 x 640 center crop sent to Core ML. The right
pane shows the shader-generated false-color depth image. Warm colors are near;
cool colors are far.

## Architecture

- `Assets/Plugins/YOLODepth/Native~/YOLODepthPlugin.mm` loads the Core ML model,
  performs inference on a serial background queue, and returns the raw Float32
  depth tensor. It drops frames while inference or output delivery is pending.
- `Assets/Resources/YOLODepth/Preprocess.shader` performs camera orientation,
  center cropping, and resizing on the GPU.
- `Assets/Resources/YOLODepth/VisualizeDepth.shader` maps raw depth values to
  false colors using the UI-selected range.
- `Assets/YOLODepth/Runtime/DepthDemoController.cs` coordinates webcam capture,
  asynchronous GPU readback, native inference, and UI Toolkit.

Rebuild the native plug-in after modifying its source:

```sh
./scripts/build-native-plugin.sh
```

## Notes

Monocular depth values are expressed in meters, but their accuracy depends on
the camera and scene. Do not use this demo for safety-critical measurement or
surveying. Review the Ultralytics model license before redistribution or
commercial use.
