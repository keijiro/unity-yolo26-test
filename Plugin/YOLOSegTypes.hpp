#pragma once

#include <cstddef>

namespace yolo_seg
{

// These constants are part of the data contract with YOLOSegNative.cs and
// ComposeMask.compute. Change all three implementations together.
constexpr int DetectionLimit = 10;
constexpr int MaskChannelCount = 32;
constexpr int PrototypeSlotCount = 3;

// Binary ABI: one float4 mask-space bounds value followed by 32 mask
// coefficients. The compute shader consumes this as nine consecutive float4s.
struct DetectionMetadata
{
    float bounds[4];
    float coefficients[MaskChannelCount];
};

static_assert(
    sizeof(DetectionMetadata) == 36 * sizeof(float),
    "DetectionMetadata must remain compatible with C# and ComposeMask.compute."
);

} // namespace yolo_seg
