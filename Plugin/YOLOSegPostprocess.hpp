#pragma once

#import <CoreML/CoreML.h>

#include "YOLOSegTypes.hpp"

#include <array>
#include <string>
#include <vector>

namespace yolo_seg
{

struct Detection
{
    float minX;
    float minY;
    float maxX;
    float maxY;
    float confidence;
    std::vector<float> coefficients;
};

struct DecodedOutput
{
    int width = 0;
    int height = 0;
    std::vector<Detection> detections;
};

// Decodes the model-specific YOLO segmentation tensors and performs NMS.
// prediction is [batch, 4 + classes + masks, candidates], while prototype is
// [batch, masks, height, width]. Class zero is treated as "person".
bool DecodePersonDetections(
    MLMultiArray *prediction,
    MLMultiArray *prototype,
    DecodedOutput &output,
    std::string &error
);

// Converts model-input-space boxes to prototype-mask-space metadata consumed
// directly by the compute shader.
void WriteDetectionMetadata(
    const DecodedOutput &output,
    int inputWidth,
    int inputHeight,
    std::array<DetectionMetadata, DetectionLimit> &metadata
);

} // namespace yolo_seg
