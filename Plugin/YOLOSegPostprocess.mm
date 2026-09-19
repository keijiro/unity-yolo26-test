#include "YOLOSegPostprocess.hpp"

#include <algorithm>
#include <cmath>

namespace yolo_seg
{
namespace
{

constexpr float ConfidenceThreshold = 0.35f;
constexpr float NmsThreshold = 0.45f;

float IntersectionOverUnion(const Detection &lhs, const Detection &rhs)
{
    auto minX = std::max(lhs.minX, rhs.minX);
    auto minY = std::max(lhs.minY, rhs.minY);
    auto maxX = std::min(lhs.maxX, rhs.maxX);
    auto maxY = std::min(lhs.maxY, rhs.maxY);
    auto intersection = std::max(0.0f, maxX - minX) * std::max(0.0f, maxY - minY);
    auto lhsArea = std::max(0.0f, lhs.maxX - lhs.minX) *
                   std::max(0.0f, lhs.maxY - lhs.minY);
    auto rhsArea = std::max(0.0f, rhs.maxX - rhs.minX) *
                   std::max(0.0f, rhs.maxY - rhs.minY);
    auto unionArea = lhsArea + rhsArea - intersection;
    return unionArea > 0 ? intersection / unionArea : 0;
}

std::vector<Detection> SelectDetections(std::vector<Detection> detections)
{
    std::sort(detections.begin(), detections.end(), [](const auto &lhs, const auto &rhs)
    {
        return lhs.confidence > rhs.confidence;
    });

    std::vector<Detection> selected;
    for (auto &detection : detections)
    {
        auto overlaps = std::any_of(selected.begin(), selected.end(), [&](const auto &other)
        {
            return IntersectionOverUnion(detection, other) >= NmsThreshold;
        });
        if (overlaps) continue;
        selected.push_back(std::move(detection));
        if (selected.size() == DetectionLimit) break;
    }
    return selected;
}

} // namespace

bool DecodePersonDetections(
    MLMultiArray *prediction,
    MLMultiArray *prototype,
    DecodedOutput &output,
    std::string &error
)
{
    if (prediction.dataType != MLMultiArrayDataTypeFloat32 ||
        (prototype.dataType != MLMultiArrayDataTypeFloat32 &&
         prototype.dataType != MLMultiArrayDataTypeFloat16) ||
        prediction.shape.count != 3 || prototype.shape.count != 4)
    {
        error = "The model does not contain supported YOLO segmentation outputs.";
        return false;
    }

    auto channelCount = prediction.shape[1].intValue;
    auto candidateCount = prediction.shape[2].intValue;
    auto maskChannels = prototype.shape[1].intValue;
    auto outputHeight = prototype.shape[2].intValue;
    auto outputWidth = prototype.shape[3].intValue;
    auto classCount = channelCount - 4 - maskChannels;
    if (classCount <= 0 || candidateCount <= 0 || maskChannels != MaskChannelCount ||
        outputWidth <= 0 || outputHeight <= 0)
    {
        error = "The YOLO segmentation output has an invalid shape.";
        return false;
    }

    auto predictionValues = static_cast<const float *>(prediction.dataPointer);
    auto predictionChannelStride = prediction.strides[1].intValue;
    auto predictionCandidateStride = prediction.strides[2].intValue;
    auto value = [&](int channel, int candidate)
    {
        return predictionValues[channel * predictionChannelStride +
                                candidate * predictionCandidateStride];
    };

    std::vector<Detection> detections;
    for (auto candidate = 0; candidate < candidateCount; candidate++)
    {
        // Channel 4 is class zero, which is "person" in the bundled model.
        auto confidence = value(4, candidate);
        if (confidence < ConfidenceThreshold) continue;
        auto centerX = value(0, candidate);
        auto centerY = value(1, candidate);
        auto width = value(2, candidate);
        auto height = value(3, candidate);
        Detection detection
        {
            centerX - width / 2,
            centerY - height / 2,
            centerX + width / 2,
            centerY + height / 2,
            confidence,
            std::vector<float>(static_cast<size_t>(maskChannels))
        };
        for (auto channel = 0; channel < maskChannels; channel++)
            detection.coefficients[static_cast<size_t>(channel)] =
                value(4 + classCount + channel, candidate);
        detections.push_back(std::move(detection));
    }

    output.width = outputWidth;
    output.height = outputHeight;
    output.detections = SelectDetections(std::move(detections));
    return true;
}

void WriteDetectionMetadata(
    const DecodedOutput &output,
    int inputWidth,
    int inputHeight,
    std::array<DetectionMetadata, DetectionLimit> &metadata
)
{
    metadata = {};
    auto scaleX = static_cast<float>(output.width) / inputWidth;
    auto scaleY = static_cast<float>(output.height) / inputHeight;
    for (size_t index = 0; index < output.detections.size(); index++)
    {
        const auto &source = output.detections[index];
        auto &destination = metadata[index];
        destination.bounds[0] = std::clamp(
            static_cast<float>(std::floor(source.minX * scaleX)),
            0.0f,
            static_cast<float>(output.width)
        );
        destination.bounds[1] = std::clamp(
            static_cast<float>(std::floor(source.minY * scaleY)),
            0.0f,
            static_cast<float>(output.height)
        );
        destination.bounds[2] = std::clamp(
            static_cast<float>(std::ceil(source.maxX * scaleX)),
            0.0f,
            static_cast<float>(output.width)
        );
        destination.bounds[3] = std::clamp(
            static_cast<float>(std::ceil(source.maxY * scaleY)),
            0.0f,
            static_cast<float>(output.height)
        );
        std::copy(
            source.coefficients.begin(),
            source.coefficients.end(),
            destination.coefficients
        );
    }
}

} // namespace yolo_seg
