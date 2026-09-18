#import <CoreML/CoreML.h>
#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#define YS_EXPORT extern "C" __attribute__((visibility("default")))

namespace
{

constexpr float ConfidenceThreshold = 0.35f;
constexpr float MaskThreshold = 0.5f;
constexpr float NmsThreshold = 0.45f;
constexpr int DetectionLimit = 10;

struct Detection
{
    float minX;
    float minY;
    float maxX;
    float maxY;
    float confidence;
    std::vector<float> coefficients;
};

struct Context
{
    __strong MLModel *model = nil;
    __strong NSString *inputName = nil;
    dispatch_queue_t queue = nullptr;
    std::mutex mutex;
    std::vector<uint8_t> output;
    std::string error;
    int inputWidth = 0;
    int inputHeight = 0;
    int outputWidth = 0;
    int outputHeight = 0;
    int personCount = 0;
    double inferenceMilliseconds = 0;
    bool busy = false;
    bool ready = false;
    bool errorReady = false;
};

void CopyString(const std::string &source, char *destination, int capacity)
{
    if (destination == nullptr || capacity <= 0) return;
    auto length = std::min(source.size(), static_cast<size_t>(capacity - 1));
    std::memcpy(destination, source.data(), length);
    destination[length] = '\0';
}

std::string ErrorString(NSError *error)
{
    if (error == nil) return "Unknown Core ML error.";
    return error.localizedDescription.UTF8String ?: "Unknown Core ML error.";
}

NSURL *ResolveModelURL(NSString *path, NSError **error)
{
    auto url = [NSURL fileURLWithPath:path];
    if ([path.pathExtension caseInsensitiveCompare:@"mlpackage"] != NSOrderedSame)
        return url;
    return [MLModel compileModelAtURL:url error:error];
}

CVPixelBufferRef CreatePixelBuffer(const uint8_t *bgra, int width, int height, int rowBytes)
{
    NSDictionary *attributes = @{
        (id)kCVPixelBufferMetalCompatibilityKey: @YES,
        (id)kCVPixelBufferIOSurfacePropertiesKey: @{}
    };
    CVPixelBufferRef buffer = nullptr;
    auto status = CVPixelBufferCreate(
        kCFAllocatorDefault,
        width,
        height,
        kCVPixelFormatType_32BGRA,
        (__bridge CFDictionaryRef)attributes,
        &buffer
    );
    if (status != kCVReturnSuccess || buffer == nullptr) return nullptr;

    CVPixelBufferLockBaseAddress(buffer, 0);
    auto destination = static_cast<uint8_t *>(CVPixelBufferGetBaseAddress(buffer));
    auto destinationRowBytes = CVPixelBufferGetBytesPerRow(buffer);
    for (auto y = 0; y < height; y++)
    {
        auto sourceRow = bgra + y * rowBytes;
        auto destinationRow = destination + y * destinationRowBytes;
        std::memcpy(destinationRow, sourceRow, static_cast<size_t>(width * 4));
    }
    CVPixelBufferUnlockBaseAddress(buffer, 0);
    return buffer;
}

void StoreError(Context *context, const std::string &message)
{
    std::lock_guard<std::mutex> lock(context->mutex);
    context->error = message;
    context->busy = false;
    context->ready = false;
    context->errorReady = true;
}

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

bool RenderPersonMasks(
    MLMultiArray *prediction,
    MLMultiArray *prototype,
    int inputWidth,
    int inputHeight,
    std::vector<uint8_t> &output,
    int &outputWidth,
    int &outputHeight,
    int &personCount,
    std::string &error
)
{
    if (prediction.dataType != MLMultiArrayDataTypeFloat32 ||
        prototype.dataType != MLMultiArrayDataTypeFloat32 ||
        prediction.shape.count != 3 || prototype.shape.count != 4)
    {
        error = "The model does not contain Float32 YOLO segmentation outputs.";
        return false;
    }

    auto channelCount = prediction.shape[1].intValue;
    auto candidateCount = prediction.shape[2].intValue;
    auto maskChannels = prototype.shape[1].intValue;
    outputHeight = prototype.shape[2].intValue;
    outputWidth = prototype.shape[3].intValue;
    auto classCount = channelCount - 4 - maskChannels;
    if (classCount <= 0 || candidateCount <= 0 || maskChannels <= 0 ||
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

    auto selected = SelectDetections(std::move(detections));
    personCount = static_cast<int>(selected.size());
    auto pixelCount = outputWidth * outputHeight;
    std::vector<float> alpha(static_cast<size_t>(pixelCount), 0);
    auto prototypeValues = static_cast<const float *>(prototype.dataPointer);
    auto prototypeChannelStride = prototype.strides[1].intValue;
    auto prototypeRowStride = prototype.strides[2].intValue;
    auto prototypeColumnStride = prototype.strides[3].intValue;
    auto scaleX = static_cast<float>(outputWidth) / inputWidth;
    auto scaleY = static_cast<float>(outputHeight) / inputHeight;

    for (const auto &detection : selected)
    {
        auto minX = std::clamp(
            static_cast<int>(std::floor(detection.minX * scaleX)), 0, outputWidth
        );
        auto maxX = std::clamp(
            static_cast<int>(std::ceil(detection.maxX * scaleX)), 0, outputWidth
        );
        auto minY = std::clamp(
            static_cast<int>(std::floor(detection.minY * scaleY)), 0, outputHeight
        );
        auto maxY = std::clamp(
            static_cast<int>(std::ceil(detection.maxY * scaleY)), 0, outputHeight
        );

        for (auto y = minY; y < maxY; y++)
        for (auto x = minX; x < maxX; x++)
        {
            auto logit = 0.0f;
            for (auto channel = 0; channel < maskChannels; channel++)
                logit += detection.coefficients[static_cast<size_t>(channel)] *
                         prototypeValues[channel * prototypeChannelStride +
                                         y * prototypeRowStride +
                                         x * prototypeColumnStride];
            auto probability = 1 / (1 + std::exp(-logit));
            if (probability < MaskThreshold) continue;
            auto index = static_cast<size_t>(y * outputWidth + x);
            alpha[index] = std::max(alpha[index], probability);
        }
    }

    output.resize(static_cast<size_t>(pixelCount * 4));
    for (auto y = 0; y < outputHeight; y++)
    for (auto x = 0; x < outputWidth; x++)
    {
        auto sourceIndex = static_cast<size_t>(y * outputWidth + x);
        auto destinationIndex = static_cast<size_t>(
            (outputHeight - 1 - y) * outputWidth + x
        );
        auto opacity = static_cast<uint8_t>(
            std::clamp(alpha[sourceIndex], 0.0f, 1.0f) * 255
        );
        output[destinationIndex * 4 + 0] = 0;
        output[destinationIndex * 4 + 1] = opacity;
        output[destinationIndex * 4 + 2] = opacity;
        output[destinationIndex * 4 + 3] = 255;
    }
    return true;
}

} // namespace

YS_EXPORT void *YOLOSegCreate(const char *modelPath, char *errorBuffer, int errorCapacity)
{
    @autoreleasepool
    {
        if (modelPath == nullptr)
        {
            CopyString("The model path is null.", errorBuffer, errorCapacity);
            return nullptr;
        }

        NSError *error = nil;
        auto path = [NSString stringWithUTF8String:modelPath];
        auto modelURL = ResolveModelURL(path, &error);
        if (modelURL == nil)
        {
            CopyString(ErrorString(error), errorBuffer, errorCapacity);
            return nullptr;
        }

        auto configuration = [MLModelConfiguration new];
        configuration.computeUnits = MLComputeUnitsAll;
        auto model = [MLModel modelWithContentsOfURL:modelURL
                                      configuration:configuration
                                              error:&error];
        if (model == nil)
        {
            CopyString(ErrorString(error), errorBuffer, errorCapacity);
            return nullptr;
        }

        NSString *inputName = nil;
        MLFeatureDescription *inputDescription = nil;
        for (NSString *name in model.modelDescription.inputDescriptionsByName)
        {
            auto description = model.modelDescription.inputDescriptionsByName[name];
            if (description.type != MLFeatureTypeImage) continue;
            inputName = name;
            inputDescription = description;
            break;
        }
        if (inputName == nil || inputDescription.imageConstraint == nil)
        {
            CopyString("The model does not contain an image input.", errorBuffer, errorCapacity);
            return nullptr;
        }

        auto context = new Context();
        context->model = model;
        context->inputName = inputName;
        context->inputWidth = static_cast<int>(inputDescription.imageConstraint.pixelsWide);
        context->inputHeight = static_cast<int>(inputDescription.imageConstraint.pixelsHigh);
        context->queue = dispatch_queue_create("jp.keijiro.yolo-seg.inference", DISPATCH_QUEUE_SERIAL);
        return context;
    }
}

YS_EXPORT void YOLOSegDestroy(void *handle)
{
    auto context = static_cast<Context *>(handle);
    if (context == nullptr) return;
    dispatch_sync(context->queue, ^{});
    delete context;
}

YS_EXPORT int YOLOSegGetInputWidth(void *handle)
{
    auto context = static_cast<Context *>(handle);
    return context == nullptr ? 0 : context->inputWidth;
}

YS_EXPORT int YOLOSegGetInputHeight(void *handle)
{
    auto context = static_cast<Context *>(handle);
    return context == nullptr ? 0 : context->inputHeight;
}

YS_EXPORT int YOLOSegCanSubmit(void *handle)
{
    auto context = static_cast<Context *>(handle);
    if (context == nullptr) return 0;
    std::lock_guard<std::mutex> lock(context->mutex);
    return !context->busy && !context->ready && !context->errorReady;
}

YS_EXPORT int YOLOSegSubmitBGRA(
    void *handle,
    const uint8_t *bgra,
    int width,
    int height,
    int rowBytes
)
{
    auto context = static_cast<Context *>(handle);
    if (context == nullptr || bgra == nullptr) return -1;
    if (width != context->inputWidth || height != context->inputHeight || rowBytes < width * 4)
        return -1;

    {
        std::lock_guard<std::mutex> lock(context->mutex);
        if (context->busy || context->ready || context->errorReady) return 0;
        context->busy = true;
    }

    auto pixelBuffer = CreatePixelBuffer(bgra, width, height, rowBytes);
    if (pixelBuffer == nullptr)
    {
        StoreError(context, "Could not allocate the Core Video input buffer.");
        return -1;
    }

    dispatch_async(context->queue, ^{
        @autoreleasepool
        {
            NSError *error = nil;
            auto value = [MLFeatureValue featureValueWithPixelBuffer:pixelBuffer];
            auto provider = [[MLDictionaryFeatureProvider alloc]
                initWithDictionary:@{context->inputName: value}
                error:&error];
            if (provider == nil)
            {
                CFRelease(pixelBuffer);
                StoreError(context, ErrorString(error));
                return;
            }

            auto start = std::chrono::steady_clock::now();
            auto prediction = [context->model predictionFromFeatures:provider error:&error];
            auto end = std::chrono::steady_clock::now();
            CFRelease(pixelBuffer);
            if (prediction == nil)
            {
                StoreError(context, ErrorString(error));
                return;
            }

            MLMultiArray *predictionArray = nil;
            MLMultiArray *prototypeArray = nil;
            for (NSString *name in prediction.featureNames)
            {
                auto array = [prediction featureValueForName:name].multiArrayValue;
                if (array.shape.count == 3) predictionArray = array;
                if (array.shape.count == 4) prototypeArray = array;
            }
            if (predictionArray == nil || prototypeArray == nil)
            {
                StoreError(context, "The model does not contain YOLO segmentation outputs.");
                return;
            }

            std::vector<uint8_t> output;
            std::string renderError;
            int outputWidth;
            int outputHeight;
            int personCount;
            if (!RenderPersonMasks(
                    predictionArray,
                    prototypeArray,
                    context->inputWidth,
                    context->inputHeight,
                    output,
                    outputWidth,
                    outputHeight,
                    personCount,
                    renderError
                ))
            {
                StoreError(context, renderError);
                return;
            }

            auto milliseconds = std::chrono::duration<double, std::milli>(end - start).count();
            std::lock_guard<std::mutex> lock(context->mutex);
            context->output = std::move(output);
            context->outputWidth = outputWidth;
            context->outputHeight = outputHeight;
            context->personCount = personCount;
            context->inferenceMilliseconds = milliseconds;
            context->busy = false;
            context->ready = true;
        }
    });
    return 1;
}

YS_EXPORT int YOLOSegTryGetOutputInfo(
    void *handle,
    int *width,
    int *height,
    int *personCount,
    double *inferenceMilliseconds,
    char *errorBuffer,
    int errorCapacity
)
{
    auto context = static_cast<Context *>(handle);
    if (context == nullptr) return -1;
    std::lock_guard<std::mutex> lock(context->mutex);
    if (context->errorReady)
    {
        CopyString(context->error, errorBuffer, errorCapacity);
        context->errorReady = false;
        return -1;
    }
    if (!context->ready) return 0;
    if (width != nullptr) *width = context->outputWidth;
    if (height != nullptr) *height = context->outputHeight;
    if (personCount != nullptr) *personCount = context->personCount;
    if (inferenceMilliseconds != nullptr)
        *inferenceMilliseconds = context->inferenceMilliseconds;
    return 1;
}

YS_EXPORT int YOLOSegCopyOutput(void *handle, uint8_t *destination, int capacity)
{
    auto context = static_cast<Context *>(handle);
    if (context == nullptr || destination == nullptr) return -1;
    std::lock_guard<std::mutex> lock(context->mutex);
    if (!context->ready) return 0;
    if (capacity < static_cast<int>(context->output.size())) return -1;
    std::memcpy(destination, context->output.data(), context->output.size());
    context->output.clear();
    context->ready = false;
    return 1;
}
