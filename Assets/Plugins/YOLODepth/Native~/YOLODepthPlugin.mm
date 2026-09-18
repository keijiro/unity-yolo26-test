#import <CoreML/CoreML.h>
#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#define YD_EXPORT extern "C" __attribute__((visibility("default")))

namespace
{

struct Context
{
    __strong MLModel *model = nil;
    __strong NSString *inputName = nil;
    dispatch_queue_t queue = nullptr;
    std::mutex mutex;
    std::vector<float> output;
    std::string error;
    int inputWidth = 0;
    int inputHeight = 0;
    int outputWidth = 0;
    int outputHeight = 0;
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

CVPixelBufferRef CreatePixelBuffer(const uint8_t *rgba, int width, int height, int rowBytes)
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
        auto sourceRow = rgba + (height - 1 - y) * rowBytes;
        auto destinationRow = destination + y * destinationRowBytes;
        for (auto x = 0; x < width; x++)
        {
            destinationRow[x * 4 + 0] = sourceRow[x * 4 + 2];
            destinationRow[x * 4 + 1] = sourceRow[x * 4 + 1];
            destinationRow[x * 4 + 2] = sourceRow[x * 4 + 0];
            destinationRow[x * 4 + 3] = sourceRow[x * 4 + 3];
        }
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

} // namespace

YD_EXPORT void *YOLODepthCreate(const char *modelPath, char *errorBuffer, int errorCapacity)
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
        context->queue = dispatch_queue_create("jp.keijiro.yolo-depth.inference", DISPATCH_QUEUE_SERIAL);
        return context;
    }
}

YD_EXPORT void YOLODepthDestroy(void *handle)
{
    auto context = static_cast<Context *>(handle);
    if (context == nullptr) return;
    dispatch_sync(context->queue, ^{});
    delete context;
}

YD_EXPORT int YOLODepthGetInputWidth(void *handle)
{
    auto context = static_cast<Context *>(handle);
    return context == nullptr ? 0 : context->inputWidth;
}

YD_EXPORT int YOLODepthGetInputHeight(void *handle)
{
    auto context = static_cast<Context *>(handle);
    return context == nullptr ? 0 : context->inputHeight;
}

YD_EXPORT int YOLODepthCanSubmit(void *handle)
{
    auto context = static_cast<Context *>(handle);
    if (context == nullptr) return 0;
    std::lock_guard<std::mutex> lock(context->mutex);
    return !context->busy && !context->ready && !context->errorReady;
}

YD_EXPORT int YOLODepthSubmitRGBA(
    void *handle,
    const uint8_t *rgba,
    int width,
    int height,
    int rowBytes
)
{
    auto context = static_cast<Context *>(handle);
    if (context == nullptr || rgba == nullptr) return -1;
    if (width != context->inputWidth || height != context->inputHeight || rowBytes < width * 4)
        return -1;

    {
        std::lock_guard<std::mutex> lock(context->mutex);
        if (context->busy || context->ready || context->errorReady) return 0;
        context->busy = true;
    }

    auto pixelBuffer = CreatePixelBuffer(rgba, width, height, rowBytes);
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

            MLMultiArray *array = nil;
            for (NSString *name in prediction.featureNames)
            {
                auto candidate = [prediction featureValueForName:name].multiArrayValue;
                if (candidate.shape.count < 2) continue;
                array = candidate;
                break;
            }
            if (array == nil || array.dataType != MLMultiArrayDataTypeFloat32)
            {
                StoreError(context, "The model does not contain a Float32 depth output.");
                return;
            }

            auto rank = array.shape.count;
            auto outputHeight = array.shape[rank - 2].intValue;
            auto outputWidth = array.shape[rank - 1].intValue;
            auto rowStride = array.strides[rank - 2].intValue;
            auto columnStride = array.strides[rank - 1].intValue;
            if (outputWidth <= 0 || outputHeight <= 0)
            {
                StoreError(context, "The model returned an empty depth output.");
                return;
            }

            std::vector<float> output(static_cast<size_t>(outputWidth * outputHeight));
            auto source = static_cast<const float *>(array.dataPointer);
            for (auto y = 0; y < outputHeight; y++)
                for (auto x = 0; x < outputWidth; x++)
                    output[static_cast<size_t>(y * outputWidth + x)] =
                        source[(outputHeight - 1 - y) * rowStride + x * columnStride];

            auto milliseconds = std::chrono::duration<double, std::milli>(end - start).count();
            std::lock_guard<std::mutex> lock(context->mutex);
            context->output = std::move(output);
            context->outputWidth = outputWidth;
            context->outputHeight = outputHeight;
            context->inferenceMilliseconds = milliseconds;
            context->busy = false;
            context->ready = true;
        }
    });
    return 1;
}

YD_EXPORT int YOLODepthTryGetOutputInfo(
    void *handle,
    int *width,
    int *height,
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
    if (inferenceMilliseconds != nullptr)
        *inferenceMilliseconds = context->inferenceMilliseconds;
    return 1;
}

YD_EXPORT int YOLODepthCopyOutput(void *handle, float *destination, int capacity)
{
    auto context = static_cast<Context *>(handle);
    if (context == nullptr || destination == nullptr) return -1;
    std::lock_guard<std::mutex> lock(context->mutex);
    if (!context->ready) return 0;
    if (capacity < static_cast<int>(context->output.size())) return -1;
    std::memcpy(destination, context->output.data(), context->output.size() * sizeof(float));
    context->output.clear();
    context->ready = false;
    return 1;
}
