#import "YOLOSegContext.hpp"

#import "YOLOSegPostprocess.hpp"

#import <CoreML/CoreML.h>
#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <mutex>
#include <string>

namespace yolo_seg
{
namespace
{

__strong id<MTLDevice> s_MetalDevice = nil;

// Native inference owns a slot through Ready. Unity takes ownership by moving
// it to GPUInFlight and releases it only after its graphics fence has passed.
enum class SlotState
{
    Free,
    Inferencing,
    Ready,
    GPUInFlight
};

struct PrototypeSlot
{
    CVPixelBufferRef pixelBuffer = nullptr;
    CVMetalTextureRef textureView = nullptr;
    __strong id<MTLTexture> texture = nil;
    SlotState state = SlotState::Free;
    uint64_t generation = 0;
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

} // namespace

// Context is opaque outside this translation unit so synchronization and
// Core Foundation ownership cannot leak into the exported C ABI layer.
struct Context
{
    __strong MLModel *model = nil;
    __strong NSString *inputName = nil;
    __strong NSString *predictionName = nil;
    __strong NSString *prototypeName = nil;
    __strong NSArray<NSNumber *> *prototypeShape = nil;
    dispatch_queue_t queue = nullptr;
    std::mutex mutex;
    CVMetalTextureCacheRef textureCache = nullptr;
    std::array<PrototypeSlot, PrototypeSlotCount> slots;
    std::array<DetectionMetadata, DetectionLimit> detections = {};
    std::string error;
    int inputWidth = 0;
    int inputHeight = 0;
    int outputWidth = 0;
    int outputHeight = 0;
    int prototypeTextureWidth = 0;
    int prototypeTextureHeight = 0;
    int personCount = 0;
    int readySlot = -1;
    int nextSlot = 0;
    uint64_t readyGeneration = 0;
    double inferenceMilliseconds = 0;
    bool busy = false;
    bool ready = false;
    bool errorReady = false;

    ~Context()
    {
        for (auto &slot : slots)
        {
            slot.texture = nil;
            if (slot.textureView != nullptr) CFRelease(slot.textureView);
            if (slot.pixelBuffer != nullptr) CFRelease(slot.pixelBuffer);
        }
        if (textureCache != nullptr) CFRelease(textureCache);
    }
};

namespace
{

bool CreatePrototypeSlots(Context *context, std::string &error)
{
    // Shared textures must use Unity's Metal device. The fallback supports
    // validation callers for which UnityPluginLoad is never invoked.
    auto device = s_MetalDevice ?: MTLCreateSystemDefaultDevice();
    if (device == nil)
    {
        error = "Could not obtain a Metal device.";
        return false;
    }

    auto status = CVMetalTextureCacheCreate(
        kCFAllocatorDefault,
        nullptr,
        device,
        nullptr,
        &context->textureCache
    );
    if (status != kCVReturnSuccess || context->textureCache == nullptr)
    {
        error = "Could not create the Core Video Metal texture cache.";
        return false;
    }

    NSDictionary *attributes = @{
        (id)kCVPixelBufferMetalCompatibilityKey: @YES,
        (id)kCVPixelBufferIOSurfacePropertiesKey: @{}
    };
    for (auto &slot : context->slots)
    {
        // The model declares a Float32 prototype output, but Core ML accepts an
        // FP16 IOSurface as a requested output backing. The request is advisory,
        // so each prediction verifies that Core ML actually used this buffer.
        status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            context->prototypeTextureWidth,
            context->prototypeTextureHeight,
            kCVPixelFormatType_OneComponent16Half,
            (__bridge CFDictionaryRef)attributes,
            &slot.pixelBuffer
        );
        if (status != kCVReturnSuccess || slot.pixelBuffer == nullptr ||
            CVPixelBufferGetIOSurface(slot.pixelBuffer) == nullptr)
        {
            error = "Could not create an IOSurface-backed prototype buffer.";
            return false;
        }

        status = CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault,
            context->textureCache,
            slot.pixelBuffer,
            nullptr,
            MTLPixelFormatR16Float,
            context->prototypeTextureWidth,
            context->prototypeTextureHeight,
            0,
            &slot.textureView
        );
        if (status != kCVReturnSuccess || slot.textureView == nullptr)
        {
            error = "Could not create a Metal prototype texture view.";
            return false;
        }
        slot.texture = CVMetalTextureGetTexture(slot.textureView);
        if (slot.texture == nil)
        {
            error = "The prototype texture view did not contain a Metal texture.";
            return false;
        }
    }
    return true;
}

// The caller holds Context::mutex. A monotonically increasing generation keeps
// a late GPU fence from releasing a slot that has already been reused.
int AcquireSlot(Context *context)
{
    for (auto offset = 0; offset < PrototypeSlotCount; offset++)
    {
        auto index = (context->nextSlot + offset) % PrototypeSlotCount;
        auto &slot = context->slots[static_cast<size_t>(index)];
        if (slot.state != SlotState::Free) continue;
        slot.state = SlotState::Inferencing;
        slot.generation++;
        context->nextSlot = (index + 1) % PrototypeSlotCount;
        return index;
    }
    return -1;
}

void StoreError(Context *context, const std::string &message, int slotIndex = -1)
{
    std::lock_guard<std::mutex> lock(context->mutex);
    if (slotIndex >= 0 && slotIndex < PrototypeSlotCount)
        context->slots[static_cast<size_t>(slotIndex)].state = SlotState::Free;
    context->error = message;
    context->busy = false;
    context->ready = false;
    context->readySlot = -1;
    context->readyGeneration = 0;
    context->errorReady = true;
}

} // namespace

void SetMetalDevice(id<MTLDevice> device)
{
    s_MetalDevice = device;
}

Context *CreateContext(const char *modelPath, char *errorBuffer, int errorCapacity)
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

        NSString *predictionName = nil;
        NSString *prototypeName = nil;
        NSArray<NSNumber *> *prototypeShape = nil;
        for (NSString *name in model.modelDescription.outputDescriptionsByName)
        {
            auto description = model.modelDescription.outputDescriptionsByName[name];
            if (description.type != MLFeatureTypeMultiArray) continue;
            auto shape = description.multiArrayConstraint.shape;
            if (shape.count == 3) predictionName = name;
            if (shape.count == 4)
            {
                prototypeName = name;
                prototypeShape = shape;
            }
        }
        if (predictionName == nil || prototypeName == nil || prototypeShape == nil)
        {
            CopyString(
                "The model does not contain YOLO segmentation outputs.",
                errorBuffer,
                errorCapacity
            );
            return nullptr;
        }

        // Flatten [batch, channel, height, width] into an R16 texture whose width
        // is the final dimension and whose height is the product of the others.
        auto prototypeWidth = prototypeShape.lastObject.intValue;
        auto prototypeHeight = 1;
        for (NSUInteger index = 0; index + 1 < prototypeShape.count; index++)
            prototypeHeight *= prototypeShape[index].intValue;
        if (prototypeWidth <= 0 || prototypeHeight <= 0)
        {
            CopyString("The prototype output has an invalid shape.", errorBuffer, errorCapacity);
            return nullptr;
        }

        auto context = new Context();
        context->model = model;
        context->inputName = inputName;
        context->predictionName = predictionName;
        context->prototypeName = prototypeName;
        context->prototypeShape = prototypeShape;
        context->inputWidth = static_cast<int>(inputDescription.imageConstraint.pixelsWide);
        context->inputHeight = static_cast<int>(inputDescription.imageConstraint.pixelsHigh);
        context->prototypeTextureWidth = prototypeWidth;
        context->prototypeTextureHeight = prototypeHeight;
        context->queue = dispatch_queue_create(
            "jp.keijiro.yolo-seg.inference",
            DISPATCH_QUEUE_SERIAL
        );
        std::string slotError;
        if (!CreatePrototypeSlots(context, slotError))
        {
            CopyString(slotError, errorBuffer, errorCapacity);
            delete context;
            return nullptr;
        }
        return context;
    }
}

void DestroyContext(Context *context)
{
    if (context == nullptr) return;
    // Submit work owns Context and its pixel buffers until the serial inference
    // queue drains, so destruction must wait before releasing either resource.
    dispatch_sync(context->queue, ^{});
    delete context;
}

int GetInputWidth(Context *context)
{
    return context == nullptr ? 0 : context->inputWidth;
}

int GetInputHeight(Context *context)
{
    return context == nullptr ? 0 : context->inputHeight;
}

int GetPrototypeSlotCount(Context *context)
{
    return context == nullptr ? 0 : PrototypeSlotCount;
}

int GetPrototypeTextureInfo(
    Context *context,
    int slotIndex,
    int *width,
    int *height,
    void **nativeTexture
)
{
    if (context == nullptr || slotIndex < 0 || slotIndex >= PrototypeSlotCount) return -1;
    const auto &slot = context->slots[static_cast<size_t>(slotIndex)];
    if (slot.texture == nil) return -1;
    if (width != nullptr) *width = context->prototypeTextureWidth;
    if (height != nullptr) *height = context->prototypeTextureHeight;
    if (nativeTexture != nullptr) *nativeTexture = (__bridge void *)slot.texture;
    return 1;
}

int CanSubmit(Context *context)
{
    if (context == nullptr) return 0;
    std::lock_guard<std::mutex> lock(context->mutex);
    auto hasFreeSlot = std::any_of(
        context->slots.begin(),
        context->slots.end(),
        [](const auto &slot) { return slot.state == SlotState::Free; }
    );
    return !context->busy && !context->ready && !context->errorReady && hasFreeSlot;
}

int SubmitBGRA(
    Context *context,
    const uint8_t *bgra,
    int width,
    int height,
    int rowBytes
)
{
    if (context == nullptr || bgra == nullptr) return -1;
    if (width != context->inputWidth || height != context->inputHeight || rowBytes < width * 4)
        return -1;

    int slotIndex;
    {
        std::lock_guard<std::mutex> lock(context->mutex);
        if (context->busy || context->ready || context->errorReady) return 0;
        slotIndex = AcquireSlot(context);
        if (slotIndex < 0) return 0;
        context->busy = true;
    }

    // Copy the Unity readback immediately; the managed source buffer does not
    // need to remain pinned while the asynchronous Core ML request executes.
    auto pixelBuffer = CreatePixelBuffer(bgra, width, height, rowBytes);
    if (pixelBuffer == nullptr)
    {
        StoreError(context, "Could not allocate the Core Video input buffer.", slotIndex);
        return -1;
    }

    dispatch_async(context->queue, ^{
        @autoreleasepool
        {
            auto &slot = context->slots[static_cast<size_t>(slotIndex)];
            NSError *error = nil;
            auto value = [MLFeatureValue featureValueWithPixelBuffer:pixelBuffer];
            auto provider = [[MLDictionaryFeatureProvider alloc]
                initWithDictionary:@{context->inputName: value}
                error:&error];
            if (provider == nil)
            {
                CFRelease(pixelBuffer);
                StoreError(context, ErrorString(error), slotIndex);
                return;
            }

            // MLMultiArray is a per-prediction wrapper; the IOSurface-backed
            // pixel buffer and Metal texture remain persistent in the slot.
            auto prototypeBacking = [[MLMultiArray alloc]
                initWithPixelBuffer:slot.pixelBuffer
                shape:context->prototypeShape];
            if (prototypeBacking == nil)
            {
                CFRelease(pixelBuffer);
                StoreError(context, "Could not wrap the prototype pixel buffer.", slotIndex);
                return;
            }
            auto options = [MLPredictionOptions new];
            options.outputBackings = @{context->prototypeName: prototypeBacking};
            auto start = std::chrono::steady_clock::now();
            auto prediction = [context->model predictionFromFeatures:provider
                                                       options:options
                                                         error:&error];
            auto end = std::chrono::steady_clock::now();
            CFRelease(pixelBuffer);
            if (prediction == nil)
            {
                StoreError(context, ErrorString(error), slotIndex);
                return;
            }

            auto predictionArray = [prediction
                featureValueForName:context->predictionName].multiArrayValue;
            auto prototypeArray = [prediction
                featureValueForName:context->prototypeName].multiArrayValue;
            if (predictionArray == nil || prototypeArray == nil)
            {
                StoreError(
                    context,
                    "The model did not return YOLO segmentation outputs.",
                    slotIndex
                );
                return;
            }
            // outputBackings is advisory. Using a different allocation would
            // leave the Metal texture stale, so reject that prediction.
            if (prototypeArray != prototypeBacking ||
                prototypeArray.pixelBuffer != slot.pixelBuffer)
            {
                StoreError(
                    context,
                    "Core ML rejected the requested prototype output backing.",
                    slotIndex
                );
                return;
            }

            std::string decodeError;
            DecodedOutput decoded;
            if (!DecodePersonDetections(
                    predictionArray,
                    prototypeArray,
                    decoded,
                    decodeError
                ))
            {
                StoreError(context, decodeError, slotIndex);
                return;
            }

            auto milliseconds = std::chrono::duration<double, std::milli>(end - start).count();
            std::lock_guard<std::mutex> lock(context->mutex);
            context->outputWidth = decoded.width;
            context->outputHeight = decoded.height;
            context->personCount = static_cast<int>(decoded.detections.size());
            context->inferenceMilliseconds = milliseconds;
            WriteDetectionMetadata(
                decoded,
                context->inputWidth,
                context->inputHeight,
                context->detections
            );
            slot.state = SlotState::Ready;
            context->readySlot = slotIndex;
            context->readyGeneration = slot.generation;
            context->busy = false;
            context->ready = true;
        }
    });
    return 1;
}

int TryGetOutputInfo(
    Context *context,
    int *width,
    int *height,
    int *personCount,
    double *inferenceMilliseconds,
    int *slotIndex,
    uint64_t *generation,
    char *errorBuffer,
    int errorCapacity
)
{
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
    if (slotIndex != nullptr) *slotIndex = context->readySlot;
    if (generation != nullptr) *generation = context->readyGeneration;
    return 1;
}

int CopyDetectionMetadata(Context *context, DetectionMetadata *destination, int capacity)
{
    if (context == nullptr || destination == nullptr) return -1;
    std::lock_guard<std::mutex> lock(context->mutex);
    if (!context->ready) return 0;
    auto byteCount = context->personCount * static_cast<int>(sizeof(DetectionMetadata));
    if (capacity < byteCount) return -1;
    std::memcpy(destination, context->detections.data(), static_cast<size_t>(byteCount));
    return context->personCount;
}

int MarkPrototypeSlotGPUInFlight(Context *context, int slotIndex, uint64_t generation)
{
    if (context == nullptr || slotIndex < 0 || slotIndex >= PrototypeSlotCount) return -1;
    std::lock_guard<std::mutex> lock(context->mutex);
    auto &slot = context->slots[static_cast<size_t>(slotIndex)];
    if (!context->ready || context->readySlot != slotIndex ||
        context->readyGeneration != generation || slot.generation != generation)
        return 0;
    if (slot.state != SlotState::Ready) return 0;
    slot.state = SlotState::GPUInFlight;
    context->ready = false;
    context->readySlot = -1;
    context->readyGeneration = 0;
    return 1;
}

int ReleasePrototypeSlot(Context *context, int slotIndex, uint64_t generation)
{
    if (context == nullptr || slotIndex < 0 || slotIndex >= PrototypeSlotCount) return -1;
    std::lock_guard<std::mutex> lock(context->mutex);
    auto &slot = context->slots[static_cast<size_t>(slotIndex)];
    if (slot.generation != generation) return -1;
    if (slot.state != SlotState::Ready && slot.state != SlotState::GPUInFlight) return 0;
    slot.state = SlotState::Free;
    if (context->readySlot == slotIndex && context->readyGeneration == generation)
    {
        context->ready = false;
        context->readySlot = -1;
        context->readyGeneration = 0;
    }
    return 1;
}

} // namespace yolo_seg
