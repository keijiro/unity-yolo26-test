#include "YOLOSegContext.hpp"

#include "IUnityGraphics.h"
#include "IUnityGraphicsMetal.h"

#define YS_EXPORT extern "C" __attribute__((visibility("default")))

// This file is intentionally limited to the stable Unity/C# ABI. All model,
// threading, post-processing, and GPU-resource behavior lives behind Context.

YS_EXPORT void UnityPluginLoad(IUnityInterfaces *interfaces)
{
    auto metal = interfaces == nullptr ? nullptr : interfaces->Get<IUnityGraphicsMetalV2>();
    yolo_seg::SetMetalDevice(metal == nullptr ? nil : metal->MetalDevice());
}

YS_EXPORT void UnityPluginUnload()
{
    yolo_seg::SetMetalDevice(nil);
}

YS_EXPORT void *YOLOSegCreate(const char *modelPath, char *errorBuffer, int errorCapacity)
{
    return yolo_seg::CreateContext(modelPath, errorBuffer, errorCapacity);
}

YS_EXPORT void YOLOSegDestroy(void *handle)
{
    yolo_seg::DestroyContext(static_cast<yolo_seg::Context *>(handle));
}

YS_EXPORT int YOLOSegGetInputWidth(void *handle)
{
    return yolo_seg::GetInputWidth(static_cast<yolo_seg::Context *>(handle));
}

YS_EXPORT int YOLOSegGetInputHeight(void *handle)
{
    return yolo_seg::GetInputHeight(static_cast<yolo_seg::Context *>(handle));
}

YS_EXPORT int YOLOSegGetPrototypeSlotCount(void *handle)
{
    return yolo_seg::GetPrototypeSlotCount(static_cast<yolo_seg::Context *>(handle));
}

YS_EXPORT int YOLOSegGetPrototypeTextureInfo(
    void *handle,
    int slotIndex,
    int *width,
    int *height,
    void **nativeTexture
)
{
    return yolo_seg::GetPrototypeTextureInfo(
        static_cast<yolo_seg::Context *>(handle),
        slotIndex,
        width,
        height,
        nativeTexture
    );
}

YS_EXPORT int YOLOSegCanSubmit(void *handle)
{
    return yolo_seg::CanSubmit(static_cast<yolo_seg::Context *>(handle));
}

YS_EXPORT int YOLOSegSubmitBGRA(
    void *handle,
    const uint8_t *bgra,
    int width,
    int height,
    int rowBytes
)
{
    return yolo_seg::SubmitBGRA(
        static_cast<yolo_seg::Context *>(handle),
        bgra,
        width,
        height,
        rowBytes
    );
}

YS_EXPORT int YOLOSegTryGetOutputInfoEx(
    void *handle,
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
    return yolo_seg::TryGetOutputInfo(
        static_cast<yolo_seg::Context *>(handle),
        width,
        height,
        personCount,
        inferenceMilliseconds,
        slotIndex,
        generation,
        errorBuffer,
        errorCapacity
    );
}

YS_EXPORT int YOLOSegCopyDetectionMetadata(
    void *handle,
    yolo_seg::DetectionMetadata *destination,
    int capacity
)
{
    return yolo_seg::CopyDetectionMetadata(
        static_cast<yolo_seg::Context *>(handle),
        destination,
        capacity
    );
}

YS_EXPORT int YOLOSegMarkPrototypeSlotGPUInFlight(
    void *handle,
    int slotIndex,
    uint64_t generation
)
{
    return yolo_seg::MarkPrototypeSlotGPUInFlight(
        static_cast<yolo_seg::Context *>(handle),
        slotIndex,
        generation
    );
}

YS_EXPORT int YOLOSegReleasePrototypeSlot(
    void *handle,
    int slotIndex,
    uint64_t generation
)
{
    return yolo_seg::ReleasePrototypeSlot(
        static_cast<yolo_seg::Context *>(handle),
        slotIndex,
        generation
    );
}
