#pragma once

#import <Metal/Metal.h>

#include "YOLOSegTypes.hpp"

#include <cstdint>

namespace yolo_seg
{

struct Context;

// The Unity Metal device must be installed before a context creates shared
// textures. A system-default device is used by non-Unity validation callers.
void SetMetalDevice(id<MTLDevice> device);

Context *CreateContext(const char *modelPath, char *errorBuffer, int errorCapacity);
void DestroyContext(Context *context);

int GetInputWidth(Context *context);
int GetInputHeight(Context *context);
int GetPrototypeSlotCount(Context *context);
int GetPrototypeTextureInfo(
    Context *context,
    int slotIndex,
    int *width,
    int *height,
    void **nativeTexture
);

// Status-returning operations use -1 for an error, 0 for unavailable/pending,
// and 1 for success unless otherwise documented by the function.
int CanSubmit(Context *context);
int SubmitBGRA(
    Context *context,
    const uint8_t *bgra,
    int width,
    int height,
    int rowBytes
);
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
);

// Returns the copied detection count on success rather than a boolean status.
int CopyDetectionMetadata(Context *context, DetectionMetadata *destination, int capacity);
int MarkPrototypeSlotGPUInFlight(Context *context, int slotIndex, uint64_t generation);
int ReleasePrototypeSlot(Context *context, int slotIndex, uint64_t generation);

} // namespace yolo_seg
