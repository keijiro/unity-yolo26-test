using System;
using System.Runtime.InteropServices;
using System.Text;

namespace YOLOSeg
{

internal static class YOLOSegNative
{
    const string LibraryName = "YOLOSegPlugin";
    const int ErrorCapacity = 1024;
    internal const int DetectionLimit = 10;

    // ABI shared with the native DetectionMetadata and the compute shader's
    // Detection: one float4 mask-space bounds value followed by 32 coefficients.
    [StructLayout(LayoutKind.Sequential)]
    internal unsafe struct DetectionMetadata
    {
        public fixed float Values[36];
    }

    internal readonly struct CreationResult
    {
        public IntPtr Handle { get; }
        public string Error { get; }

        public CreationResult(IntPtr handle, string error)
        {
            Handle = handle;
            Error = error;
        }
    }

#if UNITY_EDITOR_OSX || UNITY_STANDALONE_OSX
    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    static extern IntPtr YOLOSegCreate(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string modelPath,
        StringBuilder errorBuffer,
        int errorCapacity
    );

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void YOLOSegDestroy(IntPtr handle);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int YOLOSegGetInputWidth(IntPtr handle);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int YOLOSegGetInputHeight(IntPtr handle);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int YOLOSegGetPrototypeSlotCount(IntPtr handle);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int YOLOSegGetPrototypeTextureInfo(
        IntPtr handle,
        int slotIndex,
        out int width,
        out int height,
        out IntPtr nativeTexture
    );

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int YOLOSegCanSubmit(IntPtr handle);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int YOLOSegSubmitBGRA(
        IntPtr handle,
        IntPtr bgra,
        int width,
        int height,
        int rowBytes
    );

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    static extern int YOLOSegTryGetOutputInfoEx(
        IntPtr handle,
        out int width,
        out int height,
        out int personCount,
        out double inferenceMilliseconds,
        out int slotIndex,
        out ulong generation,
        StringBuilder errorBuffer,
        int errorCapacity
    );

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int YOLOSegCopyDetectionMetadata(
        IntPtr handle,
        IntPtr destination,
        int capacity
    );

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int YOLOSegMarkPrototypeSlotGPUInFlight(
        IntPtr handle,
        int slotIndex,
        ulong generation
    );

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int YOLOSegReleasePrototypeSlot(
        IntPtr handle,
        int slotIndex,
        ulong generation
    );

    internal static CreationResult Create(string modelPath)
    {
        var error = new StringBuilder(ErrorCapacity);
        var handle = YOLOSegCreate(modelPath, error, error.Capacity);
        return new CreationResult(handle, error.ToString());
    }

    // Unity cannot perform a native plugin's first load from the worker thread
    // used for asynchronous model creation, so force that load on the main thread.
    internal static void EnsureLoaded() => YOLOSegGetPrototypeSlotCount(IntPtr.Zero);

    internal static int TryGetOutputInfo(
        IntPtr handle,
        out int width,
        out int height,
        out int personCount,
        out double inferenceMilliseconds,
        out int slotIndex,
        out ulong generation,
        out string message
    )
    {
        var error = new StringBuilder(ErrorCapacity);
        var result = YOLOSegTryGetOutputInfoEx(
            handle,
            out width,
            out height,
            out personCount,
            out inferenceMilliseconds,
            out slotIndex,
            out generation,
            error,
            error.Capacity
        );
        message = error.ToString();
        return result;
    }
#else
    internal static void EnsureLoaded() { }

    internal static CreationResult Create(string modelPath) =>
        new(IntPtr.Zero, "YOLO26 segmentation inference is supported only on macOS.");

    internal static void YOLOSegDestroy(IntPtr handle) { }
    internal static int YOLOSegGetInputWidth(IntPtr handle) => 0;
    internal static int YOLOSegGetInputHeight(IntPtr handle) => 0;
    internal static int YOLOSegGetPrototypeSlotCount(IntPtr handle) => 0;
    internal static int YOLOSegGetPrototypeTextureInfo(
        IntPtr handle,
        int slotIndex,
        out int width,
        out int height,
        out IntPtr nativeTexture
    )
    {
        width = 0;
        height = 0;
        nativeTexture = IntPtr.Zero;
        return -1;
    }
    internal static int YOLOSegCanSubmit(IntPtr handle) => 0;
    internal static int YOLOSegSubmitBGRA(
        IntPtr handle,
        IntPtr bgra,
        int width,
        int height,
        int rowBytes
    ) => -1;

    internal static int YOLOSegCopyDetectionMetadata(
        IntPtr handle,
        IntPtr destination,
        int capacity
    ) => -1;

    internal static int YOLOSegMarkPrototypeSlotGPUInFlight(
        IntPtr handle,
        int slotIndex,
        ulong generation
    ) => -1;

    internal static int YOLOSegReleasePrototypeSlot(
        IntPtr handle,
        int slotIndex,
        ulong generation
    ) => -1;

    internal static int TryGetOutputInfo(
        IntPtr handle,
        out int width,
        out int height,
        out int personCount,
        out double inferenceMilliseconds,
        out int slotIndex,
        out ulong generation,
        out string message
    )
    {
        width = 0;
        height = 0;
        personCount = 0;
        inferenceMilliseconds = 0;
        slotIndex = -1;
        generation = 0;
        message = "YOLO26 segmentation inference is supported only on macOS.";
        return -1;
    }
#endif
}

} // namespace YOLOSeg
