using System;
using System.Runtime.InteropServices;
using System.Text;

namespace YOLOSeg
{

internal static class YOLOSegNative
{
    const string LibraryName = "YOLOSegPlugin";
    const int ErrorCapacity = 1024;

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
    static extern int YOLOSegTryGetOutputInfo(
        IntPtr handle,
        out int width,
        out int height,
        out int personCount,
        out double inferenceMilliseconds,
        StringBuilder errorBuffer,
        int errorCapacity
    );

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int YOLOSegCopyOutput(
        IntPtr handle,
        IntPtr destination,
        int capacity
    );

    internal static CreationResult Create(string modelPath)
    {
        var error = new StringBuilder(ErrorCapacity);
        var handle = YOLOSegCreate(modelPath, error, error.Capacity);
        return new CreationResult(handle, error.ToString());
    }

    internal static int TryGetOutputInfo(
        IntPtr handle,
        out int width,
        out int height,
        out int personCount,
        out double inferenceMilliseconds,
        out string message
    )
    {
        var error = new StringBuilder(ErrorCapacity);
        var result = YOLOSegTryGetOutputInfo(
            handle,
            out width,
            out height,
            out personCount,
            out inferenceMilliseconds,
            error,
            error.Capacity
        );
        message = error.ToString();
        return result;
    }
#else
    internal static CreationResult Create(string modelPath) =>
        new(IntPtr.Zero, "YOLO26 segmentation inference is supported only on macOS.");

    internal static void YOLOSegDestroy(IntPtr handle) { }
    internal static int YOLOSegGetInputWidth(IntPtr handle) => 0;
    internal static int YOLOSegGetInputHeight(IntPtr handle) => 0;
    internal static int YOLOSegCanSubmit(IntPtr handle) => 0;
    internal static int YOLOSegSubmitBGRA(IntPtr handle, IntPtr bgra, int width, int height, int rowBytes) => -1;
    internal static int YOLOSegCopyOutput(IntPtr handle, IntPtr destination, int capacity) => -1;

    internal static int TryGetOutputInfo(
        IntPtr handle,
        out int width,
        out int height,
        out int personCount,
        out double inferenceMilliseconds,
        out string message
    )
    {
        width = 0;
        height = 0;
        personCount = 0;
        inferenceMilliseconds = 0;
        message = "YOLO26 segmentation inference is supported only on macOS.";
        return -1;
    }
#endif
}

} // namespace YOLOSeg
