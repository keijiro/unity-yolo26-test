using System;
using System.Runtime.InteropServices;
using System.Text;

namespace YOLODepth
{

internal static class YOLODepthNative
{
    const string LibraryName = "YOLODepthPlugin";
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
    static extern IntPtr YOLODepthCreate(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string modelPath,
        StringBuilder errorBuffer,
        int errorCapacity
    );

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void YOLODepthDestroy(IntPtr handle);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int YOLODepthGetInputWidth(IntPtr handle);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int YOLODepthGetInputHeight(IntPtr handle);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int YOLODepthCanSubmit(IntPtr handle);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int YOLODepthSubmitRGBA(
        IntPtr handle,
        IntPtr rgba,
        int width,
        int height,
        int rowBytes
    );

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int YOLODepthTryGetOutputInfo(
        IntPtr handle,
        out int width,
        out int height,
        out double inferenceMilliseconds,
        StringBuilder errorBuffer,
        int errorCapacity
    );

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int YOLODepthCopyOutput(
        IntPtr handle,
        IntPtr destination,
        int capacity
    );

    internal static CreationResult Create(string modelPath)
    {
        var error = new StringBuilder(ErrorCapacity);
        var handle = YOLODepthCreate(modelPath, error, error.Capacity);
        return new CreationResult(handle, error.ToString());
    }

    internal static int TryGetOutputInfo(
        IntPtr handle,
        out int width,
        out int height,
        out double inferenceMilliseconds,
        out string message
    )
    {
        var error = new StringBuilder(ErrorCapacity);
        var result = YOLODepthTryGetOutputInfo(
            handle,
            out width,
            out height,
            out inferenceMilliseconds,
            error,
            error.Capacity
        );
        message = error.ToString();
        return result;
    }
#else
    internal static CreationResult Create(string modelPath) =>
        new(IntPtr.Zero, "YOLO26 depth inference is supported only on macOS.");

    internal static void YOLODepthDestroy(IntPtr handle) { }
    internal static int YOLODepthGetInputWidth(IntPtr handle) => 0;
    internal static int YOLODepthGetInputHeight(IntPtr handle) => 0;
    internal static int YOLODepthCanSubmit(IntPtr handle) => 0;
    internal static int YOLODepthSubmitRGBA(IntPtr handle, IntPtr rgba, int width, int height, int rowBytes) => -1;
    internal static int YOLODepthCopyOutput(IntPtr handle, IntPtr destination, int capacity) => -1;

    internal static int TryGetOutputInfo(
        IntPtr handle,
        out int width,
        out int height,
        out double inferenceMilliseconds,
        out string message
    )
    {
        width = 0;
        height = 0;
        inferenceMilliseconds = 0;
        message = "YOLO26 depth inference is supported only on macOS.";
        return -1;
    }
#endif
}

} // namespace YOLODepth
