using System;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using UnityEditor;
using UnityEditor.Rendering;
using UnityEngine;
using UnityEngine.UIElements;
using Debug = UnityEngine.Debug;

namespace ProjectBootstrap
{

public static class ProjectValidator
{
    const string LibraryName = "YOLOSegPlugin";
    const int ErrorCapacity = 1024;

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    static extern IntPtr YOLOSegCreate(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string modelPath,
        StringBuilder errorBuffer,
        int errorCapacity
    );

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    static extern void YOLOSegDestroy(IntPtr handle);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    static extern int YOLOSegGetInputWidth(IntPtr handle);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    static extern int YOLOSegGetInputHeight(IntPtr handle);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    static extern int YOLOSegSubmitRGBA(
        IntPtr handle,
        IntPtr rgba,
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
    static extern int YOLOSegCopyOutput(IntPtr handle, IntPtr destination, int capacity);

    public static void Validate()
    {
        try
        {
            ValidateUI();
            ValidateShader();
            ValidateNativePlugin();
            Debug.Log("[ProjectValidator] All checks passed.");
            EditorApplication.Exit(0);
        }
        catch (Exception exception)
        {
            Debug.LogException(exception);
            EditorApplication.Exit(1);
        }
    }

    static void ValidateUI()
    {
        var tree = AssetDatabase.LoadAssetAtPath<VisualTreeAsset>("Assets/UI/Main.uxml");
        if (tree == null) throw new InvalidOperationException("Main.uxml could not be loaded.");
        var root = tree.Instantiate();
        foreach (var name in new[] { "cameraImage", "segmentationImage", "statusLabel" })
            if (root.Q(name) == null)
                throw new InvalidOperationException($"UI element '{name}' is missing.");
    }

    static void ValidateShader()
    {
        const string path = "Assets/YOLOSeg/Shaders/Preprocess.shader";
        var shader = AssetDatabase.LoadAssetAtPath<Shader>(path);
        if (shader == null) throw new InvalidOperationException($"{path} could not be loaded.");
        foreach (var message in ShaderUtil.GetShaderMessages(shader))
            if (message.severity == ShaderCompilerMessageSeverity.Error)
                throw new InvalidOperationException($"{path}: {message.message}");
    }

    static void ValidateNativePlugin()
    {
        var modelPath = Path.Combine(
            Application.streamingAssetsPath,
            "Models/yolo26n-seg.mlpackage"
        );
        var error = new StringBuilder(ErrorCapacity);
        var stopwatch = Stopwatch.StartNew();
        var handle = YOLOSegCreate(modelPath, error, error.Capacity);
        stopwatch.Stop();
        if (handle == IntPtr.Zero)
            throw new InvalidOperationException($"Native model load failed: {error}");
        try
        {
            var width = YOLOSegGetInputWidth(handle);
            var height = YOLOSegGetInputHeight(handle);
            if (width != 640 || height != 640)
                throw new InvalidOperationException($"Unexpected model input: {width} x {height}.");
            Debug.Log($"[ProjectValidator] Core ML model loaded in {stopwatch.ElapsedMilliseconds} ms.");
            ValidateInference(handle, width, height);
        }
        finally
        {
            YOLOSegDestroy(handle);
        }
    }

    static void ValidateInference(IntPtr handle, int width, int height)
    {
        var pixels = new byte[width * height * 4];
        for (var y = 0; y < height; y++)
        for (var x = 0; x < width; x++)
        {
            var index = (y * width + x) * 4;
            pixels[index + 0] = (byte)(x * 255 / (width - 1));
            pixels[index + 1] = (byte)(y * 255 / (height - 1));
            pixels[index + 2] = 128;
            pixels[index + 3] = 255;
        }

        var pixelPin = GCHandle.Alloc(pixels, GCHandleType.Pinned);
        try
        {
            if (YOLOSegSubmitRGBA(handle, pixelPin.AddrOfPinnedObject(), width, height, width * 4) != 1)
                throw new InvalidOperationException("The synthetic frame could not be submitted.");
        }
        finally
        {
            pixelPin.Free();
        }

        var timeout = Stopwatch.StartNew();
        int result;
        int outputWidth;
        int outputHeight;
        int personCount;
        double milliseconds;
        var error = new StringBuilder(ErrorCapacity);
        do
        {
            result = YOLOSegTryGetOutputInfo(
                handle,
                out outputWidth,
                out outputHeight,
                out personCount,
                out milliseconds,
                error,
                error.Capacity
            );
            if (result == 0) Thread.Sleep(10);
        }
        while (result == 0 && timeout.Elapsed.TotalSeconds < 30);

        if (result < 0) throw new InvalidOperationException($"Native inference failed: {error}");
        if (result == 0) throw new TimeoutException("Native inference timed out.");
        if (outputWidth <= 0 || outputHeight <= 0)
            throw new InvalidOperationException("The segmentation output has an invalid size.");

        var output = new byte[outputWidth * outputHeight * 4];
        var outputPin = GCHandle.Alloc(output, GCHandleType.Pinned);
        try
        {
            if (YOLOSegCopyOutput(handle, outputPin.AddrOfPinnedObject(), output.Length) != 1)
                throw new InvalidOperationException("The segmentation output could not be copied.");
        }
        finally
        {
            outputPin.Free();
        }

        for (var index = 3; index < output.Length; index += 4)
            if (output[index] != 255)
                throw new InvalidOperationException("The segmentation output contains invalid pixels.");
        Debug.Log(
            $"[ProjectValidator] Inference produced {outputWidth} x {outputHeight} segmentation " +
            $"({personCount} person(s)) in {milliseconds:F1} ms."
        );
    }
}

} // namespace ProjectBootstrap
