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
    const string LibraryName = "YOLODepthPlugin";
    const int ErrorCapacity = 1024;

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    static extern IntPtr YOLODepthCreate(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string modelPath,
        StringBuilder errorBuffer,
        int errorCapacity
    );

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    static extern void YOLODepthDestroy(IntPtr handle);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    static extern int YOLODepthGetInputWidth(IntPtr handle);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    static extern int YOLODepthGetInputHeight(IntPtr handle);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    static extern int YOLODepthSubmitRGBA(
        IntPtr handle,
        IntPtr rgba,
        int width,
        int height,
        int rowBytes
    );

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    static extern int YOLODepthTryGetOutputInfo(
        IntPtr handle,
        out int width,
        out int height,
        out double inferenceMilliseconds,
        StringBuilder errorBuffer,
        int errorCapacity
    );

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    static extern int YOLODepthCopyOutput(IntPtr handle, IntPtr destination, int capacity);

    public static void Validate()
    {
        try
        {
            ValidateUI();
            ValidateShaders();
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
        foreach (var name in new[]
                 {
                     "cameraImage", "depthImage", "statusLabel",
                     "minimumSlider", "maximumSlider"
                 })
            if (root.Q(name) == null)
                throw new InvalidOperationException($"UI element '{name}' is missing.");
    }

    static void ValidateShaders()
    {
        foreach (var path in new[]
                 {
                     "Assets/YOLODepth/Shaders/Preprocess.shader",
                     "Assets/YOLODepth/Shaders/VisualizeDepth.shader"
                 })
        {
            var shader = AssetDatabase.LoadAssetAtPath<Shader>(path);
            if (shader == null) throw new InvalidOperationException($"{path} could not be loaded.");
            var messages = ShaderUtil.GetShaderMessages(shader);
            foreach (var message in messages)
                if (message.severity == ShaderCompilerMessageSeverity.Error)
                    throw new InvalidOperationException($"{path}: {message.message}");
        }
    }

    static void ValidateNativePlugin()
    {
        var modelPath = Path.Combine(
            Application.streamingAssetsPath,
            "Models/yolo26n-depth.mlpackage"
        );
        var error = new StringBuilder(ErrorCapacity);
        var stopwatch = Stopwatch.StartNew();
        var handle = YOLODepthCreate(modelPath, error, error.Capacity);
        stopwatch.Stop();
        if (handle == IntPtr.Zero)
            throw new InvalidOperationException($"Native model load failed: {error}");
        try
        {
            var width = YOLODepthGetInputWidth(handle);
            var height = YOLODepthGetInputHeight(handle);
            if (width != 640 || height != 640)
                throw new InvalidOperationException($"Unexpected model input: {width} x {height}.");
            Debug.Log($"[ProjectValidator] Core ML model loaded in {stopwatch.ElapsedMilliseconds} ms.");
            ValidateInference(handle, width, height);
        }
        finally
        {
            YOLODepthDestroy(handle);
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
            if (YOLODepthSubmitRGBA(handle, pixelPin.AddrOfPinnedObject(), width, height, width * 4) != 1)
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
        double milliseconds;
        var error = new StringBuilder(ErrorCapacity);
        do
        {
            result = YOLODepthTryGetOutputInfo(
                handle,
                out outputWidth,
                out outputHeight,
                out milliseconds,
                error,
                error.Capacity
            );
            if (result == 0) Thread.Sleep(10);
        }
        while (result == 0 && timeout.Elapsed.TotalSeconds < 30);

        if (result < 0) throw new InvalidOperationException($"Native inference failed: {error}");
        if (result == 0) throw new TimeoutException("Native inference timed out.");

        var output = new float[outputWidth * outputHeight];
        var outputPin = GCHandle.Alloc(output, GCHandleType.Pinned);
        try
        {
            if (YOLODepthCopyOutput(handle, outputPin.AddrOfPinnedObject(), output.Length) != 1)
                throw new InvalidOperationException("The depth output could not be copied.");
        }
        finally
        {
            outputPin.Free();
        }

        var finiteCount = 0;
        foreach (var value in output)
            if (!float.IsNaN(value) && !float.IsInfinity(value)) finiteCount++;
        if (finiteCount != output.Length)
            throw new InvalidOperationException("The depth output contains non-finite values.");
        Debug.Log(
            $"[ProjectValidator] Inference produced {outputWidth} x {outputHeight} depth in " +
            $"{milliseconds:F1} ms."
        );
    }
}

} // namespace ProjectBootstrap
