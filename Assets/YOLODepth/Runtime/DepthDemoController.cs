using System;
using System.Collections;
using System.IO;
using System.Threading.Tasks;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.UIElements;

namespace YOLODepth
{

[RequireComponent(typeof(PanelRenderer))]
public sealed class DepthDemoController : MonoBehaviour
{
    const float MinimumLimit = 0.1f;
    const float MaximumLimit = 10;
    const float MinimumSpan = 0.1f;

    IntPtr _plugin;
    Task<YOLODepthNative.CreationResult> _creationTask;
    WebCamTexture _webcam;
    RenderTexture _inputTexture;
    RenderTexture _visualizedTexture;
    Texture2D _depthTexture;
    Material _preprocessMaterial;
    Material _visualizeMaterial;
    bool _readbackPending;
    bool _disposed;
    int _inputWidth;
    int _inputHeight;
    int _uiVersion = -1;
    float _minimumDepth = 0.3f;
    float _maximumDepth = 2;
    string _statusMessage;

    PanelRenderer _panelRenderer;
    Image _cameraImage;
    Image _depthImage;
    Label _statusLabel;
    Label _minimumLabel;
    Label _maximumLabel;
    Slider _minimumSlider;
    Slider _maximumSlider;

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    static void Initialize()
    {
        var renderer = FindAnyObjectByType<PanelRenderer>();
        if (renderer != null && renderer.GetComponent<DepthDemoController>() == null)
            renderer.gameObject.AddComponent<DepthDemoController>();
    }

    void OnEnable()
    {
        _disposed = false;
        _uiVersion = -1;
        _panelRenderer = GetComponent<PanelRenderer>();
        _panelRenderer.RegisterUIReloadCallback(OnUIReload);
        CreateMaterials();
        StartCoroutine(StartCamera());

        var modelPath = Path.Combine(
            Application.streamingAssetsPath,
            "Models/yolo26n-depth.mlpackage"
        );
        SetStatus("Loading YOLO26 Core ML model…");
        _creationTask = Task.Run(() => YOLODepthNative.Create(modelPath));
    }

    void Update()
    {
        CompleteInitialization();
        if (_plugin == IntPtr.Zero) return;

        ReceiveDepth();
        ScheduleFrame();
    }

    void OnDisable()
    {
        _disposed = true;
        StopAllCoroutines();
        if (_panelRenderer != null)
            _panelRenderer.UnregisterUIReloadCallback(OnUIReload);
        _panelRenderer = null;
        UnbindUI();

        if (_webcam != null) _webcam.Stop();
        _webcam = null;
        ReleaseTexture(ref _inputTexture);
        ReleaseTexture(ref _visualizedTexture);
        Destroy(_depthTexture);
        Destroy(_preprocessMaterial);
        Destroy(_visualizeMaterial);
        _depthTexture = null;
        _preprocessMaterial = null;
        _visualizeMaterial = null;

        if (_plugin != IntPtr.Zero)
        {
            YOLODepthNative.YOLODepthDestroy(_plugin);
            _plugin = IntPtr.Zero;
        }

        if (_creationTask == null) return;
        var task = _creationTask;
        _creationTask = null;
        task.ContinueWith(completed =>
        {
            if (completed.Status != TaskStatus.RanToCompletion) return;
            if (completed.Result.Handle != IntPtr.Zero)
                YOLODepthNative.YOLODepthDestroy(completed.Result.Handle);
        });
    }

    void OnUIReload(PanelRenderer renderer, VisualElement root, int version)
    {
        if (root == null || version == _uiVersion) return;
        _uiVersion = version;

        UnbindUI();
        _cameraImage = root.Q<Image>("cameraImage");
        _depthImage = root.Q<Image>("depthImage");
        _statusLabel = root.Q<Label>("statusLabel");
        _minimumLabel = root.Q<Label>("minimumValue");
        _maximumLabel = root.Q<Label>("maximumValue");
        _minimumSlider = root.Q<Slider>("minimumSlider");
        _maximumSlider = root.Q<Slider>("maximumSlider");

        _cameraImage.scaleMode = ScaleMode.ScaleToFit;
        _depthImage.scaleMode = ScaleMode.ScaleToFit;
        _cameraImage.image = _inputTexture != null ? _inputTexture : _webcam;
        _depthImage.image = _visualizedTexture;
        ConfigureSlider(_minimumSlider, _minimumDepth);
        ConfigureSlider(_maximumSlider, _maximumDepth);
        _minimumSlider.RegisterValueChangedCallback(OnMinimumChanged);
        _maximumSlider.RegisterValueChangedCallback(OnMaximumChanged);
        UpdateRangeLabels();
        SetStatus(_statusMessage);
    }

    void UnbindUI()
    {
        _minimumSlider?.UnregisterValueChangedCallback(OnMinimumChanged);
        _maximumSlider?.UnregisterValueChangedCallback(OnMaximumChanged);
        _cameraImage = null;
        _depthImage = null;
        _statusLabel = null;
        _minimumLabel = null;
        _maximumLabel = null;
        _minimumSlider = null;
        _maximumSlider = null;
    }

    static void ConfigureSlider(Slider slider, float value)
    {
        slider.lowValue = MinimumLimit;
        slider.highValue = MaximumLimit;
        slider.SetValueWithoutNotify(value);
    }

    void CreateMaterials()
    {
        var preprocess = Resources.Load<Shader>("YOLODepth/Preprocess");
        var visualize = Resources.Load<Shader>("YOLODepth/VisualizeDepth");
        if (preprocess != null) _preprocessMaterial = new Material(preprocess);
        if (visualize != null) _visualizeMaterial = new Material(visualize);
        if (_preprocessMaterial == null || _visualizeMaterial == null)
            SetStatus("Required shaders could not be loaded.");
    }

    IEnumerator StartCamera()
    {
        if (!Application.HasUserAuthorization(UserAuthorization.WebCam))
            yield return Application.RequestUserAuthorization(UserAuthorization.WebCam);
        if (!Application.HasUserAuthorization(UserAuthorization.WebCam))
        {
            SetStatus("Camera access was denied.");
            yield break;
        }

        _webcam = new WebCamTexture(1280, 720, 30);
        _webcam.Play();
        if (_cameraImage != null) _cameraImage.image = _webcam;
    }

    void CompleteInitialization()
    {
        if (_creationTask == null || !_creationTask.IsCompleted) return;

        if (_creationTask.IsFaulted)
        {
            SetStatus($"Model load failed: {_creationTask.Exception?.GetBaseException().Message}");
            _creationTask = null;
            return;
        }

        var result = _creationTask.Result;
        _creationTask = null;
        if (_disposed)
        {
            if (result.Handle != IntPtr.Zero) YOLODepthNative.YOLODepthDestroy(result.Handle);
            return;
        }
        if (result.Handle == IntPtr.Zero)
        {
            SetStatus($"Model load failed: {result.Error}");
            return;
        }

        _plugin = result.Handle;
        _inputWidth = YOLODepthNative.YOLODepthGetInputWidth(_plugin);
        _inputHeight = YOLODepthNative.YOLODepthGetInputHeight(_plugin);
        if (_inputWidth <= 0 || _inputHeight <= 0)
        {
            SetStatus("The model reported an invalid input size.");
            return;
        }

        _inputTexture = new RenderTexture(
            _inputWidth,
            _inputHeight,
            0,
            RenderTextureFormat.ARGB32,
            RenderTextureReadWrite.sRGB
        )
        {
            name = "YOLO26 Input",
            filterMode = FilterMode.Bilinear,
            wrapMode = TextureWrapMode.Clamp
        };
        _inputTexture.Create();
        if (_cameraImage != null) _cameraImage.image = _inputTexture;
        SetStatus($"Ready · {_inputWidth} × {_inputHeight} input");
    }

    void ScheduleFrame()
    {
        if (_webcam == null || !_webcam.isPlaying || !_webcam.didUpdateThisFrame) return;
        if (_webcam.width <= 16 || _webcam.height <= 16) return;
        if (_readbackPending || YOLODepthNative.YOLODepthCanSubmit(_plugin) == 0) return;
        if (_preprocessMaterial == null) return;

        _preprocessMaterial.SetVector(
            "_SourceSize",
            new Vector4(_webcam.width, _webcam.height, 0, 0)
        );
        _preprocessMaterial.SetFloat("_Rotation", _webcam.videoRotationAngle);
        _preprocessMaterial.SetFloat("_MirrorY", _webcam.videoVerticallyMirrored ? 1 : 0);
        Graphics.Blit(_webcam, _inputTexture, _preprocessMaterial);

        _readbackPending = true;
        AsyncGPUReadback.Request(_inputTexture, 0, TextureFormat.RGBA32, OnReadback);
    }

    unsafe void OnReadback(AsyncGPUReadbackRequest request)
    {
        _readbackPending = false;
        if (_disposed || _plugin == IntPtr.Zero || request.hasError)
        {
            if (request.hasError) SetStatus("GPU readback failed.");
            return;
        }

        var source = request.GetData<byte>();
        var pointer = (IntPtr)NativeArrayUnsafeUtility.GetUnsafeReadOnlyPtr(source);
        var result = YOLODepthNative.YOLODepthSubmitRGBA(
            _plugin,
            pointer,
            _inputWidth,
            _inputHeight,
            _inputWidth * 4
        );
        if (result < 0) SetStatus("Could not submit the camera frame.");
    }

    unsafe void ReceiveDepth()
    {
        var result = YOLODepthNative.TryGetOutputInfo(
            _plugin,
            out var width,
            out var height,
            out var milliseconds,
            out var error
        );
        if (result < 0)
        {
            SetStatus($"Inference failed: {error}");
            return;
        }
        if (result == 0) return;

        EnsureOutputTextures(width, height);
        var pixels = _depthTexture.GetRawTextureData<float>();
        var pointer = (IntPtr)NativeArrayUnsafeUtility.GetUnsafePtr(pixels);
        if (YOLODepthNative.YOLODepthCopyOutput(_plugin, pointer, pixels.Length) != 1)
        {
            SetStatus("Could not copy the depth output.");
            return;
        }

        _depthTexture.Apply(false, false);
        RenderDepth();
        SetStatus($"Running · {milliseconds:F1} ms inference · {width} × {height} depth");
    }

    void EnsureOutputTextures(int width, int height)
    {
        if (_depthTexture != null && _depthTexture.width == width && _depthTexture.height == height)
            return;

        Destroy(_depthTexture);
        ReleaseTexture(ref _visualizedTexture);
        _depthTexture = new Texture2D(width, height, TextureFormat.RFloat, false, true)
        {
            name = "YOLO26 Raw Depth",
            filterMode = FilterMode.Bilinear,
            wrapMode = TextureWrapMode.Clamp
        };
        _visualizedTexture = new RenderTexture(
            width,
            height,
            0,
            RenderTextureFormat.ARGB32,
            RenderTextureReadWrite.sRGB
        )
        {
            name = "YOLO26 Visualized Depth",
            filterMode = FilterMode.Bilinear,
            wrapMode = TextureWrapMode.Clamp
        };
        _visualizedTexture.Create();
        if (_depthImage != null) _depthImage.image = _visualizedTexture;
    }

    void RenderDepth()
    {
        if (_visualizeMaterial == null || _depthTexture == null || _visualizedTexture == null)
            return;
        _visualizeMaterial.SetFloat("_MinimumDepth", _minimumDepth);
        _visualizeMaterial.SetFloat("_MaximumDepth", _maximumDepth);
        Graphics.Blit(_depthTexture, _visualizedTexture, _visualizeMaterial);
        _depthImage?.MarkDirtyRepaint();
    }

    void OnMinimumChanged(ChangeEvent<float> evt)
    {
        _minimumDepth = Mathf.Min(evt.newValue, _maximumDepth - MinimumSpan);
        _minimumSlider.SetValueWithoutNotify(_minimumDepth);
        UpdateRangeLabels();
        RenderDepth();
    }

    void OnMaximumChanged(ChangeEvent<float> evt)
    {
        _maximumDepth = Mathf.Max(evt.newValue, _minimumDepth + MinimumSpan);
        _maximumSlider.SetValueWithoutNotify(_maximumDepth);
        UpdateRangeLabels();
        RenderDepth();
    }

    void UpdateRangeLabels()
    {
        if (_minimumLabel != null) _minimumLabel.text = $"{_minimumDepth:F1} m";
        if (_maximumLabel != null) _maximumLabel.text = $"{_maximumDepth:F1} m";
    }

    void SetStatus(string message)
    {
        _statusMessage = message;
        if (_statusLabel != null) _statusLabel.text = message;
    }

    static void ReleaseTexture(ref RenderTexture texture)
    {
        if (texture == null) return;
        texture.Release();
        Destroy(texture);
        texture = null;
    }
}

} // namespace YOLODepth
