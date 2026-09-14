using System;
using System.Collections;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.UIElements;

namespace YOLODepth
{

[RequireComponent(typeof(UIDocument))]
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
    byte[] _readbackData;
    float[] _depthData;
    bool _readbackPending;
    bool _disposed;
    int _inputWidth;
    int _inputHeight;
    float _minimumDepth = 0.3f;
    float _maximumDepth = 2;

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
        var document = FindFirstObjectByType<UIDocument>();
        if (document != null && document.GetComponent<DepthDemoController>() == null)
            document.gameObject.AddComponent<DepthDemoController>();
    }

    void OnEnable()
    {
        _disposed = false;
        BindUI();
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

    void BindUI()
    {
        var root = GetComponent<UIDocument>().rootVisualElement;
        _cameraImage = root.Q<Image>("cameraImage");
        _depthImage = root.Q<Image>("depthImage");
        _statusLabel = root.Q<Label>("statusLabel");
        _minimumLabel = root.Q<Label>("minimumValue");
        _maximumLabel = root.Q<Label>("maximumValue");
        _minimumSlider = root.Q<Slider>("minimumSlider");
        _maximumSlider = root.Q<Slider>("maximumSlider");

        _cameraImage.scaleMode = ScaleMode.ScaleToFit;
        _depthImage.scaleMode = ScaleMode.ScaleToFit;
        ConfigureSlider(_minimumSlider, _minimumDepth);
        ConfigureSlider(_maximumSlider, _maximumDepth);
        _minimumSlider.RegisterValueChangedCallback(OnMinimumChanged);
        _maximumSlider.RegisterValueChangedCallback(OnMaximumChanged);
        UpdateRangeLabels();
    }

    void UnbindUI()
    {
        _minimumSlider?.UnregisterValueChangedCallback(OnMinimumChanged);
        _maximumSlider?.UnregisterValueChangedCallback(OnMaximumChanged);
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
        _cameraImage.image = _webcam;
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
        _cameraImage.image = _inputTexture;
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

    void OnReadback(AsyncGPUReadbackRequest request)
    {
        _readbackPending = false;
        if (_disposed || _plugin == IntPtr.Zero || request.hasError)
        {
            if (request.hasError) SetStatus("GPU readback failed.");
            return;
        }

        var source = request.GetData<byte>();
        if (_readbackData == null || _readbackData.Length != source.Length)
            _readbackData = new byte[source.Length];
        source.CopyTo(_readbackData);

        var pin = GCHandle.Alloc(_readbackData, GCHandleType.Pinned);
        try
        {
            var result = YOLODepthNative.YOLODepthSubmitRGBA(
                _plugin,
                pin.AddrOfPinnedObject(),
                _inputWidth,
                _inputHeight,
                _inputWidth * 4
            );
            if (result < 0) SetStatus("Could not submit the camera frame.");
        }
        finally
        {
            pin.Free();
        }
    }

    void ReceiveDepth()
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

        var count = width * height;
        if (_depthData == null || _depthData.Length != count)
            _depthData = new float[count];
        var pin = GCHandle.Alloc(_depthData, GCHandleType.Pinned);
        try
        {
            if (YOLODepthNative.YOLODepthCopyOutput(_plugin, pin.AddrOfPinnedObject(), count) != 1)
            {
                SetStatus("Could not copy the depth output.");
                return;
            }
        }
        finally
        {
            pin.Free();
        }

        EnsureOutputTextures(width, height);
        _depthTexture.SetPixelData(_depthData, 0);
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
        _depthImage.image = _visualizedTexture;
    }

    void RenderDepth()
    {
        if (_visualizeMaterial == null || _depthTexture == null || _visualizedTexture == null)
            return;
        _visualizeMaterial.SetFloat("_MinimumDepth", _minimumDepth);
        _visualizeMaterial.SetFloat("_MaximumDepth", _maximumDepth);
        Graphics.Blit(_depthTexture, _visualizedTexture, _visualizeMaterial);
        _depthImage.MarkDirtyRepaint();
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
        _minimumLabel.text = $"{_minimumDepth:F1} m";
        _maximumLabel.text = $"{_maximumDepth:F1} m";
    }

    void SetStatus(string message)
    {
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
