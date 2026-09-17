using System;
using System.Collections;
using System.IO;
using System.Threading.Tasks;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.UIElements;

namespace YOLOSeg
{

[RequireComponent(typeof(PanelRenderer))]
public sealed class SegDemoController : MonoBehaviour
{
    [SerializeField, HideInInspector] Shader _preprocessShader = null;

    IntPtr _plugin;
    Task<YOLOSegNative.CreationResult> _creationTask;
    WebCamTexture _webcam;
    RenderTexture _inputTexture;
    Texture2D _segmentationTexture;
    Material _preprocessMaterial;
    bool _readbackPending;
    bool _disposed;
    int _inputWidth;
    int _inputHeight;
    int _uiVersion = -1;
    string _statusMessage;

    PanelRenderer _panelRenderer;
    Image _cameraImage;
    Image _segmentationImage;
    Label _statusLabel;

    void OnEnable()
    {
        _disposed = false;
        _uiVersion = -1;
        _panelRenderer = GetComponent<PanelRenderer>();
        _panelRenderer.RegisterUIReloadCallback(OnUIReload);
        CreateMaterial();
        StartCoroutine(StartCamera());

        var modelPath = Path.Combine(
            Application.streamingAssetsPath,
            "Models/yolo26n-seg.mlpackage"
        );
        SetStatus("Loading YOLO26 segmentation model…");
        _creationTask = Task.Run(() => YOLOSegNative.Create(modelPath));
    }

    void Update()
    {
        CompleteInitialization();
        if (_plugin == IntPtr.Zero) return;

        ReceiveSegmentation();
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
        Destroy(_segmentationTexture);
        Destroy(_preprocessMaterial);
        _segmentationTexture = null;
        _preprocessMaterial = null;

        if (_plugin != IntPtr.Zero)
        {
            YOLOSegNative.YOLOSegDestroy(_plugin);
            _plugin = IntPtr.Zero;
        }

        if (_creationTask == null) return;
        var task = _creationTask;
        _creationTask = null;
        task.ContinueWith(completed =>
        {
            if (completed.Status != TaskStatus.RanToCompletion) return;
            if (completed.Result.Handle != IntPtr.Zero)
                YOLOSegNative.YOLOSegDestroy(completed.Result.Handle);
        });
    }

    void OnUIReload(PanelRenderer renderer, VisualElement root, int version)
    {
        if (root == null || version == _uiVersion) return;
        _uiVersion = version;

        UnbindUI();
        _cameraImage = root.Q<Image>("cameraImage");
        _segmentationImage = root.Q<Image>("segmentationImage");
        _statusLabel = root.Q<Label>("statusLabel");

        _cameraImage.scaleMode = ScaleMode.ScaleToFit;
        _segmentationImage.scaleMode = ScaleMode.ScaleToFit;
        _cameraImage.image = _inputTexture != null ? _inputTexture : _webcam;
        _segmentationImage.image = _segmentationTexture;
        SetStatus(_statusMessage);
    }

    void UnbindUI()
    {
        _cameraImage = null;
        _segmentationImage = null;
        _statusLabel = null;
    }

    void CreateMaterial()
    {
        if (_preprocessShader != null) _preprocessMaterial = new Material(_preprocessShader);
        if (_preprocessMaterial == null) SetStatus("The preprocessing shader could not be loaded.");
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
            if (result.Handle != IntPtr.Zero) YOLOSegNative.YOLOSegDestroy(result.Handle);
            return;
        }
        if (result.Handle == IntPtr.Zero)
        {
            SetStatus($"Model load failed: {result.Error}");
            return;
        }

        _plugin = result.Handle;
        _inputWidth = YOLOSegNative.YOLOSegGetInputWidth(_plugin);
        _inputHeight = YOLOSegNative.YOLOSegGetInputHeight(_plugin);
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
            name = "YOLO26 Segmentation Input",
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
        if (_readbackPending || YOLOSegNative.YOLOSegCanSubmit(_plugin) == 0) return;
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
        var result = YOLOSegNative.YOLOSegSubmitRGBA(
            _plugin,
            pointer,
            _inputWidth,
            _inputHeight,
            _inputWidth * 4
        );
        if (result < 0) SetStatus("Could not submit the camera frame.");
    }

    unsafe void ReceiveSegmentation()
    {
        var result = YOLOSegNative.TryGetOutputInfo(
            _plugin,
            out var width,
            out var height,
            out var personCount,
            out var milliseconds,
            out var error
        );
        if (result < 0)
        {
            SetStatus($"Inference failed: {error}");
            return;
        }
        if (result == 0) return;

        EnsureOutputTexture(width, height);
        var pixels = _segmentationTexture.GetRawTextureData<byte>();
        var pointer = (IntPtr)NativeArrayUnsafeUtility.GetUnsafePtr(pixels);
        if (YOLOSegNative.YOLOSegCopyOutput(_plugin, pointer, pixels.Length) != 1)
        {
            SetStatus("Could not copy the segmentation output.");
            return;
        }

        _segmentationTexture.Apply(false, false);
        if (_segmentationImage != null) _segmentationImage.image = _segmentationTexture;
        _segmentationImage?.MarkDirtyRepaint();
        SetStatus($"{milliseconds:F1} ms · {personCount} person(s)");
    }

    void EnsureOutputTexture(int width, int height)
    {
        if (_segmentationTexture != null &&
            _segmentationTexture.width == width &&
            _segmentationTexture.height == height)
            return;

        Destroy(_segmentationTexture);
        _segmentationTexture = new Texture2D(width, height, TextureFormat.RGBA32, false, true)
        {
            name = "YOLO26 Person Segmentation",
            filterMode = FilterMode.Bilinear,
            wrapMode = TextureWrapMode.Clamp
        };
        if (_segmentationImage != null) _segmentationImage.image = _segmentationTexture;
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

} // namespace YOLOSeg
