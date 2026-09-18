using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
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
    [SerializeField, HideInInspector] ComputeShader _composeMaskShader = null;

    readonly struct PrototypeLease
    {
        public int SlotIndex { get; }
        public ulong Generation { get; }
        public GraphicsFence Fence { get; }

        public PrototypeLease(int slotIndex, ulong generation, GraphicsFence fence)
        {
            SlotIndex = slotIndex;
            Generation = generation;
            Fence = fence;
        }
    }

    IntPtr _plugin;
    Task<YOLOSegNative.CreationResult> _creationTask;
    WebCamTexture _webcam;
    RenderTexture _inputTexture;
    RenderTexture _segmentationTexture;
    Texture2D[] _prototypeTextures;
    GraphicsBuffer _detectionBuffer;
    YOLOSegNative.DetectionMetadata[] _detectionMetadata;
    readonly List<PrototypeLease> _prototypeLeases = new();
    Material _preprocessMaterial;
    bool _readbackPending;
    bool _disposed;
    int _inputWidth;
    int _inputHeight;
    int _composeKernel = -1;
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
        YOLOSegNative.EnsureLoaded();
        _creationTask = Task.Run(() => YOLOSegNative.Create(modelPath));
    }

    void Update()
    {
        ReleaseCompletedPrototypeSlots();
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

        if (_plugin != IntPtr.Zero) ReleaseAllPrototypeSlots();
        ReleaseTexture(ref _inputTexture);
        ReleaseTexture(ref _segmentationTexture);
        DestroyPrototypeTextures();
        _detectionBuffer?.Release();
        _detectionBuffer = null;
        _detectionMetadata = null;
        Destroy(_preprocessMaterial);
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
        UpdateCameraImage();
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
        UpdateCameraImage();
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
            YOLOSegNative.YOLOSegDestroy(_plugin);
            _plugin = IntPtr.Zero;
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
        if (!CreatePrototypeTextures())
        {
            YOLOSegNative.YOLOSegDestroy(_plugin);
            _plugin = IntPtr.Zero;
            return;
        }
        if (!CreateComputeResources())
        {
            DestroyPrototypeTextures();
            YOLOSegNative.YOLOSegDestroy(_plugin);
            _plugin = IntPtr.Zero;
            return;
        }
        UpdateCameraImage();
        SetStatus(
            $"Ready · {_inputWidth} × {_inputHeight} input · " +
            $"{_prototypeTextures.Length} GPU prototype slots"
        );
    }

    bool CreateComputeResources()
    {
        if (_composeMaskShader == null)
        {
            SetStatus("The mask composition compute shader is missing.");
            return false;
        }

        _composeKernel = _composeMaskShader.FindKernel("ComposeMask");
        _detectionMetadata = new YOLOSegNative.DetectionMetadata[
            YOLOSegNative.DetectionLimit
        ];
        _detectionBuffer = new GraphicsBuffer(
            GraphicsBuffer.Target.Structured,
            YOLOSegNative.DetectionLimit,
            Marshal.SizeOf<YOLOSegNative.DetectionMetadata>()
        );
        return true;
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
        AsyncGPUReadback.Request(_inputTexture, 0, TextureFormat.BGRA32, OnReadback);
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
        var result = YOLOSegNative.YOLOSegSubmitBGRA(
            _plugin,
            pointer,
            _inputWidth,
            _inputHeight,
            _inputWidth * 4
        );
        if (result < 0) SetStatus("Could not submit the camera frame.");
    }

    void ReceiveSegmentation()
    {
        var result = YOLOSegNative.TryGetOutputInfo(
            _plugin,
            out var width,
            out var height,
            out var personCount,
            out var milliseconds,
            out var slotIndex,
            out var generation,
            out var error
        );
        if (result < 0)
        {
            SetStatus($"Inference failed: {error}");
            return;
        }
        if (result == 0) return;

        var gpuSubmitted = false;
        try
        {
            EnsureOutputTexture(width, height);
            if (!CopyDetectionMetadata(personCount))
            {
                SetStatus("Could not copy the detection metadata.");
                return;
            }

            _detectionBuffer.SetData(_detectionMetadata);
            _composeMaskShader.SetTexture(
                _composeKernel,
                "_PrototypeTexture",
                _prototypeTextures[slotIndex]
            );
            _composeMaskShader.SetBuffer(_composeKernel, "_Detections", _detectionBuffer);
            _composeMaskShader.SetTexture(_composeKernel, "_OutputTexture", _segmentationTexture);
            _composeMaskShader.SetInt("_DetectionCount", personCount);
            _composeMaskShader.SetInt("_OutputWidth", width);
            _composeMaskShader.SetInt("_OutputHeight", height);
            _composeMaskShader.Dispatch(
                _composeKernel,
                (width + 7) / 8,
                (height + 7) / 8,
                1
            );
            var fence = Graphics.CreateGraphicsFence(
                GraphicsFenceType.CPUSynchronisation,
                SynchronisationStageFlags.AllGPUOperations
            );
            // The external Texture2D aliases the native slot's IOSurface. Keep the
            // slot leased until this dispatch has finished reading from it.
            _prototypeLeases.Add(new PrototypeLease(slotIndex, generation, fence));
            gpuSubmitted = true;
            if (YOLOSegNative.YOLOSegMarkPrototypeSlotGPUInFlight(
                    _plugin,
                    slotIndex,
                    generation
                ) != 1)
            {
                SetStatus("Could not transfer the GPU prototype slot.");
                return;
            }
            if (_segmentationImage != null) _segmentationImage.image = _segmentationTexture;
            _segmentationImage?.MarkDirtyRepaint();
            SetStatus($"{milliseconds:F1} ms · GPU slot {slotIndex}");
        }
        finally
        {
            if (!gpuSubmitted && slotIndex >= 0)
                YOLOSegNative.YOLOSegReleasePrototypeSlot(_plugin, slotIndex, generation);
        }
    }

    unsafe bool CopyDetectionMetadata(int personCount)
    {
        if (personCount < 0 || personCount > _detectionMetadata.Length) return false;
        fixed (YOLOSegNative.DetectionMetadata *pointer = _detectionMetadata)
        {
            var byteCount = _detectionMetadata.Length *
                            Marshal.SizeOf<YOLOSegNative.DetectionMetadata>();
            return YOLOSegNative.YOLOSegCopyDetectionMetadata(
                _plugin,
                (IntPtr)pointer,
                byteCount
            ) == personCount;
        }
    }

    bool CreatePrototypeTextures()
    {
        if (SystemInfo.graphicsDeviceType != GraphicsDeviceType.Metal)
        {
            SetStatus("The GPU prototype output requires the Metal graphics API.");
            return false;
        }

        var count = YOLOSegNative.YOLOSegGetPrototypeSlotCount(_plugin);
        if (count <= 0)
        {
            SetStatus("The native plugin did not provide GPU prototype slots.");
            return false;
        }

        _prototypeTextures = new Texture2D[count];
        for (var index = 0; index < count; index++)
        {
            var result = YOLOSegNative.YOLOSegGetPrototypeTextureInfo(
                _plugin,
                index,
                out var width,
                out var height,
                out var pointer
            );
            if (result != 1 || width <= 0 || height <= 0 || pointer == IntPtr.Zero)
            {
                DestroyPrototypeTextures();
                SetStatus($"Could not obtain GPU prototype slot {index}.");
                return false;
            }

            _prototypeTextures[index] = Texture2D.CreateExternalTexture(
                width,
                height,
                TextureFormat.RHalf,
                false,
                true,
                pointer
            );
            _prototypeTextures[index].name = $"YOLO26 Prototype Slot {index}";
            _prototypeTextures[index].filterMode = FilterMode.Point;
            _prototypeTextures[index].wrapMode = TextureWrapMode.Clamp;
        }
        return true;
    }

    void DestroyPrototypeTextures()
    {
        if (_prototypeTextures == null) return;
        foreach (var texture in _prototypeTextures) Destroy(texture);
        _prototypeTextures = null;
    }

    void ReleaseCompletedPrototypeSlots()
    {
        if (_plugin == IntPtr.Zero) return;
        for (var index = _prototypeLeases.Count - 1; index >= 0; index--)
        {
            var lease = _prototypeLeases[index];
            if (!lease.Fence.passed) continue;
            YOLOSegNative.YOLOSegReleasePrototypeSlot(
                _plugin,
                lease.SlotIndex,
                lease.Generation
            );
            _prototypeLeases.RemoveAt(index);
        }
    }

    void ReleaseAllPrototypeSlots()
    {
        foreach (var lease in _prototypeLeases)
            YOLOSegNative.YOLOSegReleasePrototypeSlot(
                _plugin,
                lease.SlotIndex,
                lease.Generation
            );
        _prototypeLeases.Clear();
    }

    void EnsureOutputTexture(int width, int height)
    {
        if (_segmentationTexture != null &&
            _segmentationTexture.width == width &&
            _segmentationTexture.height == height)
            return;

        ReleaseTexture(ref _segmentationTexture);
        _segmentationTexture = new RenderTexture(
            width,
            height,
            0,
            RenderTextureFormat.ARGB32,
            RenderTextureReadWrite.Linear
        )
        {
            name = "YOLO26 Person Segmentation",
            filterMode = FilterMode.Bilinear,
            wrapMode = TextureWrapMode.Clamp,
            enableRandomWrite = true
        };
        _segmentationTexture.Create();
        if (_segmentationImage != null) _segmentationImage.image = _segmentationTexture;
    }

    void UpdateCameraImage()
    {
        if (_cameraImage == null) return;
        var preprocessed = _inputTexture != null;
        _cameraImage.image = preprocessed ? _inputTexture : _webcam;
        _cameraImage.uv = preprocessed ? new Rect(0, 1, 1, -1) : new Rect(0, 0, 1, 1);
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
