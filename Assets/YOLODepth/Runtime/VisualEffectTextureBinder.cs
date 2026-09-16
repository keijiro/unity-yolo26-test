using UnityEngine;
using UnityEngine.VFX;

namespace YOLODepth
{

[RequireComponent(typeof(VisualEffect))]
public sealed class VisualEffectTextureBinder : MonoBehaviour
{
    [SerializeField] TextureSource _source = null;
    [SerializeField] string _propertyName = "Texture";

    VisualEffect _target;
    int _propertyID;
    bool _missingPropertyReported;
    bool _applyCurrentTexture;

    void OnEnable()
    {
        _target = GetComponent<VisualEffect>();
        _propertyID = Shader.PropertyToID(_propertyName);

        if (_source == null)
        {
            Debug.LogError("A texture source has not been assigned.", this);
            return;
        }

        _source.TextureChanged += OnTextureChanged;
        _applyCurrentTexture = true;
    }

    void LateUpdate()
    {
        if (!_applyCurrentTexture) return;

        _applyCurrentTexture = false;
        ApplyTexture(_source.GetTexture(_propertyName));
    }

    void OnDisable()
    {
        if (_source != null) _source.TextureChanged -= OnTextureChanged;
        _applyCurrentTexture = false;
        _target = null;
    }

    void OnTextureChanged(string name, Texture texture)
    {
        if (name == _propertyName) ApplyTexture(texture);
    }

    void ApplyTexture(Texture texture)
    {
        if (_target == null || !texture) return;
        if (_target.HasTexture(_propertyID))
        {
            _target.SetTexture(_propertyID, texture);
            _applyCurrentTexture = false;
            return;
        }

        if (_missingPropertyReported) return;
        _missingPropertyReported = true;
        Debug.LogError(
            $"Visual Effect '{_target.name}' does not expose texture property '{_propertyName}'.",
            this
        );
    }
}

} // namespace YOLODepth
