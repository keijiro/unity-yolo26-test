using System;
using System.Collections.Generic;
using UnityEngine;

namespace YOLODepth
{

public abstract class TextureSource : MonoBehaviour
{
    readonly Dictionary<string, Texture> _textures = new();

    public event Action<string, Texture> TextureChanged;

    public Texture GetTexture(string name) =>
        _textures.TryGetValue(name, out var texture) ? texture : null;

    protected void PublishTexture(string name, Texture texture)
    {
        if (ReferenceEquals(GetTexture(name), texture)) return;

        if (texture == null)
            _textures.Remove(name);
        else
            _textures[name] = texture;

        TextureChanged?.Invoke(name, texture);
    }
}

} // namespace YOLODepth
