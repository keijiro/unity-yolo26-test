using System;
using UnityEngine;

namespace YOLODepth
{

public abstract class TextureSource : MonoBehaviour
{
    public Texture Texture { get; private set; }

    public event Action<Texture> TextureChanged;

    protected void PublishTexture(Texture texture)
    {
        if (ReferenceEquals(Texture, texture)) return;

        Texture = texture;
        TextureChanged?.Invoke(texture);
    }
}

} // namespace YOLODepth
