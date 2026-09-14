Shader "Hidden/YOLODepth/VisualizeDepth"
{
    Properties
    {
        _MainTex("Depth", 2D) = "black" {}
        _MinimumDepth("Minimum Depth", Float) = 0.3
        _MaximumDepth("Maximum Depth", Float) = 2.0
    }

HLSLINCLUDE

#include "UnityCG.cginc"

sampler2D _MainTex;
float _MinimumDepth;
float _MaximumDepth;

void VertBlit(float4 position : POSITION,
              float2 texCoord : TEXCOORD0,
              out float4 outPosition : SV_Position,
              out float2 outTexCoord : TEXCOORD0)
{
    outPosition = UnityObjectToClipPos(position);
    outTexCoord = texCoord;
}

float3 HsvToRgb(float3 hsv)
{
    float3 p = abs(frac(hsv.xxx + float3(0, 2.0 / 3, 1.0 / 3)) * 6 - 3);
    float3 white = float3(1, 1, 1);
    return hsv.z * (white + (saturate(p - 1) - white) * hsv.y);
}

float4 FragVisualize(float4 position : SV_Position,
                     float2 texCoord : TEXCOORD0) : SV_Target
{
    float depth = tex2D(_MainTex, texCoord).r;
    if (depth != depth || abs(depth) > 3.402823e+37) return float4(0, 0, 0, 1);
    float normalized = saturate(
        (depth - _MinimumDepth) / max(_MaximumDepth - _MinimumDepth, 1e-5)
    );
    return float4(HsvToRgb(float3(normalized * 0.67, 0.9, 1)), 1);
}

ENDHLSL

    SubShader
    {
        Tags { "RenderType" = "Opaque" "RenderPipeline" = "UniversalPipeline" }
        ZTest Always
        ZWrite Off
        Cull Off

        Pass
        {
            Name "VisualizePass"
            HLSLPROGRAM
            #pragma vertex VertBlit
            #pragma fragment FragVisualize
            ENDHLSL
        }
    }
}
