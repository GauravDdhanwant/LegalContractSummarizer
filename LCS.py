import streamlit as st
import torch
from torch import nn
from argparse import Namespace
import torch.nn.functional as F
from transformers.activations import ACT2FN
import math
from torch.nn import LayerNorm
from PIL import Image
import numpy as np

# Define the provided model components and functions

def standard_attention(query_layer, key_layer, value_layer, scaling_attention_score=True):
    if scaling_attention_score:
        query_layer = query_layer / math.sqrt(query_layer.shape[-1])
    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    attention_probs = F.softmax(attention_scores, dim=-1)
    context_layer = torch.matmul(attention_probs, value_layer)
    return context_layer

def attention_fn_default(query_layer, key_layer, value_layer, scaling_attention_score=True):
    if int(torch.__version__.split('.')[0]) >= 2 and scaling_attention_score:
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_layer, key_layer, value_layer, 
            attn_mask=None,
            dropout_p=0.,
            is_causal=False
        )
        return attn_output
    else:
        return standard_attention(
            query_layer, key_layer, value_layer, scaling_attention_score=scaling_attention_score
        )

class PatchEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj = nn.Conv2d(config.in_channels, config.hidden_size, kernel_size=config.patch_size, stride=config.patch_size)
        self.cls_embedding = nn.Parameter(torch.zeros(1, config.hidden_size))
        self.position_embedding = nn.Embedding(config.num_positions, config.hidden_size)

    def forward(self, images: "tensor(B, C, H, W)") -> "tensor(B, L, D)":
        x = self.proj(images)
        x = x.flatten(2).transpose(1, 2)
        cls_token = self.cls_embedding.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x += self.position_embedding.weight.unsqueeze(0)
        return x


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        head_dim = config.hidden_size // config.num_heads
        self.scale = head_dim ** -0.5
        self.query_key_value = nn.Linear(config.hidden_size, config.hidden_size * 3)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.output_dropout = torch.nn.Dropout(config.dropout_prob)

    def forward(self, x: "tensor(B, L, D)") -> "tensor(B, L, D)":
        B, L, _ = x.shape
        qkv = self.query_key_value(x)
        qkv = qkv.reshape(B, L, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)  # 3, B, H, L, D
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        out = attention_fn_default(
            q, k, v
        )
        output = self.dense(out.transpose(1, 2).contiguous().view(B, L, -1))
        output = self.output_dropout(output)
        return output

    def attention(self, q, k, v):
        attn_weights = torch.matmul(q * self.scale, k.transpose(-2, -1))
        attn_weights = attn_weights.softmax(dim=-1)
        output = torch.matmul(attn_weights, v)
        return output


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.fc2(x)
        return x


class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = Attention(config)
        self.mlp = MLP(config)
        self.post_attention_layernorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        attention_input = hidden_states
        attention_output = self.input_layernorm(self.attention(attention_input))
        hidden_states = attention_input + attention_output
        mlp_input = hidden_states
        mlp_output = self.post_attention_layernorm(self.mlp(mlp_input))
        output = mlp_input + mlp_output
        return output


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([TransformerLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states):
        for layer_module in self.layers:
            hidden_states = layer_module(hidden_states)
        return hidden_states


class GLU(nn.Module):
    def __init__(self, config, in_features):
        super().__init__()
        self.linear_proj = nn.Linear(in_features, config.hidden_size, bias=False)
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.act1 = nn.GELU()
        self.act2 = nn.functional.silu
        self.dense_h_to_4h = nn.Linear(config.hidden_size, config.ffn_hidden_size, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, config.ffn_hidden_size, bias=False)
        self.dense_4h_to_h = nn.Linear(config.ffn_hidden_size, config.hidden_size, bias=False)

    def forward(self, x):
        x = self.linear_proj(x)
        x = self.act1(self.norm1(x))
        x = self.act2(self.gate_proj(x)) * self.dense_h_to_4h(x)
        x = self.dense_4h_to_h(x)
        return x


class EVA2CLIPModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        vision_config = Namespace(**config.vision_config)
        self.patch_embedding = PatchEmbedding(vision_config)
        self.transformer = Transformer(vision_config)
        self.linear_proj = GLU(config, in_features=config.hidden_size)
        self.conv = nn.Conv2d(in_channels=vision_config.hidden_size, out_channels=config.hidden_size, kernel_size=2, stride=2)
        self.boi = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.eoi = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.scaling_factor = vision_config.scaling_factor

    def forward(self, images: "tensor(B, C, H, W)") -> "tensor(B, L, D)":
        x = self.patch_embedding(images)
        x = self.transformer(x)
        x = x[:, 1:]

        b, s, h = x.shape
        grid_size = int(s**0.5)
        x = x.contiguous().view(b, grid_size, grid_size, h).permute(0, 3, 1, 2)
        x = self.conv(x)

        x = x.flatten(2).transpose(1, 2)
        x = self.linear_proj(x)
        boi = self.boi.expand(x.shape[0], -1, -1)
        eoi = self.eoi.expand(x.shape[0], -1, -1)
        x = torch.cat((boi, x, eoi), dim=1)
        x = x / self.scaling_factor
        return x

# Streamlit App Interface
def main():
    st.title("EVA2CLIPModel Visual Transformer App")

    # Sidebar for configuration
    st.sidebar.header("Model Configuration")
    config = {
        "in_channels": st.sidebar.number_input("Input Channels", min_value=1, max_value=3, value=3, step=1),
        "hidden_size": st.sidebar.number_input("Hidden Size", min_value=64, max_value=1024, value=256, step=64),
        "patch_size": st.sidebar.number_input("Patch Size", min_value=2, max_value=16, value=8, step=2),
        "num_positions": st.sidebar.number_input("Number of Positions", min_value=64, max_value=1024, value=256, step=64),
        "num_heads": st.sidebar.number_input("Number of Attention Heads", min_value=1, max_value=16, value=4, step=1),
        "dropout_prob": st.sidebar.slider("Dropout Probability", min_value=0.0, max_value=0.5, value=0.1, step=0.05),
        "num_hidden_layers": st.sidebar.number_input("Number of Transformer Layers", min_value=1, max_value=12, value=6, step=1),
        "intermediate_size": st.sidebar.number_input("Intermediate Size", min_value=64, max_value=1024, value=512, step=64),
        "hidden_act": st.sidebar.selectbox("Activation Function", options=list(ACT2FN.keys()), index=0),
        "layer_norm_eps": st.sidebar.number_input("Layer Norm Epsilon", min_value=1e-12, max_value=1e-5, value=1e-6, format="%.0e"),
        "ffn_hidden_size": st.sidebar.number_input("FFN Hidden Size", min_value=64, max_value=1024, value=512, step=64),
        "scaling_factor": st.sidebar.number_input("Scaling Factor", min_value=1.0, max_value=10.0, value=2.0, step=0.5),
        "vision_config": {}
    }
    config["vision_config"] = config.copy()

    # Upload an image
    uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("")

        # Convert image to tensor
        image_tensor = preprocess_image(image, config["in_channels"])

        # Initialize model
        model = EVA2CLIPModel(config)

        # Forward pass
        with torch.no_grad():
            output = model(image_tensor.unsqueeze(0))

        st.write("Model Output:")
        st.write(output)

def preprocess_image(image, in_channels):
    # Convert image to numpy array and resize it to the appropriate input size for the model
    image = image.resize((224, 224))  # Adjust size as necessary
    image = np.array(image).transpose(2, 0, 1)  # Convert to (C, H, W)
    image = image / 255.0  # Normalize to [0, 1]
    image_tensor = torch.tensor(image, dtype=torch.float32)
    if in_channels == 1:
        image_tensor = image_tensor.mean(dim=0, keepdim=True)  # Convert to grayscale if necessary
    return image_tensor

if __name__ == "__main__":
    main()
