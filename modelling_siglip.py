import torch
import torch.nn as nn
from typing import Optional, Tuple

class SiglipVisionConfig(nn.Module):
    
    def __init__(
        self, hidden_size = 768,
        intermediate_size = 3072,
        num_hidden_layers = 12,
        num_attention_heads = 12,
        num_channels = 3,
        img_size = 224,
        patch_size = 16,
        layer_norm_eps = 1e-6,
        attention_dropout = 0.0,
        num_img_tokens: Optional[int] = None,
        **kwargs
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.img_size = img_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_img_tokens = num_img_tokens

class SiglipVisionEmbeddings(nn.Module):
    def __init__(self,config:SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_patches = config.patch_size
        self.img_size = config.img_size
        
        self.patch_embeddings = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size
            padding='valid '
        )
        self.num_patches = (config.img_size // config.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embeddings = nn.Embedding( self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1),presistent=False))
    def forward(self, pixel_values: torch.Tensor)->torch.Tensor:
        _, _, height, width = pixel_values.size()
        patch_embeds = self.patch_embeddings(pixel_values)
        embeddings = patch_embeds.flatten(2).transpose(1,2)
        embeddings = embeddings + self.position_embeddings(self.position_ids)
        return embeddings

class SiglipMLP(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
    def forward(self, hidden_states: torch.Tensor)->torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = nn.functional.gelu(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states
    
class SiglipAttention(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = config.attention_dropout
        
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        
    def forward(self, hidden_states: torch.Tensor)->Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_length, _ = hidden_states.size() # Batch, Seq Len, Embedding Dim
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        #Query- Batch Size, Num Heads, Seq Patches, Head Dim
        query_states = query_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1,2)
        key_states = key_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1,2)
        value_states = value_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1,2)
        # Batch Size, Num Heads, Seq Patches, Seq Patches 
        attn_weights = torch.matmul(query_states, key_states.transpose(2,3)) * self.scale
        
        if attn_weights.siz()!= (batch_size, self.num_heads, seq_length, seq_length):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, seq_length, seq_length)}, but is {attn_weights.size()}"
            )
            
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        
        if attn_output.size()!= (batch_size, seq_length, self.embed_dim):
            raise ValueError(
                f"Attention output should be of size {(batch_size, seq_length, self.embed_dim)}, but is {attn_output.size()}"
            )
        # batch, heads, seq len, head dim -> batch, seq len, heads, head dim  
        attn_output = attn_output.transpose(1,2).contiguous().view()
        attn_output = attn_output.reshape(batch_size, seq_length, self.embed_dim)#concat
        attn_output = self.out_proj(attn_output)
        
        return attn_output, attn_weights
        
        
class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps) 
    def forward(self, hidden_states: torch.Tensor)->torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states,_= self.self_attn(hidden_states=hidden_states)
        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual
        return hidden_states

class SiglipEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([SiglipEncoder(config) for _ in range(config.num_hidden_layers)])
        
    def forward(self, input_embeds: torch.Tensor)->torch.Tensor:
        hidden_states = input_embeds
        
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states)
        return hidden_states
        
        
    
class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        
        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        
    def forward(self, pixel_values: torch.Tensor)->torch.Tensor:
        hidden_states = self.embeddings(pixel_values)
        last_hidden_state = self.encoder(input_embeds=hidden_states)
        last_hidden_state = self.post_layernorm(last_hidden_state)
        
        return last_hidden_state    
        
class SiglipVisionModel(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)
        
        #      
    def forward(self, pixel_values) -> Tuple:
        # Batch size, channels, height, width -Batch, Num_Patches, Embedding Dim
        return self.vision_model(pixel_values=pixel_values)