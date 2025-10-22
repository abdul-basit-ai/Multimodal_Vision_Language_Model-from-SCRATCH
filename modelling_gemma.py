import torch
from torch import nn
from typing import Tuple, Optional, List
from torch.nn import CrossEntropyLoss
import math
from modelling_siglip import SiglipVisionConfig, SiglipVisionModel

class KVCache():
    def __init__(self)->None:
        self.key_cache : List[torch.Tensor] = []
        self.value_cache : List[torch.Tensor] = []
        
    def num_items(self)->int:
        if len(self.key_cache)==0:
            return 0
        return self.key_cache[0].shape[-2] # batch, num_heads, seq_len, head_dim  
    
    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: Optional[int]=None)->Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx is not None:
            if len(self.key_cache)<=layer_idx:
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
            else:
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
            return self.key_cache[layer_idx], self.value_cache[layer_idx]

class GemmaConfig():
    def __init__(self, 
                vocab_size,
                hidden_size,
                intermediate_size,
                num_attention_heads,
                num_hidden_layers,
                num_key_value_heads,
                head_dim=256,
                max_position_embeddings=8192,
                rms_norm_eps=1e-6,
                rope_theta=100000,
                attention_bias=False,
                attention_dropout=0.0,
                pad_token_id=None,
                **kwargs,):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id
        

class PaliGemmaConfig():
    def __init__(self,
        vision_config=None,
        text_config=None,
        ignore_index: int=-100,
        image_token_index=2560000,
        vocab_size: int=257152,
        projection_dim: int=2048,
        hidden_size: int=2048,
        pad_token_id: Optional[int]=None, 
        **kwargs,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.vision = vision_config
        self.is_encoder_decoder = False
        self.pad_token_id = pad_token_id
        
        self.vison_config = SiglipVisionConfig(**vision_config)
        self.text_config = text_config
        
        self.text_config = GemmaConfig(**text_config,pad_token_id=pad_token_id)
        self.vocab_size = self.text_config.vocab_size
        
        self.text_config.num_image_tokens = (self.vison_config.image_size // self.vison_config.patch_size )**2
        self.vison_config.projection_dim = projection_dim

class GemmaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config=config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
    def forward(self,x):
        return self.down_proj(nn.functional.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))

class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.zeros(dim))
    
    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1+ self.weight.float())
        
        return output.type_as(x)

def repeat_kv(hidden_states: torch.Tensor, n_repeat: int)->torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_repeat==1:
        return hidden_states
    hidden_states = hidden_states[:,:,None,:,:].expand(batch, num_key_value_heads, n_repeat, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_repeat, slen, head_dim)

class GemmaRotaryEmbedding(nn.Module):
    def __init__(self, dim,max_position_embeddings=8192, base=100000,device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2,dtype=torch.int64).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
    @torch.no_grad()
    def forward(self, x , position_ids, seq_len=None):
        self.inv_freq = self.inv_freq.to(x.device)
        inv_freq_expanded = self.inv_freq[None,:, None].float().expand(position_ids.shape[0], -1, -1)
        
        position_ids_expanded = position_ids[:, None, :].float()
        device = x.device.type
        device_type = device_type if isinstance(device, str) and device.type != 'mps' else 'cpu'
        
        with torch.autocast(device_type=device_type, enabled=False):
            freq = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1,2)
            embd = torch.cat((freq, freq), dim=-1)
            sin = embd.sin()
            cos = embd.cos()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)       

def apply_and_rotate_rotary_emb(q,k,cos,sin,un_squeeze_dim=1):
    cos = cos.unsqueeze(un_squeeze_dim)
    sin = sin.unsqueeze(un_squeeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class GemmaAttention(nn.Module):
    def __init__(self, config:GemmaConfig,layer_idx:Optional[int]=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.num_key_value_heads = config.num_key_value_heads
        self.key_value_group_size = self.num_heads // self.num_key_value_heads
        
        assert self.hidden_size % self.num_heads==0
        
        #N_head = 8, hidden_size = 1024, head_dim = 1024/8=128 assuming
        #Wq = [hidden_size, N_head * head_dim] = [1024, 8*128]
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        #Head Sharing for K,V
        # Wk = [hidden_size, num_key_value_heads * head_dim] = [1024, 4*128] assuming num_key_value_heads=4
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        # Wv = [hidden_size, num_key_value_heads * head_dim] = [1024, 4*128]
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        # Wo = [num_heads * head_dim, hidden_size] = [8*128, 1024]
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        
        self.rotary_embd = GemmaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            theta=self.rope_theta,
        )
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor]=None,
        position_ids: Optional[torch.LongTensor]=None,
        kv_cache: Optional[KVCache]=None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor,torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size() #batch_size, seq_length, hidden_size
        #Batch size, Seq_len, NUM_HEADS Q * HEAD_DIM
        query_states = self.q_proj(hidden_states)
        #Batch size, Seq_len, NUM_KEY_VALUE_HEADS KV * HEAD_DIM
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        #Batch size, , NUM_HEADS KV ,Seq_len  HEAD_DIM
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1,2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1,2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1,2)
        
        cos , sin = self.rotary_embd.get_cos_sin(value_states,position_ids, seq_len=None)
        
        query_states, key_states = apply_and_rotate_rotary_emb( query_states, key_states, cos, sin)
        
        if kv_cache is not None:
            key_states, value_states = kv_cache.update(key_states, value_states, self.layer_idx)

        key_states = repeat_kv(key_states, self.num_key_value_group_size) # Reversing Multi Query head attention
        value_states = repeat_kv(value_states, self.num_key_value_group_size) # as we have no custom cuda kernels
        
        attn_weights = torch.matmul(query_states, key_states.transpose(2,3)) # B, num_heads, q_len, kv_len
        attn_weights = attn_weights / math.sqrt(self.head_dim)
        
        assert attention_mask is not None, "Attention mask is required"
        attn_weights = attn_weights + attention_mask
        
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.config.attention_dropout, training=self.training)
        
        attn_output = torch.matmul(attn_weights, value_states) # B, num_heads, q_len, head_dim
        
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"Attention output shape {attn_output.size()} is not correct. Expected {(bsz, self.num_heads, q_len, self.head_dim)}"
            )
            
        attn_output = attn_output.transpose(1,2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)
        
        attn_output = self.out_proj(attn_output)
        
        return attn_output,attn_weights
        
class GemmaDecoderLayer(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = GemmaAttention(config=config, layer_idx=layer_idx)
        
        self.mlp = GemmaMLP(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, config.rms)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor]=None,
        position_ids: Optional[torch.LongTensor]=None,
        kv_cache: Optional[KVCache]=None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor,torch.FloatTensor]]]:
        
        residual = hidden_states
        hidden_states,_, = self.input_layernorm(hidden_states)
        attn_output = self.self_attn(
            hidden_states,  
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
        )
        
        hidden_states = residual+hidden_states
        residual =  hidden_states
        
        hidden_states= self.post_attention_layernorm(hidden_states)
        hidden_states= self.mlp(hidden_states)
        hidden_states= residual + hidden_states
        return hidden_states
        

class Gemma_model(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        
        self.config = config
        self.padding_idx = config.pad_token_id 
        self.vocab_size = config.vocab_size
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([GemmaDecoderLayer(config) for layer_idx in range(config.num_hidden_layers)])
        self.norm = GemmaRMSNorm(config.hidden_size, config.rms_norm_eps)
        
    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    def forward(
        self,
        attention_mask: Optional[torch.Tensor]=None,
        position_ids: Optional[torch.LongTensor]=None,
        inputs_embeds: Optional[torch.FloatTensor]=None,
        kv_cache: Optional[KVCache]=None,
    ) -> torch.FloatTensor:
        
        hidden_states = inputs_embeds
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer
        
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache,
            ) 
        hidden_states = self.norm(hidden_states)
        return hidden_states
        
class GemmaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Gemma_model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False) 
    
    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    def tie_weights(self):
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        attention_mask: Optional[torch.Tensor]=None,
        position_ids: Optional[torch.LongTensor]=None,
        inputs_embeds: Optional[torch.FloatTensor]=None,
        kv_cache: Optional[KVCache]=None,
    )->Tuple:
        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )
        
        hidden_states = outputs 
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        
        return logits
    
    
    
class PaliGemmaMultiModalProjector(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.linear= nn.Linear(config.vison_config.hidden_size, config.vison_config.projection_dim, bias = True)
        
    def forward(self, image_features):
        hidden_states = self.linear(image_features)
        return hidden_states
    

class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config 
        self.vision_tower =SiglipVisionModel(config.vision.config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        self.vocab_size = config.vocab_size
        
        language_model = GemmaForCausalLM(config.text_config)
        self.language_model = language_model
        
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        
    def tie_weights(self):
        return self.language_model.tie_weights()
    
    def _merge_input_ids_with_image_features(
        self, image_features: torch.Tensor, input_embeds: torch.Tensor, input_ids: torch.Tensor,attention_mask: torch.Tensor, kv_cache: Optional[KVCache]=None
    ):
        _,_,embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        dtype, device =   input_embeds.dtype, input_embeds.device
        
        scaled_iamge_features = image_features/(self.config.hidden_size**0.5)
        final_embeddings =torch.zeros( batch_size, sequence_length, embed_dim, dtype=input_embeds.dtype, device=input_embeds.device)

        task_mask = (input_ids!=self.config.image_token_index) & (input_ids!=self.config.pad_token_id) # [0,0,01,1,1,1,1,1]
        image_mask = input_ids==self.config.image_token_index #[1,1,1,1,0,0,0,0]
        pad_mask = input_ids==self.config.pad_token_id # [0,0,0,00,00,0] 
        
        text_mask_expanded = task_mask.unsqueeze(-1).expand(-1,-1,embed_dim)# to be able to use in tensor in torch.where
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1,-1,embed_dim)# put the mask over embed_dimension
        pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1,-1,embed_dim)
        
        #Add the Embeddings
        final_embeddings = torch.where(
            text_mask_expanded,
            input_embeds,
            final_embeddings
        )# if 1 is true , second will be copied or vise versa
        final_embeddings = torch.masked_scatter(
            image_mask_expanded,
            scaled_iamge_features)# do same as where but for image features cuz seq_len may differ surely, does
        final_embeddings = torch.where(
            pad_mask_expanded,
            torch.zeros_like(final_embeddings),
            final_embeddings
        )
        
        #Creating attention mask
        type, device = attention_mask.dtype, attention_mask.device
        min_dtype = torch.finfo(dtype).min
        q_len = input_embeds.shape[1]
        
        if kv_cache is not None or kv_cache.num_items()==0:
            #do not mask any toekn as we are prefilling
            #only without paddings
            causal_mask = torch.full(
                (batch_size, q_len,q_len), fill_value = 0, dtype=type, device=device
            )
    
        else:
            q_len ==1
            kv_len = kv_cache.num_items()+q_len
            causal_mask= torch.full(
                (batch_size, q_len, kv_len), fill_value = min_dtype, dtype=dtype, device=device
            )
            
        causal_mask =causal_mask.unsqueeze(1) # B,num_heads,QLen,KvLen
        
        
        if kv_cache is not None or kv_cache.num_items()>0:
            
            position_ids = attention_mask.cumsum(-1)[:, -1]
            if position_ids.dim()==1:
                position_ids = position_ids.unsqueeze(0)
        else:
            position_ids=(attention_mask.cumsum(-1)).masked_fill_((attention_mask==0),1).to(device)
            
        return final_embeddings, causal_mask, position_ids
        
        
        
    def forward(
        self,
        input_ids : torch.LongTensor=None,
        pixel_values: torch.FloatTensor=None,
        attention_mask: Optional[torch.FloatTensor]=None,
        kv_cache : Optional[KVCache]=None,
    )->Tuple: 
        assert torch.all(attention_mask==1), "Attention mask for image tokens must be 1"
        
        #Extra input embedding
        #batch, seq_len, hidden_size
        input_embeds = self.language_model.get_input_embeddings()(input_ids)
        
        #Merge Text, Image
        #BCHW->B,NP,Embed
        selected_image_features = self.vision_tower(pixel_values=pixel_values)
        
        #B , NP, Embed -> B, NP, Hidden_size
        image_features = self.multi_modal_projector(selected_image_features)
        
        input_embeds, attention_mask, position_ids = self._merge_input_ids_with_image_features(
            image_features,
            input_embeds,
            input_ids,
            attention_mask,
            kv_cache
        )
        
        outputs = self.language_model(
            inputs_embeds=input_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            kv_cache=kv_cache
        )
        
        return outputs
print("Modelling Gemma Done")