import torch
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from torch import nn, Tensor
import numpy as np
from torch.nn import functional as F
import einops
import math
# patches : patch_size
# path_embedder  # concatenates embeddings of each byte
# gloabl module that operates over patches like they were tokens
# local module, predicts the next bytes within each patch (like a language model)


@dataclass
class Megabyte_Config:
    debug : bool = False
    dtype : torch.dtype = torch.float16

    #initialization
    init_range: float = 0.02
    layer_norm_eps: float = 1e-5

    #patch_embedder
    patch_size: int = 4

    #global model
    global_d_pre_patch: int = 32
    global_d_model =  global_d_pre_patch * patch_size
    global_n_heads = 8
    global_d_head = 8
    global_n_layers = 2
    global_d_mlp = 64

    d_vocab : int = 256
    
    #local model
    #global_dropout = 0.1
    local_d_model = 16
    local_n_heads = 4
    local_d_head = 4
    local_n_layers = 2
    local_d_mlp = 8

    #task
    classification : bool = True
    n_classes = 10

#TODO: should there be special bytes for image_start, image_end, text_start, text_end?

class Megabyte(nn.Module):
    def __init__(self, cfg: Megabyte_Config):
        super().__init__()
        self.cfg = cfg
        self._name = "Megabyte"
        self.patch_embedder = Patch_Embedder(cfg)
        self.global_model = GlobalModel(cfg)
        self.local_model = LocalModel(cfg)
        self.local_pad = nn.Parameter(torch.randn(1, 1, cfg.local_d_model))
        self.global_to_local_proj = nn.Linear(cfg.global_d_pre_patch, cfg.local_d_model)
        self.byte_embedding_local = nn.Embedding(256, cfg.local_d_model)
        self.unembed = nn.Linear(cfg.local_d_model, 256)

        if self.cfg.classification:
            self.global_class_token = nn.Parameter(torch.randn(1, cfg.global_d_model))
            self.local_class_token = nn.Parameter(torch.randn(1, cfg.local_d_model))
            self.classification_head = nn.Linear(cfg.local_d_model, cfg.n_classes)

    def get_param_count(self) -> int:
        '''returns the number of parameters in the model'''
        return sum(p.numel() for p in self.parameters() if p.requires_grad) # all params with gradients

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Embedding and global model processing
        if self.cfg.classification:
            #add class token
            batch_size = x.shape[0]
            global_class_token = self.global_class_token.unsqueeze(0).repeat(batch_size, 1, 1)#.t  # Shape: [batch_size, 1, local_d_model]
            local_class_token = self.local_class_token.unsqueeze(0).repeat(batch_size, self.cfg.patch_size, 1)#.transpose(1,2)  # Shape: [batch_size, 1, local_d_model]

        #print("input x.shape", x.shape)
        #compute bytes_embedding for local model and offset by 1
        byte_embeddings_local = self.byte_embedding_local(x) # dim local_d_model

        # Insert class token at the start of the sequence

        if self.cfg.classification:
            byte_embeddings_local = torch.cat([local_class_token, byte_embeddings_local], dim=1)
        #print("after cat byte_embeddings_local.shape", byte_embeddings_local.shape)
        if self.cfg.debug : print("byte_embeddings_local.shape", byte_embeddings_local.shape)
        offset_byte_embeddings_local = F.pad(byte_embeddings_local, (0, 0, 1, 0), "constant", 0)
        if self.cfg.debug :  print("offset_byte_embeddings_local.shape", offset_byte_embeddings_local.shape)
        offset_byte_embeddings_local[:, 0, :] = self.local_pad
        offset_byte_embeddings_local = offset_byte_embeddings_local[:, :-1, :] # remove last byte
    
        #print("offset_byte_embeddings_local.shape", offset_byte_embeddings_local.shape)
    
        if self.cfg.debug : print("input tensor", x.shape)
        embedded = self.patch_embedder(x) 
        # add class token
        #print("pre embedded.shape", embedded.shape)
        if self.cfg.classification:
            embedded = torch.cat([global_class_token, embedded], dim=1)
        #print("post embedded.shape", embedded.shape)
        global_out = self.global_model(embedded)
        batch_size, num_patches, _ = global_out.shape


        reshaped = global_out.view(batch_size, num_patches, self.cfg.patch_size , self.cfg.global_d_pre_patch)
        if self.cfg.debug : print("shape offset_byte_embeddings_local", offset_byte_embeddings_local.shape)
        offset_byte_embeddings_local = offset_byte_embeddings_local.view(batch_size, num_patches, self.cfg.patch_size , self.cfg.local_d_model)

        if self.cfg.debug : print("reshaped.shape", reshaped.shape)
        # Project each position to the dimension of the local model
        projected = self.global_to_local_proj(reshaped)

        if self.cfg.debug : print("projected.shape", projected.shape)
        # Combine with byte embeddings
        if self.cfg.debug : print("offset_byte_embeddings_local.shape", offset_byte_embeddings_local.shape)
        if self.cfg.debug : print("projected.shape", projected.shape)
        
        combined = projected + offset_byte_embeddings_local

        # Process with local model
        if self.cfg.debug :  print("combined.shape", combined.shape)
        local_out = self.local_model(combined) # shpae [batch, n_patches, patch_size, local_d_model]
        return self.classification_head(local_out[:, 0, 0, :]) # the first token of the first patch ..
        #[batch, local_d_model] -> [batch, n_classes]
        
        """ unembedded = self.unembed(local_out)

        batch_size, num_patches, patch_size, d_local_model = unembedded.shape
        unembedded_flat = unembedded.view(batch_size * num_patches * patch_size, d_local_model)
        # Apply softmax to compute probability distribution over the vocabulary
        probs_flat = F.softmax(unembedded_flat, dim=-1)
        if self.cfg.debug :  print("probs_flat.shape", probs_flat.shape)
        # Reshape back to original shape
        probs = probs_flat.reshape(batch_size, num_patches, patch_size, d_local_model)
        if self.cfg.debug : print("probs.shape", probs.shape) 
        return probs """

class Patch_Embedder(nn.Module):
    def __init__(self, cfg: Megabyte_Config):
        super().__init__()
        self.cfg = cfg
        self.patch_size = cfg.patch_size
        self.byte_embedding_global = nn.Embedding(256, cfg.global_d_pre_patch)
        self.global_pos_embed = nn.Parameter(torch.randn(1, 1, cfg.global_d_pre_patch))
        # Trainable padding embedding
        #self.global_pad = nn.Parameter(torch.randn(1, cfg.patch_size, cfg.global_d_model))
        self.global_pad = nn.Parameter(torch.randn(1, 1, cfg.patch_size * cfg.global_d_pre_patch))
        self.local_pad = nn.Parameter(torch.randn(1,1, cfg.local_d_model))
    def forward(self, x: Tensor) -> Tensor:
        '''embeds from bytes and adds positional embedding and create padding_token_in_beginning and removes the last patch'''
        # x is [batch, byte_sequence]
        # divide byte_sequence into patches of size patch_size
        byte_embeddings = self.byte_embedding_global(x)
        seq_length = x.size(1)
        positional_embeddings = self.global_pos_embed[:, :seq_length, :]
        if self.cfg.debug :  print("positional_embeddings.shape", positional_embeddings.shape)
        byte_embeddings += positional_embeddings
        if self.cfg.debug :  print("byte_embeddings.shape", byte_embeddings.shape)
        batch_size = byte_embeddings.size(0)
        # Ensure the sequence can be divided into patches
        padded_length = (seq_length + self.patch_size - 1) // self.patch_size * self.patch_size
        pad_length = padded_length - seq_length
        padded_byte_embeddings = F.pad(byte_embeddings, (0, 0, 0, pad_length), "constant", 0) # add pad to the start
        if self.cfg.debug :  print("padded_byte_embeddings.shape", padded_byte_embeddings.shape)
        patched_embeddings = padded_byte_embeddings.reshape(batch_size, -1, self.patch_size * self.cfg.global_d_pre_patch)
        if self.cfg.debug :  print("patched_embeddings.shape", patched_embeddings.shape)
        padded_embeddings = torch.cat([self.global_pad.expand(batch_size, -1, -1), patched_embeddings], dim=1)
        if self.cfg.debug :  print("padded_embeddings.shape", padded_embeddings.shape)
        if self.cfg.debug :  print("padded_embeddings[:, :-1, :]", padded_embeddings[:, :-1, :].shape)
        return padded_embeddings[:, :-1, :]  # Remove the last patch


class GlobalModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList([Block(cfg, is_local=False) for _ in range(cfg.global_n_layers)])
        self.norm = LayerNorm(cfg, is_local=False)
        
    def forward(self, x: Tensor) -> Tensor:
        residual = x
        for block in self.blocks:
            if self.cfg.debug: print("residual.shape", residual.shape)
            residual = block(residual)
        residual_normalized = self.norm(residual)
        return residual_normalized

class LocalModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList([Block(cfg, is_local=True) for _ in range(cfg.local_n_layers)])
        self.norm = LayerNorm(cfg, is_local=True)
        
    def forward(self, x: Tensor) -> Tensor:
        #x: [batch, num_patches, patch_size, local_d_model]
        if self.cfg.debug: print("x.shape", x.shape)
       
        batch_size, num_patches, patch_size, _ = x.shape

        # Reshape x to: [batch * num_patches, patch_size, local_d_model]
        x = x.view(batch_size * num_patches, patch_size, -1)

        # Process each patch independently and in parallel
        for block in self.blocks: #TODO think about whether this is generally correct!!! 
            #TODO .. because isnt layernorm normalizing over entire batch?? is this a problem?
            x = block(x)
        x = self.norm(x)

        # Reshape x back to: [batch, num_patches, patch_size, local_d_model]
        x = x.view(batch_size, num_patches, patch_size, -1)

        return x


class MLP(nn.Module):
    def __init__(self, cfg, is_local : bool):
        super().__init__()
        self.cfg = cfg
        if is_local:
            self._in = nn.Linear(cfg.local_d_model, cfg.local_d_mlp)
            self._nonlin = nn.GELU()
            self._out = nn.Linear(cfg.local_d_mlp, cfg.local_d_model)
        else:
            self._in = nn.Linear(cfg.global_d_model, cfg.global_d_mlp)
            self._nonlin = nn.GELU()
            self._out = nn.Linear(cfg.global_d_mlp, cfg.global_d_model)

        #init
        for module in [self._in, self._out]:
            nn.init.uniform_(module.weight, -cfg.init_range, cfg.init_range)
            nn.init.uniform_(module.bias, -cfg.init_range, cfg.init_range) # initialize to 0?
    
    def forward(self, x: Tensor) -> Tensor:
        x = self._in(x)
        x = self._nonlin(x)
        x = self._out(x)
        return x
    
class LayerNorm(nn.Module):
    def __init__(self, cfg, is_local : bool):
        super().__init__()
        self.cfg = cfg
        if is_local:
            self._w = nn.Parameter(torch.ones(cfg.local_d_model)) # add cfg.local_d_model * cfg.patch_size?
            self._b = nn.Parameter(torch.zeros(cfg.local_d_model))
        else:
            self._w = nn.Parameter(torch.ones(cfg.global_d_model))
            self._b = nn.Parameter(torch.zeros(cfg.global_d_model))
        
    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        if self.cfg.debug:
            print("_w.shape", self._w.shape)
            print("mean.shape", mean.shape)
            print("std.shape", std.shape)
            print("x.shape", x.shape)
        return self._w * (x - mean) / torch.sqrt(std + self.cfg.layer_norm_eps) + self._b
   

class MultiHeadAttention(nn.Module):
    def __init__(self, cfg, is_local : bool):
        super().__init__()
        self.cfg = cfg
        if is_local:
            self.proj_Q = nn.ModuleList([nn.Linear(cfg.local_d_model, cfg.local_d_head) for _ in range(cfg.local_n_heads)])
            self.proj_K = nn.ModuleList([nn.Linear(cfg.local_d_model, cfg.local_d_head) for _ in range(cfg.local_n_heads)])
            self.proj_V = nn.ModuleList([nn.Linear(cfg.local_d_model, cfg.local_d_head) for _ in range(cfg.local_n_heads)])
            self.proj_O = nn.ModuleList([nn.Linear(cfg.local_d_head, cfg.local_d_model) for _ in range(cfg.local_n_heads)])
        else:
            self.proj_Q = nn.ModuleList([nn.Linear(cfg.global_d_model, cfg.global_d_head) for _ in range(cfg.global_n_heads)])
            self.proj_K = nn.ModuleList([nn.Linear(cfg.global_d_model, cfg.global_d_head) for _ in range(cfg.global_n_heads)])
            self.proj_V = nn.ModuleList([nn.Linear(cfg.global_d_model, cfg.global_d_head) for _ in range(cfg.global_n_heads)])
            self.proj_O = nn.ModuleList([nn.Linear(cfg.global_d_head, cfg.global_d_model) for _ in range(cfg.global_n_heads)])
        
      #proper init
        for modules in [self.proj_Q, self.proj_K, self.proj_V, self.proj_O]:
            for module in modules:
                nn.init.uniform_(module.weight, -cfg.init_range, cfg.init_range)

    def forward(self, normalized_resid_pre):
        batch_size, seq_len, _ = normalized_resid_pre.size()

        if self.cfg.debug: print("normalized_resid_pre:", normalized_resid_pre.shape)
        # Compute Q, K, V for all heads
        
        if self.cfg.debug: print("proj_q weight", self.proj_Q[0].weight.shape)
        Q = [proj(normalized_resid_pre) for proj in self.proj_Q]
        K = [proj(normalized_resid_pre) for proj in self.proj_K]
        V = [proj(normalized_resid_pre) for proj in self.proj_V]

        heads = []
        for q, k, v in zip(Q, K, V):
            attn_scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.cfg.global_d_head)
            attn_weights = attn_scores.softmax(dim=-1)
            attn_output = torch.bmm(attn_weights, v)
            heads.append(attn_output)

        # Process each head's output separately through its corresponding projection in self.proj_O
        outputs = [proj(head) for head, proj in zip(heads, self.proj_O)]

        # Sum the outputs from all heads to get the final attn_out
        attn_out = sum(outputs)
        if self.cfg.debug: print("attn_out:", attn_out.shape)
        return attn_out

class Block(nn.Module):
    def __init__(self, cfg, is_local : bool):
        super().__init__()
        self.cfg = cfg
        self.norm1 = LayerNorm(cfg, is_local)
        self.attn = MultiHeadAttention(cfg, is_local)
        self.norm2 = LayerNorm(cfg, is_local)
        self.mlp = MLP(cfg, is_local)

    def forward(self, residual : Tensor) -> Tensor:
        normalized_resid_pre = self.norm1(residual)
        if self.cfg.debug: print("shape normalized_resid_pre", normalized_resid_pre.shape)
        attn_out = self.attn(normalized_resid_pre)
        normalized_resid_mid = normalized_resid_pre + attn_out
        normalized_resid_post = self.norm2(normalized_resid_mid)
        mlp_out = self.mlp(normalized_resid_post)
        residual = normalized_resid_mid + mlp_out
        return residual



@dataclass
class VisualTransformer_Config:
    debug : bool = False
    dtype : torch.dtype = torch.float16

    #initialization
    init_range: float = 0.02
    layer_norm_eps: float = 1e-5

    #patches dimension
    patch_dim : int = 4
    global_d_model =  128
    global_n_heads = 8
    global_d_head = 4
    global_n_layers = 3
    global_d_mlp = 64
    n_classes = 10 



class VisualTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._name = "Vit"
        self.projection = nn.Linear(cfg.patch_dim*cfg.patch_dim, cfg.global_d_model)
        self.blocks = nn.ModuleList([Block(cfg, is_local=False) for _ in range(cfg.global_n_layers)])
        self.norm = LayerNorm(cfg, is_local=False)
        self.pos_embedding_layer = nn.Linear(2, cfg.global_d_model)
        self.last_layer = nn.Sequential(
            nn.Linear(cfg.global_d_model, cfg.global_d_model),
            nn.GELU(),
            nn.Linear(cfg.global_d_model, cfg.n_classes),
        )
        self.class_token = nn.Parameter(torch.randn(cfg.global_d_model, 1))

    def get_param_count(self) -> int:
        '''returns the number of parameters in the model'''
        return sum(p.numel() for p in self.parameters() if p.requires_grad) # all params with gradients


    def forward(self, images : Tensor)-> Tensor:
        #images : [batch, num_patches, patch_dim]
        #returns [batch, 1]

        #positional encoding
        # Compute 2D positions for the current image size
        batch , grid_w, grid_h, p_h, p_w = images.shape  # Patch height, patch width
        pos_x = torch.arange(grid_w).repeat(grid_h, 1).flatten().to(images.device)
        pos_y = torch.arange(grid_h).repeat(grid_w, 1).t().flatten().to(images.device)
        pos = torch.stack((pos_x, pos_y), dim=1).float()

        # Add a position for the class token
        class_token_pos = torch.zeros(1, 2).to(images.device)
        pos = torch.cat([class_token_pos, pos], dim=0)

        pos_embeddings = self.pos_embedding_layer(pos)  # Shape: [num_patches + 1, d_model]

        images = images.flatten(1,2).flatten(2,3) # [batch, num_patches, patch_dim * patch_dim]
        residual = self.projection(images)
        if self.cfg.debug: print("residual.shape", residual.shape)
        #add class token at the start
        batch_size = residual.shape[0]
        class_tokens = self.class_token.unsqueeze(0).repeat(batch_size, 1, 1)  # Shape: [batch_size, 1, global_d_model]
        class_tokens = class_tokens.transpose(1, 2)
        # Concatenate along the patches dimension (dim=1)
        residual = torch.cat([class_tokens, residual], dim=1)

        pos_embeddings = pos_embeddings.unsqueeze(0).repeat(residual.size(0), 1, 1)  # Repeat for batch

        # Shape: [batch_size, num_patches + 1, global_d_model]
        if self.cfg.debug:  print("residual.shape", residual.shape)
        if self.cfg.debug:  print("pos_embeddings.shape", pos_embeddings.shape)

        residual += pos_embeddings


        for block in self.blocks:
            residual = block(residual)
        residual_normalized = self.norm(residual)
        class_token_output = residual_normalized[:, 0] # [batch, global_d_model]
        last_hidden_state = self.last_layer(class_token_output)
        logits = F.softmax(last_hidden_state, dim = -1)
        return logits
 

import torch
import torch.nn as  nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels*self.expansion)
        
        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()
        
    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))
        
        x = self.relu(self.batch_norm2(self.conv2(x)))
        
        x = self.conv3(x)
        x = self.batch_norm3(x)
        
        #downsample if needed
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        #add identity
        x+=identity
        x=self.relu(x)
        
        return x

class ResBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(ResBlock, self).__init__()
       

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
      identity = x.clone()

      x = self.relu(self.batch_norm2(self.conv1(x)))
      x = self.batch_norm2(self.conv2(x))

      if self.i_downsample is not None:
          identity = self.i_downsample(identity)
      print(x.shape)
      print(identity.shape)
      x += identity
      x = self.relu(x)
      return x


@dataclass
class ResNet_Config:
    layers = [2]
    num_classes : int = 10
    num_channels : int = 1

        
class ResNet(nn.Module):
    def __init__(self, ResBlock, cfg: ResNet_Config):
        super(ResNet, self).__init__()
        self.cfg = cfg
        self._name = "ResNet"
        layer_list = cfg.layers
        num_classes = cfg.num_classes
        num_channels = cfg.num_channels
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size = 3, stride=2, padding=1)
        self.blocks = nn.ModuleList([])
        for i in range(len(layer_list)):
            if i == 0:
                self.blocks.append(self._make_layer(ResBlock, layer_list[i], planes=64))
            else:
                self.blocks.append(self._make_layer(ResBlock, layer_list[i], planes=64*(2**i), stride=2))
            
        out_channels = 64*(2**(len(layer_list)-1))
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(out_channels*ResBlock.expansion, num_classes)
        
    def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)

        for layer in self.blocks:
            x = layer(x)
        
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        
        return x
        
    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != planes*ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes*ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes*ResBlock.expansion)
            )
            
        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes*ResBlock.expansion
        
        for i in range(blocks-1):
            layers.append(ResBlock(self.in_channels, planes))
            
        return nn.Sequential(*layers)
    
    def get_param_count(self):
        '''returns the number of parameters in the model'''
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

