import torch
import torch.nn as nn
import math
from enum import Enum

class PhaseType(Enum):
    PRETRAIN = 'pretrain'
    FINETUNE = 'finetune'


class GlobalFeatureEmbedding(nn.Module):
    """Embeds global features (Mach, Reynolds, z-coordinate) into the model dimension"""

    def __init__(self, d_model):
        super().__init__()
        self.embedding = nn.Linear(3, d_model)

    def forward(self, mach, reynolds, z_coord):
        global_features = torch.stack([mach, reynolds, z_coord], dim=-1)
        return self.embedding(global_features)


class SplineFeatureEmbedding(nn.Module):
    """Projects spline features into the model dimension"""

    def __init__(self, d_model, include_pressure=True):
        super().__init__()
        self.input_dim = (
            4 if include_pressure else 3
        )  # 3 features: arc_length, x, y (optionally pressure)
        self.embedding = nn.Linear(self.input_dim, d_model)

    def forward(self, spline_features):
        return self.embedding(spline_features)

# Transformer decoder for traditional training. This decoder uses
# traditional training without cross-attention.
# It uses the geometry features to predict to the pressure distribution.
class AirfoilTransformerDecoder(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_decoder_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
    ):
        super().__init__()

        # Embeddings for source (2D) features including pressure
        self.source_embedding = SplineFeatureEmbedding(
            d_model, include_pressure=False
        )
        # Embeddings for target (3D) geometric features only
        self.target_embedding = SplineFeatureEmbedding(
            d_model, include_pressure=False
        )
        self.global_embedding = GlobalFeatureEmbedding(d_model)

        # Create decoder layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        # Stack decoder layers
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )

        # Output projection to single pressure value
        self.output_projection = nn.Linear(d_model, 1)  # Project to pressure only
        
        # Track frozen layers
        self.frozen_layers = []

    def forward(
        self,
        tgt_geometry,  # [batch_size, tgt_seq_len, 3] - 3D geometry without pressure
        mach,  # [batch_size]
        reynolds,  # [batch_size]
        z_coord,  # [batch_size]
    ):
        # Embed source (2D) features including pressure
        mem_embedded = self.source_embedding(tgt_geometry)
        # Embed target (3D) geometric features
        tgt_embedded = self.target_embedding(tgt_geometry)

        # Embed and add global features
        global_embedded = self.global_embedding(mach, reynolds, z_coord)
        global_embedded = global_embedded.unsqueeze(1)  # [batch_size, 1, d_model]

        # Add global features to both source and target embeddings
        mem_embedded = mem_embedded + global_embedded
        tgt_embedded = tgt_embedded + global_embedded

        # Pass through decoder
        output = self.decoder(
            tgt=tgt_embedded,
            memory=mem_embedded,
        )

        # Project to output dimension
        return self.output_projection(output)
    
# Transformer decoder for transfer learning with cross-attention
# This decoder uses cross-attention to attend to the source features
# while decoding the target features.
class AirfoilTransformerDecoder_TL_CrossAttention(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_decoder_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
    ):
        super().__init__()

        # Embeddings for source (2D) features including pressure
        self.source_embedding = SplineFeatureEmbedding(
            d_model, include_pressure=True
        )
        # Embeddings for target (3D) geometric features only
        self.target_embedding = SplineFeatureEmbedding(
            d_model, include_pressure=False
        )
        self.global_embedding = GlobalFeatureEmbedding(d_model)

        # Create decoder layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        # Stack decoder layers
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )

        # Output projection to single pressure value
        self.output_projection = nn.Linear(d_model, 1)  # Project to pressure only
        
        # Track frozen layers
        self.frozen_layers = []

    def forward(
        self,
        src_features,  # [batch_size, src_seq_len, 4] - 2D airfoil with pressure
        tgt_geometry,  # [batch_size, tgt_seq_len, 3] - 3D geometry without pressure
        mach,  # [batch_size]
        reynolds,  # [batch_size]
        z_coord,  # [batch_size]
    ):
        # Embed source (2D) features including pressure
        src_embedded = self.source_embedding(src_features)
        # Embed target (3D) geometric features
        tgt_embedded = self.target_embedding(tgt_geometry)

        # Embed and add global features
        global_embedded = self.global_embedding(mach, reynolds, z_coord)
        global_embedded = global_embedded.unsqueeze(1)  # [batch_size, 1, d_model]

        # Add global features to both source and target embeddings
        src_embedded = src_embedded + global_embedded
        tgt_embedded = tgt_embedded + global_embedded

        # Pass through decoder
        output = self.decoder(
            tgt=tgt_embedded,
            memory=src_embedded,
        )

        # Project to output dimension
        return self.output_projection(output)


class AirfoilTransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.decoder = AirfoilTransformerDecoder(
            d_model=config.d_model,
            nhead=config.n_head,
            num_decoder_layers=config.n_layers,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
        )

    def forward(
        self,
        tgt_spline_features,
        mach,
        reynolds,
        z_coord,
    ):
        return self.decoder(
            tgt_spline_features,
            mach,
            reynolds,
            z_coord,
        )
    
class AirfoilTransformerModel_TL_CrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.decoder = AirfoilTransformerDecoder_TL_CrossAttention(
            d_model=config.d_model,
            nhead=config.n_head,
            num_decoder_layers=config.n_layers,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
        )

    def forward(
        self,
        src_spline_features,
        tgt_spline_features,
        mach,
        reynolds,
        z_coord,
    ):
        return self.decoder(
            src_spline_features,
            tgt_spline_features,
            mach,
            reynolds,
            z_coord,
        )
        
# Decoder for pretraining and finetuning approach
class AirfoilTransformerDecoder_Pretrain_Finetune(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_decoder_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
    ):
        super().__init__()

        # Embeddings for 2D/3D geometries features
        self.source_embedding = SplineFeatureEmbedding(
            d_model, include_pressure=False
        )
        # Embeddings for 2D/3D geometries features
        self.target_embedding = SplineFeatureEmbedding(
            d_model, include_pressure=False
        )
        self.global_embedding = GlobalFeatureEmbedding(d_model)

        # Create decoder layers explicitly for better control over freezing
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            ) for _ in range(num_decoder_layers)
        ])
        
        # Output projection to single pressure value
        self.output_projection = nn.Linear(d_model, 1)  # Project to pressure only
        
        # Track frozen layers
        self.frozen_layers = []
        
    def forward(
        self,
        tgt_geometry,  # [batch_size, tgt_seq_len, 3] - geometry without pressure
        mach,          # [batch_size]
        reynolds,      # [batch_size]
        z_coord,       # [batch_size]
    ):
        # Embed source (2D) features including pressure
        mem_embedded = self.source_embedding(tgt_geometry)
        # Embed target (3D) geometric features
        tgt_embedded = self.target_embedding(tgt_geometry)

        # Embed and add global features
        global_embedded = self.global_embedding(mach, reynolds, z_coord)
        global_embedded = global_embedded.unsqueeze(1)  # [batch_size, 1, d_model]

        # Add global features to both source and target embeddings
        mem_embedded = mem_embedded + global_embedded
        tgt_embedded = tgt_embedded + global_embedded
        
        # Pass through decoder layers manually (for better control when freezing)
        memory = mem_embedded
        output = tgt_embedded
        
        for i, layer in enumerate(self.layers):
            output = layer(output, memory)

        # Project to output dimension
        return self.output_projection(output)

# Model for pretraining on 2D and finetuning on 3D data
class AirfoilTransformerModel_Pretrain_Finetune(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.decoder = AirfoilTransformerDecoder_Pretrain_Finetune(
            d_model=config.d_model,
            nhead=config.n_head,
            num_decoder_layers=config.n_layers,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
        )
        
        self.phase = PhaseType.PRETRAIN
        self.config = config

    def forward(
        self,
        tgt_spline_features,
        mach,
        reynolds,
        z_coord,
        phase=None,
    ):
        # If phase is provided, use it; otherwise use the current model phase
        current_phase = phase if phase is not None else self.phase
        
        # In pretraining phase, predict 2D airfoil pressure using 2D features only
        if current_phase == PhaseType.PRETRAIN:
            inv_z_coord = 0.0*z_coord
        else:
            inv_z_coord = z_coord/2.5

        return self.decoder(
                tgt_spline_features,  # 3D geometry (as target)
                mach,
                reynolds,
                inv_z_coord,
            )
            
    def freeze_layers(self, num_layers_to_freeze=4):
        """Freeze the first num_layers_to_freeze layers of the decoder"""
        if not hasattr(self.decoder, 'layers'):
            return
            
        # Freeze embeddings
        for param in self.decoder.source_embedding.parameters():
            param.requires_grad = False
        for param in self.decoder.target_embedding.parameters():
            param.requires_grad = False
        for param in self.decoder.global_embedding.parameters():
            param.requires_grad = False
            
        # Freeze the specified decoder layers
        self.decoder.frozen_layers = list(range(num_layers_to_freeze))
        for i in range(num_layers_to_freeze):
            if i < len(self.decoder.layers):
                for param in self.decoder.layers[i].parameters():
                    param.requires_grad = False
        
        # Set phase to finetuning
        self.phase = PhaseType.FINETUNE
                
    def unfreeze_all_layers(self):
        """Unfreeze all layers of the model"""
        for param in self.parameters():
            param.requires_grad = True
        self.decoder.frozen_layers = []
        
        # Set phase to pretraining
        self.phase = PhaseType.PRETRAIN
        
    def set_phase(self, phase):
        """Set the current phase (pretraining or finetuning)"""
        if phase == PhaseType.PRETRAIN:
            self.unfreeze_all_layers()
        elif phase == PhaseType.FINETUNE:
            self.freeze_layers(num_layers_to_freeze=4)
