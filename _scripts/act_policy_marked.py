"""
ACTION CHUNKING WITH TRANSFORMERS (ACT) - FULLY ANNOTATED

This is a detailed, line-by-line annotated version of the ACT policy for educational purposes.
ACT predicts a sequence of T future actions given L recent observations (typically L=1).

Key Architecture Components:
1. Vision Backbone: ResNet18 processes RGB-D images from multiple cameras
2. VAE Encoder (training only): Encodes ground-truth action sequences into latent z
3. Transformer Decoder: Generates action predictions conditioned on visual features,
   proprioception, and latent z

Paper: "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware"
       https://arxiv.org/abs/2304.13705

Data Dimensions Reference (from inspect_data.py output):
- B = Batch size (e.g., 128)
- L = Observation window (e.g., 1 for ACT)
- T = Action horizon (e.g., 20 future timesteps)
- Prop_dim = 37 (3 base_vel + 4 torso + 7 left_arm + 1 left_gripper + 7 right_arm + 1 right_gripper + 3 left_pos + 4 left_quat + 3 right_pos + 4 right_quat)
- A = Action dimension = 23 (3 base + 4 torso + 7 left_arm + 1 left_gripper + 7 right_arm + 1 right_gripper)
- hidden_dim = 512 (default transformer feature dimension)
- latent_dim = 32 (VAE latent space dimension)
- num_queries = 20 (number of action slots to predict)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from hydra.utils import instantiate
from il_lib.optim import CosineScheduleFunction, default_optimizer_groups
from il_lib.nn.transformers import (
    build_position_encoding,
    Transformer, TransformerEncoderLayer, TransformerEncoder
)
from il_lib.policies.policy_base import BasePolicy
from il_lib.utils.array_tensor_utils import any_concat, get_batch_size
from omegaconf import DictConfig
from omnigibson.learning.utils.obs_utils import MAX_DEPTH, MIN_DEPTH
from torch.autograd import Variable
from typing import Any, List, Optional

__all__ = ["ACTMod"]

#==Start
class ACTMod(BasePolicy):
    """
    Action Chunking with Transformers (ACT) Policy

    ACT is a behavior cloning method that:
    1. Processes visual observations (RGB-D from multiple cameras) through a ResNet
    2. Combines visual features with proprioceptive state (joint positions, velocities, etc.)
    3. Uses a Conditional VAE (CVAE) to model action distributions
    4. Generates multiple future actions (action chunking) in one forward pass
    5. Applies temporal ensemble at test time for smooth execution

    Training Flow:
        Observations → Vision Backbone → Visual Features ────┐
        Ground Truth Actions → VAE Encoder → Latent z        ├→ Transformer Decoder → Predicted Actions
        Proprioception ───────────────────────────────────────┘

    Inference Flow:
        Observations → Vision Backbone → Visual Features ────┐
        Random z ~ N(0,1) ────────────────────────────────────├→ Transformer Decoder → Predicted Actions
        Proprioception ───────────────────────────────────────┘
    """

    def __init__(
        self,
        *args,
        # ==== Input/Output Dimensions ====
        prop_dim: int,                    # Proprioception dimension (37 for R1Pro robot)
        prop_keys: List[str],             # Which proprioceptive features to use (e.g., ['qpos/torso', 'eef/left_pos'])
        action_dim: int,                  # Action dimension (23 for R1Pro robot)
        action_keys: List[str],           # Which action components to predict (e.g., ['base', 'torso', 'left_arm'])
        features: List[str],              # Input modalities (e.g., ['proprioception', 'rgbd'])
        obs_backbone: DictConfig,         # Config for vision backbone (ResNet18)
        pos_encoding: DictConfig,         # Config for positional encoding
        # ==== Policy Architecture ====
        num_queries: int,                 # Number of action slots (T=20, predicts 20 future actions)
        hidden_dim: int,                  # Transformer hidden dimension (512)
        dropout: float,                   # Dropout probability for regularization
        n_heads: int,                     # Number of attention heads (8)
        dim_feedforward: int,             # FFN intermediate dimension (3200)
        num_encoder_layers: int,          # VAE encoder layers (4)
        num_decoder_layers: int,          # Transformer decoder layers (7)
        pre_norm: bool,                   # Whether to use pre-normalization
        kl_weight: float,                 # Weight for KL divergence loss (10)
        temporal_ensemble: bool,          # Whether to use temporal ensemble at test time
        # ==== Learning Hyperparameters ====
        lr: float,                        # Learning rate
        use_cosine_lr: bool = False,      # Use cosine annealing schedule
        lr_warmup_steps: Optional[int] = None,     # Warmup steps for lr schedule
        lr_cosine_steps: Optional[int] = None,     # Total steps for cosine schedule
        lr_cosine_min: Optional[float] = None,     # Minimum lr for cosine schedule
        lr_layer_decay: float = 1.0,      # Layer-wise lr decay factor
        weight_decay: float = 0.0,        # L2 regularization weight
        **kwargs,
    ) -> None:
        # Call parent class (BasePolicy) constructor
        # BasePolicy handles observation normalization, action denormalization, device management
        super().__init__(*args, **kwargs)
#==End

#==Start
        # ============================================================================
        # SAVE CONFIGURATION PARAMETERS
        # ============================================================================
        # Store which proprioceptive features to use
        # Example: ['odom/base_velocity', 'qpos/torso', 'qpos/left_arm', 'eef/left_pos', ...]
        self._prop_keys = prop_keys

        # Store which action components to predict
        # Example: ['base', 'torso', 'left_arm', 'left_gripper', 'right_arm', 'right_gripper']
        self._action_keys = action_keys

        # Total action dimension (A=23 for R1Pro)
        self.action_dim = action_dim

        # Input modalities to use (e.g., ['proprioception', 'rgbd'])
        # Could also include ['task'] for task conditioning
        self._features = features

        # Whether we're using depth images in addition to RGB
        # If True, inputs are 4-channel (RGB-D), else 3-channel (RGB)
        self._use_depth = obs_backbone.include_depth
#==End

#==Start
        # ============================================================================
        # VISION BACKBONE (ResNet18)
        # ============================================================================
        # Instantiate the vision backbone from config
        # Input: 3 cameras × (B, L=1, 4, 240, 240) for RGBD images
        # Output: 3 cameras × (B*L, 512, h, w) spatial feature maps
        # The backbone is pretrained on ImageNet and adapted for RGBD input
        self.obs_backbone = instantiate(obs_backbone)
#==End

#==Start
        # ============================================================================
        # TRANSFORMER DECODER
        # ============================================================================
        # The main transformer that generates action predictions
        # Architecture: Cross-attention decoder (similar to DETR object detector)
        # - Query: Learnable action slot embeddings (num_queries=20)
        # - Key/Value: Visual tokens + proprioception + latent z
        # - Outputs: (B, num_queries=20, hidden_dim=512) features → actions
        self.transformer = Transformer(
            d_model=hidden_dim,              # Feature dimension (512)
            dropout=dropout,                  # Dropout for regularization (0.1)
            nhead=n_heads,                    # Number of attention heads (8)
            dim_feedforward=dim_feedforward,  # FFN hidden dimension (3200)
            num_encoder_layers=num_encoder_layers,  # Not used in standard ACT (kept for compatibility)
            num_decoder_layers=num_decoder_layers,  # Number of decoder layers (7)
            normalize_before=pre_norm,        # Pre-norm vs post-norm architecture
            return_intermediate_dec=True,     # Return outputs from all decoder layers
        )
#==End

#==Start
        # ============================================================================
        # VAE ENCODER (for training only)
        # ============================================================================
        # Encodes ground-truth action sequences into a latent distribution
        # Input: (B, T=20, A=23) action sequence
        # Output: (B, latent_dim=32) latent vector z
        # Purpose: Allows the model to learn a structured latent space for action generation
        # At test time, we sample z ~ N(0,1) instead of using this encoder
        self.encoder = TransformerEncoder(
            encoder_layer=TransformerEncoderLayer(
                d_model=hidden_dim,           # Feature dimension (512)
                nhead=n_heads,                # Number of attention heads (8)
                dim_feedforward=dim_feedforward,  # FFN hidden dimension (3200)
                dropout=dropout,              # Dropout (0.1)
                activation="relu",            # Activation function
                normalize_before=pre_norm,    # Pre-norm architecture
            ),
            num_layers=num_encoder_layers,    # Number of encoder layers (4)
            norm=nn.LayerNorm(hidden_dim) if pre_norm else None,  # Final layer norm
        )
#==End

#==Start
        # ============================================================================
        # POSITIONAL ENCODING
        # ============================================================================
        # Adds spatial position information to visual features
        # Type: Sine/cosine positional encoding (similar to original Transformer paper)
        # Applied to: Visual feature maps (h×w spatial positions)
        # Output: (1, hidden_dim=512, h, w) positional encoding
        self.position_embedding = build_position_encoding(pos_encoding)

        # ============================================================================
        # LEARNABLE PARAMETERS - Action Generation
        # ============================================================================
        # Number of action slots to predict (T=20 future actions)
        self.num_queries = num_queries

        # Linear head that projects transformer features to actions
        # Input: (B, num_queries=20, hidden_dim=512)
        # Output: (B, num_queries=20, action_dim=23)
        self.action_head = nn.Linear(hidden_dim, action_dim)

        # Learnable action slot embeddings (queries for decoder)
        # These are similar to "object queries" in DETR
        # Shape: (num_queries=20, hidden_dim=512)
        # Purpose: Each embedding represents one future timestep's action
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
#==End

#==Start
        # ============================================================================
        # PROJECTION LAYERS - Vision to Transformer
        # ============================================================================
        # Projects ResNet features to transformer dimension
        # Input: (B*L, resnet_output_dim=512, h, w) per camera
        # Output: (B*L, hidden_dim=512, h, w) per camera
        # Purpose: Match feature dimensions for transformer input
        self.input_proj = nn.Conv2d(obs_backbone.resnet_output_dim, hidden_dim, kernel_size=1)

        # Projects proprioception to transformer dimension
        # Input: (B*L, prop_dim=37)
        # Output: (B*L, hidden_dim=512)
        # Purpose: Embed robot state into transformer feature space
        self.input_proj_robot_state = nn.Linear(prop_dim, hidden_dim)
#==End

#==Start
        # ============================================================================
        # VAE ENCODER COMPONENTS
        # ============================================================================
        # Latent dimension for VAE (32-dimensional latent space)
        # This is much smaller than the action dimension (23) or hidden_dim (512)
        # Purpose: Bottleneck that forces learning of compressed action representation
        self.latent_dim = 32

        # Special [CLS] token for encoder (similar to BERT)
        # Shape: (1, hidden_dim=512)
        # Purpose: The [CLS] output will be used to produce latent z
        self.cls_embed = nn.Embedding(1, hidden_dim)

        # Projects actions to transformer embedding space for encoder input
        # Input: (B, T=20, action_dim=23)
        # Output: (B, T=20, hidden_dim=512)
        self.encoder_action_proj = nn.Linear(action_dim, hidden_dim)

        # Projects proprioception for encoder input
        # Input: (B, prop_dim=37)
        # Output: (B, hidden_dim=512)
        # Purpose: Condition latent z on current robot state
        self.encoder_prop_proj = nn.Linear(prop_dim, hidden_dim)

        # Projects [CLS] token output to latent distribution parameters
        # Input: (B, hidden_dim=512)
        # Output: (B, latent_dim*2=64) where first 32 dims = μ, last 32 dims = log(σ²)
        # Purpose: Parameterize Gaussian distribution q(z|actions, proprioception)
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim * 2)

        # Sinusoidal positional encoding table for encoder inputs
        # Shape: (1, 1+1+num_queries=22, hidden_dim=512)
        # Breakdown: [CLS token=1] + [proprioception=1] + [action sequence=20]
        # Purpose: Give encoder information about temporal order of inputs
        # Note: register_buffer means this is saved with model but not trained
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(1+1+num_queries, hidden_dim))
#==End

#==Start
        # ============================================================================
        # DECODER LATENT CONDITIONING
        # ============================================================================
        # Projects sampled latent z back to transformer dimension
        # Input: (B, latent_dim=32)
        # Output: (B, hidden_dim=512)
        # Purpose: Inject latent information into decoder
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim)

        # Learnable position embeddings for [proprioception, latent_z] tokens
        # Shape: (2, hidden_dim=512)
        # Purpose: Distinguish between proprio and latent inputs to decoder
        self.additional_pos_embed = nn.Embedding(2, hidden_dim)
#==End

#==Start
        # ============================================================================
        # TEMPORAL ENSEMBLE SETUP
        # ============================================================================
        # At test time, ACT predicts 20 actions but executes only 1
        # Temporal ensemble averages predictions from multiple forward passes
        # This improves smoothness and reduces noise
        self.temporal_ensemble = temporal_ensemble

        if temporal_ensemble:
            # Save horizon for buffer management
            self._horizon = num_queries  # 20

            # Buffer to store recent action predictions
            # Size: deque of length 20, each element is (20, action_dim=23)
            # Element i contains the action prediction for timestep i from past forward passes
            self._action_buffer = deque(maxlen=self._horizon)

            # Initialize buffer with zeros
            # Each entry is a (horizon=20, action_dim=23) tensor
            for _ in range(self._horizon):
                self._action_buffer.append(
                    torch.zeros((self._horizon, self.action_dim), dtype=torch.float32).to(self.device)
                )
#==End

#==Start
        # ============================================================================
        # SAVE TRAINING HYPERPARAMETERS
        # ============================================================================
        # Weight for KL divergence term in CVAE loss
        # Total loss = L1_loss + kl_weight * KL_loss
        # Typical value: 10
        self.kl_weight = kl_weight

        # Learning rate and schedule parameters
        self.lr = lr
        self.use_cosine_lr = use_cosine_lr
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_cosine_steps = lr_cosine_steps
        self.lr_cosine_min = lr_cosine_min
        self.lr_layer_decay = lr_layer_decay
        self.weight_decay = weight_decay

        # Save all hyperparameters to checkpoint
        # This is a PyTorch Lightning feature that stores config with model
        self.save_hyperparameters()
#==End

#==Start
    # ============================================================================
    # FORWARD PASS - Main computation
    # ============================================================================
    def forward(
        self,
        obs: dict,                          # Observations: {'rgbd': dict, 'qpos': dict, 'eef': dict, 'odom': dict}
        actions: Optional[torch.Tensor] = None,  # Ground truth actions (B, T=20, A=23), None during inference
        is_pad: Optional[torch.Tensor] = None    # Padding mask (B, T=20), True where padded
    ) -> torch.Tensor:
        """
        Forward pass through ACT model.

        Training mode (actions provided):
            1. Encode actions → latent z via VAE encoder
            2. Process observations through vision backbone
            3. Decode latent z + observations → predicted actions
            4. Return predictions and (μ, log_var) for KL loss

        Inference mode (actions=None):
            1. Sample z ~ N(0,1)
            2. Process observations through vision backbone
            3. Decode z + observations → predicted actions
            4. Return predictions and (None, None)

        Args:
            obs: Dictionary of observations
                - 'rgbd': dict of tensors, keys like 'robot_r1::robot_r1:left_realsense_link:Camera:0'
                          Each tensor is (B, L=1, 4, 240, 240) for RGBD images
                - 'qpos': dict of tensors like {'torso': (B,L,4), 'left_arm': (B,L,7), ...}
                - 'eef': dict of tensors like {'left_pos': (B,L,3), 'left_quat': (B,L,4), ...}
                - 'odom': dict of tensors like {'base_velocity': (B,L,3)}
            actions: Ground truth action sequence (B, T=20, A=23), normalized to [-1, 1]
                     Only provided during training
            is_pad: Boolean mask (B, T=20) indicating which timesteps are padding
                    True = padding, False = valid data

        Returns:
            a_hat: Predicted actions (B, num_queries=20, action_dim=23)
            [mu, logvar]: VAE distribution parameters (training) or [None, None] (inference)
        """

        # ========================================================================
        # DETERMINE MODE (training vs inference)
        # ========================================================================
        # If actions are provided, we're in training mode and will use VAE encoder
        # Otherwise, we're in inference mode and will sample z from prior N(0,1)
        is_training = actions is not None

        # Get batch size from observations
        # get_batch_size() is a utility that extracts B from nested dict structure
        bs = get_batch_size(obs, strict=True)
#==End

#==Start
        # ========================================================================
        # PROCESS PROPRIOCEPTION (robot state)
        # ========================================================================
        # Concatenate all proprioceptive features specified in self._prop_keys
        # Keys are like 'odom/base_velocity', 'qpos/torso', 'eef/left_pos'
        # These contain joint positions, velocities, end-effector poses
        prop_obs = []
        for prop_key in self._prop_keys:
            if "/" in prop_key:
                # Handle nested dictionary keys (e.g., 'qpos/torso')
                group, key = prop_key.split("/")  # Split 'qpos/torso' → 'qpos', 'torso'
                prop_obs.append(obs[group][key])  # Extract obs['qpos']['torso']
            else:
                # Handle flat keys (e.g., 'proprioception')
                prop_obs.append(obs[prop_key])

        # Concatenate all proprioceptive features along last dimension
        # Each component has shape (B, L=1, feat_dim)
        # Result: (B, L=1, prop_dim=37) where 37 = sum of all feature dimensions
        prop_obs = torch.cat(prop_obs, dim=-1)

        # Flatten batch and observation window dimensions
        # Before: (B, L=1, prop_dim=37)
        # After: (B*L, prop_dim=37)
        # For ACT with L=1, this is just (B, 37)
        prop_obs = prop_obs.reshape(-1, prop_obs.shape[-1])
#==End

#==Start
        # ========================================================================
        # VAE ENCODER PATH (TRAINING ONLY)
        # ========================================================================
        if is_training:
            # ---- Project Actions to Embedding Space ----
            # Input: (B, T=20, action_dim=23) normalized actions
            # Output: (B, T=20, hidden_dim=512)
            # Purpose: Convert actions to transformer-compatible features
            action_embed = self.encoder_action_proj(actions)

            # ---- Project Proprioception to Embedding Space ----
            # Input: (B, prop_dim=37)
            # Output: (B, hidden_dim=512)
            prop_embed = self.encoder_prop_proj(prop_obs)

            # Add time/sequence dimension to proprioception
            # Before: (B, hidden_dim=512)
            # After: (B, 1, hidden_dim=512)
            prop_embed = torch.unsqueeze(prop_embed, dim=1)

            # ---- Get [CLS] Token Embedding ----
            # self.cls_embed.weight has shape (1, hidden_dim=512)
            # This is a learnable embedding similar to BERT's [CLS] token
            cls_embed = self.cls_embed.weight

            # Expand [CLS] token for entire batch
            # Before: (1, hidden_dim=512)
            # After: (B, 1, hidden_dim=512)
            cls_embed = torch.unsqueeze(cls_embed, dim=0).repeat(bs, 1, 1)

            # ---- Construct Encoder Input Sequence ----
            # Concatenate: [CLS] + proprioception + action_sequence
            # cls_embed: (B, 1, hidden_dim=512)
            # prop_embed: (B, 1, hidden_dim=512)
            # action_embed: (B, T=20, hidden_dim=512)
            # Result: (B, 1+1+20=22, hidden_dim=512)
            encoder_input = torch.cat([cls_embed, prop_embed, action_embed], dim=1)

            # Transpose for transformer (expects seq_len first)
            # Before: (B, seq_len=22, hidden_dim=512)
            # After: (seq_len=22, B, hidden_dim=512)
            encoder_input = encoder_input.permute(1, 0, 2)

            # ---- Create Attention Mask for Padding ----
            # Don't mask [CLS] or proprioception tokens
            # Create (B, 2) tensor of False (not padded)
            cls_joint_is_pad = torch.full((bs, 2), False).to(prop_obs.device)

            # Concatenate with action sequence padding mask
            # cls_joint_is_pad: (B, 2) for [CLS] and proprioception
            # is_pad: (B, T=20) for action sequence
            # Result: (B, 22) full padding mask
            is_pad = torch.cat([cls_joint_is_pad, is_pad], dim=1)

            # ---- Get Positional Encoding ----
            # self.pos_table: (1, seq_len=22, hidden_dim=512)
            # Provides temporal/positional information to encoder
            pos_embed = self.pos_table.clone().detach()  # Detach to avoid gradients

            # Transpose to match encoder input format
            # Before: (1, seq_len=22, hidden_dim=512)
            # After: (seq_len=22, 1, hidden_dim=512)
            pos_embed = pos_embed.permute(1, 0, 2)
#==End

#==Start
            # ---- Run VAE Encoder ----
            # Input: encoder_input (seq_len=22, B, hidden_dim=512)
            # Positional encoding: pos_embed (seq_len=22, 1, hidden_dim=512)
            # Attention mask: is_pad (B, seq_len=22) where True = ignore
            # Output: (seq_len=22, B, hidden_dim=512) encoded representations
            encoder_output = self.encoder(
                encoder_input,
                pos=pos_embed,
                src_key_padding_mask=is_pad
            )

            # Extract [CLS] token output (first position)
            # encoder_output[0]: (B, hidden_dim=512)
            # This aggregates information from entire sequence
            encoder_output = encoder_output[0]

            # ---- Project to Latent Distribution Parameters ----
            # Input: (B, hidden_dim=512)
            # Output: (B, latent_dim*2=64)
            latent_info = self.latent_proj(encoder_output)

            # Split into mean and log-variance
            # First 32 dims = μ (mean)
            # Last 32 dims = log(σ²) (log-variance)
            mu = latent_info[:, :self.latent_dim]              # (B, 32)
            logvar = latent_info[:, self.latent_dim:]          # (B, 32)

            # ---- Sample Latent Variable using Reparameterization Trick ----
            # z = μ + σ * ε, where ε ~ N(0,1)
            # This allows backpropagation through sampling
            # Result: (B, latent_dim=32)
            latent_sample = self._reparametrize(mu, logvar)

        else:
            # ========================================================================
            # INFERENCE PATH (no VAE encoder)
            # ========================================================================
            # Set distribution parameters to None (no KL loss in inference)
            mu = logvar = None

            # Sample latent from standard normal prior
            # z ~ N(0, 1)
            # Shape: (B, latent_dim=32)
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(self.device)
#==End

#==Start
        # ========================================================================
        # PROJECT LATENT TO DECODER INPUT
        # ========================================================================
        # Transform latent z to transformer feature space
        # Input: (B, latent_dim=32)
        # Output: (B, hidden_dim=512)
        latent_input = self.latent_out_proj(latent_sample)
#==End

#==Start
        # ========================================================================
        # PROCESS VISUAL OBSERVATIONS
        # ========================================================================
        # Extract visual observations (RGB or RGBD)
        # vs is dict with camera names as keys
        # Each value is (B, L=1, C=4, H=240, W=240) for RGBD
        vs = obs["rgbd"] if self._use_depth else obs["rgb"]

        # Run vision backbone (ResNet18)
        # Input: dict of (B, L=1, C=4, H=240, W=240) tensors (one per camera)
        # Output: dict of (B*L, resnet_output_dim=512, h, w) spatial feature maps
        # where h, w are reduced spatial dimensions (e.g., 7×7 after downsampling)
        resnet_output = self.obs_backbone(vs)

        # Collect features and positional encodings from all cameras
        all_cam_features = []  # Will store projected features
        all_cam_pos = []       # Will store positional encodings

        # Process each camera's features
        for features in resnet_output.values():
            # features: (B*L, resnet_output_dim=512, h, w)

            # ---- Generate Positional Encoding for Spatial Locations ----
            # Input: (B*L, 512, h, w)
            # Output: (B*L, hidden_dim=512, h, w)
            # Encodes (x, y) spatial positions using sine/cosine functions
            pos = self.position_embedding(features)

            # ---- Project Features to Transformer Dimension ----
            # Input: (B*L, 512, h, w)
            # Output: (B*L, hidden_dim=512, h, w)
            # 1×1 convolution to match transformer feature dimension
            all_cam_features.append(self.input_proj(features))

            # Store positional encoding
            all_cam_pos.append(pos)
#==End

#==Start
        # ========================================================================
        # PROCESS PROPRIOCEPTION FOR DECODER
        # ========================================================================
        # Project proprioception to transformer feature space
        # Input: (B*L, prop_dim=37)
        # Output: (B*L, hidden_dim=512)
        proprio_input = self.input_proj_robot_state(prop_obs)

        # ========================================================================
        # PREPARE DECODER INPUTS
        # ========================================================================
        # Concatenate features from all cameras along width dimension
        # Each camera: (B*L, hidden_dim=512, h, w)
        # If 3 cameras: (B*L, hidden_dim=512, h, 3*w)
        # This creates a "panoramic" view of all cameras
        src = torch.cat(all_cam_features, axis=3)

        # Concatenate positional encodings similarly
        # Result: (B*L, hidden_dim=512, h, 3*w)
        pos = torch.cat(all_cam_pos, axis=3)
#==End

#==Start
        # ========================================================================
        # RUN TRANSFORMER DECODER
        # ========================================================================
        # The decoder performs cross-attention:
        # - Query: Action slot embeddings (self.query_embed.weight)
        # - Key/Value: Visual features + proprioception + latent
        #
        # Inputs:
        #   src: Visual features (B*L, hidden_dim=512, h, 3*w)
        #   mask: None (no masking)
        #   query_embed: Action queries (num_queries=20, hidden_dim=512)
        #   pos: Positional encoding (B*L, hidden_dim=512, h, 3*w)
        #   latent_input: Latent z (B, hidden_dim=512)
        #   proprio_input: Robot state (B*L, hidden_dim=512)
        #   additional_pos_embed: Position IDs for proprio/latent (2, hidden_dim=512)
        #
        # Output: List of decoder layer outputs, we take the last one [-1]
        # Shape: (B, num_queries=20, hidden_dim=512)
        hs = self.transformer(
            src,                                    # Visual features
            None,                                   # No mask
            self.query_embed.weight,               # Learnable action queries
            pos,                                    # Spatial positional encoding
            latent_input,                          # Latent variable z
            proprio_input,                         # Proprioception
            self.additional_pos_embed.weight       # Position IDs for conditioning
        )[-1]  # Take output from last decoder layer

        # ========================================================================
        # GENERATE ACTION PREDICTIONS
        # ========================================================================
        # Project decoder outputs to action space
        # Input: (B, num_queries=20, hidden_dim=512)
        # Output: (B, num_queries=20, action_dim=23)
        # These are normalized actions in [-1, 1]
        a_hat = self.action_head(hs)

        # Return predictions and distribution parameters
        # During training: [mu, logvar] are tensors for KL loss
        # During inference: [None, None]
        return a_hat, [mu, logvar]
#==End

#==Start
    # ============================================================================
    # INFERENCE - Generate actions for deployment
    # ============================================================================
    @torch.no_grad()  # Disable gradient computation for inference
    def act(self, obs: dict) -> torch.Tensor:
        """
        Generate action for deployment (called at every control timestep).

        Flow:
        1. Preprocess observations (normalize, format)
        2. Run forward pass to get action predictions
        3. Apply temporal ensemble (if enabled)
        4. Denormalize actions to actual robot commands

        Args:
            obs: Raw observations from environment

        Returns:
            action: Denormalized action to execute (1, 1, action_dim=23)
        """
        # ========================================================================
        # PREPROCESS OBSERVATIONS
        # ========================================================================
        # Normalize and format observations to match training data
        # Input: Raw observations from environment
        # Output: Processed dict matching training format
        obs = self.process_data(obs, extract_action=False)

        # ========================================================================
        # FORWARD PASS
        # ========================================================================
        # Run model to predict action sequence
        # Input: Processed observations
        # Output: a_hat (1, T=20, A=23), [None, None]
        # We only need the action predictions, not the VAE parameters
        a_hat = self.forward(obs=obs)[0]  # Shape: (1, T=20, A=23)
#==End

#==Start
        # ========================================================================
        # TEMPORAL ENSEMBLE (if enabled)
        # ========================================================================
        if self.temporal_ensemble:
            # ACT predicts T=20 future actions but executes only 1
            # Temporal ensemble maintains a buffer of recent predictions
            # and averages them for smooth execution

            # Add new predictions to buffer
            # a_hat[0]: (T=20, A=23) current prediction
            self._action_buffer.append(a_hat[0])

            # Extract the action for current timestep from all buffered predictions
            # Buffer structure after n steps (n ≥ 20):
            #   buffer[0]: [a₀, a₁, ..., a₁₉] from t=0
            #   buffer[1]: [a₀, a₁, ..., a₁₉] from t=1
            #   ...
            #   buffer[19]: [a₀, a₁, ..., a₁₉] from t=19
            #
            # For current timestep, we want:
            #   buffer[0][19], buffer[1][18], ..., buffer[19][0]
            # These all predict the action for the same timestep from different times
            actions_for_curr_step = torch.stack([
                self._action_buffer[i][self._horizon - i - 1]  # Diagonal extraction
                for i in range(self._horizon)
            ])  # Shape: (20, A=23)

            # Filter out zero-initialized entries (from startup)
            # actions_populated: (20,) boolean mask, True where non-zero
            actions_populated = torch.all(actions_for_curr_step != 0, axis=1)

            # Keep only valid predictions
            # Shape: (n_valid, A=23) where n_valid ≤ 20
            actions_for_curr_step = actions_for_curr_step[actions_populated]

            # ---- Compute Exponentially Weighted Average ----
            # More recent predictions get higher weight
            # Weight decay factor k=0.01
            k = 0.01
            # Compute weights: [e^0, e^(-k), e^(-2k), ...]
            # More recent = higher weight
            exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))

            # Normalize weights to sum to 1
            exp_weights = exp_weights / exp_weights.sum()

            # Convert to torch tensor and add dimension for broadcasting
            # Shape: (n_valid, 1)
            exp_weights = torch.from_numpy(exp_weights).unsqueeze(dim=1).to(self.device)

            # Compute weighted average
            # actions_for_curr_step: (n_valid, A=23)
            # exp_weights: (n_valid, 1)
            # Result: (1, A=23) → (1, 1, A=23)
            a_hat = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True).unsqueeze(0)

        # ========================================================================
        # DENORMALIZE ACTIONS
        # ========================================================================
        # Convert from normalized [-1, 1] to actual robot commands
        # Move to CPU for environment execution
        a_hat = a_hat.cpu()

        # Denormalize using joint ranges
        # Input: (1, 1, A=23) in [-1, 1]
        # Output: (1, 1, A=23) in actual joint units
        return self._denormalize_action(a_hat)
#==End

#==Start
    # ============================================================================
    # RESET - Clear internal state
    # ============================================================================
    def reset(self) -> None:
        """
        Reset policy internal state.
        Called at the beginning of each episode.

        Purpose: Clear temporal ensemble buffer to avoid using
                 predictions from previous episode.
        """
        if self.temporal_ensemble:
            # Re-initialize action buffer with zeros
            self._action_buffer = deque(maxlen=self._horizon)
            for _ in range(self._horizon):
                self._action_buffer.append(
                    torch.zeros((self._horizon, self.action_dim), dtype=torch.float32).to(self.device)
                )
#==End

#==Start
    # ============================================================================
    # TRAINING STEP - Compute loss for one batch
    # ============================================================================
    def policy_training_step(self, batch, batch_idx) -> Any:
        """
        Training step called by PyTorch Lightning.

        Args:
            batch: Dictionary with keys ['obs', 'actions', 'masks']
            batch_idx: Index of batch in epoch

        Returns:
            loss: Scalar loss value
            log_dict: Dictionary of metrics to log
            B: Batch size (for averaging across GPUs)
        """
        # ========================================================================
        # PREPARE ACTION DATA
        # ========================================================================
        # batch['actions'] is a dict: {'base': (B,T,3), 'torso': (B,T,4), ...}
        # Concatenate all action components into single tensor
        # Result: (B, T=20, A=23)
        batch["actions"] = any_concat(
            [batch["actions"][k] for k in self._action_keys], dim=-1
        )

        # Get batch size for later metric averaging
        B = batch["actions"].shape[0]

        # ========================================================================
        # PREPROCESS BATCH DATA
        # ========================================================================
        # Normalize observations, format correctly
        # Input: Raw batch data
        # Output: Processed batch ready for model
        batch = self.process_data(batch, extract_action=True)

        # ========================================================================
        # EXTRACT AND FORMAT PADDING MASK
        # ========================================================================
        # Remove mask from batch dict (will pass separately to forward)
        pad_mask = batch.pop("masks")  # (B, T=20)

        # Flatten batch dimension if needed
        # Before: (B, T=20)
        # After: (B*T,) or (B, T) depending on shape
        pad_mask = pad_mask.reshape(-1, pad_mask.shape[-1])

        # Invert mask: ACT assumes True=padding, False=valid
        # But data uses True=valid, False=padding
        # So we invert it
        pad_mask = ~pad_mask

        # ========================================================================
        # EXTRACT GROUND TRUTH ACTIONS
        # ========================================================================
        # Actions are already normalized to [-1, 1] by process_data
        # Shape: (B, T=20, A=23)
        gt_actions = batch.pop("actions")

        # ========================================================================
        # COMPUTE LOSS
        # ========================================================================
        # Run forward pass and compute L1 + KL loss
        loss_dict = self._compute_loss(
            obs=batch,           # Processed observations
            actions=gt_actions,  # Ground truth actions (B, T=20, A=23)
            is_pad=pad_mask,     # Padding mask (B, T=20)
        )

        # Extract total loss for backprop
        loss = loss_dict["loss"]

        # Prepare logging dictionary (excluding total loss)
        log_dict = {
            "l1": loss_dict["l1"],    # L1 reconstruction loss
            "kl": loss_dict["kl"],    # KL divergence loss
        }

        # Return loss, metrics to log, and batch size
        return loss, log_dict, B
#==End

#==Start
    # ============================================================================
    # VALIDATION STEP - Evaluate without gradients
    # ============================================================================
    def policy_evaluation_step(self, batch, batch_idx) -> Any:
        """
        Validation step (identical to training but without gradients).

        Args:
            batch: Dictionary with keys ['obs', 'actions', 'masks']
            batch_idx: Index of batch in epoch

        Returns:
            Same as policy_training_step
        """
        with torch.no_grad():
            return self.policy_training_step(batch, batch_idx)
#==End

#==Start
    # ============================================================================
    # OPTIMIZER SETUP
    # ============================================================================
    def configure_optimizers(self):
        """
        Setup optimizer and learning rate scheduler.
        Called by PyTorch Lightning.

        Returns:
            If use_cosine_lr=True: ([optimizer], [scheduler_config])
            Otherwise: optimizer
        """
        # ========================================================================
        # CREATE OPTIMIZER PARAMETER GROUPS
        # ========================================================================
        # Groups parameters by weight decay settings
        # Some layers (like LayerNorm, biases) typically don't use weight decay
        optimizer_groups = self._get_optimizer_groups(
            weight_decay=self.weight_decay,     # L2 regularization weight (0.1)
            lr_layer_decay=self.lr_layer_decay, # Layer-wise LR decay (1.0 = no decay)
            lr_scale=1.0,                        # Global LR scale
        )

        # Create AdamW optimizer
        # AdamW is Adam with decoupled weight decay
        optimizer = torch.optim.AdamW(
            optimizer_groups,
            lr=self.lr,                          # Base learning rate (7e-4)
            weight_decay=self.weight_decay,      # L2 regularization (0.1)
        )

        # ========================================================================
        # SETUP LEARNING RATE SCHEDULER (if enabled)
        # ========================================================================
        if self.use_cosine_lr:
            # Cosine annealing schedule with warmup
            # LR schedule: warmup → cosine decay → constant minimum
            scheduler_kwargs = dict(
                base_value=1.0,                           # Start at base LR (multiplier)
                final_value=self.lr_cosine_min / self.lr, # Minimum LR (as fraction of base)
                epochs=self.lr_cosine_steps,             # Total steps for schedule (300k)
                warmup_start_value=self.lr_cosine_min / self.lr, # Start warmup at min LR
                warmup_epochs=self.lr_warmup_steps,      # Warmup duration (1000 steps)
                steps_per_epoch=1,                        # Call scheduler every step
            )

            # Create cosine schedule function
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=optimizer,
                lr_lambda=CosineScheduleFunction(**scheduler_kwargs),
            )

            # Return optimizer and scheduler
            # interval='step' means update LR every optimizer step (not epoch)
            return (
                [optimizer],
                [{"scheduler": scheduler, "interval": "step"}],
            )

        # If no scheduler, just return optimizer
        return optimizer
#==End

#==Start
    # ============================================================================
    # DATA PREPROCESSING
    # ============================================================================
    def process_data(self, data_batch: dict, extract_action: bool = False) -> Any:
        """
        Preprocess observations and optionally actions.

        Converts raw data to model-ready format:
        - Normalizes RGB images to [0, 1]
        - Normalizes depth images to [0, 1]
        - Concatenates RGB+D into 4-channel input
        - Extracts specified proprioceptive features

        Args:
            data_batch: Raw batch from dataloader
            extract_action: Whether to include actions in output

        Returns:
            data: Processed dictionary ready for model
        """
        # ========================================================================
        # EXTRACT PROPRIOCEPTIVE DATA
        # ========================================================================
        # Start with basic robot state
        data = {
            "qpos": data_batch["obs"]["qpos"],  # Joint positions
            "eef": data_batch["obs"]["eef"],    # End-effector poses
        }

        # Add odometry if available
        if "odom" in data_batch["obs"]:
            data["odom"] = data_batch["obs"]["odom"]  # Base velocity

        # ========================================================================
        # PROCESS RGB OBSERVATIONS (if used)
        # ========================================================================
        if "rgb" in self._features:
            # Find all RGB camera observations
            # Keys look like 'robot_r1::robot_r1:left_realsense_link:Camera:0::rgb'

            # Extract camera name (remove '::rgb' suffix)
            # Normalize pixel values from [0, 255] to [0, 1]
            data["rgb"] = {
                k.rsplit("::", 1)[0]: data_batch["obs"][k].float() / 255.0
                for k in data_batch["obs"] if "rgb" in k
            }
            # Result: dict of (B, L=1, 3, 240, 240) tensors

        # ========================================================================
        # PROCESS RGB-D OBSERVATIONS (if used)
        # ========================================================================
        if "rgbd" in self._features:
            # ---- Process RGB ----
            # Same as above: normalize to [0, 1]
            rgb = {
                k.rsplit("::", 1)[0]: data_batch["obs"][k].float() / 255.0
                for k in data_batch["obs"] if "rgb" in k
            }
            # rgb: dict of (B, L=1, 3, 240, 240) tensors

            # ---- Process Depth ----
            # Normalize depth from [MIN_DEPTH, MAX_DEPTH] to [0, 1]
            # MIN_DEPTH and MAX_DEPTH are constants from OmniGibson
            depth = {
                k.rsplit("::", 1)[0]: (data_batch["obs"][k].float() - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH)
                for k in data_batch["obs"] if "depth" in k
            }
            # depth: dict of (B, L=1, 1, 240, 240) tensors (note: 1 channel)

            # ---- Concatenate RGB + D ----
            # For each camera, concatenate RGB (3 channels) and depth (1 channel)
            # Result: 4-channel RGBD image
            data["rgbd"] = {
                k: torch.cat([rgb[k], depth[k].unsqueeze(-3)], dim=-3)
                for k in rgb
            }
            # rgbd: dict of (B, L=1, 4, 240, 240) tensors

        # ========================================================================
        # PROCESS TASK CONDITIONING (if used)
        # ========================================================================
        if "task" in self._features:
            # Task information could be one-hot encoding of task ID
            # or language embedding of task description
            data["task"] = data_batch["obs"]["task"]

        # ========================================================================
        # EXTRACT ACTIONS AND MASKS (if requested)
        # ========================================================================
        if extract_action:
            # Actions are already normalized in [-1, 1] by data pipeline
            data.update({
                "actions": data_batch["actions"],  # (B, T=20, A=23)
                "masks": data_batch["masks"],      # (B, T=20) True=valid
            })

        return data
#==End

#==Start
    # ============================================================================
    # OPTIMIZER UTILITIES
    # ============================================================================
    def _get_optimizer_groups(self, weight_decay, lr_layer_decay, lr_scale=1.0):
        """
        Create parameter groups for optimizer.

        Separates parameters by whether they should use weight decay.
        Typically biases and normalization parameters don't use weight decay.

        Args:
            weight_decay: L2 regularization weight
            lr_layer_decay: Layer-wise learning rate decay factor
            lr_scale: Global learning rate scale

        Returns:
            List of parameter group dictionaries
        """
        # Use default grouping from il_lib.optim
        # This handles layer normalization, biases, etc.
        head_pg, _ = default_optimizer_groups(
            self,
            weight_decay=weight_decay,
            lr_scale=lr_scale,
        )
        return head_pg
#==End

#==Start
    # ============================================================================
    # LOSS COMPUTATION
    # ============================================================================
    def _compute_loss(self, obs, actions, is_pad):
        """
        Compute ACT training loss (L1 + KL).

        Loss = L1(predicted_actions, gt_actions) + β * KL(q(z|x) || p(z))
        where:
        - L1 is action reconstruction error
        - KL is divergence from prior N(0,1)
        - β is self.kl_weight (typically 10)

        Args:
            obs: Processed observations
            actions: Ground truth actions (B, T=20, A=23)
            is_pad: Padding mask (B, T=20), True=padding

        Returns:
            loss_dict: Dictionary with 'l1', 'kl', and 'loss' keys
        """
        # ========================================================================
        # TRUNCATE TO PREDICTION HORIZON
        # ========================================================================
        # Ensure we only predict/supervise num_queries timesteps
        # If data has T=20 and num_queries=20, no change
        # If data has T>20, truncate to 20
        actions = actions[:, :self.num_queries]     # (B, 20, A=23)
        is_pad = is_pad[:, :self.num_queries]       # (B, 20)

        # ========================================================================
        # FORWARD PASS
        # ========================================================================
        # Run model to get predictions and VAE distribution
        # a_hat: (B, num_queries=20, action_dim=23)
        # mu, logvar: (B, latent_dim=32) each
        a_hat, (mu, logvar) = self.forward(
            obs=obs,
            actions=actions,
            is_pad=is_pad,
        )

        # ========================================================================
        # COMPUTE KL DIVERGENCE
        # ========================================================================
        # KL(q(z|x) || N(0,1)) for each sample in batch
        # Returns: (B,) tensor of KL values
        # We take [0] to get scalar for this batch
        total_kld = self._kl_divergence(mu, logvar)[0]

        # ========================================================================
        # COMPUTE ACTION RECONSTRUCTION LOSS
        # ========================================================================
        loss_dict = dict()

        # Compute L1 loss for all timesteps (including padded)
        # all_l1: (B, num_queries=20, action_dim=23)
        all_l1 = F.l1_loss(actions, a_hat, reduction="none")

        # Mask out padded timesteps and average
        # ~is_pad: (B, 20) boolean, True where valid
        # unsqueeze(-1): (B, 20, 1) for broadcasting across action dims
        # Result: only compute loss on valid (non-padded) timesteps
        l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()

        # ========================================================================
        # COMBINE LOSSES
        # ========================================================================
        loss_dict["l1"] = l1           # Action reconstruction error
        loss_dict["kl"] = total_kld[0] # KL divergence (scalar)

        # Total loss = L1 + β * KL
        # β = self.kl_weight (typically 10)
        loss_dict["loss"] = loss_dict["l1"] + loss_dict["kl"] * self.kl_weight

        return loss_dict
#==End

#==Start
    # ============================================================================
    # VAE UTILITIES
    # ============================================================================
    def _reparametrize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for VAE.

        Instead of sampling z ~ N(μ, σ²) directly (not differentiable),
        we use: z = μ + σ * ε where ε ~ N(0, 1)

        This allows gradients to flow through μ and σ.

        Args:
            mu: Mean of latent distribution (B, latent_dim=32)
            logvar: Log-variance (B, latent_dim=32)

        Returns:
            z: Sampled latent variable (B, latent_dim=32)
        """
        # Compute standard deviation from log-variance
        # σ = exp(log(σ²) / 2) = exp(log(σ))
        std = logvar.div(2).exp()

        # Sample ε ~ N(0, 1) with same shape as std
        eps = Variable(std.data.new(std.size()).normal_())

        # Reparameterization: z = μ + σ * ε
        return mu + std * eps
#==End

#==Start
    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """
        Create sinusoidal positional encoding table.

        Same as original Transformer paper (Vaswani et al., 2017).
        For position i and dimension j:
        - PE(i, 2j) = sin(i / 10000^(2j/d))
        - PE(i, 2j+1) = cos(i / 10000^(2j/d))

        Args:
            n_position: Number of positions (sequence length)
            d_hid: Hidden dimension (feature size)

        Returns:
            Positional encoding table (1, n_position, d_hid)
        """
        def get_position_angle_vec(position):
            """Compute angle for one position across all dimensions."""
            return [
                position / np.power(10000, 2 * (hid_j // 2) / d_hid)
                for hid_j in range(d_hid)
            ]

        # Create table of angles for all positions
        # Shape: (n_position, d_hid)
        sinusoid_table = np.array(
            [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
        )

        # Apply sin to even indices
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])

        # Apply cos to odd indices
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

        # Convert to torch tensor and add batch dimension
        # Shape: (1, n_position, d_hid)
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)
#==End

#==Start
    def _kl_divergence(self, mu, logvar):
        """
        Compute KL divergence KL(q(z|x) || p(z)) for VAE.

        For q(z|x) = N(μ, σ²) and p(z) = N(0, 1):
        KL = 0.5 * Σ(1 + log(σ²) - μ² - σ²)

        Negative sign because we're minimizing the ELBO.

        Args:
            mu: Mean (B, latent_dim=32)
            logvar: Log-variance (B, latent_dim=32)

        Returns:
            total_kld: Sum over dimensions, mean over batch (scalar)
            dimension_wise_kld: KL per dimension (latent_dim,)
            mean_kld: Mean KL over batch (scalar)
        """
        batch_size = mu.size(0)
        assert batch_size != 0

        # Handle 4D tensors (e.g., from convolutional VAEs)
        if mu.data.ndimension() == 4:
            mu = mu.view(mu.size(0), mu.size(1))
        if logvar.data.ndimension() == 4:
            logvar = logvar.view(logvar.size(0), logvar.size(1))

        # Compute KL divergence per dimension and sample
        # Formula: -0.5 * (1 + log(σ²) - μ² - σ²)
        # Shape: (B, latent_dim=32)
        klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

        # Sum over dimensions, mean over batch
        # Result: (1,) tensor
        total_kld = klds.sum(1).mean(0, True)

        # Mean KL per dimension (for analysis/logging)
        # Result: (latent_dim=32,) tensor
        dimension_wise_kld = klds.mean(0)

        # Mean KL over both dimensions and batch
        # Result: (1,) tensor
        mean_kld = klds.mean(1).mean(0, True)

        return total_kld, dimension_wise_kld, mean_kld
#==End
