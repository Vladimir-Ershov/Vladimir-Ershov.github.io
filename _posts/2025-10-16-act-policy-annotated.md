---
layout: annotated-code
title: "ACT Policy: Complete Annotated Walkthrough"
date: 2025-10-16
categories: [machine-learning]
tags: [ACT, transformer, imitation-learning, robotics, VAE, deep-learning]
description: "A comprehensive, line-by-line annotated walkthrough of Action Chunking with Transformers (ACT) - covering architecture, training, inference, and implementation details"
intro: |
  This is a **complete walkthrough** of the Action Chunking with Transformers (ACT) policy implementation. Every line of code is accompanied by detailed explanations covering:

  - **Architecture**: Vision backbone, VAE encoder, Transformer decoder
  - **Training**: Loss computation, optimizer setup, data preprocessing
  - **Inference**: Temporal ensemble, action generation
  - **Implementation Details**: Tensor shapes, design decisions, PyTorch Lightning integration

  ACT combines visual observations (RGB-D from multiple cameras), proprioception, and a Conditional VAE to generate smooth, multi-step action predictions for robot manipulation tasks.

  **Paper**: [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705)

  The two-column format pairs code (left) with detailed annotations (right). As you scroll, the focused code block and its annotation are highlighted!
---
<p></p>
<div class="code-block" data-index="0">
<pre><code class="language-python">
class ACTMod(BasePolicy):
    def __init__(
        self,
        *args,
        prop_dim: int,                    # Proprioception dimension (37 for R1Pro robot)
        prop_keys: List[str],             # Which proprioceptive features to use (e.g., ['qpos/torso', 'eef/left_pos'])
        action_dim: int,                  # Action dimension (23 for R1Pro robot)
        action_keys: List[str],           # Which action components to predict (e.g., ['base', 'torso', 'left_arm'])
        features: List[str],              # Input modalities (e.g., ['proprioception', 'rgbd'])
        obs_backbone: DictConfig,         # Config for vision backbone (ResNet18)
        pos_encoding: DictConfig,         # Config for positional encoding
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
        lr: float,                        # Learning rate
        use_cosine_lr: bool = False,      # Use cosine annealing schedule
        lr_warmup_steps: Optional[int] = None,     # Warmup steps for lr schedule
        lr_cosine_steps: Optional[int] = None,     # Total steps for cosine schedule
        lr_cosine_min: Optional[float] = None,     # Minimum lr for cosine schedule
        lr_layer_decay: float = 1.0,      # Layer-wise lr decay factor
        weight_decay: float = 0.0,        # L2 regularization weight
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
</code></pre>
</div>

<div class="code-block" data-index="1">
<pre><code class="language-python">
self._prop_keys = prop_keys
        self._action_keys = action_keys
        self.action_dim = action_dim
        self._features = features
        self._use_depth = obs_backbone.include_depth
</code></pre>
</div>

<div class="code-block" data-index="2">
<pre><code class="language-python">
self.obs_backbone = instantiate(obs_backbone)
</code></pre>
</div>

<div class="code-block" data-index="3">
<pre><code class="language-python">
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
</code></pre>
</div>

<div class="code-block" data-index="4">
<pre><code class="language-python">
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
</code></pre>
</div>

<div class="code-block" data-index="5">
<pre><code class="language-python">
self.position_embedding = build_position_encoding(pos_encoding)
        self.num_queries = num_queries
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
</code></pre>
</div>

<div class="code-block" data-index="6">
<pre><code class="language-python">
self.input_proj = nn.Conv2d(obs_backbone.resnet_output_dim, hidden_dim, kernel_size=1)
        self.input_proj_robot_state = nn.Linear(prop_dim, hidden_dim)
</code></pre>
</div>

<div class="code-block" data-index="7">
<pre><code class="language-python">
self.latent_dim = 32
        self.cls_embed = nn.Embedding(1, hidden_dim)
        self.encoder_action_proj = nn.Linear(action_dim, hidden_dim)
        self.encoder_prop_proj = nn.Linear(prop_dim, hidden_dim)
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim * 2)
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(1+1+num_queries, hidden_dim))
</code></pre>
</div>

<div class="code-block" data-index="8">
<pre><code class="language-python">
self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim)
        self.additional_pos_embed = nn.Embedding(2, hidden_dim)
</code></pre>
</div>

<div class="code-block" data-index="9">
<pre><code class="language-python">
self.temporal_ensemble = temporal_ensemble
        if temporal_ensemble:
            self._horizon = num_queries  # 20
            self._action_buffer = deque(maxlen=self._horizon)
            for _ in range(self._horizon):
                self._action_buffer.append(
                    torch.zeros((self._horizon, self.action_dim), dtype=torch.float32).to(self.device)
                )
</code></pre>
</div>

<div class="code-block" data-index="10">
<pre><code class="language-python">
self.kl_weight = kl_weight
        self.lr = lr
        self.use_cosine_lr = use_cosine_lr
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_cosine_steps = lr_cosine_steps
        self.lr_cosine_min = lr_cosine_min
        self.lr_layer_decay = lr_layer_decay
        self.weight_decay = weight_decay
        self.save_hyperparameters()
</code></pre>
</div>

<div class="code-block" data-index="11">
<pre><code class="language-python">
def forward(
        self,
        obs: dict,                          # Observations: {'rgbd': dict, 'qpos': dict, 'eef': dict, 'odom': dict}
        actions: Optional[torch.Tensor] = None,  # Ground truth actions (B, T=20, A=23), None during inference
        is_pad: Optional[torch.Tensor] = None    # Padding mask (B, T=20), True where padded
    ) -> torch.Tensor:
        is_training = actions is not None
        bs = get_batch_size(obs, strict=True)
</code></pre>
</div>

<div class="code-block" data-index="12">
<pre><code class="language-python">
prop_obs = []
        for prop_key in self._prop_keys:
            if "/" in prop_key:
                group, key = prop_key.split("/")  # Split 'qpos/torso' → 'qpos', 'torso'
                prop_obs.append(obs[group][key])  # Extract obs['qpos']['torso']
            else:
                prop_obs.append(obs[prop_key])
        prop_obs = torch.cat(prop_obs, dim=-1)
        prop_obs = prop_obs.reshape(-1, prop_obs.shape[-1])
</code></pre>
</div>

<div class="code-block" data-index="13">
<pre><code class="language-python">
if is_training:
            action_embed = self.encoder_action_proj(actions)
            prop_embed = self.encoder_prop_proj(prop_obs)
            prop_embed = torch.unsqueeze(prop_embed, dim=1)
            cls_embed = self.cls_embed.weight
            cls_embed = torch.unsqueeze(cls_embed, dim=0).repeat(bs, 1, 1)
            encoder_input = torch.cat([cls_embed, prop_embed, action_embed], dim=1)
            encoder_input = encoder_input.permute(1, 0, 2)
            cls_joint_is_pad = torch.full((bs, 2), False).to(prop_obs.device)
            is_pad = torch.cat([cls_joint_is_pad, is_pad], dim=1)
            pos_embed = self.pos_table.clone().detach()  # Detach to avoid gradients
            pos_embed = pos_embed.permute(1, 0, 2)
</code></pre>
</div>

<div class="code-block" data-index="14">
<pre><code class="language-python">
encoder_output = self.encoder(
                encoder_input,
                pos=pos_embed,
                src_key_padding_mask=is_pad
            )
            encoder_output = encoder_output[0]
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, :self.latent_dim]              # (B, 32)
            logvar = latent_info[:, self.latent_dim:]          # (B, 32)
            latent_sample = self._reparametrize(mu, logvar)
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(self.device)
</code></pre>
</div>

<div class="code-block" data-index="15">
<pre><code class="language-python">
latent_input = self.latent_out_proj(latent_sample)
</code></pre>
</div>

<div class="code-block" data-index="16">
<pre><code class="language-python">
vs = obs["rgbd"] if self._use_depth else obs["rgb"]
        resnet_output = self.obs_backbone(vs)
        all_cam_features = []  # Will store projected features
        all_cam_pos = []       # Will store positional encodings
        for features in resnet_output.values():
            pos = self.position_embedding(features)
            all_cam_features.append(self.input_proj(features))
            all_cam_pos.append(pos)
</code></pre>
</div>

<div class="code-block" data-index="17">
<pre><code class="language-python">
proprio_input = self.input_proj_robot_state(prop_obs)
        src = torch.cat(all_cam_features, axis=3)
        pos = torch.cat(all_cam_pos, axis=3)
</code></pre>
</div>

<div class="code-block" data-index="18">
<pre><code class="language-python">
hs = self.transformer(
            src,                                    # Visual features
            None,                                   # No mask
            self.query_embed.weight,               # Learnable action queries
            pos,                                    # Spatial positional encoding
            latent_input,                          # Latent variable z
            proprio_input,                         # Proprioception
            self.additional_pos_embed.weight       # Position IDs for conditioning
        )[-1]  # Take output from last decoder layer
        a_hat = self.action_head(hs)
        return a_hat, [mu, logvar]
</code></pre>
</div>

<div class="code-block" data-index="19">
<pre><code class="language-python">
@torch.no_grad()  # Disable gradient computation for inference
    def act(self, obs: dict) -> torch.Tensor:
        obs = self.process_data(obs, extract_action=False)
        a_hat = self.forward(obs=obs)[0]  # Shape: (1, T=20, A=23)
</code></pre>
</div>

<div class="code-block" data-index="20">
<pre><code class="language-python">
if self.temporal_ensemble:
            self._action_buffer.append(a_hat[0])
            actions_for_curr_step = torch.stack([
                self._action_buffer[i][self._horizon - i - 1]  # Diagonal extraction
                for i in range(self._horizon)
            ])  # Shape: (20, A=23)
            actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
            actions_for_curr_step = actions_for_curr_step[actions_populated]
            k = 0.01
            exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
            exp_weights = exp_weights / exp_weights.sum()
            exp_weights = torch.from_numpy(exp_weights).unsqueeze(dim=1).to(self.device)
            a_hat = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True).unsqueeze(0)
        a_hat = a_hat.cpu()
        return self._denormalize_action(a_hat)
</code></pre>
</div>

<div class="code-block" data-index="21">
<pre><code class="language-python">
def reset(self) -> None:
        if self.temporal_ensemble:
            self._action_buffer = deque(maxlen=self._horizon)
            for _ in range(self._horizon):
                self._action_buffer.append(
                    torch.zeros((self._horizon, self.action_dim), dtype=torch.float32).to(self.device)
                )
</code></pre>
</div>

<div class="code-block" data-index="22">
<pre><code class="language-python">
def policy_training_step(self, batch, batch_idx) -> Any:
        batch["actions"] = any_concat(
            [batch["actions"][k] for k in self._action_keys], dim=-1
        )
        B = batch["actions"].shape[0]
        batch = self.process_data(batch, extract_action=True)
        pad_mask = batch.pop("masks")  # (B, T=20)
        pad_mask = pad_mask.reshape(-1, pad_mask.shape[-1])
        pad_mask = ~pad_mask
        gt_actions = batch.pop("actions")
        loss_dict = self._compute_loss(
            obs=batch,           # Processed observations
            actions=gt_actions,  # Ground truth actions (B, T=20, A=23)
            is_pad=pad_mask,     # Padding mask (B, T=20)
        )
        loss = loss_dict["loss"]
        log_dict = {
            "l1": loss_dict["l1"],    # L1 reconstruction loss
            "kl": loss_dict["kl"],    # KL divergence loss
        }
        return loss, log_dict, B
</code></pre>
</div>

<div class="code-block" data-index="23">
<pre><code class="language-python">
def policy_evaluation_step(self, batch, batch_idx) -> Any:
        with torch.no_grad():
            return self.policy_training_step(batch, batch_idx)
</code></pre>
</div>

<div class="code-block" data-index="24">
<pre><code class="language-python">
def configure_optimizers(self):
        optimizer_groups = self._get_optimizer_groups(
            weight_decay=self.weight_decay,     # L2 regularization weight (0.1)
            lr_layer_decay=self.lr_layer_decay, # Layer-wise LR decay (1.0 = no decay)
            lr_scale=1.0,                        # Global LR scale
        )
        optimizer = torch.optim.AdamW(
            optimizer_groups,
            lr=self.lr,                          # Base learning rate (7e-4)
            weight_decay=self.weight_decay,      # L2 regularization (0.1)
        )
        if self.use_cosine_lr:
            scheduler_kwargs = dict(
                base_value=1.0,                           # Start at base LR (multiplier)
                final_value=self.lr_cosine_min / self.lr, # Minimum LR (as fraction of base)
                epochs=self.lr_cosine_steps,             # Total steps for schedule (300k)
                warmup_start_value=self.lr_cosine_min / self.lr, # Start warmup at min LR
                warmup_epochs=self.lr_warmup_steps,      # Warmup duration (1000 steps)
                steps_per_epoch=1,                        # Call scheduler every step
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=optimizer,
                lr_lambda=CosineScheduleFunction(**scheduler_kwargs),
            )
            return (
                [optimizer],
                [{"scheduler": scheduler, "interval": "step"}],
            )
        return optimizer
</code></pre>
</div>

<div class="code-block" data-index="25">
<pre><code class="language-python">
def process_data(self, data_batch: dict, extract_action: bool = False) -> Any:
        data = {
            "qpos": data_batch["obs"]["qpos"],  # Joint positions
            "eef": data_batch["obs"]["eef"],    # End-effector poses
        }
        if "odom" in data_batch["obs"]:
            data["odom"] = data_batch["obs"]["odom"]  # Base velocity
        if "rgb" in self._features:
            data["rgb"] = {
                k.rsplit("::", 1)[0]: data_batch["obs"][k].float() / 255.0
                for k in data_batch["obs"] if "rgb" in k
            }
        if "rgbd" in self._features:
            rgb = {
                k.rsplit("::", 1)[0]: data_batch["obs"][k].float() / 255.0
                for k in data_batch["obs"] if "rgb" in k
            }
            depth = {
                k.rsplit("::", 1)[0]: (data_batch["obs"][k].float() - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH)
                for k in data_batch["obs"] if "depth" in k
            }
            data["rgbd"] = {
                k: torch.cat([rgb[k], depth[k].unsqueeze(-3)], dim=-3)
                for k in rgb
            }
        if "task" in self._features:
            data["task"] = data_batch["obs"]["task"]
        if extract_action:
            data.update({
                "actions": data_batch["actions"],  # (B, T=20, A=23)
                "masks": data_batch["masks"],      # (B, T=20) True=valid
            })
        return data
</code></pre>
</div>

<div class="code-block" data-index="26">
<pre><code class="language-python">
def _get_optimizer_groups(self, weight_decay, lr_layer_decay, lr_scale=1.0):
        head_pg, _ = default_optimizer_groups(
            self,
            weight_decay=weight_decay,
            lr_scale=lr_scale,
        )
        return head_pg
</code></pre>
</div>

<div class="code-block" data-index="27">
<pre><code class="language-python">
def _compute_loss(self, obs, actions, is_pad):
        actions = actions[:, :self.num_queries]     # (B, 20, A=23)
        is_pad = is_pad[:, :self.num_queries]       # (B, 20)
        a_hat, (mu, logvar) = self.forward(
            obs=obs,
            actions=actions,
            is_pad=is_pad,
        )
        total_kld = self._kl_divergence(mu, logvar)[0]
        loss_dict = dict()
        all_l1 = F.l1_loss(actions, a_hat, reduction="none")
        l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
        loss_dict["l1"] = l1           # Action reconstruction error
        loss_dict["kl"] = total_kld[0] # KL divergence (scalar)
        loss_dict["loss"] = loss_dict["l1"] + loss_dict["kl"] * self.kl_weight
        return loss_dict
</code></pre>
</div>

<div class="code-block" data-index="28">
<pre><code class="language-python">
def _reparametrize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = logvar.div(2).exp()
        eps = Variable(std.data.new(std.size()).normal_())
        return mu + std * eps
</code></pre>
</div>

<div class="code-block" data-index="29">
<pre><code class="language-python">
def _get_sinusoid_encoding_table(self, n_position, d_hid):
        def get_position_angle_vec(position):
            return [
                position / np.power(10000, 2 * (hid_j // 2) / d_hid)
                for hid_j in range(d_hid)
            ]
        sinusoid_table = np.array(
            [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
        )
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)
</code></pre>
</div>

<div class="code-block" data-index="30">
<pre><code class="language-python">
def _kl_divergence(self, mu, logvar):
        batch_size = mu.size(0)
        assert batch_size != 0
        if mu.data.ndimension() == 4:
            mu = mu.view(mu.size(0), mu.size(1))
        if logvar.data.ndimension() == 4:
            logvar = logvar.view(logvar.size(0), logvar.size(1))
        klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        total_kld = klds.sum(1).mean(0, True)
        dimension_wise_kld = klds.mean(0)
        mean_kld = klds.mean(1).mean(0, True)
        return total_kld, dimension_wise_kld, mean_kld
</code></pre>
</div>


<script>
document.addEventListener('DOMContentLoaded', function() {
  const annotationColumn = document.querySelector('.ac-annotation-column');
  if (!annotationColumn) return;

  const annotations = [{"index": 0, "html": "<p>Action Chunking with Transformers (ACT) Policy</p>\n<p>ACT is a behavior cloning method that:\n1. Processes visual observations (RGB-D from multiple cameras) through a ResNet\n2. Combines visual features with proprioceptive state (joint positions, velocities, etc.)\n3. Uses a Conditional VAE (CVAE) to model action distributions\n4. Generates multiple future actions (action chunking) in one forward pass\n5. Applies temporal ensemble at test time for smooth execution</p>\n"}, {"index": 1, "html": "<p>SAVE CONFIGURATION PARAMETERS\nStore which proprioceptive features to use\nExample: ['odom/base_velocity', 'qpos/torso', 'qpos/left_arm', 'eef/left_pos', ...]\nStore which action components to predict\nExample: ['base', 'torso', 'left_arm', 'left_gripper', 'right_arm', 'right_gripper']\nTotal action dimension (A=23 for R1Pro)\nInput modalities to use (e.g., ['proprioception', 'rgbd'])\nCould also include ['task'] for task conditioning\nWhether we're using depth images in addition to RGB\nIf True, inputs are 4-channel (RGB-D), else 3-channel (RGB)</p>"}, {"index": 2, "html": "<p>VISION BACKBONE (ResNet18)\nInstantiate the vision backbone from config\nInput: 3 cameras \u00d7 (B, L=1, 4, 240, 240) for RGBD images\nOutput: 3 cameras \u00d7 (B*L, 512, h, w) spatial feature maps\nThe backbone is pretrained on ImageNet and adapted for RGBD input</p>"}, {"index": 3, "html": "<p>TRANSFORMER DECODER\nThe main transformer that generates action predictions\nArchitecture: Cross-attention decoder (similar to DETR object detector)\n- Query: Learnable action slot embeddings (num_queries=20)\n- Key/Value: Visual tokens + proprioception + latent z\n- Outputs: (B, num_queries=20, hidden_dim=512) features \u2192 actions</p>"}, {"index": 4, "html": "<p>VAE ENCODER (for training only)\nEncodes ground-truth action sequences into a latent distribution\nInput: (B, T=20, A=23) action sequence\nOutput: (B, latent_dim=32) latent vector z\nPurpose: Allows the model to learn a structured latent space for action generation\nAt test time, we sample z ~ N(0,1) instead of using this encoder</p>"}, {"index": 5, "html": "<p>POSITIONAL ENCODING\nAdds spatial position information to visual features\nType: Sine/cosine positional encoding (similar to original Transformer paper)\nApplied to: Visual feature maps (h\u00d7w spatial positions)\nOutput: (1, hidden_dim=512, h, w) positional encoding\nLEARNABLE PARAMETERS - Action Generation\nNumber of action slots to predict (T=20 future actions)\nLinear head that projects transformer features to actions\nInput: (B, num_queries=20, hidden_dim=512)\nOutput: (B, num_queries=20, action_dim=23)\nLearnable action slot embeddings (queries for decoder)\nThese are similar to \"object queries\" in DETR\nShape: (num_queries=20, hidden_dim=512)\nPurpose: Each embedding represents one future timestep's action</p>"}, {"index": 6, "html": "<p>PROJECTION LAYERS - Vision to Transformer\nProjects ResNet features to transformer dimension\nInput: (B*L, resnet_output_dim=512, h, w) per camera\nOutput: (B*L, hidden_dim=512, h, w) per camera\nPurpose: Match feature dimensions for transformer input\nProjects proprioception to transformer dimension\nInput: (B*L, prop_dim=37)\nOutput: (B*L, hidden_dim=512)\nPurpose: Embed robot state into transformer feature space</p>"}, {"index": 7, "html": "<p>VAE ENCODER COMPONENTS\nLatent dimension for VAE (32-dimensional latent space)\nThis is much smaller than the action dimension (23) or hidden_dim (512)\nPurpose: Bottleneck that forces learning of compressed action representation\nSpecial [CLS] token for encoder (similar to BERT)\nShape: (1, hidden_dim=512)\nPurpose: The [CLS] output will be used to produce latent z\nProjects actions to transformer embedding space for encoder input\nInput: (B, T=20, action_dim=23)\nOutput: (B, T=20, hidden_dim=512)\nProjects proprioception for encoder input\nInput: (B, prop_dim=37)\nOutput: (B, hidden_dim=512)\nPurpose: Condition latent z on current robot state\nProjects [CLS] token output to latent distribution parameters\nInput: (B, hidden_dim=512)\nOutput: (B, latent_dim*2=64) where first 32 dims = \u03bc, last 32 dims = log(\u03c3\u00b2)\nPurpose: Parameterize Gaussian distribution q(z|actions, proprioception)\nSinusoidal positional encoding table for encoder inputs\nShape: (1, 1+1+num_queries=22, hidden_dim=512)\nBreakdown: [CLS token=1] + [proprioception=1] + [action sequence=20]\nPurpose: Give encoder information about temporal order of inputs\nNote: register_buffer means this is saved with model but not trained</p>"}, {"index": 8, "html": "<p>DECODER LATENT CONDITIONING\nProjects sampled latent z back to transformer dimension\nInput: (B, latent_dim=32)\nOutput: (B, hidden_dim=512)\nPurpose: Inject latent information into decoder\nLearnable position embeddings for [proprioception, latent_z] tokens\nShape: (2, hidden_dim=512)\nPurpose: Distinguish between proprio and latent inputs to decoder</p>"}, {"index": 9, "html": "<p>TEMPORAL ENSEMBLE SETUP\nAt test time, ACT predicts 20 actions but executes only 1\nTemporal ensemble averages predictions from multiple forward passes\nThis improves smoothness and reduces noise\nSave horizon for buffer management\nBuffer to store recent action predictions\nSize: deque of length 20, each element is (20, action_dim=23)\nElement i contains the action prediction for timestep i from past forward passes\nInitialize buffer with zeros\nEach entry is a (horizon=20, action_dim=23) tensor</p>"}, {"index": 10, "html": "<p>SAVE TRAINING HYPERPARAMETERS\nWeight for KL divergence term in CVAE loss\nTotal loss = L1_loss + kl_weight * KL_loss\nTypical value: 10\nLearning rate and schedule parameters\nSave all hyperparameters to checkpoint\nThis is a PyTorch Lightning feature that stores config with model</p>"}, {"index": 11, "html": "<p>FORWARD PASS - Main computation\nForward pass through ACT model.</p>\n<p>Training mode (actions provided):\n1. Encode actions \u2192 latent z via VAE encoder\n2. Process observations through vision backbone\n3. Decode latent z + observations \u2192 predicted actions\n4. Return predictions and (\u03bc, log_var) for KL loss</p>\n<p>Inference mode (actions=None):\n1. Sample z ~ N(0,1)\n2. Process observations through vision backbone\n3. Decode z + observations \u2192 predicted actions\n4. Return predictions and (None, None)</p>\n<p>Args:\nobs: Dictionary of observations\n- 'rgbd': dict of tensors, keys like 'robot_r1::robot_r1:left_realsense_link:Camera:0'\nEach tensor is (B, L=1, 4, 240, 240) for RGBD images\n- 'qpos': dict of tensors like {'torso': (B,L,4), 'left_arm': (B,L,7), ...}\n- 'eef': dict of tensors like {'left_pos': (B,L,3), 'left_quat': (B,L,4), ...}\n- 'odom': dict of tensors like {'base_velocity': (B,L,3)}\nactions: Ground truth action sequence (B, T=20, A=23), normalized to [-1, 1]\nOnly provided during training\nis_pad: Boolean mask (B, T=20) indicating which timesteps are padding\nTrue = padding, False = valid data</p>\n<p>Returns:\na_hat: Predicted actions (B, num_queries=20, action_dim=23)\n[mu, logvar]: VAE distribution parameters (training) or [None, None] (inference)\nDETERMINE MODE (training vs inference)\nIf actions are provided, we're in training mode and will use VAE encoder\nOtherwise, we're in inference mode and will sample z from prior N(0,1)\nGet batch size from observations\nget_batch_size() is a utility that extracts B from nested dict structure</p>"}, {"index": 12, "html": "<p>PROCESS PROPRIOCEPTION (robot state)\nConcatenate all proprioceptive features specified in self._prop_keys\nKeys are like 'odom/base_velocity', 'qpos/torso', 'eef/left_pos'\nThese contain joint positions, velocities, end-effector poses\nHandle nested dictionary keys (e.g., 'qpos/torso')\nHandle flat keys (e.g., 'proprioception')\nConcatenate all proprioceptive features along last dimension\nEach component has shape (B, L=1, feat_dim)\nResult: (B, L=1, prop_dim=37) where 37 = sum of all feature dimensions\nFlatten batch and observation window dimensions\nBefore: (B, L=1, prop_dim=37)\nAfter: (B*L, prop_dim=37)\nFor ACT with L=1, this is just (B, 37)</p>"}, {"index": 13, "html": "<p>VAE ENCODER PATH (TRAINING ONLY)\n---- Project Actions to Embedding Space ----\nInput: (B, T=20, action_dim=23) normalized actions\nOutput: (B, T=20, hidden_dim=512)\nPurpose: Convert actions to transformer-compatible features\n---- Project Proprioception to Embedding Space ----\nInput: (B, prop_dim=37)\nOutput: (B, hidden_dim=512)\nAdd time/sequence dimension to proprioception\nBefore: (B, hidden_dim=512)\nAfter: (B, 1, hidden_dim=512)\n---- Get [CLS] Token Embedding ----\nself.cls_embed.weight has shape (1, hidden_dim=512)\nThis is a learnable embedding similar to BERT's [CLS] token\nExpand [CLS] token for entire batch\nBefore: (1, hidden_dim=512)\nAfter: (B, 1, hidden_dim=512)\n---- Construct Encoder Input Sequence ----\nConcatenate: [CLS] + proprioception + action_sequence\ncls_embed: (B, 1, hidden_dim=512)\nprop_embed: (B, 1, hidden_dim=512)\naction_embed: (B, T=20, hidden_dim=512)\nResult: (B, 1+1+20=22, hidden_dim=512)\nTranspose for transformer (expects seq_len first)\nBefore: (B, seq_len=22, hidden_dim=512)\nAfter: (seq_len=22, B, hidden_dim=512)\n---- Create Attention Mask for Padding ----\nDon't mask [CLS] or proprioception tokens\nCreate (B, 2) tensor of False (not padded)\nConcatenate with action sequence padding mask\ncls_joint_is_pad: (B, 2) for [CLS] and proprioception\nis_pad: (B, T=20) for action sequence\nResult: (B, 22) full padding mask\n---- Get Positional Encoding ----\nself.pos_table: (1, seq_len=22, hidden_dim=512)\nProvides temporal/positional information to encoder\nTranspose to match encoder input format\nBefore: (1, seq_len=22, hidden_dim=512)\nAfter: (seq_len=22, 1, hidden_dim=512)</p>"}, {"index": 14, "html": "<p>---- Run VAE Encoder ----\nInput: encoder_input (seq_len=22, B, hidden_dim=512)\nPositional encoding: pos_embed (seq_len=22, 1, hidden_dim=512)\nAttention mask: is_pad (B, seq_len=22) where True = ignore\nOutput: (seq_len=22, B, hidden_dim=512) encoded representations\nExtract [CLS] token output (first position)\nencoder_output[0]: (B, hidden_dim=512)\nThis aggregates information from entire sequence\n---- Project to Latent Distribution Parameters ----\nInput: (B, hidden_dim=512)\nOutput: (B, latent_dim*2=64)\nSplit into mean and log-variance\nFirst 32 dims = \u03bc (mean)\nLast 32 dims = log(\u03c3\u00b2) (log-variance)\n---- Sample Latent Variable using Reparameterization Trick ----\nz = \u03bc + \u03c3 * \u03b5, where \u03b5 ~ N(0,1)\nThis allows backpropagation through sampling\nResult: (B, latent_dim=32)\nINFERENCE PATH (no VAE encoder)\nSet distribution parameters to None (no KL loss in inference)\nSample latent from standard normal prior\nz ~ N(0, 1)\nShape: (B, latent_dim=32)</p>"}, {"index": 15, "html": "<p>PROJECT LATENT TO DECODER INPUT\nTransform latent z to transformer feature space\nInput: (B, latent_dim=32)\nOutput: (B, hidden_dim=512)</p>"}, {"index": 16, "html": "<p>PROCESS VISUAL OBSERVATIONS\nExtract visual observations (RGB or RGBD)\nvs is dict with camera names as keys\nEach value is (B, L=1, C=4, H=240, W=240) for RGBD\nRun vision backbone (ResNet18)\nInput: dict of (B, L=1, C=4, H=240, W=240) tensors (one per camera)\nOutput: dict of (B*L, resnet_output_dim=512, h, w) spatial feature maps\nwhere h, w are reduced spatial dimensions (e.g., 7\u00d77 after downsampling)\nCollect features and positional encodings from all cameras\nProcess each camera's features\nfeatures: (B*L, resnet_output_dim=512, h, w)\n---- Generate Positional Encoding for Spatial Locations ----\nInput: (B*L, 512, h, w)\nOutput: (B*L, hidden_dim=512, h, w)\nEncodes (x, y) spatial positions using sine/cosine functions\n---- Project Features to Transformer Dimension ----\nInput: (B*L, 512, h, w)\nOutput: (B*L, hidden_dim=512, h, w)\n1\u00d71 convolution to match transformer feature dimension\nStore positional encoding</p>"}, {"index": 17, "html": "<p>PROCESS PROPRIOCEPTION FOR DECODER\nProject proprioception to transformer feature space\nInput: (B*L, prop_dim=37)\nOutput: (B*L, hidden_dim=512)\nPREPARE DECODER INPUTS\nConcatenate features from all cameras along width dimension\nEach camera: (B*L, hidden_dim=512, h, w)\nIf 3 cameras: (B*L, hidden_dim=512, h, 3*w)\nThis creates a \"panoramic\" view of all cameras\nConcatenate positional encodings similarly\nResult: (B*L, hidden_dim=512, h, 3*w)</p>"}, {"index": 18, "html": "<p>RUN TRANSFORMER DECODER\nThe decoder performs cross-attention:\n- Query: Action slot embeddings (self.query_embed.weight)\n- Key/Value: Visual features + proprioception + latent\nInputs:\nsrc: Visual features (B*L, hidden_dim=512, h, 3*w)\nmask: None (no masking)\nquery_embed: Action queries (num_queries=20, hidden_dim=512)\npos: Positional encoding (B*L, hidden_dim=512, h, 3*w)\nlatent_input: Latent z (B, hidden_dim=512)\nproprio_input: Robot state (B*L, hidden_dim=512)\nadditional_pos_embed: Position IDs for proprio/latent (2, hidden_dim=512)\nOutput: List of decoder layer outputs, we take the last one [-1]\nShape: (B, num_queries=20, hidden_dim=512)\nGENERATE ACTION PREDICTIONS\nProject decoder outputs to action space\nInput: (B, num_queries=20, hidden_dim=512)\nOutput: (B, num_queries=20, action_dim=23)\nThese are normalized actions in [-1, 1]\nReturn predictions and distribution parameters\nDuring training: [mu, logvar] are tensors for KL loss\nDuring inference: [None, None]</p>"}, {"index": 19, "html": "<p>INFERENCE - Generate actions for deployment\nGenerate action for deployment (called at every control timestep).</p>\n<p>Flow:\n1. Preprocess observations (normalize, format)\n2. Run forward pass to get action predictions\n3. Apply temporal ensemble (if enabled)\n4. Denormalize actions to actual robot commands</p>\n<p>Args:\nobs: Raw observations from environment</p>\n<p>Returns:\naction: Denormalized action to execute (1, 1, action_dim=23)\nPREPROCESS OBSERVATIONS\nNormalize and format observations to match training data\nInput: Raw observations from environment\nOutput: Processed dict matching training format\nFORWARD PASS\nRun model to predict action sequence\nInput: Processed observations\nOutput: a_hat (1, T=20, A=23), [None, None]\nWe only need the action predictions, not the VAE parameters</p>"}, {"index": 20, "html": "<p>TEMPORAL ENSEMBLE (if enabled)\nACT predicts T=20 future actions but executes only 1\nTemporal ensemble maintains a buffer of recent predictions\nand averages them for smooth execution\nAdd new predictions to buffer\na_hat[0]: (T=20, A=23) current prediction\nExtract the action for current timestep from all buffered predictions\nBuffer structure after n steps (n \u2265 20):\nbuffer[0]: [a\u2080, a\u2081, ..., a\u2081\u2089] from t=0\nbuffer[1]: [a\u2080, a\u2081, ..., a\u2081\u2089] from t=1\n...\nbuffer[19]: [a\u2080, a\u2081, ..., a\u2081\u2089] from t=19\nFor current timestep, we want:\nbuffer[0][19], buffer[1][18], ..., buffer[19][0]\nThese all predict the action for the same timestep from different times\nFilter out zero-initialized entries (from startup)\nactions_populated: (20,) boolean mask, True where non-zero\nKeep only valid predictions\nShape: (n_valid, A=23) where n_valid \u2264 20\n---- Compute Exponentially Weighted Average ----\nMore recent predictions get higher weight\nWeight decay factor k=0.01\nCompute weights: [e^0, e^(-k), e^(-2k), ...]\nMore recent = higher weight\nNormalize weights to sum to 1\nConvert to torch tensor and add dimension for broadcasting\nShape: (n_valid, 1)\nCompute weighted average\nactions_for_curr_step: (n_valid, A=23)\nexp_weights: (n_valid, 1)\nResult: (1, A=23) \u2192 (1, 1, A=23)\nDENORMALIZE ACTIONS\nConvert from normalized [-1, 1] to actual robot commands\nMove to CPU for environment execution\nDenormalize using joint ranges\nInput: (1, 1, A=23) in [-1, 1]\nOutput: (1, 1, A=23) in actual joint units</p>"}, {"index": 21, "html": "<p>RESET - Clear internal state\nReset policy internal state.\nCalled at the beginning of each episode.</p>\n<p>Purpose: Clear temporal ensemble buffer to avoid using\npredictions from previous episode.\nRe-initialize action buffer with zeros</p>"}, {"index": 22, "html": "<p>TRAINING STEP - Compute loss for one batch\nTraining step called by PyTorch Lightning.</p>\n<p>Args:\nbatch: Dictionary with keys ['obs', 'actions', 'masks']\nbatch_idx: Index of batch in epoch</p>\n<p>Returns:\nloss: Scalar loss value\nlog_dict: Dictionary of metrics to log\nB: Batch size (for averaging across GPUs)\nPREPARE ACTION DATA\nbatch['actions'] is a dict: {'base': (B,T,3), 'torso': (B,T,4), ...}\nConcatenate all action components into single tensor\nResult: (B, T=20, A=23)\nGet batch size for later metric averaging\nPREPROCESS BATCH DATA\nNormalize observations, format correctly\nInput: Raw batch data\nOutput: Processed batch ready for model\nEXTRACT AND FORMAT PADDING MASK\nRemove mask from batch dict (will pass separately to forward)\nFlatten batch dimension if needed\nBefore: (B, T=20)\nAfter: (B*T,) or (B, T) depending on shape\nInvert mask: ACT assumes True=padding, False=valid\nBut data uses True=valid, False=padding\nSo we invert it\nEXTRACT GROUND TRUTH ACTIONS\nActions are already normalized to [-1, 1] by process_data\nShape: (B, T=20, A=23)\nCOMPUTE LOSS\nRun forward pass and compute L1 + KL loss\nExtract total loss for backprop\nPrepare logging dictionary (excluding total loss)\nReturn loss, metrics to log, and batch size</p>"}, {"index": 23, "html": "<p>VALIDATION STEP - Evaluate without gradients\nValidation step (identical to training but without gradients).</p>\n<p>Args:\nbatch: Dictionary with keys ['obs', 'actions', 'masks']\nbatch_idx: Index of batch in epoch</p>\n<p>Returns:\nSame as policy_training_step</p>"}, {"index": 24, "html": "<p>OPTIMIZER SETUP\nSetup optimizer and learning rate scheduler.\nCalled by PyTorch Lightning.</p>\n<p>Returns:\nIf use_cosine_lr=True: ([optimizer], [scheduler_config])\nOtherwise: optimizer\nCREATE OPTIMIZER PARAMETER GROUPS\nGroups parameters by weight decay settings\nSome layers (like LayerNorm, biases) typically don't use weight decay\nCreate AdamW optimizer\nAdamW is Adam with decoupled weight decay\nSETUP LEARNING RATE SCHEDULER (if enabled)\nCosine annealing schedule with warmup\nLR schedule: warmup \u2192 cosine decay \u2192 constant minimum\nCreate cosine schedule function\nReturn optimizer and scheduler\ninterval='step' means update LR every optimizer step (not epoch)\nIf no scheduler, just return optimizer</p>"}, {"index": 25, "html": "<p>DATA PREPROCESSING\nPreprocess observations and optionally actions.</p>\n<p>Converts raw data to model-ready format:\n- Normalizes RGB images to [0, 1]\n- Normalizes depth images to [0, 1]\n- Concatenates RGB+D into 4-channel input\n- Extracts specified proprioceptive features</p>\n<p>Args:\ndata_batch: Raw batch from dataloader\nextract_action: Whether to include actions in output</p>\n<p>Returns:\ndata: Processed dictionary ready for model\nEXTRACT PROPRIOCEPTIVE DATA\nStart with basic robot state\nAdd odometry if available\nPROCESS RGB OBSERVATIONS (if used)\nFind all RGB camera observations\nKeys look like 'robot_r1::robot_r1:left_realsense_link:Camera:0::rgb'\nExtract camera name (remove '::rgb' suffix)\nNormalize pixel values from [0, 255] to [0, 1]\nResult: dict of (B, L=1, 3, 240, 240) tensors\nPROCESS RGB-D OBSERVATIONS (if used)\n---- Process RGB ----\nSame as above: normalize to [0, 1]\nrgb: dict of (B, L=1, 3, 240, 240) tensors\n---- Process Depth ----\nNormalize depth from [MIN_DEPTH, MAX_DEPTH] to [0, 1]\nMIN_DEPTH and MAX_DEPTH are constants from OmniGibson\ndepth: dict of (B, L=1, 1, 240, 240) tensors (note: 1 channel)\n---- Concatenate RGB + D ----\nFor each camera, concatenate RGB (3 channels) and depth (1 channel)\nResult: 4-channel RGBD image\nrgbd: dict of (B, L=1, 4, 240, 240) tensors\nPROCESS TASK CONDITIONING (if used)\nTask information could be one-hot encoding of task ID\nor language embedding of task description\nEXTRACT ACTIONS AND MASKS (if requested)\nActions are already normalized in [-1, 1] by data pipeline</p>"}, {"index": 26, "html": "<p>OPTIMIZER UTILITIES\nCreate parameter groups for optimizer.</p>\n<p>Separates parameters by whether they should use weight decay.\nTypically biases and normalization parameters don't use weight decay.</p>\n<p>Args:\nweight_decay: L2 regularization weight\nlr_layer_decay: Layer-wise learning rate decay factor\nlr_scale: Global learning rate scale</p>\n<p>Returns:\nList of parameter group dictionaries\nUse default grouping from il_lib.optim\nThis handles layer normalization, biases, etc.</p>"}, {"index": 27, "html": "<p>LOSS COMPUTATION\nCompute ACT training loss (L1 + KL).</p>\n<p>Loss = L1(predicted_actions, gt_actions) + \u03b2 * KL(q(z|x) || p(z))\nwhere:\n- L1 is action reconstruction error\n- KL is divergence from prior N(0,1)\n- \u03b2 is self.kl_weight (typically 10)</p>\n<p>Args:\nobs: Processed observations\nactions: Ground truth actions (B, T=20, A=23)\nis_pad: Padding mask (B, T=20), True=padding</p>\n<p>Returns:\nloss_dict: Dictionary with 'l1', 'kl', and 'loss' keys\nTRUNCATE TO PREDICTION HORIZON\nEnsure we only predict/supervise num_queries timesteps\nIf data has T=20 and num_queries=20, no change\nIf data has T>20, truncate to 20\nFORWARD PASS\nRun model to get predictions and VAE distribution\na_hat: (B, num_queries=20, action_dim=23)\nmu, logvar: (B, latent_dim=32) each\nCOMPUTE KL DIVERGENCE\nKL(q(z|x) || N(0,1)) for each sample in batch\nReturns: (B,) tensor of KL values\nWe take [0] to get scalar for this batch\nCOMPUTE ACTION RECONSTRUCTION LOSS\nCompute L1 loss for all timesteps (including padded)\nall_l1: (B, num_queries=20, action_dim=23)\nMask out padded timesteps and average\n~is_pad: (B, 20) boolean, True where valid\nunsqueeze(-1): (B, 20, 1) for broadcasting across action dims\nResult: only compute loss on valid (non-padded) timesteps\nCOMBINE LOSSES\nTotal loss = L1 + \u03b2 * KL\n\u03b2 = self.kl_weight (typically 10)</p>"}, {"index": 28, "html": "<p>VAE UTILITIES\nReparameterization trick for VAE.</p>\n<p>Instead of sampling z ~ N(\u03bc, \u03c3\u00b2) directly (not differentiable),\nwe use: z = \u03bc + \u03c3 * \u03b5 where \u03b5 ~ N(0, 1)</p>\n<p>This allows gradients to flow through \u03bc and \u03c3.</p>\n<p>Args:\nmu: Mean of latent distribution (B, latent_dim=32)\nlogvar: Log-variance (B, latent_dim=32)</p>\n<p>Returns:\nz: Sampled latent variable (B, latent_dim=32)\nCompute standard deviation from log-variance\n\u03c3 = exp(log(\u03c3\u00b2) / 2) = exp(log(\u03c3))\nSample \u03b5 ~ N(0, 1) with same shape as std\nReparameterization: z = \u03bc + \u03c3 * \u03b5</p>"}, {"index": 29, "html": "<p>Create sinusoidal positional encoding table.</p>\n<p>Same as original Transformer paper (Vaswani et al., 2017).\nFor position i and dimension j:\n- PE(i, 2j) = sin(i / 10000^(2j/d))\n- PE(i, 2j+1) = cos(i / 10000^(2j/d))</p>\n<p>Args:\nn_position: Number of positions (sequence length)\nd_hid: Hidden dimension (feature size)</p>\n<p>Returns:\nPositional encoding table (1, n_position, d_hid)\nCompute angle for one position across all dimensions.\nCreate table of angles for all positions\nShape: (n_position, d_hid)\nApply sin to even indices\nApply cos to odd indices\nConvert to torch tensor and add batch dimension\nShape: (1, n_position, d_hid)</p>"}, {"index": 30, "html": "<p>Compute KL divergence KL(q(z|x) || p(z)) for VAE.</p>\n<p>For q(z|x) = N(\u03bc, \u03c3\u00b2) and p(z) = N(0, 1):\nKL = 0.5 * \u03a3(1 + log(\u03c3\u00b2) - \u03bc\u00b2 - \u03c3\u00b2)</p>\n<p>Negative sign because we're minimizing the ELBO.</p>\n<p>Args:\nmu: Mean (B, latent_dim=32)\nlogvar: Log-variance (B, latent_dim=32)</p>\n<p>Returns:\ntotal_kld: Sum over dimensions, mean over batch (scalar)\ndimension_wise_kld: KL per dimension (latent_dim,)\nmean_kld: Mean KL over batch (scalar)\nHandle 4D tensors (e.g., from convolutional VAEs)\nCompute KL divergence per dimension and sample\nFormula: -0.5 * (1 + log(\u03c3\u00b2) - \u03bc\u00b2 - \u03c3\u00b2)\nShape: (B, latent_dim=32)\nSum over dimensions, mean over batch\nResult: (1,) tensor\nMean KL per dimension (for analysis/logging)\nResult: (latent_dim=32,) tensor\nMean KL over both dimensions and batch\nResult: (1,) tensor</p>"}];

  annotations.forEach(function(annot) {
    const div = document.createElement('div');
    div.className = 'annotation-block';
    div.setAttribute('data-index', annot.index);
    div.innerHTML = annot.html;
    annotationColumn.appendChild(div);
  });
});
</script>
