---
layout: post
title: "Training ML Models in Computer Vision (2025)"
date: 2025-10-01
categories: [machine-learning, computer-vision]
tags: [ai-generated, pytorch, tensorflow, yolo, huggingface, ml-training]
---

#  Training ML Models in Computer Vision (2025)

When working with computer vision models, there are many ways to train or use pre-trained models. This guide covers all the common approaches in the Python ecosystem with practical examples, pros/cons, and when to use each.

## Summary Comparison Table

<div class="table-responsive">
<table>
<thead>
<tr>
<th>Approach</th>
<th>Ease of Use</th>
<th>Flexibility</th>
<th>Best For</th>
<th>Auth Required</th>
<th>Fine-tuning Support</th>
</tr>
</thead>
<tbody>
<tr>
<td>PyTorch Raw</td>
<td>⭐⭐</td>
<td>⭐⭐⭐⭐⭐</td>
<td>Custom research</td>
<td>No</td>
<td>Yes (full control)</td>
</tr>
<tr>
<td>Timm</td>
<td>⭐⭐⭐⭐</td>
<td>⭐⭐⭐⭐</td>
<td>Classification</td>
<td>No</td>
<td>Yes</td>
</tr>
<tr>
<td>Hugging Face</td>
<td>⭐⭐⭐⭐⭐</td>
<td>⭐⭐⭐</td>
<td>Transformers, sharing</td>
<td>Optional</td>
<td>Yes (excellent)</td>
</tr>
<tr>
<td>Ultralytics</td>
<td>⭐⭐⭐⭐⭐</td>
<td>⭐⭐⭐</td>
<td>Detection, production</td>
<td>No</td>
<td>Yes (very easy)</td>
</tr>
<tr>
<td>PyTorch Lightning</td>
<td>⭐⭐⭐</td>
<td>⭐⭐⭐⭐</td>
<td>Scalable training</td>
<td>No</td>
<td>Yes</td>
</tr>
<tr>
<td>FastAI</td>
<td>⭐⭐⭐⭐⭐</td>
<td>⭐⭐⭐</td>
<td>Learning, prototyping</td>
<td>No</td>
<td>Yes (excellent)</td>
</tr>
<tr>
<td>OpenMMLab</td>
<td>⭐⭐</td>
<td>⭐⭐⭐⭐</td>
<td>SOTA detection/seg</td>
<td>No</td>
<td>Yes</td>
</tr>
<tr>
<td>TF/Keras</td>
<td>⭐⭐⭐⭐</td>
<td>⭐⭐⭐</td>
<td>Production, mobile</td>
<td>No</td>
<td>Yes</td>
</tr>
<tr>
<td>Detectron2</td>
<td>⭐⭐</td>
<td>⭐⭐⭐⭐</td>
<td>Research detection</td>
<td>No</td>
<td>Yes</td>
</tr>
<tr>
<td>Hugging Face Hub</td>
<td>⭐⭐⭐⭐⭐</td>
<td>⭐⭐⭐</td>
<td>Model sharing</td>
<td>Yes (uploads only)</td>
<td>Yes</td>
</tr>
</tbody>
</table>
</div>

---

## 1. PyTorch/TensorFlow from Scratch

Build and train models with complete control over every aspect of the training process.

```python
import torch
import torch.nn as nn
import torchvision.models as models

# Define custom model
class CustomModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = models.resnet50(pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)

# Train loop
model = CustomModel(num_classes=96)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        outputs = model(batch['images'])
        loss = criterion(outputs, batch['labels'])
        loss.backward()
        optimizer.step()
```

<div class="pros-cons-grid">
<div class="pros">
<h4>✅ Pros</h4>
<ul>
<li>Maximum flexibility and control</li>
<li>No platform dependencies</li>
<li>Deep understanding of training process</li>
<li>Easy debugging</li>
</ul>
</div>
<div class="cons">
<h4>❌ Cons</h4>
<ul>
<li>Most code to write</li>
<li>Need to handle data loading, augmentation, metrics manually</li>
<li>Longer development time</li>
<li>Easy to make mistakes</li>
</ul>
</div>
</div>

---

## 2. Torchvision/Timm Models (Transfer Learning)

Access to 1000+ pre-trained models with minimal code using `timm` (PyTorch Image Models).

```python
import timm
import torch.nn as nn

# Timm has 1000+ pretrained models
model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=96)

# Or modify existing model
model = timm.create_model('vit_base_patch16_224', pretrained=True)
model.head = nn.Linear(model.head.in_features, 96)

# Simple training
model.train()
# ... your training loop
```

<div class="pros-cons-grid">
<div class="pros">
<h4>✅ Pros</h4>
<ul>
<li>Easy access to SOTA models</li>
<li>Well-maintained, optimized implementations</li>
<li>Extensive model zoo (1000+ models)</li>
<li>Good for research and experimentation</li>
</ul>
</div>
<div class="cons">
<h4>❌ Cons</h4>
<ul>
<li>Still need to write training loop</li>
<li>Need to handle data pipelines</li>
<li>Less batteries-included than other options</li>
</ul>
</div>
</div>

---

## 3. Hugging Face Transformers + Trainers

High-level API for transformers and vision models with built-in training utilities.

```python
from transformers import AutoModelForImageClassification, Trainer, TrainingArguments
from datasets import load_dataset

# Load pretrained model
model = AutoModelForImageClassification.from_pretrained(
    "microsoft/resnet-50",
    num_labels=96,
    ignore_mismatched_sizes=True
)

# Or Vision Transformer
model = AutoModelForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=96
)

# High-level training
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=10,
    per_device_train_batch_size=16,
    logging_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()
```

<div class="pros-cons-grid">
<div class="pros">
<h4>✅ Pros</h4>
<ul>
<li>Minimal boilerplate code</li>
<li>Built-in logging, checkpointing, evaluation</li>
<li>Easy model sharing via Hub</li>
<li>Excellent documentation</li>
<li>Works with <code>datasets</code> library</li>
</ul>
</div>
<div class="cons">
<h4>❌ Cons</h4>
<ul>
<li>Less flexibility than raw PyTorch</li>
<li>Primarily designed for transformers</li>
<li>Abstractions can be confusing initially</li>
<li>Some overhead for simple tasks</li>
</ul>
</div>
</div>

---

## 4. Ultralytics (YOLO Ecosystem)

Production-ready object detection, segmentation, and classification with minimal code.

```python
from ultralytics import YOLO

# Object detection
model = YOLO("yolov8n.pt")  # nano, small, medium, large, xlarge
results = model.train(
    data="coco128.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device='0'  # GPU
)

# Inference
results = model.predict("image.jpg", conf=0.5)

# Classification
model = YOLO("yolov8n-cls.pt")
model.train(data="imagenet10", epochs=10)

# Segmentation
model = YOLO("yolov8n-seg.pt")
```

<div class="pros-cons-grid">
<div class="pros">
<h4>✅ Pros</h4>
<ul>
<li>Extremely user-friendly, production-ready</li>
<li>Built-in data augmentation, metrics, visualization</li>
<li>Fast training and inference</li>
<li>Supports detection, segmentation, classification, pose</li>
<li>Active community</li>
</ul>
</div>
<div class="cons">
<h4>❌ Cons</h4>
<ul>
<li>Less flexibility for custom architectures</li>
<li>Primarily YOLO-focused</li>
<li>Config files can be limiting</li>
<li>Black box for some operations</li>
</ul>
</div>
</div>

---

## 5. PyTorch Lightning + Lightning Flash

Organized, scalable training with automatic multi-GPU support and best practices built-in.

```python
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

class LitModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(2048, 96)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

# Train
trainer = L.Trainer(max_epochs=10, accelerator='gpu')
trainer.fit(model, train_loader, val_loader)
```

<div class="pros-cons-grid">
<div class="pros">
<h4>✅ Pros</h4>
<ul>
<li>Clean, organized code structure</li>
<li>Automatic multi-GPU, mixed precision</li>
<li>Great for research and production</li>
<li>Reduces boilerplate significantly</li>
<li>Excellent logging and debugging</li>
</ul>
</div>
<div class="cons">
<h4>❌ Cons</h4>
<ul>
<li>Learning curve for Lightning conventions</li>
<li>Overhead for simple projects</li>
<li>Sometimes too opinionated</li>
</ul>
</div>
</div>

---

## 6. FastAI

Beginner-friendly with excellent defaults and automatic best practices.

```python
from fastai.vision.all import *

# High-level API
path = untar_data(URLs.PETS)
dls = ImageDataLoaders.from_folder(path, valid_pct=0.2, item_tfms=Resize(224))

learn = vision_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(4)  # Automatic fine-tuning with frozen/unfrozen stages

# Prediction
learn.predict('image.jpg')

# Custom model
learn = Learner(dls, model, loss_func=CrossEntropyLossFlat(), metrics=accuracy)
```

<div class="pros-cons-grid">
<div class="pros">
<h4>✅ Pros</h4>
<ul>
<li>Extremely beginner-friendly</li>
<li>Excellent defaults and best practices built-in</li>
<li>Strong focus on practical deep learning</li>
<li>Great documentation and course materials</li>
<li>Learning rate finder and other utilities</li>
</ul>
</div>
<div class="cons">
<h4>❌ Cons</h4>
<ul>
<li>Less popular in research community</li>
<li>Abstraction can hide important details</li>
<li>Smaller ecosystem than PyTorch/TensorFlow</li>
<li>Less customization options</li>
</ul>
</div>
</div>

---

## 7. MMDetection / MMSegmentation (OpenMMLab)

State-of-the-art detection and segmentation implementations from OpenMMLab.

```python
from mmdet.apis import init_detector, inference_detector
from mmdet.apis import train_detector
from mmcv import Config

# Load config and checkpoint
config = Config.fromfile('configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py')
config.model.roi_head.bbox_head.num_classes = 96

# Train
train_detector(model, datasets, cfg, validate=True)

# Inference
model = init_detector(config, checkpoint, device='cuda:0')
result = inference_detector(model, 'test.jpg')
```

<div class="pros-cons-grid">
<div class="pros">
<h4>✅ Pros</h4>
<ul>
<li>State-of-the-art implementations</li>
<li>Comprehensive model zoo</li>
<li>Production-grade quality</li>
<li>Excellent for detection/segmentation</li>
<li>Modular design</li>
</ul>
</div>
<div class="cons">
<h4>❌ Cons</h4>
<ul>
<li>Steep learning curve</li>
<li>Config system can be complex</li>
<li>Heavier framework</li>
<li>Documentation can be challenging</li>
</ul>
</div>
</div>

---

## 8. TensorFlow/Keras with Hub

Google's framework with easy access to TensorFlow Hub models.

```python
import tensorflow as tf
import tensorflow_hub as hub

# Using TF Hub
model = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5",
                   trainable=True),
    tf.keras.layers.Dense(96, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit(train_data, epochs=10)

# Or Keras Applications
base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False)
x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
output = tf.keras.layers.Dense(96, activation='softmax')(x)
model = tf.keras.Model(inputs=base_model.input, outputs=output)
```

<div class="pros-cons-grid">
<div class="pros">
<h4>✅ Pros</h4>
<ul>
<li>Easy to use, intuitive API</li>
<li>Good integration with TensorFlow ecosystem</li>
<li>Strong mobile/edge deployment (TFLite)</li>
<li>Good for production</li>
</ul>
</div>
<div class="cons">
<h4>❌ Cons</h4>
<ul>
<li>PyTorch more popular in research</li>
<li>TF Hub smaller than Hugging Face</li>
<li>TensorFlow 2.x breaking changes</li>
<li>Can be verbose</li>
</ul>
</div>
</div>

---

## 9. Hugging Face Hub (Direct Model Loading)

Central repository for sharing and discovering ML models.

```python
from huggingface_hub import hf_hub_download
import torch

# Download model directly
model_path = hf_hub_download(repo_id="facebook/detr-resnet-50", filename="pytorch_model.bin")

# Or use with transformers
from transformers import AutoModel
model = AutoModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

# Upload your own model
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(folder_path="./my_model", repo_id="username/my-model")
```

<div class="pros-cons-grid">
<div class="pros">
<h4>✅ Pros</h4>
<ul>
<li>Massive model repository</li>
<li>Easy sharing and collaboration</li>
<li>Version control for models</li>
<li>Community-driven</li>
<li>Great discoverability</li>
</ul>
</div>
<div class="cons">
<h4>❌ Cons</h4>
<ul>
<li>Requires account for uploads</li>
<li>Download speeds can vary</li>
<li>Quality varies across models</li>
<li>Storage limits on free tier</li>
</ul>
</div>
</div>

---

## 10. Detectron2 (Facebook Research)

Facebook's platform for object detection and segmentation research.

```python
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 96

trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
```

<div class="pros-cons-grid">
<div class="pros">
<h4>✅ Pros</h4>
<ul>
<li>Excellent for detection/segmentation</li>
<li>Research-quality implementations</li>
<li>Flexible and modular</li>
<li>Used by Facebook Research</li>
</ul>
</div>
<div class="cons">
<h4>❌ Cons</h4>
<ul>
<li>Steeper learning curve</li>
<li>Less maintained than Ultralytics</li>
<li>Primarily detection-focused</li>
<li>Complex configuration system</li>
</ul>
</div>
</div>

---

## My Recommendations

**For learning ML/CV:** Start with **FastAI** or **Hugging Face Transformers**. They have excellent tutorials and hide complexity while teaching best practices.

**For production CV tasks:** Use **Ultralytics YOLO**. It's production-ready, well-maintained, and covers most CV needs (detection, segmentation, classification, pose).

**For maximum flexibility:** Use **PyTorch + Timm**. This gives you complete control while still providing access to pre-trained models.

**For model sharing & collaboration:** Use **Hugging Face Hub**. It's become the de facto standard for sharing ML models.

**For research:** **PyTorch Lightning** or **raw PyTorch** for full control with organized code.

**For transformers/attention models:** **Hugging Face Transformers** is unmatched.


---

**Last Updated:** October 2025  
**Tags:** `#MachineLearning` `#ComputerVision` `#PyTorch` `#TensorFlow` `#DeepLearning`