---
layout: post
title: "Neural Style Transfer: From Classic to Modern Approaches (2025)"
date: 2025-10-02
categories: [machine-learning, computer-vision]
tags: [ai-generated, pytorch, style-transfer, neural-networks, AdaIN, diffusion-models]
---

# Neural Style Transfer: A Comparative Guide

Neural style transfer combines the content of one image with the artistic style of another. This post reviews my implementation of the classic approach and compares it with five modern alternatives.

## Quick Comparison Table

<div class="table-responsive">
<table>
<thead>
<tr>
<th>Method</th>
<th>Year</th>
<th>Speed</th>
<th>Quality</th>
<th>GPU Memory</th>
<th>Training Required</th>
<th>Arbitrary Styles</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Gatys et al. (Classic)</strong></td>
<td>2015</td>
<td>⚠️ Slow (2-5 min)</td>
<td>⭐⭐⭐ Good</td>
<td>Low (2GB)</td>
<td>No</td>
<td>✅ Yes</td>
</tr>
<tr>
<td><strong>Fast Neural Style</strong></td>
<td>2016</td>
<td>⚡ Real-time (&lt;0.1s)</td>
<td>⭐⭐⭐ Good</td>
<td>Low (2GB)</td>
<td>✅ Per style</td>
<td>❌ No</td>
</tr>
<tr>
<td><strong>AdaIN</strong></td>
<td>2017</td>
<td>⚡ Real-time (&lt;0.1s)</td>
<td>⭐⭐⭐⭐ Great</td>
<td>Medium (4GB)</td>
<td>✅ Once</td>
<td>✅ Yes</td>
</tr>
<tr>
<td><strong>PyTorch Hub</strong></td>
<td>2019</td>
<td>⚡ Fast (0.5s)</td>
<td>⭐⭐⭐ Good</td>
<td>Low (2GB)</td>
<td>❌ Pre-trained</td>
<td>✅ Yes</td>
</tr>
<tr>
<td><strong>StyleGAN-NADA</strong></td>
<td>2021</td>
<td>⚡ Fast (1s)</td>
<td>⭐⭐⭐⭐⭐ Excellent</td>
<td>High (8GB)</td>
<td>✅ Per domain</td>
<td>✅ Text-guided</td>
</tr>
<tr>
<td><strong>Stable Diffusion</strong></td>
<td>2022</td>
<td>⏱️ Moderate (5-10s)</td>
<td>⭐⭐⭐⭐⭐ Excellent</td>
<td>High (8-12GB)</td>
<td>❌ Pre-trained</td>
<td>✅ Text-guided</td>
</tr>
</tbody>
</table>
</div>

### Pros & Cons Summary

<div class="pros-cons-grid">
<div class="pros">
<h4>✅ Classic Gatys Method (My Implementation)</h4>
<ul>
<li>Simple to understand and implement</li>
<li>Works with any style without training</li>
<li>Low GPU memory requirements</li>
</ul>
</div>
<div class="cons">
<h4>❌ Limitations</h4>
<ul>
<li>Very slow (optimization per image)</li>
<li>Limited resolution (typically 512×512 max)</li>
<li>No semantic understanding</li>
</ul>
</div>
</div>

<div class="pros-cons-grid">
<div class="pros">
<h4>✅ Modern Approaches</h4>
<ul>
<li>100-1000× faster</li>
<li>Higher quality results</li>
<li>Better semantic preservation</li>
<li>Some support text prompts</li>
</ul>
</div>
<div class="cons">
<h4>❌ Trade-offs</h4>
<ul>
<li>Require more GPU memory</li>
<li>May need training phase</li>
</ul>
</div>
</div>

---

## 1. Classic Approach: Gatys et al. (2015)

This is the foundational method I implemented. It optimizes a target image to match content features from one image and style features (via Gram matrices) from another.

### My Implementation

```python
import torch
from torch.optim import Adam
from torchvision import models, transforms
from torch.nn.functional import mse_loss
from PIL import Image
import numpy as np

# Load pre-trained VGG19
vgg = models.vgg19(pretrained=True).features

# Feature extraction layers
LOSS_LAYERS = {
    '0': 'conv1_1', '5': 'conv2_1', '10': 'conv3_1',
    '19': 'conv4_1', '21': 'conv4_2', '28': 'conv5_1'
}

def extract_features(x, model):
    features = {}
    for name, layer in model._modules.items():
        x = layer(x)
        if name in LOSS_LAYERS:
            features[LOSS_LAYERS[name]] = x
    return features

def calc_gram_matrix(tensor):
    """Captures style by computing feature correlations"""
    _, C, H, W = tensor.size()
    tensor = tensor.view(C, H * W)
    gram_matrix = torch.mm(tensor, tensor.t())
    return gram_matrix.div(C * H * W)  # Normalize

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
])

content_img = preprocess(Image.open('./data/target.png').convert('RGB')).unsqueeze(0)
style_img = preprocess(Image.open('./data/night_style.png').convert('RGB')).unsqueeze(0)

# Extract features
content_features = extract_features(content_img, vgg)
style_features = extract_features(style_img, vgg)
style_grams = {layer: calc_gram_matrix(style_features[layer])
               for layer in style_features}

# Optimization
target = content_img.clone().requires_grad_(True)
optimizer = Adam([target], lr=0.03)

weights = {'conv1_1': 1.0, 'conv2_1': 0.8, 'conv3_1': 0.6,
           'conv4_1': 0.4, 'conv5_1': 0.2}

for i in range(100):
    target_features = extract_features(target, vgg)

    # Content loss: preserve content structure
    content_loss = mse_loss(target_features['conv4_2'],
                           content_features['conv4_2'])

    # Style loss: match style correlations
    style_loss = 0
    for layer in weights:
        target_gram = calc_gram_matrix(target_features[layer])
        style_loss += mse_loss(target_gram, style_grams[layer]) * weights[layer]

    total_loss = 1e6 * style_loss + content_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
```

### Why Gram Matrices?

The Gram matrix captures **style** by measuring which features co-occur:

```
gram[i,j] = correlation between channel i and channel j
```

This encodes texture patterns (like "swirly brushstrokes + blue colors") while discarding spatial layout. It's why we can transfer artistic style without copying content structure.

<div class="todo-block">
<p><strong>TODO:</strong> </p>


<div style="background: #f5f5f5; padding: 15px; border-radius: 5px; margin: 20px 0;">
<strong>Interactive Demo:</strong> Try adjusting the style weight below to see how it affects the balance between content and style.

<label for="styleWeight">Style Weight: <span id="weightValue">1000000</span></label><br>
<input type="range" id="styleWeight" min="100000" max="10000000" value="1000000" step="100000"
       style="width: 100%;" oninput="updateWeight(this.value)">

<script>
function updateWeight(value) {
    document.getElementById('weightValue').textContent = value;
    // In practice, this would trigger re-optimization
}
</script>
</div>
</div>

---

## 2. Fast Neural Style Transfer (2016) ⚡

Instead of optimizing per image, train a **feed-forward network** once per style.

### Approach
```python
# Training phase (once per style)
transform_net = TransformNet()
for content_img in dataset:
    output = transform_net(content_img)
    loss = content_loss + style_loss  # Same losses as Gatys
    # Train network...

# Inference (instant!)
stylized = transform_net(new_content_img)  # Single forward pass
```

**Use case**: Real-time video processing, mobile apps

**Resources**: [PyTorch Example](https://github.com/pytorch/examples/tree/master/fast_neural_style)

---

## 3. AdaIN: Adaptive Instance Normalization (2017) 🔥

The breakthrough: Style transfer = aligning feature statistics!

### Core Idea
```python
def AdaIN(content_features, style_features):
    # Normalize content features
    normalized = (content_features - content_features.mean()) / content_features.std()

    # Match style statistics
    return normalized * style_features.std() + style_features.mean()

# Full pipeline
content_feat = encoder(content_img)
style_feat = encoder(style_img)
stylized_feat = AdaIN(content_feat, style_feat)
output = decoder(stylized_feat)
```

**Why it's better**: Arbitrary styles without retraining, real-time speed, better quality.

**Implementation**:
```bash
pip install torch torchvision
git clone https://github.com/naoto0804/pytorch-AdaIN
python test.py --content input.jpg --style style.jpg
```

---

## 4. PyTorch Hub Pre-trained Models 🎯

The easiest approach for quick experiments:

```python
import torch

model = torch.hub.load('pytorch/vision:v0.10.0',
                       'mobilenet_v2',
                       pretrained=True)

# Or use pre-trained style transfer
stylized = model(content, style)
```

**Best for**: Prototyping, educational demos

---

## 5. StyleGAN-NADA & StyleCLIP (2021) 🚀

Combines CLIP (language-vision model) with StyleGAN for text-guided style:

```python
# Text-to-style
generator.transfer("A portrait in Pixar animation style")

# Or use reference image
generator.transfer(style_img)
```

**Innovation**: Semantic understanding via CLIP allows precise control like "make it more abstract" or "add cyberpunk aesthetic."

**Resource**: [StyleGAN-NADA GitHub](https://github.com/rinongal/StyleGAN-nada)

---

## 6. Stable Diffusion img2img (2022) 🤖

The current state-of-the-art for artistic quality:

```python
from diffusers import StableDiffusionImg2ImgPipeline
import torch

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16
).to("cuda")

result = pipe(
    prompt="oil painting in Van Gogh style, starry night",
    image=content_img,
    strength=0.75,  # How much to transform (0=original, 1=full restyle)
    guidance_scale=7.5
).images[0]
```

### Interactive Demo: Try Different Strengths

<div style="background: #f5f5f5; padding: 20px; border-radius: 5px;">
<label for="strength">Transformation Strength: <span id="strengthValue">0.75</span></label><br>
<input type="range" id="strength" min="0" max="1" value="0.75" step="0.05"
       style="width: 100%;" oninput="updateStrength(this.value)">

<div id="strengthDescription" style="margin-top: 10px; font-style: italic;">
Moderate transformation - balances content and style
</div>

<script>
function updateStrength(value) {
    const val = parseFloat(value);
    document.getElementById('strengthValue').textContent = val.toFixed(2);

    let desc = "";
    if (val < 0.3) desc = "Subtle style hints - mostly preserves original";
    else if (val < 0.6) desc = "Light transformation - clear style influence";
    else if (val < 0.8) desc = "Moderate transformation - balances content and style";
    else desc = "Heavy transformation - strong artistic interpretation";

    document.getElementById('strengthDescription').textContent = desc;
}
</script>
</div>

**Why it wins**:
- Semantic understanding of both content and style
- Text control for precise artistic direction
- Highest quality results
- Active community with thousands of pre-trained models

---

## Practical Recommendations

### For Learning 📚
Start with my classic implementation - it teaches the fundamentals of:
- Feature extraction with pre-trained CNNs
- Gram matrices for texture representation
- Optimization-based image generation

### For Production 🚀

| Use Case | Recommended Method | Why |
|----------|-------------------|-----|
| Real-time video | Fast Neural Style | Fastest inference |
| Flexible app | AdaIN | Arbitrary styles, good speed |
| Highest quality | Stable Diffusion | Best results, text control |
| Mobile deployment | PyTorch Mobile + Fast Style | Optimized for edge devices |
| Research | All of the above | Compare and innovate |

### Resource Requirements

```bash
# Classic Gatys (my implementation)
- GPU: Optional (2GB sufficient)
- Time: 2-5 minutes per image
- Code: ~100 lines

# AdaIN
- GPU: Recommended (4GB)
- Training: 4-6 hours once
- Inference: <100ms
- Code: ~300 lines + pre-trained weights

# Stable Diffusion
- GPU: Required (8GB minimum, 12GB+ recommended)
- Setup: Download 4GB model
- Inference: 5-10 seconds
- Code: 10 lines with diffusers library
```

---

## Code Repository

All implementations available at: [github.com/Vladimir-Ershov/ml-recap](https://github.com/Vladimir-Ershov)

<div class="todo-block">
<p><strong>TODO:</strong>  interactive Colab notebook: <a href="#">Open in Colab</a></p>
</div>

---

## Further Reading

1. [Gatys et al. - Original Paper (2015)](https://arxiv.org/abs/1508.06576)
2. [Johnson et al. - Fast Neural Style (2016)](https://arxiv.org/abs/1603.08155)
3. [Huang & Belongie - AdaIN (2017)](https://arxiv.org/abs/1703.06868)
4. [Rombach et al. - Stable Diffusion (2022)](https://arxiv.org/abs/2112.10752)

---

## Conclusion

Neural style transfer has evolved from a slow optimization process to real-time, high-quality artistic transformation. While my classic implementation provides excellent learning value, modern approaches like AdaIN and Stable Diffusion offer practical advantages for real applications.

---

*Last updated: October 2025*
