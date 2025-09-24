---
layout: post
title: "Interactive Convolution Dashboard"
date: 2025-08-25
categories: computer-vision deep-learning
author: Vladimir Ershov
description: "Interactive dashboard for visualizing how image convolutions work with different kernels"
---

# Understanding Convolutions

Convolutions are fundamental operations in computer vision and deep learning. This interactive dashboard lets you explore how different convolution kernels transform images and see the mathematical operations happening at each pixel.

Use the controls below to select different pixels and kernel types to see how convolutions work in real-time.

<div id="convolution-dashboard">
  <!-- Dashboard will be rendered here -->
</div>

<style>
#convolution-dashboard {
  max-width: 1200px;
  margin: 20px auto;
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

.dashboard-container {
  display: grid;
  grid-template-columns: 250px 1fr;
  gap: 20px;
  background: #f8f9fa;
  border-radius: 10px;
  padding: 20px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.controls-panel {
  background: white;
  padding: 20px;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.control-group {
  margin-bottom: 20px;
}

.control-group label {
  display: block;
  font-weight: 600;
  margin-bottom: 8px;
  color: #333;
}

.slider-container {
  position: relative;
}

.slider {
  width: 100%;
  height: 6px;
  background: #ddd;
  border-radius: 3px;
  outline: none;
  -webkit-appearance: none;
  appearance: none;
  cursor: pointer;
}

.slider::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 20px;
  height: 20px;
  background: #007bff;
  border-radius: 50%;
  cursor: pointer;
}

.slider::-moz-range-thumb {
  width: 20px;
  height: 20px;
  background: #007bff;
  border-radius: 50%;
  cursor: pointer;
  border: none;
}

.slider-value {
  display: inline-block;
  margin-left: 10px;
  font-weight: bold;
  color: #007bff;
}

.dropdown {
  width: 100%;
  padding: 8px 12px;
  border: 2px solid #ddd;
  border-radius: 4px;
  background: white;
  font-size: 14px;
  cursor: pointer;
}

.dropdown:focus {
  border-color: #007bff;
  outline: none;
}

.main-content {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.images-container {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
}

.image-panel {
  background: white;
  padding: 15px;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  text-align: center;
}

.image-panel h3 {
  margin-top: 0;
  color: #333;
  font-size: 16px;
}

.image-container {
  position: relative;
  display: inline-block;
  border: 1px solid #ddd;
  border-radius: 4px;
  overflow: hidden;
}

.image-canvas {
  display: block;
  image-rendering: pixelated;
  cursor: crosshair;
}

.pixel-marker {
  position: absolute;
  width: 12px;
  height: 12px;
  border: 2px solid #ff0000;
  background: rgba(255, 0, 0, 0.3);
  pointer-events: none;
  transform: translate(-50%, -50%);
}

.data-table-container {
  background: white;
  padding: 20px;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.data-table-container h3 {
  margin-top: 0;
  color: #333;
  text-align: center;
}

.tables-grid {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 20px;
  margin-top: 15px;
}

.table-section {
  text-align: center;
}

.table-section h4 {
  margin-bottom: 10px;
  color: #666;
  font-size: 14px;
}

.data-table {
  width: 100%;
  border-collapse: collapse;
  font-family: monospace;
  font-size: 12px;
}

.data-table td {
  border: 1px solid #ddd;
  padding: 6px;
  text-align: center;
  background: #f9f9f9;
  min-width: 40px;
}

.data-table td.center-cell {
  background: #fff3cd;
  font-weight: bold;
}

.processed-value {
  font-size: 24px;
  font-weight: bold;
  color: #007bff;
  padding: 20px;
  background: #f8f9fa;
  border-radius: 8px;
  margin-top: 10px;
}

@media (max-width: 768px) {
  .dashboard-container {
    grid-template-columns: 1fr;
  }
  
  .images-container {
    grid-template-columns: 1fr;
  }
  
  .tables-grid {
    grid-template-columns: 1fr;
  }
}
</style>

<script>
class ConvolutionDashboard {
  constructor() {
    this.imageSize = 50;
    this.pixelX = 25;
    this.pixelY = 25;
    this.kernelType = 'blur';
    this.imageData = null;
    this.processedData = null;
    
    this.kernels = {
      'blur': [
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9]
      ],
      'sharpen': [
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
      ],
      'edge_detection': [
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]
      ],
      'emboss': [
        [-1, -1, 0],
        [-1, 0, 1],
        [0, 1, 1]
      ],
      'identity': [
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
      ]
    };
    
    this.init();
  }
  
  init() {
    this.createDashboard();
    this.generateSampleImage();
    this.setupEventListeners();
    this.updateDisplay();
  }
  
  createDashboard() {
    const container = document.getElementById('convolution-dashboard');
    container.innerHTML = `
      <div class="dashboard-container">
        <div class="controls-panel">
          <div class="control-group">
            <label for="x-slider">X Pixel</label>
            <div class="slider-container">
              <input type="range" id="x-slider" class="slider" min="2" max="47" value="25">
              <span id="x-value" class="slider-value">25</span>
            </div>
          </div>
          
          <div class="control-group">
            <label for="y-slider">Y Pixel</label>
            <div class="slider-container">
              <input type="range" id="y-slider" class="slider" min="2" max="47" value="25">
              <span id="y-value" class="slider-value">25</span>
            </div>
          </div>
          
          <div class="control-group">
            <label for="kernel-select">Kernel Method</label>
            <select id="kernel-select" class="dropdown">
              <option value="blur">Blur</option>
              <option value="sharpen">Sharpen</option>
              <option value="edge_detection">Edge Detection</option>
              <option value="emboss">Emboss</option>
              <option value="identity">Identity</option>
            </select>
          </div>
        </div>
        
        <div class="main-content">
          <div class="images-container">
            <div class="image-panel">
              <h3>Raw Image</h3>
              <div class="image-container">
                <canvas id="raw-canvas" class="image-canvas" width="200" height="200"></canvas>
                <div id="raw-marker" class="pixel-marker"></div>
              </div>
            </div>
            
            <div class="image-panel">
              <h3>Processed Image</h3>
              <div class="image-container">
                <canvas id="processed-canvas" class="image-canvas" width="200" height="200"></canvas>
                <div id="processed-marker" class="pixel-marker"></div>
              </div>
            </div>
          </div>
          
          <div class="data-table-container">
            <h3>Convolution Mathematics</h3>
            <div class="tables-grid">
              <div class="table-section">
                <h4>Raw Values (3×3)</h4>
                <table class="data-table" id="raw-table">
                  <!-- Will be populated by JS -->
                </table>
              </div>
              
              <div class="table-section">
                <h4>Convolution Kernel</h4>
                <table class="data-table" id="kernel-table">
                  <!-- Will be populated by JS -->
                </table>
              </div>
              
              <div class="table-section">
                <h4>Result</h4>
                <div class="processed-value" id="processed-result">0</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    `;
  }
  
  generateSampleImage() {
    // Create a sample image with some patterns
    this.imageData = [];
    for (let y = 0; y < this.imageSize; y++) {
      const row = [];
      for (let x = 0; x < this.imageSize; x++) {
        // Create a pattern with circles, lines, and gradients
        let value = 128; // Base gray
        
        // Add some geometric shapes
        const centerX = this.imageSize / 2;
        const centerY = this.imageSize / 2;
        const dist = Math.sqrt((x - centerX) ** 2 + (y - centerY) ** 2);
        
        if (dist < 8) {
          value = 200; // Bright circle in center
        } else if (dist < 12) {
          value = 100; // Dark ring
        }
        
        // Add some diagonal lines
        if (Math.abs(x - y) < 2) {
          value = Math.min(255, value + 50);
        }
        
        // Add noise and variations
        value += (Math.random() - 0.5) * 30;
        value = Math.max(0, Math.min(255, value));
        
        row.push(Math.round(value));
      }
      this.imageData.push(row);
    }
  }
  
  applyConvolution() {
    const kernel = this.kernels[this.kernelType];
    this.processedData = [];
    
    for (let y = 0; y < this.imageSize; y++) {
      const row = [];
      for (let x = 0; x < this.imageSize; x++) {
        let sum = 0;
        
        for (let ky = -1; ky <= 1; ky++) {
          for (let kx = -1; kx <= 1; kx++) {
            const imageY = y + ky;
            const imageX = x + kx;
            
            // Handle edge cases with padding
            let pixelValue = 0;
            if (imageY >= 0 && imageY < this.imageSize && 
                imageX >= 0 && imageX < this.imageSize) {
              pixelValue = this.imageData[imageY][imageX];
            }
            
            sum += pixelValue * kernel[ky + 1][kx + 1];
          }
        }
        
        // Clamp the result
        sum = Math.max(0, Math.min(255, sum));
        row.push(Math.round(sum));
      }
      this.processedData.push(row);
    }
  }
  
  drawImageToCanvas(canvasId, data) {
    const canvas = document.getElementById(canvasId);
    const ctx = canvas.getContext('2d');
    const imageData = ctx.createImageData(canvas.width, canvas.height);
    
    const scaleX = canvas.width / this.imageSize;
    const scaleY = canvas.height / this.imageSize;
    
    for (let y = 0; y < canvas.height; y++) {
      for (let x = 0; x < canvas.width; x++) {
        const srcX = Math.floor(x / scaleX);
        const srcY = Math.floor(y / scaleY);
        const value = data[srcY][srcX];
        
        const index = (y * canvas.width + x) * 4;
        imageData.data[index] = value;     // R
        imageData.data[index + 1] = value; // G
        imageData.data[index + 2] = value; // B
        imageData.data[index + 3] = 255;   // A
      }
    }
    
    ctx.putImageData(imageData, 0, 0);
  }
  
  updatePixelMarkers() {
    const scaleX = 200 / this.imageSize;
    const scaleY = 200 / this.imageSize;
    
    const rawMarker = document.getElementById('raw-marker');
    const processedMarker = document.getElementById('processed-marker');
    
    const markerX = (this.pixelX + 0.5) * scaleX;
    const markerY = (this.pixelY + 0.5) * scaleY;
    
    rawMarker.style.left = markerX + 'px';
    rawMarker.style.top = markerY + 'px';
    processedMarker.style.left = markerX + 'px';
    processedMarker.style.top = markerY + 'px';
  }
  
  updateDataTables() {
    // Update raw values table
    const rawTable = document.getElementById('raw-table');
    let rawHTML = '';
    for (let dy = -1; dy <= 1; dy++) {
      rawHTML += '<tr>';
      for (let dx = -1; dx <= 1; dx++) {
        const y = this.pixelY + dy;
        const x = this.pixelX + dx;
        let value = 0;
        if (y >= 0 && y < this.imageSize && x >= 0 && x < this.imageSize) {
          value = this.imageData[y][x];
        }
        const className = (dx === 0 && dy === 0) ? 'center-cell' : '';
        rawHTML += `<td class="${className}">${value}</td>`;
      }
      rawHTML += '</tr>';
    }
    rawTable.innerHTML = rawHTML;
    
    // Update kernel table
    const kernelTable = document.getElementById('kernel-table');
    const kernel = this.kernels[this.kernelType];
    let kernelHTML = '';
    for (let y = 0; y < 3; y++) {
      kernelHTML += '<tr>';
      for (let x = 0; x < 3; x++) {
        const className = (x === 1 && y === 1) ? 'center-cell' : '';
        const value = kernel[y][x];
        const displayValue = Math.abs(value) < 0.001 ? '0' : 
                           Number.isInteger(value) ? value.toString() : 
                           value.toFixed(3);
        kernelHTML += `<td class="${className}">${displayValue}</td>`;
      }
      kernelHTML += '</tr>';
    }
    kernelTable.innerHTML = kernelHTML;
    
    // Calculate and display processed result
    const kernel_flat = this.kernels[this.kernelType];
    let result = 0;
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        const y = this.pixelY + dy;
        const x = this.pixelX + dx;
        let value = 0;
        if (y >= 0 && y < this.imageSize && x >= 0 && x < this.imageSize) {
          value = this.imageData[y][x];
        }
        result += value * kernel_flat[dy + 1][dx + 1];
      }
    }
    result = Math.max(0, Math.min(255, Math.round(result)));
    document.getElementById('processed-result').textContent = result;
  }
  
  setupEventListeners() {
    const xSlider = document.getElementById('x-slider');
    const ySlider = document.getElementById('y-slider');
    const kernelSelect = document.getElementById('kernel-select');
    const xValue = document.getElementById('x-value');
    const yValue = document.getElementById('y-value');
    
    xSlider.addEventListener('input', (e) => {
      this.pixelX = parseInt(e.target.value);
      xValue.textContent = this.pixelX;
      this.updateDisplay();
    });
    
    ySlider.addEventListener('input', (e) => {
      this.pixelY = parseInt(e.target.value);
      yValue.textContent = this.pixelY;
      this.updateDisplay();
    });
    
    kernelSelect.addEventListener('change', (e) => {
      this.kernelType = e.target.value;
      this.updateDisplay();
    });
  }
  
  updateDisplay() {
    this.applyConvolution();
    this.drawImageToCanvas('raw-canvas', this.imageData);
    this.drawImageToCanvas('processed-canvas', this.processedData);
    this.updatePixelMarkers();
    this.updateDataTables();
  }
}

// Initialize the dashboard when the page loads
document.addEventListener('DOMContentLoaded', () => {
  new ConvolutionDashboard();
});
</script>

## How Convolutions Work

Convolutions are mathematical operations that combine two functions to produce a third function. In image processing, we slide a small matrix (called a kernel or filter) across an image and compute the dot product at each position.

### Key Concepts:

1. **Kernel/Filter**: A small matrix (usually 3×3) that defines the operation
2. **Stride**: How many pixels to move the kernel each step (usually 1)
3. **Padding**: How to handle edges of the image
4. **Feature Detection**: Different kernels detect different features (edges, blur, etc.)

### Common Kernels:

- **Blur**: Smooths the image by averaging neighboring pixels
- **Sharpen**: Enhances edges and details
- **Edge Detection**: Highlights boundaries between different regions
- **Emboss**: Creates a 3D raised effect
- **Identity**: Returns the original image unchanged

Try different kernels and pixel positions above to see how each operation transforms the image data!

## Applications

Convolutions are fundamental to:
- Computer vision preprocessing
- Convolutional Neural Networks (CNNs)
- Image filtering and enhancement
- Feature extraction
- Pattern recognition

Understanding how convolutions work at the pixel level helps build intuition for more complex deep learning architectures.