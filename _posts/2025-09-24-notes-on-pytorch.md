---
layout: post
title: "Notes on PyTorch"
date: 2025-09-24
categories: [machine-learning, pytorch, WIP]
author: Vladimir Ershov
description: "TBD"
---

### PyTorch API notes. 

https://jeancochrane.com/blog/pytorch-functional-api - Why are model parameters shared and mutated between three distinct objects?
"In order to inspect the gradient at a particular point in the training loop, for example, the user would need to drop some form of debugger at the right moment and introspect the internal state of model". 
Artcicle suggested the functional programming API, but as noted, replacing the set of parameters on each step instead of mutating is an extra load for gc, which we don't want.

### from torch.utils.data import Dataset
  Avoid to include logic in __get_item__ except the direct loading. We don't want that to be a potential source of data mutation 

### YOLO vs R-CNN
When an image goes in, YOLO simultaneously predicts bounding boxes and class probabilities for all objects in one shot.

`
YOLO divides the input image into an S × S grid (for example, 7×7 or 13×13 depending on the version). Each grid cell is responsible for detecting objects whose center falls within that cell.
Each grid cell predicts:

Multiple bounding boxes (typically 2-5 per cell), where each box includes:

x, y coordinates of the box center (relative to the grid cell)
width and height of the box (relative to the whole image)
Confidence score - how confident the model is that the box contains an object AND how accurate the box is (calculated as: probability of object × IoU with ground truth)

Class probabilities - the likelihood that the object belongs to each possible class (person, car, dog, etc.)


`
`
Non-Maximum Suppression solves this by:

Filtering by confidence: First, throw out any predictions below a confidence threshold (e.g., 0.3)
For each class, iteratively:

Select the box with the highest confidence score
Keep it as a final detection
Remove all other boxes that overlap significantly with it (typically IoU > 0.5)
Repeat with the next highest confidence box among the remaining ones
`



R-CNN or Faster R-CNN break the problem into two separate steps:

First stage: Generate region proposals - essentially asking "where might objects be?" This creates a bunch of candidate bounding boxes that might contain objects.
Second stage: For each proposed region, classify what's in it and refine the bounding box coordinates.

<div class="todo-block">
<p><strong>TODO:</strong> 
Here are some resources for practicing tensor operations: Interactive Sites:
einops.rocks - Visual, interactive tensor operation (supports PyTorch syntax) https://einops.rocks/
Tensor Puzzles by Sasha Rush - Coding challenges specifically for practicing reshape, gather, etc. - https://github.com/srush/Tensor-Puzzles
PyTorch Tutorials - Tensor Operations - Official interactive notebooks  https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html
</p>
</div>
