---
layout: post
title: "Notes on PyTorch"
date: 2025-09-24
categories: ml
author: Vladimir Ershov
description: "TBD"
---

### PyTorch API notes. 

https://jeancochrane.com/blog/pytorch-functional-api - Why are model parameters shared and mutated between three distinct objects?
"In order to inspect the gradient at a particular point in the training loop, for example, the user would need to drop some form of debugger at the right moment and introspect the internal state of model". 
Artcicle suggested the functional programming API, but as noted, replacing the set of parameters on each step instead of mutating is an extra load for gc, which we don't want.

### from torch.utils.data import Dataset
  Avoid to include logic in __get_item__ except the direct loading. We don't want that to be a potential source of data mutation 