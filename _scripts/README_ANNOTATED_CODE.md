# Annotated Code Post System

A reusable Jekyll template system for creating two-column code walkthroughs with visual connectors between code and annotations.

## Overview

This system allows you to create blog posts where:
- **Left column**: Code blocks
- **Right column**: Detailed annotations/explanations
- **Visual connectors**: SVG lines link code to its explanation
- **Interactive**: Hover effects highlight connected blocks

Perfect for detailed code walkthroughs, tutorials, and educational content.

## Components

### 1. Layout Template
**Location**: `_layouts/annotated-code.html`

Features:
- Two-column responsive grid layout
- SVG connector lines between columns
- Hover effects for highlighting
- Mobile-friendly (stacks columns on small screens)
- Dark mode support

### 2. Parser Script
**Location**: `_scripts/parse_annotated_code.py`

Converts annotated Python files into Jekyll posts.

**Usage**:
```bash
python3 _scripts/parse_annotated_code.py input.py output.md
```

### 3. Annotation Markers

Use these markers in your source code to define annotated sections:

```python
#==Start
# This comment becomes an annotation (right column)
def my_function():
    # This also becomes an annotation
    code_here()  # This code goes to the left column
#==End
```

**Rules**:
- `#==Start`: Begin an annotated block
- `#==End`: End an annotated block
- Within a block:
  - Comments (lines starting with `#`) → Right column (annotations)
  - Docstrings (`"""..."""`) → Right column (annotations)
  - Code (everything else) → Left column

Each `#==Start`/`#==End` pair creates **one** synchronized code-annotation block pair.

## Creating an Annotated Post

### Step 1: Annotate Your Code

Create or modify your Python file with annotation markers:

```python
"""
My Amazing Algorithm
"""

import torch

#==Start
class MyModel(nn.Module):
    """
    This is the main model class.
    It implements a neural network for X task.
    """

    def __init__(self, input_dim, hidden_dim):
        # Initialize the parent class
        super().__init__()

        # Create the linear layer
        # input_dim: size of input features
        # hidden_dim: size of hidden layer
        self.layer = nn.Linear(input_dim, hidden_dim)
#==End

#==Start
    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor (batch_size, input_dim)

        Returns:
            Output tensor (batch_size, hidden_dim)
        """
        # Apply linear transformation
        output = self.layer(x)

        # Apply activation
        return F.relu(output)
#==End
```

### Step 2: Run the Parser

```bash
python3 _scripts/parse_annotated_code.py my_code.py _posts/YYYY-MM-DD-my-post.md
```

### Step 3: Customize the Front Matter

Edit the generated markdown file to update metadata:

```yaml
---
layout: annotated-code
title: "Your Custom Title"
date: 2025-10-16
categories: [category1, category2]
tags: [tag1, tag2, tag3]
description: "A brief description for SEO"
intro: |
  Optional introduction paragraph that appears before the code.
  Supports **markdown** formatting.
---
```

### Step 4: Build and Preview

```bash
bundle exec jekyll serve
```

Visit `http://localhost:4000` to preview your post.

## Customization

### Styling

The layout includes CSS variables for easy theming:

```css
--accent-color: Connector line color
--code-bg: Code block background
--card-bg: Annotation block background
--border-color: Border colors
--text-color: Text color
```

Modify `_layouts/annotated-code.html` to adjust styling.

### Connector Lines

The JavaScript in `annotated-code.html` draws curved SVG paths between blocks. To customize:

- **Curve intensity**: Adjust the Bézier control points in the `path` generation
- **Line style**: Modify `.connector-line` CSS class
- **Animation**: Add CSS transitions or JavaScript animations

### Mobile Behavior

On screens < 1024px:
- Columns stack vertically
- Connectors are hidden
- Code appears first, then annotations

To change the breakpoint, modify the `@media` queries in the layout.

## Tips for Good Annotations

1. **Keep blocks focused**: Each `#==Start`/`#==End` should cover one logical concept
2. **Use clear comments**: Write annotations that explain *why*, not just *what*
3. **Balance content**: Try to keep code and annotation blocks roughly similar in size
4. **Add context**: Use docstrings for high-level explanations, inline comments for details
5. **Test responsiveness**: Preview on different screen sizes

## Example Structure

```python
#==Start
# High-level explanation of what this section does
# Why this approach was chosen
# Key concepts to understand

class MyClass:
    def method(self):
        # Detailed explanation of this specific code
        code_here()
#==End

#==Start
# Explanation for the next logical section
def another_function():
    # More detailed comments
    more_code()
#==End
```

## Troubleshooting

**Issue**: Too many small blocks
- **Solution**: Consolidate related code into single `#==Start`/`#==End` pairs

**Issue**: Indentation looks wrong
- **Solution**: Check that your source file uses consistent indentation (spaces or tabs)

**Issue**: Connectors don't align
- **Solution**: Refresh the page or resize the window to trigger redraw

**Issue**: Annotations missing
- **Solution**: Verify comments start with `#` and are inside `#==Start`/`#==End` blocks

## Advanced Usage

### Adding Images to Annotations

In your source code comments:

```python
#==Start
# Here's how the architecture works:
# ![Architecture Diagram](/assets/images/architecture.png)
# As you can see, the model has three components...

code_here()
#==End
```

The parser preserves markdown in annotations, so images will render.

### Code Syntax Highlighting

The template uses Prism.js compatible classes. Ensure your Jekyll `_config.yml` has:

```yaml
markdown: kramdown
kramdown:
  syntax_highlighter: rouge
```

### Multiple Languages

To support languages other than Python:
1. Modify the parser to handle language-specific comments
2. Update the language class in `generate_jekyll_post()`:
   ```python
   code_content.append('<pre><code class="language-javascript">')
   ```

## Files Reference

- `_layouts/annotated-code.html` - Jekyll layout template
- `_scripts/parse_annotated_code.py` - Python parser
- `_scripts/act_annotated_sample.py` - Example annotated source
- `_posts/2025-10-16-act-policy-annotated.md` - Example generated post

## Future Enhancements

Potential improvements:
- Support for other programming languages
- Jupyter notebook converter
- Automatic section numbering
- Collapsible code blocks
- Copy code button
- Line highlighting
- Jump-to-section navigation

## License

This template system is part of the blog codebase and follows the same license.
