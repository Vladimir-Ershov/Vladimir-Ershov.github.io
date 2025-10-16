#!/usr/bin/env python3
"""
Parser for converting Python files with annotation markers into Jekyll annotated code posts.

Usage:
    python parse_annotated_code.py input.py output.md

Annotation Markers:
    #==Start   - Begin an annotated block
    #==End     - End an annotated block

Within an annotated block:
    - Comments (lines starting with #) become annotations (right column)
    - Code (non-comment lines) becomes code blocks (left column)
    - Docstrings are treated as annotations
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple


class AnnotatedBlock:
    def __init__(self):
        self.code_lines = []
        self.annotation_lines = []

    def add_code(self, line: str):
        self.code_lines.append(line)

    def add_annotation(self, line: str):
        self.annotation_lines.append(line)

    def has_content(self):
        return bool(self.code_lines or self.annotation_lines)

    def get_code(self):
        return '\n'.join(self.code_lines).strip()

    def get_annotation(self):
        return '\n'.join(self.annotation_lines).strip()


def extract_comment_text(line: str) -> str:
    """Extract text from a comment line, removing the # prefix."""
    line = line.strip()
    if line.startswith('#'):
        # Remove leading # and optional space
        text = line[1:].lstrip()
        return text
    return line


def is_comment_line(line: str) -> bool:
    """Check if a line is a comment (starts with #)."""
    return line.strip().startswith('#')


def is_docstring_delimiter(line: str) -> bool:
    """Check if a line contains a docstring delimiter (triple quotes)."""
    stripped = line.strip()
    return stripped.startswith('"""') or stripped.startswith("'''")


def parse_annotated_python(content: str) -> List[Tuple[str, str]]:
    """
    Parse Python file with annotation markers into (code, annotation) pairs.

    Each #==Start / #==End pair creates ONE block containing all code and all comments.

    Returns:
        List of (code_block, annotation_block) tuples
    """
    lines = content.split('\n')
    blocks = []
    in_annotated_section = False
    in_docstring = False
    docstring_delimiter = None

    # Accumulate code and annotations for current section
    section_code = []
    section_annotations = []

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Check for annotation markers
        if stripped == '#==Start':
            in_annotated_section = True
            section_code = []
            section_annotations = []
            i += 1
            continue
        elif stripped == '#==End':
            in_annotated_section = False
            # Create ONE block for this entire section
            if section_code or section_annotations:
                code_text = '\n'.join(section_code).strip()
                annotation_text = '\n'.join(section_annotations).strip()
                blocks.append((code_text, annotation_text))
            section_code = []
            section_annotations = []
            i += 1
            continue

        # Only process lines within annotated sections
        if not in_annotated_section:
            i += 1
            continue

        # Handle docstrings
        if is_docstring_delimiter(line):
            if not in_docstring:
                # Start of docstring
                in_docstring = True
                docstring_delimiter = '"""' if '"""' in line else "'''"

                # Check if it's a single-line docstring
                if line.count(docstring_delimiter) >= 2:
                    # Single-line docstring
                    text = line.strip().replace(docstring_delimiter, '').strip()
                    if text:
                        section_annotations.append(text)
                    in_docstring = False
                else:
                    # Multi-line docstring start
                    text = line.replace(docstring_delimiter, '').strip()
                    if text:
                        section_annotations.append(text)
            else:
                # End of docstring
                in_docstring = False
                text = line.replace(docstring_delimiter, '').strip()
                if text:
                    section_annotations.append(text)
                docstring_delimiter = None
            i += 1
            continue

        # Inside a docstring
        if in_docstring:
            section_annotations.append(line.strip())
            i += 1
            continue

        # Regular comment line
        if is_comment_line(line):
            # Check if it's a section separator (like # ====)
            if re.match(r'^#\s*=+\s*$', stripped):
                # Skip separator lines
                i += 1
                continue

            comment_text = extract_comment_text(line)
            if comment_text:
                section_annotations.append(comment_text)
        else:
            # Code line
            if stripped:  # Only add non-empty code lines
                section_code.append(line)

        i += 1

    # Add final block if exists (in case file ends without #==End)
    if in_annotated_section and (section_code or section_annotations):
        code_text = '\n'.join(section_code).strip()
        annotation_text = '\n'.join(section_annotations).strip()
        blocks.append((code_text, annotation_text))

    return blocks


def generate_jekyll_post(blocks: List[Tuple[str, str]], title: str, description: str) -> str:
    """
    Generate a Jekyll markdown post from parsed blocks.

    Args:
        blocks: List of (code, annotation) pairs
        title: Post title
        description: Post description

    Returns:
        Complete Jekyll post content
    """
    import json

    # Generate front matter
    front_matter = f"""---
layout: annotated-code
title: "{title}"
date: 2025-10-16
categories: [machine-learning, reinforcement-learning]
tags: [ACT, transformer, imitation-learning, robotics]
description: "{description}"
---
"""

    # Generate code blocks (left column)
    code_content = []
    for i, (code, annotation) in enumerate(blocks):
        if code:
            code_content.append(f'<div class="code-block" data-index="{i}">')
            code_content.append('<pre><code class="language-python">')
            code_content.append(code)
            code_content.append('</code></pre>')
            code_content.append('</div>')
            code_content.append('')

    # Generate annotation script (right column populated by JS)
    annotation_data = []
    for i, (code, annotation) in enumerate(blocks):
        if annotation:
            # Convert annotation text to HTML paragraphs
            paragraphs = annotation.split('\n\n')
            annotation_html = '\n'.join([f'<p>{p.strip()}</p>' for p in paragraphs if p.strip()])
        else:
            annotation_html = '<p><em>No annotation</em></p>'

        annotation_data.append({
            'index': i,
            'html': annotation_html
        })

    # Use json.dumps for proper escaping
    annotations_json = json.dumps(annotation_data)

    # Create JavaScript to populate annotations
    js_script = f"""
<script>
document.addEventListener('DOMContentLoaded', function() {{
  const annotationColumn = document.querySelector('.ac-annotation-column');
  if (!annotationColumn) return;

  const annotations = {annotations_json};

  annotations.forEach(function(annot) {{
    const div = document.createElement('div');
    div.className = 'annotation-block';
    div.setAttribute('data-index', annot.index);
    div.innerHTML = annot.html;
    annotationColumn.appendChild(div);
  }});
}});
</script>
"""

    return front_matter + '\n'.join(code_content) + '\n' + js_script


def main():
    if len(sys.argv) != 3:
        print("Usage: python parse_annotated_code.py input.py output.md")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2])

    if not input_file.exists():
        print(f"Error: Input file '{input_file}' not found")
        sys.exit(1)

    # Read input file
    content = input_file.read_text(encoding='utf-8')

    # Parse the file
    blocks = parse_annotated_python(content)

    if not blocks:
        print("Warning: No annotated blocks found. Make sure to use #==Start and #==End markers.")
        sys.exit(1)

    print(f"Found {len(blocks)} annotated blocks")

    # Extract title from first docstring or filename
    title = input_file.stem.replace('_', ' ').title()
    description = "A detailed, line-by-line walkthrough of the implementation"

    # Generate Jekyll post
    post_content = generate_jekyll_post(blocks, title, description)

    # Write output file
    output_file.write_text(post_content, encoding='utf-8')
    print(f"Generated Jekyll post: {output_file}")


if __name__ == '__main__':
    main()
