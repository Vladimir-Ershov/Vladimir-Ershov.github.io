# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a Jekyll-based personal blog and AI research showcase site hosted on GitHub Pages. The site features blog posts about AI research, machine learning implementations, and interactive demos.

## Commands

### Jekyll Site Development
- **Run local development server**: `bundle exec jekyll serve`
- **Build site**: `bundle exec jekyll build`
- **Install dependencies**: `bundle install`
- **Clean build artifacts**: `bundle exec jekyll clean`

### Machine Learning Projects (MLRecap)
- **Activate pixi environment**: `cd projects/MLRecap && pixi shell`
- **Install ML dependencies**: `cd projects/MLRecap && pixi install`
- **Work with PyTorch environment**: `cd projects/MLRecap && pixi shell -e pytorch`

## Architecture & Structure

### Jekyll Blog Structure
- **_posts/**: Blog posts in Markdown format with YAML front matter. Follow naming convention: `YYYY-MM-DD-title.md`
- **_config.yml**: Site configuration including plugins, theme (minima), and Jekyll settings
- **_site/**: Generated static site (excluded from git)

### Content Organization
- Blog posts support interactive JavaScript demos embedded directly in Markdown
- Posts use Kramdown with GFM input and Rouge syntax highlighting
- HTML and JavaScript are allowed in Markdown files (`parse_block_html: true`)

### MLRecap Project
- Machine learning reimplementations located in `projects/MLRecap/`
- Uses Pixi for dependency management with PyTorch, scikit-learn, and visualization libraries
- Contains both Jupyter notebooks (`.ipynb`) and Python scripts
- Organized into `torch/` for PyTorch implementations and `local/` for custom ML models

## Key Configuration Details

- **Jekyll version**: 4.3.2 with Minima theme
- **Plugins**: jekyll-feed, jekyll-sitemap, jekyll-seo-tag
- **Python environment**: Managed via Pixi with Python 3.9 and PyTorch 2.5 with CUDA 12.4 support
- **Excluded from Jekyll build**: `.pixi/` directories and MLRecap project files