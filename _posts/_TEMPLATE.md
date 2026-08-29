---
# ============================================================================
# ARTICLE TEMPLATE — copy this file to start a new article.
#   1. Copy:  cp _posts/_TEMPLATE.md "_posts/2025-08-12-my-title.md"
#   2. Filename MUST be: _posts/<year>-<month>-<day>-<slug>.md
#   3. Edit the YAML below + write Markdown in the body.
#   4. Permalink is automatic: /blog/<year>/<slug>/
# ============================================================================

layout: post
title: "Your article title"            # shown on the article + listings
date: 2025-08-12 10:00:00 +0530        # publish date (timezone optional)
description: "A one-line summary."      # shown under the title in lists
tags: [machine-learning, ai]            # free-form; appear as filter chips
categories: [articles]                  # articles | whitepapers
featured: false                         # true = pinned to top of the list

# Optional:
# published: false                              # hide a draft
# related_publications: sharma2024beyondimagery # cite a key from _bibliography/papers.bib
# image: /assets/img/x.png                      # social/preview image
---

# Markdown is fully supported

Write the body in **Markdown** (kramdown / GFM). Headings, lists, tables,
**bold**, *italic*, `inline code`, links, images, and blockquotes all work.

## Headings
Use `##` for sections, `###` for subsections.

## Code blocks (syntax highlighted)
```python
def hello(name: str) -> str:
    return f"hi {name}"
```

## Math / LaTeX (MathJax)
Both inline and display math render. Use `$...$` for inline and `$$...$$` for
display equations.

Inline: the loss is $f(x) = x^2$, or $\mathcal{L} = -\log p(y \mid x)$.

Display:
$$
\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N} \log p(y_i \mid x_i)
$$

## Images
![Alt text](/assets/img/prof_pic.jpg)

## Notes
- Drafts: set `published: false` in the YAML to hide a post.
- To feature an article on the **About** page, add an entry to
  `selected_writing` in `_data/content.yml`.
- Publications/whitepapers (with cite/PDF buttons) live in
  `_bibliography/papers.bib` — they also appear on the Articles page.
