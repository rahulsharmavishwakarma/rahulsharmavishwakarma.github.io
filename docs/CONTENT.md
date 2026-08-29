# Editing guide

The site is two pages: **`/` (Articles)** and **`/about/` (About)**. Here's where to change each thing.

---

## 1. About page profile → `_data/content.yml`

All About-page copy lives under `about:` in **`_data/content.yml`**:

```yaml
about:
  eyebrow: "ML Engineer"            # small label above the name
  name: "Rahul Sharma"              # the big headline
  tagline: "I build and fine-tune…" # one-liner under the name
  photo: "/assets/img/prof_pic.jpg"
  cta_primary:   { label: "Read articles", url: "/" }
  cta_secondary: { label: "Get in touch",  url: "mailto:you@email.com" }
  bio:                             # list of paragraphs (order = display order)
    - "Paragraph one…"
    - "Paragraph two. Inline HTML allowed: <a href='…'>link</a>."
```

- **`about.bio`** is a YAML list — each item is one paragraph. Inline HTML (links) is allowed.
- **`selected_writing`** (same file) = the curated list shown under "Selected Writing" on About.
- Edit this one file, save, and the About page updates on rebuild.

## 2. Adding an article (Markdown + YAML)

Articles are Markdown files in **`_posts/`**, each with a small YAML header (frontmatter).

**To add one:**
```bash
cp _posts/_TEMPLATE.md "_posts/2025-08-12-my-title.md"
```
Then edit the file. The filename **must** be `_posts/YYYY-MM-DD-slug.md`.

**The YAML header (frontmatter):**
```yaml
---
layout: post
title: "Your title"
date: 2025-08-12 10:00:00 +0530
description: "One-line summary shown in lists."
tags: [machine-learning, ai]
categories: [essays]      # essays | notes | whitepapers
featured: false           # true = pinned to top
---
```

**The body is plain Markdown** (kramdown/GFM): headings, lists, tables, bold/italic, links, images, blockquotes.

## 3. LaTeX / math + Markdown in articles

Both **Markdown** and **LaTeX math** render inside any article (powered by MathJax):

| Type | Syntax | Example |
|---|---|---|
| Inline math | `$ … $` | `$f(x) = x^2$` |
| Display math | `$$ … $$` | `$$\mathcal{L} = -\frac{1}{N}\sum \log p$$` |
| Code block | ` ``` ` fenced | ` ```python … ``` ` |
| Inline code | `` `code` `` | `` `print()` `` |

- Markdown is processed by **kramdown** (GFM) — all standard syntax works.
- LaTeX is rendered by **MathJax**; `$...$` (inline) and `$$...$$` (display) are both enabled.
- See **`_posts/_TEMPLATE.md`** for a copy-paste starting point with live examples of each.

## 4. Photo & CV

| What | How |
|---|---|
| **Profile photo** | Replace `assets/img/prof_pic.jpg` (keep the same filename), or drop a new image in `assets/img/` and point `about.photo` in `_data/content.yml` at it |
| **CV / resume** | Replace `assets/pdf/Rahul_Sharma_CV.pdf` with your new PDF (keep the same filename) — the About page links to it automatically |

No code changes needed for either — same path in = updated site out.

## 5. Publications / whitepapers → `_bibliography/papers.bib`

Publications shown under "Whitepapers & Publications" on the homepage are BibTeX entries:

```bibtex
@article{my_citation_key,
  title    = {Paper Title},
  author   = {Sharma, Rahul and Coauthor, Name},
  journal  = {Venue Name},
  year     = {2025},
  abbr     = {VENUE},        # short label badge shown in the list (see _data/venues.yml)
  url      = {https://…},    # link shown on the entry
  pdf      = {https://…},    # direct PDF link
  selected = {true},        # highlight it
  abstract = {One-paragraph abstract.}
}
```

- Add an entry → save → rebuild. Sorted/grouped by `year` (newest first).
- The citation key (`my_citation_key`) can also be referenced from a post's frontmatter via `related_publications: my_citation_key`.

## 6. Social links & site settings → `_config.yml`

| Setting | Key |
|---|---|
| GitHub / LinkedIn / X handles | `github_username`, `linkedin_username`, `twitter_username` |
| Email (footer icon + contact) | `email` |
| Site title / SEO description | `title`, `description` |
| Homepage heading text | `blog_name`, `blog_description` |

## 7. Everything else

| I want to change… | Edit |
|---|---|
| About bio / tagline / photo | `_data/content.yml` → `about` |
| Profile photo file | `assets/img/prof_pic.jpg` (replace in place) |
| CV PDF | `assets/pdf/Rahul_Sharma_CV.pdf` (replace in place) |
| Selected writing on About | `_data/content.yml` → `selected_writing` |
| Footer social icons | `_config.yml` → `*_username` |
| A blog post | `_posts/<file>.md` |
| Add a new article | copy `_posts/_TEMPLATE.md` |
| Add a publication/whitepaper | `_bibliography/papers.bib` |
| Colors (light/dark) | `_sass/_themes.scss` |
| Fonts | `_sass/_themes.scss` (`--font-*`) + `_includes/head.html` |
| Default theme (dark/light) | `assets/js/theme.js` |
| Nav labels | `_data/content.yml` → `nav` |
| Site name / email / description | `_config.yml` |
| Favicon / logo | `_config.yml` → `icon` + files in `assets/img/favicon*` (see §8) |

## 8. Favicon / logo

The browser-tab icon is the custom "forward pass" mark. Four files, one source of truth:

| File | Role |
|---|---|
| `assets/img/favicon.svg` | SVG source of truth |
| `assets/img/favicon-32.png` | used by most desktop browsers in the tab |
| `assets/img/favicon-180.png` | Apple touch icon (iOS home screen) |
| `assets/img/favicon-512.png` | highest-res master (not directly referenced) |

Which icon is used is set by **`_config.yml` → `icon: favicon.svg`**; the `<link>` tags themselves live in **`_includes/head.html`** (the three `rel="icon"` / `apple-touch-icon` lines).

**Changing the artwork:** edit/replace `favicon.svg`, then regenerate the PNG clones at the same three sizes (32/180/512) so the tab icon and the Apple touch icon stay in sync.

**The `?v=N` cache-buster — bump it every time the artwork changes.** Browsers cache favicons *separately* from the page, and a hard refresh (Cmd+Shift+R) usually does **not** re-fetch the favicon. That's why the `<link>` tags end in `?v=3`. When you change the artwork, bump the version on **all three** `<link>` tags in `_includes/head.html` at once (e.g. `?v=3` → `?v=4`) — a new URL is the only reliable way to make every browser drop the stale cached copy. Leave the version alone otherwise; bumping it without an artwork change just churns the URL for no benefit.

**If a browser still shows the old icon:** open `http://localhost:4000/assets/img/favicon-32.png?v=3` (current version) once, then open the site in a fresh tab. Stubborn tabs: Chrome → `chrome://history` → search `localhost` → **Remove**; Safari → Develop → disable caches, or quit and reopen.

## Local preview
```bash
export PATH=/opt/homebrew/opt/ruby/bin:$PATH
bundle exec jekyll serve   # http://127.0.0.1:4000
```
See `../AGENTS.md` for full build/deploy details.
