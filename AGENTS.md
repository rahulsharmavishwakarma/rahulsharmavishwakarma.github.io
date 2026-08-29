# AGENTS.md — build & run

## Stack
Jekyll (Ruby) + al-folio theme, restyled (editorial). Deployed via GitHub Actions to GitHub Pages (`gh-pages` branch).

## Site structure
Two pages only:
- **`/`** — Articles feed (posts + whitepapers). Source: `blog/index.html` (permalink `/`).
- **`/about/`** — About page (hero + bio + selected writing). Source: `_pages/about.md` (uses `_layouts/home.html`).
- Article permalinks: `/blog/<year>/<slug>/`.

## Local dev
```bash
export PATH=/opt/homebrew/opt/ruby/bin:$PATH      # Ruby 4.x via Homebrew (system Ruby is too old)
bundle install                                   # first time only
bundle exec jekyll serve                         # http://127.0.0.1:4000
```
Production build: `JEKYLL_ENV=production bundle exec jekyll build` → outputs to `_site/`.

## Ruby version notes
- The Homebrew Ruby is at `/opt/homebrew/opt/ruby/bin` and is **not** on the default shell PATH — export it (above) before any `bundle` command.
- The `Gemfile` pins a few stdlib gems (`ostruct`, `logger`, `csv`, `base64`) that Ruby 4.0 removed from the default set; these are no-ops on Ruby 3.2.2 (the CI deploy version).
- `Gemfile.lock` is gitignored — CI resolves fresh.

## Deploy
Push to `master` → `.github/workflows/deploy.yml` builds with Ruby 3.2.2 and deploys `_site/` to `gh-pages`. No manual deploy step.

## Where things are
- **Copy/content:** `_data/content.yml` (see `docs/CONTENT.md`).
- **Settings/socials:** `_config.yml`.
- **Articles:** `_posts/` (template: `_posts/_TEMPLATE.md`).
- **Publications:** `_bibliography/papers.bib` (venue labels in `_data/venues.yml`).
- **Look (tokens/colors/fonts):** `_sass/_themes.scss` (light + dark), `_sass/_editorial.scss` (layout/components), `_includes/head.html` (font link).
- **Default theme:** `assets/js/theme.js` (dark by default).
- **Layouts/includes:** `_layouts/`, `_includes/` (al-folio theme layer, kept intact).

## Theme tokens (`_sass/_themes.scss`)
- **Dark (default):** bg `#161617`, surfaces `#1d1d1f`, divider `#2a2a2e`, accent `#fdba74`.
- **Light:** bg `#fafafa`, surfaces `#ffffff`, divider `#e5e5e5`, accent `#9a3412`.
- **Fonts:** Geist (`--font-display` / `--font-body`) + JetBrains Mono (`--font-mono`). Swap by editing these tokens + the Google Fonts `<link>` in `_includes/head.html`.

## Common gotchas
- `agent-browser` CLI is at `/opt/homebrew/bin/agent-browser` for browser QA (use `--viewport WxH` for mobile).
- ImageMagick `convert` isn't installed locally → the WebP image plugin logs warnings; harmless (CI has it).
- Font Awesome must be ≥ 6.4.2 for the `fa-x-twitter` glyph (currently 6.6.0); the integrity attr was removed from `_includes/head.html` to allow the version bump.
- A visitor's last-chosen theme is stored in `localStorage("theme")`; clear site data if a theme seems stuck.
