# rahulsharmavishwakarma.github.io

Personal portfolio + writing site. Two pages: **Articles** (`/`) and **About** (`/about/`). Jekyll (Ruby) on the al-folio theme, restyled, deployed to GitHub Pages via GitHub Actions.

## Edit

| What | Where |
|---|---|
| Copy (bio, tagline, selected writing) | `_data/content.yml` |
| Site settings (name, email, socials, RSS) | `_config.yml` |
| Articles | `_posts/` (copy `_posts/_TEMPLATE.md`) |
| Publications / whitepapers | `_bibliography/papers.bib` |
| Colors (light/dark) + fonts | `_sass/_themes.scss` |
| Layout / components | `_sass/_editorial.scss` |

See **`docs/CONTENT.md`** for the full editing guide.

## Theme
- **Dark by default** (neutral grey `#161617`) — toggle in the nav.
- **Light** — neutral `#fafafa` with white surfaces.
- **Fonts** — Geist + JetBrains Mono.
- Accent: terracotta.

## Run locally

```bash
export PATH=/opt/homebrew/opt/ruby/bin:$PATH   # Homebrew Ruby (system Ruby is too old)
bundle install                                 # first time only
bundle exec jekyll serve                       # http://127.0.0.1:4000
```

Production build: `JEKYLL_ENV=production bundle exec jekyll build` → `_site/`.

## Deploy

Push to `master` → `.github/workflows/deploy.yml` builds with Ruby 3.2.2 and deploys `_site/` to the `gh-pages` branch. No manual step.

See **`AGENTS.md`** for build/deploy details and gotchas.
