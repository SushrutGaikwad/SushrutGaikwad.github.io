# How to use the blog-writing workflow

This folder holds the tooling that lets Claude Code write blog posts for this Quarto site. This README is for you (the human). The two companion files are:

- `instructions.md`: the operating instructions Claude follows when writing a post.
- `index.qmd`: your Quarto style guide (math notation, figures, tables, citations, etc.) that Claude matches.

The whole `_for_claude_code/` folder is ignored by Quarto (the leading underscore), so nothing here is ever published to the site.

## Quick start

Tell Claude Code, in one message:

> Read `_for_claude_code/instructions.md` and `_for_claude_code/index.qmd`, then write a blog post on [topic]. Sources are in `_sources/[slug]/`.

Drop the "Sources are in ..." part if the post is written from scratch with no lecture materials. Claude then shows a plan, waits for your feedback, and only after that writes the files.

Note: the instructions are not loaded automatically (this was a deliberate choice). So each blog session starts by pointing Claude at them, as above.

## Where everything lives

| What                   | Location                           | Notes                                             |
| ---------------------- | ---------------------------------- | ------------------------------------------------- |
| Operating instructions | `_for_claude_code/instructions.md` | Quarto-ignored                                    |
| Style guide            | `_for_claude_code/index.qmd`       | Quarto-ignored                                    |
| Source materials       | `_sources/<slug>/`                 | Gitignored, local-only (see `_sources/README.md`) |
| The post               | `blog/published/<slug>/index.qmd`  | Published                                         |
| Visual scripts         | `blog/published/<slug>/_src/*.py`  | Quarto-ignored, version-controlled                |
| Rendered visuals       | `blog/published/<slug>/images/`    | Published, referenced as `images/...`             |
| Citations              | `references/references.bib`        | Shared bib file                                   |

`<slug>` is a kebab-case name reused across all of these, for example `gradient-descent`.

## The full flow

1. **(Optional) Add source materials.** For a post based on a video lecture, put the subtitles (`.srt`, `.vtt`, or `.txt`) and lecture notes (`.pdf`) in `_sources/<slug>/`. These stay on your machine only and are never committed.
2. **Ask Claude to write the post**, pointing it at the instructions and any sources.
3. **Claude presents a plan** in chat: outline, narrative arc, the concrete example, every recommended visual (with its type), and the sources to cite. No files are written yet.
4. **You give feedback**: pick an approach, approve or cut visuals, or redirect.
5. **Claude produces the files**: the post `index.qmd`, the visual scripts in `_src/`, the generated assets in `images/`, and BibTeX entries appended to `references/references.bib`.

## How visuals are made

All animations and plots are generated locally with Python, never notebooks. The scripts live in the post's `_src/` folder and run in the shared root uv environment. Claude runs them for you; the commands are:

- Matplotlib plot or animation: `uv run python blog/published/<slug>/_src/foo.py`
- Manim static PNG: `uv run manim -qh -s blog/published/<slug>/_src/foo.py SceneName`
- Manim video: `uv run manim -qh blog/published/<slug>/_src/foo.py SceneName` (use `-ql` while iterating)

Matplotlib scripts save straight into the post's `images/` folder. Manim writes to its own `media/` folder, and the single output file is then moved into `images/`.

## Previewing and publishing

- **Preview locally**: `quarto preview` (live reload) or `quarto render` for a one-off build. Visuals are pre-rendered files in `images/`, so this is fast.
- **Publish**: commit and push to `main`. The GitHub Action (`.github/workflows/quarto-publish.yml`) renders the site and deploys to GitHub Pages. The `_for_claude_code/`, `_src/`, and `_sources/` folders are all ignored by Quarto, so none of them reach the published site.

## The environment (already set up)

You do not need to redo any of this. It is recorded here for reference.

- Root uv project (`pyproject.toml`, `uv.lock`, `.venv`) with numpy, scipy, matplotlib, manim, and the Quarto render stack (jupyter, nbformat, nbclient).
- FFmpeg and TeX Live are installed and on PATH, so Manim `Tex`/`MathTex` and video export work.
- To add a new local package later: `uv add <package>`.

A note on `requirements.txt`: it is the smaller dependency set used by the CI deploy (`pip install -r requirements.txt` on the GitHub runner). It does not include manim or scipy, which are only needed locally for authoring. The local source of truth is `pyproject.toml`.

## If you want the instructions to load automatically

The current setup requires pointing Claude at `instructions.md` each session. If that gets tedious, the instructions could instead become a Claude Code skill (invoked with a slash command on demand) or the project's root `CLAUDE.md` (loaded every session). Ask Claude to switch the setup if you change your mind.
