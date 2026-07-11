# Source materials for blog posts

Local-only input materials for posts that are based on online video lectures (subtitles and lecture notes PDFs). These are inputs, not outputs, and they are often copyrighted, so they are never committed to this public repo.

## How to use

1. Create a subfolder named for the post's eventual slug, e.g. `_sources/gradient-descent/`.
2. Drop the materials in it: subtitles as `.srt`, `.vtt`, or `.txt`, and lecture notes as `.pdf`. Filenames are free-form.
3. Ask Claude Code to write the post, pointing it at `_sources/<slug>/` (or just give the slug).

Everything under `_sources/` is gitignored except this README, so the materials stay on the local machine only. The post credits the original lecture via a BibTeX entry in `references/references.bib`; the raw materials are not republished.
