"""Fail the build if the MathJax wiring in the rendered site is wrong.

Background: for a long time every page loaded MathJax twice, once from Quarto's
own injection and once from a hand-written <script> tag in _quarto.yml's
include-in-header. Both happened to be version 3, so the duplicate was harmless
and invisible. When Quarto 1.10 switched its default injection to MathJax 4, the
pages started loading v3 and v4 together, the two fought over window.MathJax, and
every equation on the site silently stopped rendering. Nothing failed: the build
was green, the HTML looked fine, and only a human looking at the page could tell.

This script turns that class of failure into a red build. It reads the expected
MathJax URL out of _quarto.yml so there is exactly one place to change it, then
checks every rendered page against these invariants:

  1. No page loads more than one MathJax script.
  2. Every MathJax script URL is the one _quarto.yml declares.
  3. Every page that contains math loads MathJax.
  4. Math is emitted as raw LaTeX for MathJax to typeset, not pre-rendered into
     HTML by Pandoc (which is what html-math-method: plain silently does).

Run locally with:
    uv run python .github/scripts/check_math_wiring.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "_site"
QUARTO_YML = ROOT / "_quarto.yml"

# Any script whose src looks like a MathJax bundle, whatever CDN or version.
MATHJAX_SRC = re.compile(
    r"""<script[^>]*\bsrc=["']([^"']*(?:mathjax|tex-chtml|tex-mml)[^"']*)["'][^>]*>""",
    re.IGNORECASE,
)
MATH_SPAN = re.compile(r'<span class="math (inline|display)">(.{0,12})', re.DOTALL)


def expected_url() -> str:
    """Pull html-math-method.url out of _quarto.yml without a YAML dependency."""
    text = QUARTO_YML.read_text(encoding="utf-8")
    block = re.search(
        r"^\s*html-math-method:\s*\n(?:\s+#.*\n)*\s+method:\s*\S+\s*\n(?:\s+#.*\n)*\s+url:\s*[\"']([^\"']+)[\"']",
        text,
        re.MULTILINE,
    )
    if not block:
        sys.exit(
            "check_math_wiring: could not find html-math-method.url in _quarto.yml.\n"
            "That URL is what pins the MathJax version independently of which Quarto\n"
            "builds the site. If you removed it on purpose, update this script too."
        )
    return block.group(1)


def main() -> int:
    if not SITE.is_dir():
        sys.exit(f"check_math_wiring: {SITE} does not exist. Run `quarto render` first.")

    url = expected_url()
    pages = sorted(SITE.rglob("*.html"))
    problems: list[str] = []
    with_math = 0

    for page in pages:
        html = page.read_text(encoding="utf-8", errors="replace")
        rel = page.relative_to(SITE).as_posix()
        scripts = MATHJAX_SRC.findall(html)
        spans = MATH_SPAN.findall(html)

        # (1) and (2): exactly one MathJax, and it is the one we declared.
        if len(scripts) > 1:
            problems.append(
                f"{rel}: loads {len(scripts)} MathJax scripts, expected at most 1.\n"
                f"    {chr(10).join('    ' + s for s in scripts)}\n"
                "    Two MathJax builds on one page fight over window.MathJax and\n"
                "    nothing typesets. Usually this means Quarto is injecting its own\n"
                "    on top of one added by hand in include-in-header."
            )
        for src in scripts:
            if src != url:
                problems.append(
                    f"{rel}: unexpected MathJax URL\n"
                    f"      got:      {src}\n"
                    f"      expected: {url}\n"
                    "    Quarto's default MathJax version changes between releases;\n"
                    "    html-math-method.url in _quarto.yml is what pins it."
                )

        if not spans:
            continue
        with_math += 1

        # (3): a page with math must actually load MathJax.
        if not scripts:
            problems.append(
                f"{rel}: contains math but loads no MathJax script."
            )

        # (4): the math must still be LaTeX. html-math-method: plain makes Pandoc
        # render it into unicode and <sub>/<sup> instead, destroying the source.
        for kind, head in spans:
            opener = "\\(" if kind == "inline" else "\\["
            if not head.lstrip().startswith(opener):
                problems.append(
                    f"{rel}: {kind} math is not raw LaTeX (starts with {head.strip()!r}).\n"
                    "    Pandoc has pre-rendered it, which means html-math-method is set\n"
                    "    to a transforming method such as `plain`. Use `mathjax`."
                )
                break

    print(f"check_math_wiring: {len(pages)} pages, {with_math} contain math")
    print(f"check_math_wiring: expected MathJax URL {url}")

    if problems:
        print(f"\ncheck_math_wiring: FAILED with {len(problems)} problem(s)\n")
        for p in problems:
            print(f"  - {p}\n")
        return 1

    print("check_math_wiring: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
