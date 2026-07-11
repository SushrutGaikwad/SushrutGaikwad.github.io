# Linear Algebra Blog Series Implementation Plan

> **For the writer (you, in a future session):** This is a content plan, not a code plan. Each "task" below is one blog post. Execute one post at a time: present the per-post mini-plan in chat for approval, then write `index.qmd`, write the `_src/` visual scripts, render them into `images/`, append BibTeX, and preview. Checkbox (`- [ ]`) syntax tracks progress across sessions.

**Goal:** A faithful, story-driven 18-post blog series teaching the whole of Gilbert Strang's MIT OCW 18.06 linear algebra course, built from concrete examples up to abstraction, following `_for_claude_code/instructions.md`.

**Architecture:** 18 posts grouped by natural story unit (not one-per-lecture, not one-per-chapter), organized into three acts. Each post is numbered ("Linear Algebra, Part N: ..."), opens with the shared master roadmap (current post highlighted), and ends by handing off to the next. Posts complement (not duplicate) the existing `linear-algebra-for-ml` crash-course post and cross-link to it and to `linear-regression`.

**Tech stack:** Quarto `.qmd`, MathJax (physics + upgreek packages), Mermaid (dark theme via YAML), Graphviz `{dot}` (manual dark template), Manim Community Edition (PNG + animation), Matplotlib (dark style). All visuals built locally with `uv run`.

**Source materials:** `_sources/linear-algebra/Lec_*.txt` (MIT OCW 18.06 subtitles). Lectures 13 and 32 were exam reviews and are intentionally absent. The series covers lectures 1-12, 14-31, 33.

---

## Global Constraints

These apply to every post. Copied from `_for_claude_code/instructions.md` and `_for_claude_code/index.qmd`.

- **Folder layout:** post at `blog/published/<slug>/index.qmd`; visual scripts at `blog/published/<slug>/_src/*.py`; rendered assets at `blog/published/<slug>/images/`; BibTeX appended to `references/references.bib`.
- **YAML header:** copy verbatim from `_for_claude_code/index.qmd` (lines 1-23); change only `title`, `date`, `description`, `categories`, `tags`. Keep `bibliography: ../../../references/references.bib` and `csl: ../../../references/apa.csl`.
- **Categories (every post):** `[Linear Algebra, Mathematics]`. Tags vary per post but always include `Linear Algebra`.
- **Titles:** `"Linear Algebra, Part N: <Topic>"`.
- **Dates:** assigned at write time, incrementing so posts sort in reading order on the blog (start 2026-06-23). Record the date used in this plan as each post is completed.
- **Math notation (from index.qmd):** vectors `$\vb{x}$` (lowercase bold non-italic); matrices `$\vb{A}$` (uppercase bold non-italic); greek vectors `$\boldsymbol{\uptheta}$`; sets `$\mathcal{A}$`, known sets `$\mathbb{R}$`; transpose `$\vb{x}^{\intercal}$` (never `^T`); always auto-sizing `\left( \right)` etc.; full stop after a display equation that ends a sentence, comma otherwise; `align*` with aligned `&=`. Column vectors by default; row vector is `$\vb{x}^{\intercal}$`.
- **Figures:** `![Caption.](images/name.svg){#fig-name fig-align="center" width=100% .invert}`. File name == label name. Cross-ref `@fig-name`.
- **Tables:** `::: {.tbl tbl-colwidths="auto"}` ... `: Caption. {#tbl-name style="width:..%; margin:auto;"}` ... `:::`.
- **Mermaid:** ` ```{mermaid} ` with `%%| fig-align: center` as first line. No custom dark styling (YAML handles it). Concise labels. No LaTeX in nodes (use Manim PNG if math is needed).
- **Graphviz:** ` ```{dot} ` with `//| label: fig-x` and `//| fig-cap: "..."`. MUST include the dark template at the top:
  ```
  bgcolor="transparent"
  node [fontcolor="#e6e6e6", style=filled, color="#e6e6e6", fillcolor="#333333", fontname="Helvetica"]
  edge [color="#e6e6e6", fontcolor="#e6e6e6"]
  graph [fontcolor="#e6e6e6", fontname="Helvetica"]
  ```
- **Manim:** background `self.camera.background_color = "#222222"`; ALL text via `Tex`/`MathTex` (never `Text`/`MarkupText`); raw strings; default Computer Modern font. Static PNG via `construct` with no `self.play`, render `uv run manim -qh -s <path> SceneName`, output at `media/images/<file>/SceneName.png`, move to `images/`. Animation render `uv run manim -qh <path> SceneName` (use `-ql` while iterating), output at `media/videos/<file>/<quality>/SceneName.mp4`, move to `images/`; embed short loops (<10s) as GIF, longer as `{{< video images/x.mp4 >}}`.
- **Matplotlib:** `plt.style.use('dark_background')`, `fig.patch.set_facecolor('#222222')`, `ax.set_facecolor('#222222')`, light ticks/labels/spines `#e6e6e6`, `plt.rcParams.update({"text.usetex": False, "mathtext.fontset": "cm"})`. Save into `images/` via `Path(__file__).resolve().parent.parent / "images" / "name.gif"` with `savefig_kwargs={"facecolor": "#222222"}`. Run `uv run python <path>`.
- **Writing style:** no em dashes, no en dashes, no emojis. Warm, conversational, precise. Anchor every new symbol to an already-seen quantity. Signpost transitions, periodic recaps, foreshadow derivations, label deep-dive sidebars with callouts.
- **Roadmap pattern:** every post embeds the master roadmap (see below) with its own node styled `fill:#10a37f,color:#fff`. Add a `style` line for the current node only.
- **Cross-link pattern:** prev/next links using `[text](../<slug>/index.qmd)`. Cite other posts the same way (e.g. `[linear regression post](../linear-regression/index.qmd)`).

## Shared citations (already in `references/references.bib`, reuse these keys)

No new BibTeX entries are needed; both sources already exist in the bib file:

- Textbook: `@Strang2023IntroductionLinearAlgebra` (6th ed., 2023).
- Lectures: `@MIT18.06SCLinearAlgebra2011` (MIT OCW 18.06SC, Fall 2011).

Weave them in (intro acknowledgment + at points where a reader wants depth), do not dump a references section. Do not paste verbatim subtitle text.

## The master narrative (three acts)

- **Act I, Solving Ax = b (Posts 1-4):** from two lines crossing, to elimination as factorization, to solvability as a question about spaces.
- **Act II, Structure and orthogonality (Posts 5-9):** basis/dimension/the four subspaces, networks, projection, least squares, determinants.
- **Act III, Eigenvalues and beyond (Posts 10-18):** eigenvalues, diagonalization/ODEs, Markov/Fourier, positive definiteness, complex/FFT, Jordan, SVD, transformations, pseudoinverse.

## The master roadmap (reuse in every post; highlight the current node)

```{mermaid}
%%| fig-align: center
flowchart LR
    subgraph ACT1["Act I: Solving Ax = b"]
        P1["1. Geometry"] --> P2["2. Elimination & LU"] --> P3["3. Column Space & Nullspace"] --> P4["4. Rank & Complete Solution"]
    end
    subgraph ACT2["Act II: Structure & Orthogonality"]
        P5["5. Basis & Four Subspaces"] --> P6["6. Graphs & Networks"] --> P7["7. Projections"] --> P8["8. Least Squares & QR"] --> P9["9. Determinants"]
    end
    subgraph ACT3["Act III: Eigenvalues & Beyond"]
        P10["10. Eigenvalues"] --> P11["11. Diagonalization & ODEs"] --> P12["12. Markov & Fourier"] --> P13["13. Positive Definite"] --> P14["14. Complex & FFT"] --> P15["15. Jordan Form"] --> P16["16. SVD"] --> P17["17. Transformations"] --> P18["18. Pseudoinverse"]
    end
    ACT1 --> ACT2 --> ACT3
```

## Per-post workflow (every task)

For each post: (1) present the mini-plan in chat and get approval; (2) write `index.qmd`; (3) write `_src/*.py` visual scripts; (4) render with `uv run` and move outputs into `images/`; (5) append BibTeX if new; (6) `quarto preview` / sanity check; (7) report and pause for review.

---

## Slug + lecture map (quick reference)

| #   | Slug                                                | Lectures |
| --- | --------------------------------------------------- | -------- |
| 1   | `the-geometry-of-linear-equations`                  | 1        |
| 2   | `elimination-inverses-and-lu-factorization`         | 2,3,4    |
| 3   | `vector-spaces-column-space-and-nullspace`          | 5,6      |
| 4   | `rank-and-the-complete-solution`                    | 7,8      |
| 5   | `basis-dimension-and-the-four-subspaces`            | 9,10     |
| 6   | `matrix-spaces-rank-one-and-graphs`                 | 11,12    |
| 7   | `orthogonality-and-projections`                     | 14,15    |
| 8   | `least-squares-and-gram-schmidt`                    | 16,17    |
| 9   | `determinants`                                      | 18,19,20 |
| 10  | `eigenvalues-and-eigenvectors`                      | 21       |
| 11  | `diagonalization-powers-and-differential-equations` | 22,23    |
| 12  | `markov-matrices-and-fourier-series`                | 24       |
| 13  | `symmetric-and-positive-definite-matrices`          | 25,27    |
| 14  | `complex-matrices-and-the-fft`                      | 26       |
| 15  | `similar-matrices-and-jordan-form`                  | 28       |
| 16  | `the-singular-value-decomposition`                  | 29       |
| 17  | `linear-transformations-and-change-of-basis`        | 30,31    |
| 18  | `the-pseudoinverse`                                 | 33       |

---

## Act I: Solving Ax = b

### - [x] Post 1: The Geometry of Linear Equations  (DONE, date 2026-06-23)

- **Slug:** `the-geometry-of-linear-equations`. **Lectures:** 1. **Tags:** `[Linear Algebra, Mathematics, Systems of Equations, Vectors]`.
- **Running example (as written):** coffee-and-muffin prices giving the clean positive-coefficient system $x + 2y = 8$, $3x + y = 9$ with solution $(2, 3)$ (coffee $2, muffin $3). Deviated from Strang's exact $2x-y=0,\,-x+2y=3$ because negative price coefficients do not match the everyday story; the row/column-picture pedagogy is identical. Column combination: $2\vb{a}_1 + 3\vb{a}_2 = \vb{b}$ with $\vb{a}_1=(1,3),\vb{a}_2=(2,1),\vb{b}=(8,9)$.
- **As-built notes:** V5 (singular vs invertible) folded into prose, not a separate figure. The column-picture animation was rendered as MP4 (`{{< video >}}`), not GIF: a -qh GIF was 354 MB; MP4 is 416 KB and matches the site's other Manim clips. Assets: `images/row_picture.svg` (matplotlib), `images/column_picture.png` (Manim PNG), `images/column_picture_sweep.mp4` (Manim). Scripts in `_src/`.
- **Concepts:** the problem $\vb{A}\vb{x} = \vb{b}$; row picture (lines/planes intersecting); column picture (linear combination of columns); matrix times vector as a combination of columns; invertible vs singular (do the columns fill the space?).
- **Outline:** (1) everyday two-price problem; (2) master roadmap + journey; (3) row picture (plot the lines, find the crossing); (4) signpost to the same equations seen as combining column vectors; (5) introduce $\vb{A}, \vb{x}, \vb{b}$ notation and define matrix-vector product as a column combination; (6) up to 3D (planes vs three vectors in $\mathbb{R}^3$), then general $n$; (7) can the columns reach every $\vb{b}$? invertible vs singular, foreshadowing column space/rank; (8) recap + cliffhanger to elimination.
- **Visuals:**
  - V1 essential, Mermaid: master roadmap, Part 1 highlighted.
  - V2 essential, Matplotlib static: row picture, two lines crossing at $(1,2)$, point marked.
  - V3 essential, Manim PNG (signature): column picture, $1\cdot\vb{a}_1 + 2\cdot\vb{a}_2 = \vb{b}$ tip-to-tail with `MathTex` labels.
  - V4 nice-to-have, Manim animation: sweep weights $x,y$ so $x\vb{a}_1 + y\vb{a}_2$ traces the plane, then snaps to $\vb{b}$.
  - V5 nice-to-have, Manim PNG: invertible vs singular, three columns spanning $\mathbb{R}^3$ vs trapped in a plane.
- **Citations:** intro acknowledgment of `@mitocw1806` and `@strang2016linearalgebra` (append both to `.bib` here).
- **Cross-links:** next -> Post 2; mention the `linear-algebra-for-ml` overview post.

### - [x] Post 2: Elimination, Inverses, and A = LU  (DONE, date 2026-06-24)

- **Slug:** `elimination-inverses-and-lu-factorization`. **Lectures:** 2,3,4. **Tags:** `[Linear Algebra, Mathematics, Gaussian Elimination, Matrix Factorization]`.
- **As-built notes:** Used Strang's 3x3 example $\vb{A}=[[1,2,1],[3,8,1],[0,4,1]]$, $\vb{b}=(2,12,2)$, pivots $1,2,5$, solution $(2,1,-2)$; Gauss-Jordan inverse shown on 2x2 $[[1,3],[2,7]]$. Gauss-Jordan kept as inline augmented-matrix math (no figure). Visuals: `images/a_equals_lu.png` (Manim PNG, signature), `images/elimination_cost.svg` (Matplotlib), plus Mermaid roadmap and Mermaid elimination flowchart (fig-elimination_flow). Roadmap uses `flowchart LR` (user preference applied from Post 1 onward).
- **Running example:** 3x3 system whose elimination gives pivots $1, 2, 5$; a 2x2 inverse via Gauss-Jordan (e.g. $\begin{bmatrix}1&3\\2&7\end{bmatrix}$); 2x2 $\vb{A}=\begin{bmatrix}2&8\\1&7\end{bmatrix}$ for $\vb{A}=\vb{L}\vb{U}$.
- **Concepts:** Gaussian elimination to upper triangular $\vb{U}$, pivots, failure/row-exchange; elimination steps as elementary matrices $\vb{E}_{ij}$; matrix multiplication (four views); back-substitution; inverses, Gauss-Jordan $[\vb{A}\,|\,\vb{I}] \to [\vb{I}\,|\,\vb{A}^{-1}]$, $(\vb{A}\vb{B})^{-1}=\vb{B}^{-1}\vb{A}^{-1}$; $\vb{A}=\vb{L}\vb{U}$ with multipliers sitting in $\vb{L}$; cost $\sim n^3/3$; permutations $\vb{P}\vb{A}=\vb{L}\vb{U}$.
- **Outline:** (1) recap Post 1 cliffhanger: we need a systematic method; (2) roadmap; (3) elimination by hand on the 3x3; (4) encode each step as a matrix $\vb{E}_{ij}$, so $\vb{E}\vb{A}=\vb{U}$ (signpost: elimination IS matrix multiplication); (5) the four ways to multiply matrices; (6) inverses and Gauss-Jordan; (7) the punchline: undo the $\vb{E}$'s and the multipliers fall into $\vb{L}$, giving $\vb{A}=\vb{L}\vb{U}$; (8) cost and permutations; (9) recap + handoff to spaces.
- **Visuals:**
  - V1 essential, Mermaid roadmap (Part 2).
  - V2 essential, Mermaid flowchart: the elimination algorithm (forward elimination -> back-substitution).
  - V3 essential, Manim PNG (signature): $\vb{E}\vb{A}=\vb{U}$ and $\vb{A}=\vb{L}\vb{U}$ with the multipliers highlighted in $\vb{L}$.
  - V4 nice-to-have, Manim PNG: Gauss-Jordan $[\vb{A}\,|\,\vb{I}] \to [\vb{I}\,|\,\vb{A}^{-1}]$ steps.
  - V5 nice-to-have, Matplotlib: operation-count curve $n^3/3$ vs $n$.
- **Cross-links:** prev Post 1, next Post 3.

### - [x] Post 3: Vector Spaces, Column Space, and Nullspace  (DONE, date 2026-06-25)

- **Slug:** `vector-spaces-column-space-and-nullspace`. **Lectures:** 5,6. **Tags:** `[Linear Algebra, Mathematics, Vector Spaces, Subspaces]`.
- **As-built notes:** Swapped the planned 4x3 example for a cleaner 3x3 rank-2 matrix $\vb{A}=[[1,1,2],[1,2,3],[1,3,4]]$ (columns $(1,1,1),(1,2,3),(2,3,4)$, third = first + second). Reason: in $\mathbb{R}^3$ both $C(\vb{A})$ (plane $b_1-2b_2+b_3=0$) and $N(\vb{A})$ (line through $(1,1,-1)$) are actually drawable. Covered transpose/symmetric/$\vb{R}^\intercal\vb{R}$ and permutations ($\vb{P}^\intercal=\vb{P}^{-1}$, closing the $\vb{P}\vb{A}=\vb{L}\vb{U}$ debt) first, then subspaces, column space, nullspace. Visuals: `images/subspace_closure.png` (Manim PNG, line through vs not through origin), `images/column_space.png` (Manim PNG, signature, 2D-schematic tilted plane), Mermaid roadmap, and a markdown subspace-test table (tbl-subspace_test). Nullspace shown via the column dependency (no separate figure).
- **Running example:** transpose/symmetric examples; permutation matrices; a 4x3 matrix with columns $(1,2,3,4),(2,3,4,5),(3,4,5,6)$ where col 3 = col 1 + col 2 (rank 2), so $C(\vb{A})$ is a plane in $\mathbb{R}^4$ and $(1,1,-1)$ is in the nullspace.
- **Concepts:** transpose, symmetric matrices ($\vb{R}^{\intercal}\vb{R}$ is symmetric); permutations as a group with $\vb{P}^{\intercal}\vb{P}=\vb{I}$; vector space and subspace axioms (closed under addition and scalar multiplication, contains $\vb{0}$); column space $C(\vb{A})$; nullspace $N(\vb{A})$; solvability $\vb{A}\vb{x}=\vb{b}$ iff $\vb{b}\in C(\vb{A})$.
- **Outline:** (1) recap: rank appeared in Post 2, what is it really; (2) roadmap; (3) transposes and permutations finish the LU story; (4) what is a vector space / subspace (line and plane through origin; union is not a subspace, intersection is); (5) column space as all combinations of columns; (6) solvability restated as $\vb{b}\in C(\vb{A})$; (7) nullspace as all solutions of $\vb{A}\vb{x}=\vb{0}$; (8) recap + handoff: how to compute these spaces.
- **Visuals:**
  - V1 essential, Mermaid roadmap (Part 3).
  - V2 essential, Manim PNG (signature): column space as a plane inside $\mathbb{R}^4$ (drawn schematically in 3D), reachable vs unreachable $\vb{b}$.
  - V3 essential, Manim PNG or Graphviz: subspace closure (line/plane through origin; why union fails).
  - V4 nice-to-have, table: subspace test checklist.
- **Cross-links:** prev Post 2, next Post 4.

### - [x] Post 4: Rank and the Complete Solution to Ax = b  (DONE, date 2026-06-26)

- **Slug:** `rank-and-the-complete-solution`. **Lectures:** 7,8. **Tags:** `[Linear Algebra, Mathematics, Rank, Linear Systems]`.
- **As-built notes:** Used Strang's $\vb{A}=[[1,2,2,2],[2,4,6,8],[3,6,8,10]]$ (rank 2, pivot cols 1,3, special solutions $(-2,1,0,0),(2,0,-2,1)$); chose $\vb{b}=(1,6,7)$ so $\vb{x}_p=(-3,0,2,0)$ is clean integers. Four-rank-cases rendered as a markdown table with inline MathJax block forms (NOT Graphviz: cells need matrix block forms $[\vb{I}\,\vb{F};\vb{0}\,\vb{0}]$ that Graphviz cannot typeset; a math table is the right tool). Visuals: `images/complete_solution.png` (Manim PNG, signature, affine shifted-line picture with 1-D nullspace), Mermaid roadmap, Mermaid solver flowchart (fig-solver_flow), tbl-rank_cases. Completes Act I.
- **Running example:** $\vb{A}=\begin{bmatrix}2&4&6&8\\3&6&8&10\\1&2&2&2\end{bmatrix}$ (rank 2): pivot columns 1,3; free columns 2,4; special solutions $(-2,1,0,0)^{\intercal}$ and $(2,0,-2,1)^{\intercal}$; RREF $\vb{R}$; particular solution with free vars 0.
- **Concepts:** pivot vs free variables; echelon form; special solutions (one per free variable); RREF $\vb{R}=\begin{bmatrix}\vb{I}&\vb{F}\\\vb{0}&\vb{0}\end{bmatrix}$; nullspace basis; complete solution $\vb{x}=\vb{x}_p+\vb{x}_n$; solvability condition; the four rank cases ($r=m=n$, $r=m<n$, $r<m=n$, $r<m,\,n$) and their solution counts; solution set as an affine subspace.
- **Outline:** (1) recap: we have the spaces, now solve; (2) roadmap; (3) solve $\vb{A}\vb{x}=\vb{0}$: free variables -> special solutions -> nullspace; (4) RREF cleans it up; (5) solve $\vb{A}\vb{x}=\vb{b}$: particular + nullspace; (6) when is there a solution at all; (7) the master table: rank decides existence and uniqueness; (8) geometric picture (shifted nullspace); (9) recap + handoff to basis/dimension.
- **Visuals:**
  - V1 essential, Mermaid roadmap (Part 4).
  - V2 essential, Graphviz (signature): the four rank cases mapped to solution counts (0 / 1 / infinitely many).
  - V3 essential, Manim PNG: $\vb{x}=\vb{x}_p+\vb{x}_n$ as a line/plane shifted off the origin (affine).
  - V4 nice-to-have, Mermaid/pseudocode: algorithm to produce the complete solution.
- **Cross-links:** prev Post 3, next Post 5.

## Act II: Structure and Orthogonality

### - [x] Post 5: Basis, Dimension, and the Four Subspaces

- **Slug:** `basis-dimension-and-the-four-subspaces`. **Lectures:** 9,10. **Tags:** `[Linear Algebra, Mathematics, Basis, Dimension, Four Subspaces]`.
- **Running example:** the rank-2 four-column matrix from Posts 3/4 reused to exhibit all four subspaces and their dimensions.
- **Concepts:** linear independence; span; basis (independent + spanning); dimension (all bases same size); rank + nullity = $n$; the four fundamental subspaces with dimensions ($C(\vb{A})$ dim $r$, $N(\vb{A})$ dim $n-r$, $C(\vb{A}^{\intercal})$ dim $r$, $N(\vb{A}^{\intercal})$ dim $m-r$) and their pairing; bases from RREF.
- **Outline:** (1) recap: special solutions were a basis, name it; (2) roadmap; (3) independence/span/basis on the example; (4) dimension is well-defined; (5) the four subspaces, where each lives, its dimension and basis; (6) the big-picture diagram; (7) recap (this is the spine of Act II) + handoff to applications/networks.
- **Visuals:**
  - V1 essential, Mermaid roadmap (Part 5).
  - V2 essential, Manim PNG (signature): Strang's four-subspaces "big picture" with dimensions $r$, $n-r$, $r$, $m-r$ and the $\mathbb{R}^n$/$\mathbb{R}^m$ split (orthogonality drawn but proved in Post 7).
  - V3 nice-to-have, Manim PNG: independent vs dependent vectors (span a plane vs a line).
  - V4 nice-to-have, table: the four subspaces (space, ambient, dimension, basis source).
- **Cross-links:** prev Post 4, next Post 6.

### - [x] Post 6: Matrix Spaces, Rank-One Matrices, and Graphs

- **Slug:** `matrix-spaces-rank-one-and-graphs`. **Lectures:** 11,12. **Tags:** `[Linear Algebra, Mathematics, Graphs, Networks, Incidence Matrix]`.
- **Running example:** the space of $3\times 3$ matrices (dim 9), symmetric (dim 6), upper triangular (dim 6), diagonal (dim 3); a rank-1 matrix $\vb{u}\vb{v}^{\intercal}$; a 4-node, 5-edge directed graph with $5\times 4$ incidence matrix (rank 3).
- **Concepts:** vector spaces beyond $\mathbb{R}^n$ (matrices, functions); $\dim(\mathcal{S}\cap\mathcal{U})+\dim(\mathcal{S}+\mathcal{U})=\dim\mathcal{S}+\dim\mathcal{U}$; rank-1 matrices as building blocks; solution space of $y''+y=0$ (dim 2); incidence matrix; $\vb{A}\vb{x}$ as potential differences; $\vb{A}^{\intercal}\vb{y}$ as Kirchhoff's current law; nullspace = constant potentials; left nullspace = loops; Euler's formula nodes - edges + loops = 1.
- **Outline:** (1) recap: subspaces, now stretch the idea; (2) roadmap; (3) matrix spaces and dimension counting; (4) rank-1 matrices (foreshadow SVD); (5) signpost to a real application: graphs; (6) incidence matrix and the four subspaces on a network (potentials, loops, KCL); (7) Euler's formula falls out; (8) recap + handoff to orthogonality.
- **Visuals:**
  - V1 essential, Mermaid roadmap (Part 6).
  - V2 essential, Graphviz (signature): the directed graph with node/edge labels alongside its incidence matrix layout.
  - V3 essential, Manim PNG: rank-1 matrix as column times row $\vb{u}\vb{v}^{\intercal}$.
  - V4 nice-to-have, table: matrix subspaces and their dimensions.
- **Cross-links:** prev Post 5, next Post 7.

### - [x] Post 7: Orthogonality and Projections

- **Slug:** `orthogonality-and-projections`. **Lectures:** 14,15. **Tags:** `[Linear Algebra, Mathematics, Orthogonality, Projection]`.
- **Running example:** orthogonal vectors $(1,2,3)$ and $(2,-1,0)$; projecting a vector $\vb{b}$ onto a line through $\vb{a}$; then projecting onto the plane spanned by the columns of a 3x2 $\vb{A}$.
- **Concepts:** orthogonal vectors $\vb{x}^{\intercal}\vb{y}=0$; Pythagoras; orthogonal subspaces; row space $\perp$ nullspace (completes the Post 5 picture); orthogonal complement; projection onto a line $\vb{p}=\dfrac{\vb{a}^{\intercal}\vb{b}}{\vb{a}^{\intercal}\vb{a}}\vb{a}$, projection matrix $\vb{P}=\dfrac{\vb{a}\vb{a}^{\intercal}}{\vb{a}^{\intercal}\vb{a}}$; error $\vb{e}=\vb{b}-\vb{p}\perp\vb{a}$; projection onto a subspace $\vb{p}=\vb{A}(\vb{A}^{\intercal}\vb{A})^{-1}\vb{A}^{\intercal}\vb{b}$; normal equations $\vb{A}^{\intercal}\vb{A}\hat{\vb{x}}=\vb{A}^{\intercal}\vb{b}$; $\vb{P}$ symmetric and idempotent.
- **Outline:** (1) recap: the big picture promised perpendicularity, prove it; (2) roadmap; (3) orthogonal vectors and subspaces, row space $\perp$ nullspace; (4) the projection problem: closest point on a line; (5) the projection matrix and why $\vb{e}\perp$ the space; (6) projecting onto a subspace, normal equations appear; (7) properties of $\vb{P}$; (8) recap + handoff: this solves unsolvable systems (least squares).
- **Visuals:**
  - V1 essential, Mermaid roadmap (Part 7).
  - V2 essential, Manim animation (signature): $\vb{b}$ projecting down to $\vb{p}$ on a line, the error $\vb{e}$ shown perpendicular.
  - V3 essential, Manim PNG: projection onto a plane, $\vb{b}=\vb{p}+\vb{e}$ with $\vb{p}\in C(\vb{A})$, $\vb{e}\in N(\vb{A}^{\intercal})$.
  - V4 nice-to-have, Manim PNG: $\vb{P}=\vb{P}^{\intercal}$, $\vb{P}^2=\vb{P}$ shown geometrically.
- **Cross-links:** prev Post 6, next Post 8.

### - [x] Post 8: Least Squares and Gram-Schmidt

- **Slug:** `least-squares-and-gram-schmidt`. **Lectures:** 16,17. **Tags:** `[Linear Algebra, Mathematics, Least Squares, Gram-Schmidt, QR]`.
- **Running example:** fit a line to $(1,1),(2,2),(3,2)$, giving $y=\tfrac{2}{3}+\tfrac{1}{2}t$, projection $(\tfrac{7}{6},\tfrac{5}{3},\tfrac{13}{6})$, error $(-\tfrac16,\tfrac26,-\tfrac16)$; Gram-Schmidt on $(1,1,1)$ and $(1,0,2)$.
- **Concepts:** least squares minimize $\lVert\vb{A}\vb{x}-\vb{b}\rVert^2$; normal equations from calculus; errors sum to zero and are $\perp C(\vb{A})$; orthonormal vectors; orthogonal matrix $\vb{Q}^{\intercal}\vb{Q}=\vb{I}$ (permutations, rotations, Hadamard); Gram-Schmidt; $\vb{A}=\vb{Q}\vb{R}$; normal equations simplify to $\hat{\vb{x}}=\vb{Q}^{\intercal}\vb{b}$.
- **Outline:** (1) recap: projection gives the best approximate solution; (2) roadmap; (3) the line-fitting problem and the sum of squared errors; (4) derive and solve the normal equations on the example; (5) connect to the probability view (link to the `linear-regression` post); (6) orthonormal bases are nicer: orthogonal matrices; (7) Gram-Schmidt and $\vb{A}=\vb{Q}\vb{R}$; (8) recap + handoff: a single number that tests invertibility (determinants).
- **Visuals:**
  - V1 essential, Mermaid roadmap (Part 8).
  - V2 essential, Matplotlib animation (signature, math-heavy plot): the fitting line rotating/shifting to minimize total squared vertical error, error bars shrinking.
  - V3 essential, Manim animation: Gram-Schmidt turning two skew vectors into an orthonormal pair.
  - V4 nice-to-have, Manim PNG: $\vb{A}=\vb{Q}\vb{R}$ structure.
- **Cross-links:** prev Post 7, next Post 9; cite `[linear regression post](../linear-regression/index.qmd)`.

### - [x] Post 9: Determinants

- **Slug:** `determinants`. **Lectures:** 18,19,20. **Tags:** `[Linear Algebra, Mathematics, Determinants, Cofactors, Cramer's Rule]`.
- **Running example:** 2x2 $ad-bc$; a 3x3 by cofactors; a parallelogram area and a 3D box volume.
- **Concepts:** three defining properties ($\det\vb{I}=1$; row swap flips sign; linear in each row separately) and the derived ones (equal rows -> 0, row operations leave it unchanged, triangular -> product of pivots, $\det(\vb{A}\vb{B})=\det\vb{A}\det\vb{B}$, $\det(\vb{A}^{\intercal})=\det\vb{A}$); the big formula (sum over $n!$ signed permutations); cofactors $C_{ij}=(-1)^{i+j}M_{ij}$ and cofactor expansion; $\vb{A}^{-1}=\frac{1}{\det\vb{A}}\vb{C}^{\intercal}$; Cramer's rule $x_j=\det(\vb{B}_j)/\det\vb{A}$; $\lvert\det\rvert$ = volume.
- **Outline:** (1) recap: invertibility kept coming up, here is the one-number test; (2) roadmap; (3) the three properties as axioms; (4) consequences, including det = product of pivots (ties back to elimination); (5) the big formula; (6) cofactors; (7) applications: inverse formula, Cramer's rule, volume; (8) recap + handoff: determinants set up the eigenvalue equation $\det(\vb{A}-\lambda\vb{I})=0$.
- **Visuals:**
  - V1 essential, Mermaid roadmap (Part 9).
  - V2 essential, Manim PNG (signature): area of a parallelogram = $\lvert ad-bc\rvert$ and a 3D box volume.
  - V3 essential, Manim PNG: cofactor expansion structure (deleting row $i$, column $j$).
  - V4 nice-to-have, table: the determinant properties with one-line consequences.
- **Cross-links:** prev Post 8, next Post 10.

## Act III: Eigenvalues and Beyond

### - [x] Post 10: Eigenvalues and Eigenvectors

- **Slug:** `eigenvalues-and-eigenvectors`. **Lectures:** 21. **Tags:** `[Linear Algebra, Mathematics, Eigenvalues, Eigenvectors]`.
- **Running example:** a projection matrix (eigenvalues 0, 1); the permutation $\begin{bmatrix}0&1\\1&0\end{bmatrix}$ (eigenvalues 1, -1; eigenvectors $(1,1),(1,-1)$); a 90-degree rotation (complex $i, -i$); a repeated-eigenvalue matrix lacking a full set of eigenvectors.
- **Concepts:** $\vb{A}\vb{x}=\lambda\vb{x}$; eigenvectors as directions only scaled, not turned; characteristic equation $\det(\vb{A}-\lambda\vb{I})=0$; eigenvector from $N(\vb{A}-\lambda\vb{I})$; trace $=\sum\lambda$, determinant $=\prod\lambda$; complex eigenvalues; repeated eigenvalues and eigenvector shortfall.
- **Outline:** (1) hook: which vectors does a matrix leave pointing the same way; (2) roadmap; (3) the eigenvalue equation, geometric meaning; (4) find them via $\det(\vb{A}-\lambda\vb{I})=0$ on the examples; (5) trace/determinant shortcuts; (6) when things go wrong (complex, repeated); (7) recap + handoff: use eigenvectors to take powers.
- **Visuals:**
  - V1 essential, Mermaid roadmap (Part 10).
  - V2 essential, Matplotlib animation (signature): a unit circle of vectors $\vb{x}$ mapped to $\vb{A}\vb{x}$; eigen-directions stay on their line while others rotate.
  - V3 essential, Manim PNG: the characteristic-equation pipeline $\det(\vb{A}-\lambda\vb{I})=0$.
  - V4 nice-to-have, table: example matrices and their eigenvalues/eigenvectors.
- **Cross-links:** prev Post 9, next Post 11.

### - [x] Post 11: Diagonalization, Powers, and Differential Equations

- **Slug:** `diagonalization-powers-and-differential-equations`. **Lectures:** 22,23. **Tags:** `[Linear Algebra, Mathematics, Diagonalization, Difference Equations, Differential Equations]`.
- **Running example:** Fibonacci recurrence as a 2x2 system with eigenvalues $\frac{1\pm\sqrt5}{2}$ (golden ratio); a 2x2 with eigenvalues 0 and -3 for $\dfrac{d\vb{u}}{dt}=\vb{A}\vb{u}$ reaching a steady state.
- **Concepts:** diagonalization $\vb{A}=\vb{S}\boldsymbol{\Lambda}\vb{S}^{-1}$; powers $\vb{A}^k=\vb{S}\boldsymbol{\Lambda}^k\vb{S}^{-1}$; difference equations $\vb{u}_{k+1}=\vb{A}\vb{u}_k$, stability $|\lambda|<1$; eigenvector expansion of $\vb{u}_0$; differential equations $\dot{\vb{u}}=\vb{A}\vb{u}$, matrix exponential $e^{\vb{A}t}=\vb{S}e^{\boldsymbol{\Lambda}t}\vb{S}^{-1}$ (power series), stability $\mathrm{Re}(\lambda)<0$, steady state at $\lambda=0$.
- **Outline:** (1) recap: computing $\vb{A}^k$ directly is painful; (2) roadmap; (3) diagonalization and the power formula; (4) Fibonacci solved cleanly, growth rate is the dominant eigenvalue; (5) discrete stability $|\lambda|<1$; (6) the continuous analogue $\dot{\vb{u}}=\vb{A}\vb{u}$ and $e^{\vb{A}t}$; (7) continuous stability $\mathrm{Re}(\lambda)<0$ vs discrete disk; (8) recap + handoff: special matrices where eigenvalues behave nicely.
- **Visuals:**
  - V1 essential, Mermaid roadmap (Part 11).
  - V2 essential, Matplotlib: Fibonacci values and the dominant-eigenvalue growth curve.
  - V3 essential, Matplotlib (signature): stability regions, unit disk $|\lambda|<1$ (discrete) vs left half-plane $\mathrm{Re}(\lambda)<0$ (continuous), trajectories overlaid.
  - V4 nice-to-have, Manim PNG: $\vb{A}^k=\vb{S}\boldsymbol{\Lambda}^k\vb{S}^{-1}$.
- **Cross-links:** prev Post 10, next Post 12.

### - [x] Post 12: Markov Matrices and Fourier Series

- **Slug:** `markov-matrices-and-fourier-series`. **Lectures:** 24. **Tags:** `[Linear Algebra, Mathematics, Markov Matrices, Fourier Series]`.
- **Running example:** California/Massachusetts migration Markov matrix (eigenvalue 1 with steady state $\propto(2,1)$, other eigenvalue 0.7 decaying); Fourier coefficients of a function via inner products.
- **Concepts:** Markov matrix (entries $\ge 0$, columns sum to 1); guaranteed eigenvalue 1, others $|\lambda|\le 1$; steady state from the $\lambda=1$ eigenvector; orthonormal expansion $\vb{v}=\sum(\vb{q}_i^{\intercal}\vb{v})\vb{q}_i$; Fourier series $f(x)=a_0+\sum a_n\cos nx+\sum b_n\sin nx$ with orthogonality $\int_0^{2\pi}\sin mx\cos nx\,\dd{x}=0$; inner product of functions.
- **Outline:** (1) recap: eigenvalues predict long-run behavior; (2) roadmap; (3) Markov matrices and why $\lambda=1$ always exists; (4) the population model converging to steady state; (5) signpost: the same orthonormal-basis idea, now for functions; (6) Fourier series as projection onto an infinite orthonormal basis; (7) recap + handoff: symmetric/positive-definite matrices.
- **Visuals:**
  - V1 essential, Mermaid roadmap (Part 12).
  - V2 essential, Mermaid state diagram: two-state Markov transitions with probabilities.
  - V3 essential, Matplotlib (signature): population vector converging to the steady state over iterations.
  - V4 nice-to-have, Manim PNG: orthogonality of $\sin nx, \cos nx$ and the coefficient formula $a_n$.
- **Cross-links:** prev Post 11, next Post 13.

### - [x] Post 13: Symmetric and Positive Definite Matrices

- **Slug:** `symmetric-and-positive-definite-matrices`. **Lectures:** 25,27. **Tags:** `[Linear Algebra, Mathematics, Symmetric Matrices, Positive Definite]`.
- **Running example:** $\begin{bmatrix}2&6\\6&c\end{bmatrix}$ with $c=18$ (semidefinite) vs $c=20$ (definite); the 3x3 tridiagonal $\begin{bmatrix}2&-1&0\\-1&2&-1\\0&-1&2\end{bmatrix}$ (determinants 2,3,4; pivots $2,\tfrac32,\tfrac43$); completing the square $2x^2+12xy+20y^2=2(x+3y)^2+2y^2$.
- **Concepts:** symmetric $\vb{A}=\vb{A}^{\intercal}$ has real eigenvalues and orthogonal eigenvectors; spectral theorem $\vb{A}=\vb{Q}\boldsymbol{\Lambda}\vb{Q}^{\intercal}$; pivot signs match eigenvalue signs; positive-definite tests (all eigenvalues > 0, all pivots > 0, all leading minors > 0, $\vb{x}^{\intercal}\vb{A}\vb{x}>0$); quadratic forms (bowl vs saddle); completing the square (pivots as the coefficients); $\vb{A}^{\intercal}\vb{A}$ positive definite iff full column rank.
- **Outline:** (1) recap: symmetric matrices showed up as $\vb{A}^{\intercal}\vb{A}$, they are special; (2) roadmap; (3) the spectral theorem; (4) positive definiteness and its four equivalent tests; (5) the geometry: $\vb{x}^{\intercal}\vb{A}\vb{x}$ as a bowl vs a saddle, tie to second-derivative tests; (6) completing the square exposes the pivots; (7) why $\vb{A}^{\intercal}\vb{A}$ matters (back to least squares); (8) recap + handoff: complex matrices and the FFT.
- **Visuals:**
  - V1 essential, Mermaid roadmap (Part 13).
  - V2 essential, Matplotlib 3D (signature): $\vb{x}^{\intercal}\vb{A}\vb{x}$ as a bowl (positive definite) vs a saddle (indefinite).
  - V3 essential, Manim PNG: the four positive-definiteness tests side by side.
  - V4 nice-to-have, Manim PNG: ellipse $\vb{x}^{\intercal}\vb{A}\vb{x}=1$ with eigenvectors as principal axes.
- **Cross-links:** prev Post 12, next Post 14.

### - [x] Post 14: Complex Matrices and the FFT

- **Slug:** `complex-matrices-and-the-fft`. **Lectures:** 26. **Tags:** `[Linear Algebra, Mathematics, Complex Matrices, Fourier Transform, FFT]`.
- **Running example:** the $4\times 4$ Fourier matrix $\vb{F}_4$ with $w=i$; the recursive factorization of $\vb{F}_{64}$ into two $\vb{F}_{32}$ blocks plus a permutation and a diagonal.
- **Concepts:** complex length $\vb{z}^{H}\vb{z}$; Hermitian $\vb{A}^{H}=\vb{A}$ (real eigenvalues, orthogonal eigenvectors); unitary $\vb{Q}^{H}\vb{Q}=\vb{I}$; Fourier matrix columns are powers of $w=e^{2\pi i/n}$ (orthogonal); FFT factorization reducing $n^2$ to $\tfrac{n}{2}\log_2 n$.
- **Outline:** (1) recap: orthogonality and eigenvalues, now over complex numbers; (2) roadmap; (3) why real dot products need conjugates; Hermitian and unitary; (4) the Fourier matrix and its orthogonal columns; (5) the FFT recursion (the punchline: huge speedup); (6) recap + handoff: when diagonalization fails (Jordan).
- **Visuals:**
  - V1 essential, Mermaid roadmap (Part 14).
  - V2 essential, Graphviz (signature): the FFT butterfly recursion $\vb{F}_n \to \vb{F}_{n/2}$ + permutation + diagonal.
  - V3 essential, Manim PNG: roots of unity $w=e^{2\pi i/n}$ on the unit circle / Fourier matrix structure.
  - V4 nice-to-have, Matplotlib: complexity $n^2$ vs $n\log n$.
- **Cross-links:** prev Post 13, next Post 15.

### - [x] Post 15: Similar Matrices and Jordan Form

- **Slug:** `similar-matrices-and-jordan-form`. **Lectures:** 28. **Tags:** `[Linear Algebra, Mathematics, Similar Matrices, Jordan Form]`.
- **Running example:** $\begin{bmatrix}2&1\\1&2\end{bmatrix}$ (eigenvalues 3, 1) and a similar non-symmetric matrix; the contrast between $4\vb{I}$ and $\begin{bmatrix}4&1\\0&4\end{bmatrix}$; a 4x4 nilpotent example with different block structures.
- **Concepts:** similar matrices $\vb{B}=\vb{M}^{-1}\vb{A}\vb{M}$ share eigenvalues; diagonalization as the special case $\vb{M}=\vb{S}$, $\vb{B}=\boldsymbol{\Lambda}$; non-diagonalizable matrices; Jordan blocks ($\lambda$ on the diagonal, 1s on the superdiagonal, one eigenvector each); number of blocks = number of independent eigenvectors.
- **Outline:** (1) recap: diagonalization needed enough eigenvectors, what if there aren't; (2) roadmap; (3) similarity and the invariance of eigenvalues; (4) the families of similar matrices; (5) Jordan form as the best you can do; (6) counting blocks; (7) recap + handoff: a decomposition that works for every matrix, the SVD.
- **Visuals:**
  - V1 essential, Mermaid roadmap (Part 15).
  - V2 essential, Manim PNG (signature): a Jordan block, $\lambda$ on the diagonal and 1s above, with its single eigenvector.
  - V3 nice-to-have, Manim PNG: similarity transform $\vb{B}=\vb{M}^{-1}\vb{A}\vb{M}$ preserving eigenvalues.
  - V4 nice-to-have, table: block structures for repeated eigenvalues.
- **Cross-links:** prev Post 14, next Post 16.

### - [x] Post 16: The Singular Value Decomposition

- **Slug:** `the-singular-value-decomposition`. **Lectures:** 29. **Tags:** `[Linear Algebra, Mathematics, SVD, Singular Values]`.
- **Running example:** $\begin{bmatrix}4&4\\-3&3\end{bmatrix}$ with singular values $\sqrt{32},\sqrt{18}$; a rank-1 matrix $\begin{bmatrix}4&3\\8&6\end{bmatrix}$.
- **Concepts:** $\vb{A}=\vb{U}\boldsymbol{\Sigma}\vb{V}^{\intercal}$ for any matrix; singular values $\sigma_i=\sqrt{\text{eig}(\vb{A}^{\intercal}\vb{A})}$; $\vb{V}$ an orthonormal basis for row space + nullspace, $\vb{U}$ for column space + left nullspace; generalizes the spectral theorem; exactly $r$ nonzero singular values.
- **Outline:** (1) recap: spectral theorem needed symmetric, Jordan was fragile, here is the universal one; (2) roadmap; (3) the idea: orthonormal input basis mapped to orthogonal output directions, scaled by $\sigma_i$; (4) compute $\vb{U},\boldsymbol{\Sigma},\vb{V}$ from $\vb{A}^{\intercal}\vb{A}$ and $\vb{A}\vb{A}^{\intercal}$ on the example; (5) the four subspaces inside the SVD; (6) recap + handoff: matrices ARE transformations.
- **Visuals:**
  - V1 essential, Mermaid roadmap (Part 16).
  - V2 essential, Manim animation (signature): a unit circle rotated by $\vb{V}^{\intercal}$, stretched by $\boldsymbol{\Sigma}$ into an ellipse, rotated by $\vb{U}$.
  - V3 essential, Manim PNG: SVD mapping the four subspaces (row space $\to$ column space via $\sigma_i$).
  - V4 nice-to-have, table: $\sigma_i$ linked to eigenvalues of $\vb{A}^{\intercal}\vb{A}$.
- **Cross-links:** prev Post 15, next Post 17.

### - [x] Post 17: Linear Transformations and Change of Basis

- **Slug:** `linear-transformations-and-change-of-basis`. **Lectures:** 30,31. **Tags:** `[Linear Algebra, Mathematics, Linear Transformations, Change of Basis, Image Compression]`.
- **Running example:** projection and rotation as coordinate-free maps; the derivative operator on quadratics ($c_1+c_2x+c_3x^2 \mapsto c_2+2c_3x$, matrix $\begin{bmatrix}0&1&0\\0&0&2\end{bmatrix}$); image compression of an 8x8 block in a Fourier/wavelet basis.
- **Concepts:** linear transformation $T(c\vb{v}+d\vb{w})=cT(\vb{v})+dT(\vb{w})$; a matrix represents $T$ once bases are chosen (column $j$ = $T(\vb{v}_j)$ in output coordinates); eigenvector basis gives a diagonal matrix; change of basis $\vb{x}=\vb{W}\vb{c}$; $\vb{B}=\vb{M}^{-1}\vb{A}\vb{M}$ (same as similarity); image compression by changing basis and thresholding small coefficients; wavelet basis.
- **Outline:** (1) recap: we kept multiplying by matrices, what IS a matrix; (2) roadmap; (3) transformations without coordinates; (4) choosing a basis produces the matrix; the derivative example; (5) change of basis and why it is similarity; (6) the payoff: JPEG/wavelet image compression; (7) recap + handoff: the final piece, inverses for every matrix.
- **Visuals:**
  - V1 essential, Mermaid roadmap (Part 17).
  - V2 essential, Manim animation (signature): a "house" shape under rotation, reflection, and a general matrix.
  - V3 essential, Matplotlib: image compression demo on a real grayscale image (keep top-$k$ Fourier/wavelet coefficients, show reconstruction at several $k$).
  - V4 nice-to-have, Manim PNG: change-of-basis relation $\vb{B}=\vb{M}^{-1}\vb{A}\vb{M}$; or wavelet basis vectors (Matplotlib).
- **Cross-links:** prev Post 16, next Post 18.

### - [x] Post 18: The Pseudoinverse

- **Slug:** `the-pseudoinverse`. **Lectures:** 33. **Tags:** `[Linear Algebra, Mathematics, Pseudoinverse, Generalized Inverse]`.
- **Running example:** a tall full-column-rank $\vb{A}$ (least squares, left inverse $(\vb{A}^{\intercal}\vb{A})^{-1}\vb{A}^{\intercal}$); a wide full-row-rank $\vb{A}$ (right inverse $\vb{A}^{\intercal}(\vb{A}\vb{A}^{\intercal})^{-1}$); a general rank-deficient $\vb{A}$ via SVD.
- **Concepts:** two-sided inverse (square, full rank); left inverse (full column rank); right inverse (full row rank); pseudoinverse $\vb{A}^{+}=\vb{V}\boldsymbol{\Sigma}^{+}\vb{U}^{\intercal}$; $\vb{A}^{+}\vb{A}$ = projection onto the row space, $\vb{A}\vb{A}^{+}$ = projection onto the column space; Moore-Penrose conditions; null spaces are exactly why true inverses fail.
- **Outline:** (1) recap: "invertible or not" was too crude; (2) roadmap; (3) left and right inverses and the rank conditions that allow them; (4) the pseudoinverse from the SVD; (5) it inverts the part that can be inverted (row space to column space) and kills the null spaces; (6) ties back to least squares (Posts 7-8); (7) **series finale recap**: revisit Post 1's question (can the columns reach every $\vb{b}$?) and show how the whole journey answered it; the four subspaces and the right basis were the recurring theme.
- **Visuals:**
  - V1 essential, Mermaid roadmap (Part 18, final node highlighted; could also show all 18 as "done").
  - V2 essential, Manim PNG (signature): $\vb{A}$ mapping row space to column space and $\vb{A}^{+}$ mapping back, null spaces collapsed.
  - V3 essential, table: inverse type vs rank condition (two-sided / left / right / pseudo).
  - V4 nice-to-have, Manim PNG: $\boldsymbol{\Sigma}^{+}$ structure (reciprocate nonzero $\sigma_i$, zero the rest).
- **Cross-links:** prev Post 17; closing links back to Post 1 and to the `linear-algebra-for-ml` and `linear-regression` posts.

---

## Self-review (checked against the lecture digests)

- **Coverage:** all of lectures 1-12, 14-31, 33 are mapped to a post (see slug/lecture table); 13 and 32 correctly excluded as exam reviews.
- **Narrative continuity:** every post has an explicit recap-in / handoff-out, and the three-act spine is preserved; the recurring "can the columns reach every $\vb{b}$" question opens (Post 1) and closes (Post 18) the series.
- **Visual-tool rules:** math-bearing diagrams use Manim PNG; structural/no-math diagrams use Mermaid or Graphviz; math-heavy plot animations (least squares fit, eigen-action, stability, Markov convergence, image compression) use Matplotlib; geometric/structural animations (projection, Gram-Schmidt, SVD rotate-stretch-rotate, house transform) use Manim. Each conforms to the dark-theme constraints above.
- **Open items to confirm at execution:** exact OCW URL/year and textbook edition in the two BibTeX entries; whether to include the nice-to-have visuals per post (decide in each mini-plan); final dates as posts are published.
