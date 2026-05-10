# Agent operating instructions for this repo

This document is the non-negotiable policy for any agent (Claude
Code or otherwise) doing work in this codebase. It exists because
the codebase has accumulated specific, validated dependencies on
mature symmetry / geometry / crystallography libraries, and any
new work that re-implements what these libraries already do is
producing AI sprawl, not value.

If a task crosses into a domain where the libraries below do not
provide coverage, **the agent must surface that gap to the user
explicitly before writing replacement code**. Rolling our own is
the last resort, not the first.

## Hard-required libraries

The following libraries must be used whenever their domain is
relevant to a task. "Used" means imported and called — not
reimplemented, not "inspired by", not "we wrote our own to avoid
the dependency".

### Geometry / algebra

- **`clifford`** — Clifford / geometric algebra Cl(p, q, r).
  Already used in `hopf_controller.py` (Cl(3,0) rotor verification
  of Berry phase). Any new rotor / multivector / Berry-phase /
  bivector work goes through `clifford`. Do not hand-roll quaternion
  or rotor algebra when `clifford` (or the existing `qmul`/`qnormalize`
  helpers in `hopf_controller.py` that wrap a specific Cl(3,0)
  convention) already handles it.
- **`numpy-quaternion`** — quaternion algebra at numpy speed.
  Already in `requirements.txt`. Use for bulk quaternion ops where
  the loop overhead in pure-numpy hand-coded `qmul` matters.
- **`mpmath`** — arbitrary-precision real / complex arithmetic.
  Use any time a calculation needs more than double precision —
  irrep character tables, rep-theory eigenvalue identification,
  group-theoretic constants, exact Hodge-star verification,
  numerical-zero checks where machine epsilon is not tight enough
  to distinguish "true zero" from "numerical residual". Do NOT
  use Python's built-in `decimal` for these — it does not handle
  transcendentals correctly.

### Crystallography (the 230 space groups)

- **`spglib`** — symmetry detection, refinement, primitive cell,
  symmetry operations for the 230 3D space groups. The default
  for "what is the symmetry of this atomic structure".
- **`moyo`** (pip: `moyopy`) — faster modern Rust-backed
  alternative to spglib. Use when running spglib on bulk datasets
  becomes a bottleneck. API is similar.
- **`spgrep`** — irreducible representations of space groups at any
  k-point (the *little group* irreps). This is the actual
  Bloch-decomposition rep-theory engine. Use whenever an irrep
  block-diagonalization is wanted on a periodic system.
- **`pyxtal`** — random / structured crystal-structure generation
  conditioned on a space group. Use for any structure-search,
  data-augmentation-by-symmetry, or structure-prediction task on
  periodic systems.
- **`gemmi`** — CIF / mmCIF / reflection file I/O and validation.
  Use for any read/write of crystallographic file formats. Do
  not parse CIF by hand or with `re.match`.

### General

- **`scipy`** — already in requirements; sparse linear algebra,
  ODE integration (LSODA in `experiments/stellarator_lab/`),
  optimization. Reach for it before hand-rolling.
- **`networkx`** — graph data structures.
- **`h5py`** / **`pyvista`** — HDF5 and VTK I/O for 3D scientific
  data products. Already used by `experiments/stellarator_lab/`.
- **`pytest`** — every new module under `experiments/` ships with
  a test file; the bar set by `experiments/stellarator_lab/tests/`
  (42 tests, machine-precision assertions) is the bar.

## Quasicrystal / superspace coverage gap (known)

The clean Python equivalent of `spglib + spgrep + pyxtal` does
not exist for quasicrystals or for (3 + d)-dimensional superspace
groups (SSGs). The rep theory IS fully tabulated (Stokes' SSESG
tables, ISOTROPY suite), but it lives behind:

- **JANA2020** — desktop GUI program for SSG refinement. Driven
  via CIF I/O if at all.
- **Bilbao Crystallographic Server** — CGI / web-form-based
  endpoints for SSG operators and irrep matrices. No public
  Python API. Some groups scrape via `requests` + `BeautifulSoup`
  against the CGI endpoints.
- **ISOTROPY / ISODISTORT** (Stokes, Campbell, BYU) — closed-source
  executable + web frontend with the actual SSG irrep machinery.
- **QCSTRUC / QCDIFF** (Yamamoto) — Fortran-era reference
  implementation for icosahedral and decagonal nD space groups.

If a task in this repo requires SSG / quasicrystal symmetry
analysis, the agent's response is **not** to silently roll a
new implementation. The agent must:

1. Name the gap explicitly to the user.
2. Identify which of the above tools / databases would cover
   the gap if invoked / scraped.
3. Wait for direction on whether to:
   (a) drive JANA2020 via CIF I/O,
   (b) scrape Bilbao CGI endpoints,
   (c) port Stokes' SSESG tables and write the (3 + d)D Bloch
       decomposition manually with `spgrep`-style bookkeeping,
   (d) defer the SSG-aware piece and use 230-group machinery on
       the rational approximant of the quasicrystal,
4. Or, if directed to roll our own: do it in a self-contained
   subdirectory with a `README.md` noting what canonical tool
   it would be replaced by once available, and a test suite
   that verifies output against the canonical tool's published
   tables (Stokes etc.) for at least three non-trivial cases.

## Surfacing new tooling needs

When a task crosses into a differential-geometry or
operation-space *not covered* by the libraries above, the agent
must explicitly surface the coverage gap to the user before
writing code, of the form:

> "This needs X (e.g., topological band-structure invariants,
> Wannier function construction, n-D persistent homology, ...).
> The libraries currently declared in this repo (`spglib`,
> `spgrep`, `clifford`, `mpmath`, …) do not cover this. Candidate
> tools: [name them, e.g. `pymatgen.analysis.topology`,
> `wannierberri`, `gudhi`, `dionysus`]. Should I (a) add one of
> them as a dep and use it, (b) defer this piece, (c) roll our
> own as a contained sub-experiment with tests vs published
> tables?"

Default to (a). (c) is only acceptable when no canonical tool
exists (the SSG case above) AND the user has explicitly approved
rolling our own.

## What "embedded across the repo" means concretely

These libraries are not a wishlist — they are how the repo
operates.

- New crystallographic feature pipelines start with
  `spglib.get_symmetry_dataset(...)` or `moyopy` equivalent for
  symmetry detection. Not a custom symmetry detector.
- New irrep-decomposed pipelines build their irrep basis from
  `spgrep` output. Not from hand-rolled character tables.
- New CIF / mmCIF file readers use `gemmi`. Not `re`-based parsing.
- New high-precision verification tests use `mpmath` for the
  reference values. Not `decimal` and not extended-precision
  numpy floats.
- New Berry / Pancharatnam / rotor / Cl(p,q) work uses
  `clifford`. Not from-scratch index gymnastics in numpy.

When existing code in this repo has not yet been migrated to a
required library (e.g. `hopf_controller.py`'s pure-numpy `qmul`),
**do not retrofit it speculatively**. Validated benches stay
validated. Migration is a separate, explicit, user-approved task.
The hard requirement applies to *new* code unless the user
specifically asks for a refactor.

## Other expectations

- Pre-registration discipline applies to any benchmark claim.
  See `experiments/mnist_geometric/PRE_REGISTRATION.md` for the
  template and the kind of failure mode it exists to prevent
  (post-hoc venue choice, baseline drift, success-criterion
  redefinition).
- No FINDINGS / RETRACTION / META-ANALYSIS docs unless the user
  asks for them. Rigor lives in code, commits, and benches that
  reproduce — not in personal-failure-log markdown files.
- Background runs that exceed a few minutes must be reported
  back with profiling evidence before being launched, not "let me
  see if it finishes overnight". The user's time is finite.
