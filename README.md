# 🧬 GenoFlow: The Visual IDE for Genetic Regulatory AI
** Note this application is still in development. Please pin or note bugs. 

**GenoFlow** (formerly *Genome Studio*) is a visual, node-based development environment for designing, training, and tuning **GENREG** (Genetic Regulatory) AI models.

Inspired by biological gene regulatory networks, GenoFlow lets users visually construct complex, self-adapting AI architectures that combine a functional neural network layer with a stateful, trust-modulating protein network.

---

## Key Features

### Visual, Node-Based Architecture

Build and visualize your entire AI model using an intuitive, ComfyUI-like graph editor powered by **LiteGraph.js**.

### Biologically Inspired Layers (GENREG)

**Protein Network (Regulatory Layer)**
Stateless, self-adapting *proteins* (e.g., `Sensor`, `Trend`, `Comparator`, `TrustModifier`) process environmental signals to generate a **Trust Delta** (fitness signal).

**Controller Network (Functional Layer)**
A simple feed-forward neural network that selects actions based on processed signals.

### Trust-Based Evolution

Genome fitness is determined by accumulated **Trust**, promoting robust and adaptive behaviors through evolutionary pressure.

### Real-Time Environment

Train and observe AI behavior in a simulated **Snake Environment**, with real-time visualization via a separate **Pygame** window.

### Client–Server Architecture

A Python **FastAPI / WebSocket** backend handles all heavy processing (evolution, environment steps), while the JavaScript frontend provides the interactive IDE.

---

## Quick Start

### 1. Prerequisites

GenoFlow requires **Python** and several libraries, including `fastapi` and `uvicorn` for the server, and optionally `pygame` for environment visualization.

```bash
# Recommended: Create and activate a virtual environment
python -m venv venv
source venv/bin/activate
```

### 2. Installation

Install the required Python packages:

```bash
pip install -r requirements.txt

# To enable environment visualization
pip install pygame
```

> **Note**
> `requirements.txt` includes core dependencies such as `fastapi`, `uvicorn`, and `websockets`.

### 3. Running the Server

Start the backend server using `uvicorn` as defined in `start_server.py`:

```bash
python start_server.py
```

The server typically starts at:

```
http://0.0.0.0:8000
```

### 4. Accessing the IDE

Open your browser and navigate to:

```
http://localhost:8000
```

Load the provided `snake_training_template.json` to begin exploring the training flow.

---

## Architecture Overview

GenoFlow is divided into distinct layers reflecting its biological inspiration and client–server design.

| Component        | Technology                  | Role                                               | Core Files                                 |
| ---------------- | --------------------------- | -------------------------------------------------- | ------------------------------------------ |
| Frontend IDE     | LiteGraph.js, Vanilla JS    | Visual graph editor, monitoring, real-time control | `static/js/*.js`, `static/index.html`      |
| Backend Server   | Python (FastAPI, WebSocket) | AI processing, evolution, simulation               | `start_server.py`                          |
| Controller Layer | Python (`Controller` class) | Feed-forward NN for action selection               | `genreg_controller.py`                     |
| Regulatory Layer | Python (Protein classes)    | Generates Trust Delta fitness signals              | `genreg_proteins.py`                       |
| Evolution Core   | Python (Genome, Population) | Evolutionary process and selection                 | `genreg_genome.py`, `genreg_population.py` |

---

## The GENREG Model Flow

The training process is orchestrated by the node graph in a continuous loop:

1. **Environment Step** (`SnakeEnvironment`) outputs signals.
2. **Protein Network** processes signals and calculates **Trust Delta**.
3. **Controller Network** selects an **Action**.
4. **Action** is fed back into the environment.
5. Episode ends and total **Trust** determines Genome fitness.
6. **Generation Manager** triggers population evolution.

---

## Node Types

The IDE provides modular nodes to construct and monitor AI behavior.

| Category    | Example Nodes                         | Purpose                                    |
| ----------- | ------------------------------------- | ------------------------------------------ |
| Regulatory  | Sensor, Trend, Trust Modifier         | Process signals and influence Genome Trust |
| Functional  | Controller Network                    | Select actions from processed signals      |
| Environment | Snake Environment, Visualize (Pygame) | Simulate and visualize behavior            |
| Evolution   | Population Controller, Episode Runner | Manage training and mutation lifecycle     |

---

---

## Controller backends

`Genome` supports three controller backends; choose via
`controller_type` in {`"mlp"`, `"hopf"`, `"snake_hopf"`}.

| Backend | Source | What it does | Where it shines |
|---|---|---|---|
| `mlp` (default) | `genreg_controller.py` | Tanh feed-forward, weights mutated | Abstract-signal control; smooth fitness landscapes |
| `hopf` | `hopf_controller.py` + `cell600.py` + `ade_geometry.py` | 600-cell ADE-eigenspace features over a vMF-soft-assigned vertex activation | 2D-spatial / image input where the pixel kernel makes geometric sense |
| `snake_hopf` | `snake_hopf_controller.py` + `hopf_decagon.py` | Multi-channel directional embedding of Snake's signals → 600-cell activation; Hopf-decagon + vertex-cardinal-affinity geometric action prior; learned readout adds refinement | Snake-style control where actions live in a Cartesian 4-direction symmetry that the icosahedral geometry can express |

### Snake benchmark

`bench_snake.py` runs five controller types head-to-head under the
same GENREG evolutionary loop (5 seeds × 50 generations × 50
population × 200 steps/life, fitness = food eaten). Random and
Greedy are fixed-policy baselines (no learning, 0 parameters); the
rest evolve.

| Metric | random | **greedy** | mlp | hopf | **snake_hopf** |
|---|---:|---:|---:|---:|---:|
| final best food | 0.80 | **37.20** | 1.20 | 1.20 | **37.40** |
| final avg food | 0.06 | **29.97** | 0.06 | 0.04 | 27.98 |
| max best food ever | 2.40 | **41.40** | 2.40 | 1.80 | **41.20** |
| max best trust ever | 226 | **994.3** | 136 | 121 | **993.5** |
| param count | 0 | 0 | 260 | 276 | 277 |

**Result**: SnakeHopf matches the greedy heuristic ceiling within
noise on every metric (4 ties, 1 at 93%). Vs the GENREG default MLP
controller: **31× more food on average, 31× final best food**.
Vs random: **466× more food**.

**Why**: SnakeHopf's vertex-cardinal-affinity action prior is
greedy-equivalent by construction — each 600-cell vertex's Hopf
projection on S² has a fixed cosine similarity to each cardinal
direction, so an activation peaked toward where the food lies
maps directly to the right cardinal action. The food-direction
embedding channel dominates the activation; the readout starts
small (`readout_init_scale=0.05`) so the prior runs gen 0 at
~96% of greedy performance, and evolution refines from there
without first having to fight a noisy random readout.

The MLP controller, with the same evolutionary loop, never finds
the food-direction-to-action mapping that the geometric prior
encodes structurally. After 50 generations, MLP eats roughly the
same amount as random — the default GENREG GA budget is too small
for the MLP to learn the mapping from data alone.

Raw numbers in `checkpoints/snake_ab/results.json`.

## Math substrate (top level)

These modules underpin the geometric controllers and are imported
by the GENREG runtime:

- `cell600.py` — 120 vertices of the 600-cell on S³ as quaternions,
  oriented edges, triangles, tetrahedra, discrete coboundaries
  d₀/d₁/d₂, scalar/curl/face/cell eigenspaces with the McKay
  correspondence to E₈ wired in.
- `hopf_controller.py` — Hopf map S³ → S², section, Pancharatnam
  phase, Cl(3,0) rotor-composition Berry phase verified to <1e-8;
  three geometric controllers (v6 `HopfController`, v7
  `VertexHopfController`, v8/v9 `ADEHopfController`).
- `ade_geometry.py` — orbit-method irrep copy decomposition on the
  cell600 spectra; CG projector to V₁ via character formula.
- `hopf_decagon.py` — 12-fibre partition of the 120 vertices into
  great-circle decagons under a C₁₀ subgroup of 2I (used by the
  Snake action geometry).
- `snake_hopf_controller.py` — multi-channel directional embedding
  for Snake signals; Hopf-decagon + vertex-cardinal-affinity action
  prior.

## Experimental sublibraries (`experiments/`)

Exploratory work that builds on the math substrate but is independent
of the GenoFlow runtime. See `experiments/README.md` for the layout.

- `experiments/mnist_geometric/` — MNIST classification via fixed
  Hopf-geometric features + kernel ridge. **97.39% test accuracy
  (v10)** with no learned nonlinearities. v11/v12 extensions did not
  improve. See `FINDINGS.md` inside the directory for the honest
  record (and a retracted chirality claim).
- `experiments/stellarator_lab/` — discrete exterior calculus on the
  600-cell with machine-precision irrep-graded Hodge decomposition,
  closed-form circumcentric Hodge stars, generalized-Hopf seed
  fields, field-line Berry diagnostics. Frozen at git tag
  `stellarator-lab-foundations`. 42-test suite passes.

Run experiment scripts from the repo root with `PYTHONPATH=.`.

---

## Contribution

GenoFlow is an evolving project. Contributions are welcome.

Areas of interest:

* New protein types
* Additional environments
* Visualization tools
* Extensions to evolutionary algorithms

Pull requests and design discussions are encouraged.
