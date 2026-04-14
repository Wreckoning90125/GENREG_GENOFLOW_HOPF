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

## Hopf Geometric Feature Extractor (v8 – v12)

The repository also contains a separate experimental subsystem that sits
alongside the GenoFlow IDE: a closed-form kernel-ridge classifier on top
of a hand-designed geometric feature basis derived from the 600-cell,
its ADE eigenspace decomposition, and the discrete de Rham ladder up
through Ω³ cell forms via the Hopf fibration S³ → S².

Current best result: **97.39% MNIST test accuracy (v10)**, via a
multi-scale polynomial kernel ridge over 879 fixed geometric features
(three pixel-kernel softness scales × 293 features). No learned
nonlinearities in the trained path; the feature extractor is fixed
geometry and only the kernel ridge readout is fit. v11 and v12 extend
the geometry (face / Ω² and cell / Ω³ eigenspaces) but **do not
improve** MNIST accuracy over v10 — they sit at 97.32% and 97.24%
respectively.

- Current trainer: `train_ade_hopf.py` (v10)
- Extended trainers: `train_v11.py`, `train_v12.py`
- Feature extractor and geometric primitives: `hopf_controller.py`,
  `ade_geometry.py`, `cell600.py`
- Checkpoints: `checkpoints/hopf_v8_ade/` … `checkpoints/hopf_v12_ade/`

**Read `FINDINGS.md` before citing or building on these results.** It
documents what the numbers do and do not support, retracts an earlier
chirality interpretation of the signed Berry phase that the rigorous
cross-digit comparison did not reproduce, lists the experimental
choices that are not pinned in this repo, and explains what has not
yet been built (the multi-stage Hopf architecture from the original v2
sketch). The next planned test for the Hopf subsystem is
pre-registered in `PRE_REGISTRATION.md`.

The Hopf subsystem is independent of the GenoFlow Snake IDE described
above — it shares the repository but not the runtime. The Snake
environment, protein network, and LiteGraph frontend remain as
documented.

---

## Contribution

GenoFlow is an evolving project. Contributions are welcome.

Areas of interest:

* New protein types
* Additional environments
* Visualization tools
* Extensions to evolutionary algorithms

Pull requests and design discussions are encouraged.
