Below is a **complete, production‑quality `README.md`** for your CAS Dashboard application.

This documentation contains:

*   Clear setup instructions
*   How to run the app
*   Module‑by‑module explanations
*   Example workflow
*   Screenshots placeholders
*   Testing instructions
*   Deployment instructions

No factual claims from web search are needed → **no citations included**.

***

# 📘 **README.md — Unified CAS Simulation Dashboard**

## 🛰️ Overview

The **Unified CAS Simulation Dashboard** is an integrated analytical platform combining:

*   **Evolutionary Game Theory (CAS Engine)**
*   **Temporal Coalition Dynamics** (α(t), β(t))
*   **Agent‑Based Modeling (ABM)**
*   **DST Evidence Fusion**
*   **Info‑Gap Robustness Analysis**
*   **Network Controllability Mapping**
*   **Market Impact Proxies**
*   **OSINT Event Injection**
*   **Scenario Loader + Scenario Persistence**

The dashboard allows analysts to simulate multi‑player geopolitical competition, explore coalition dynamics, visualize emergent behavior, run robustness checks, and study influence networks — all inside a single Streamlit application.

***

# 🚀 Features

### ✔ CAS Engine (4‑player, multi‑strategy)

Dynamic evolutionary simulation with:

*   Nonlinear utilities
*   Misperception matrices
*   Influence weights
*   Temporal coalitions (α(t), β(t))
*   Monte‑Carlo ensemble simulation

### ✔ Scenario Loader (Option A)

Pre‑configured scenarios automatically adjust:

*   Payoffs
*   Confidence
*   Loss aversion
*   Coalition presets

### ✔ Coalition Visualization (PyVis)

Interactive network‑layout visualization of coalitions.

### ✔ ABM Simulation

Agent‑based dynamic visualization showing:

*   Diffusion
*   Interaction
*   Imitation
*   Strategy clustering

### ✔ Evolutionary Winners

Monte‑Carlo fixation analysis showing:

*   Dominant strategies
*   Extinction patterns
*   Coalition dominance indices

### ✔ Dempster‑Shafer DST Fusion

*   Slider‑based evidence input
*   JSON mass input
*   Belief/Plausibility tables

### ✔ Info‑Gap Robustness

*   α‑curve robustness analysis
*   KPI‑driven satisficing threshold

### ✔ Network Controllability

*   Directed graph editor
*   PageRank & Betweenness
*   Minimum driver nodes

### ✔ Market‑Impact Proxy Model

*   Energy
*   Industrial stress
*   Defense pressure
*   Space economics
*   Trade‑war probability

### ✔ OSINT Monitor

*   Inject events
*   Fuse LNG shock
*   Perturb CAS payoff matrices

### ✔ Scenario Persistence

Save and load model states via SQLite.

***

# 📦 **Installation**

### 1. Clone or download the project

    git clone <your repo URL>
    cd cas_dashboard_project

### 2. Install dependencies

    pip install -r requirements.txt

**requirements.txt** should include:

    streamlit
    numpy
    pandas
    matplotlib
    networkx
    pyvis

(Optionally also `pytest` for unit tests)

***

# ▶️ **Running the Dashboard**

From the project directory, run:

```bash
streamlit run app_main.py
```

Then open your browser:

    http://localhost:8501

If Streamlit doesn’t auto‑open, visit it manually.

***

# 🕹️ **How to Use the Dashboard**

## 🔹 1. Simulation Engine

Navigate to:

**Module → Simulation Engine**

Configure:

*   Payoff matrices
*   C‑matrices (misperception)
*   Influence weights
*   Dynamics (replicator / logit response)
*   Utility curvature & loss‑aversion
*   Time horizon
*   Monte‑Carlo options

Click **Run Simulation**.

This populates:

*   `traj_to_plot`
*   `mc_bundle`

These power ABM and evolutionary modules.

***

## 🔹 2. Load a Scenario Preset

Each scenario auto‑sets:

*   Payoff shifts
*   Confidence
*   Loss aversion
*   Coalition presets

Use in sidebar:
**Scenario Loader → Apply Scenario**

***

## 🔹 3. Coalition Visualization

Shows:

*   Coalition structure
*   Influence graph (modified by β(t))
*   Actor node colors

Requires `pyvis` installed.

***

## 🔹 4. ABM Visualization

Simulates micro‑agents:

*   Moving
*   Interacting
*   Imitating strategies

Reveals emergent macro‑patterns.

Requires that the Simulation Engine has run.

***

## 🔹 5. Evolutionary Winners

Shows:

*   Strategy fixation
*   Strategy extinction
*   Coalition dominance patterns

Requires Monte‑Carlo simulation.

***

## 🔹 6. DST Fusion

Supports:

*   Slider input
*   JSON input

Displays:

*   Belief
*   Plausibility

Used by Market Impact module.

***

## 🔹 7. Info‑Gap

Analyzes:

*   KPI behaviour vs α
*   Acceptance probability vs threshold

Shows robustness radius.

***

## 🔹 8. Network Control

Allows you to:

*   Edit a directed geopolitical network
*   Compute PageRank
*   Compute betweenness
*   Compute minimum driver nodes

***

## 🔹 9. Market Impact

Uses fused DST mass to infer approximate:

*   Oil price proxy
*   Industrial stress index
*   Defense pressure
*   Space launch cost multiplier
*   Trade‑war probability

***

## 🔹 10. OSINT Monitor

Lets you:

*   Inject predefined events
*   Fuse LNG shock
*   Perturb CAS payoff matrices

Useful for scenario exploration.

***

## 🔹 11. Scenario Persistence

Allows:

*   Saving scenario JSON
*   Loading previous scenarios

Stores data in:

*   SQLite (`app_data.db`)
*   JSON fallback (`fallback.json`)

***

# 🧪 **Running Unit Tests**

Make sure `pytest` is installed:

    pip install pytest

Then run:

    pytest tests/

Included tests check:

*   CAS core functions
*   DST parsing + fusion
*   Coalition weight adjustment
*   Network controllability

***

# 🧯 **Resetting Simulation State**

Inside the Simulation Engine:

Click:

    🔄 Reset Simulation State

This clears:

*   Payoffs
*   C‑matrices
*   Weights
*   Trajectories
*   MC data

***

# 📁 Directory Structure

    cas_dashboard_project/
    │
    ├── app_main.py
    ├── requirements.txt
    ├── tests/
    │   ├── test_cas_engine.py
    │   ├── test_dst.py
    │   ├── test_coalitions.py
    │   └── test_networks.py
    └── (optional additional modules)

***

# 📬 Support

If you need:

*   Architecture refactor
*   GPU‑accelerated Monte‑Carlo
*   Integration with live market feeds
*   Scenario‑based policy brief generation
*   Deployment on Streamlit Cloud or Vercel

Just ask — I'm here to help.

***

# 🎉 **README.md generated successfully**

If you’d like a **PDF manual**, **API documentation**, or a **diagram-heavy version**, just say:

**“Generate documentation pack.”**
