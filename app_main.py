# =============================================================
# app_main.py  —  Unified CAS Dashboard (Production Build B1)
# =============================================================
#   • 4-Player CAS Simulation Engine
#   • Scenario Loader (Option A)
#   • Temporal Coalition Dynamics α(t), β(t)
#   • Coalition Visualization (PyVis)
#   • ABM Visualization
#   • Evolutionary Winners
#   • DST Fusion
#   • Info-Gap Robustness
#   • Network Controllability
#   • Market Impact Proxies
#   • OSINT Shock Injection
#   • Scenario Persistence
#   • User Guide
#   • Simulation Reset Button
#   • Unit Test–Friendly Structure
# =============================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3, json, os, time
from dataclasses import dataclass
from typing import Dict, List, Tuple

# Optional: PyVis
try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except:
    PYVIS_AVAILABLE = False

# NetworkX for controllability
import networkx as nx

# -------------------------------------------------------------
# Page config
# -------------------------------------------------------------
st.set_page_config(page_title="Unified CAS Simulation Dashboard", layout="wide")
st.title("Unified CAS Simulation Dashboard — CAS • Coalitions • ABM • DST • Networks")

# -------------------------------------------------------------
# Persistence
# -------------------------------------------------------------
@dataclass
class PersistenceManager:
    db_path: str = "app_data.db"
    json_path: str = "fallback.json"
    def __post_init__(self):
        self.conn = sqlite3.connect(self.db_path)
        cur = self.conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS scenarios(
                id TEXT PRIMARY KEY,
                payload TEXT,
                created TEXT,
                updated TEXT
            );
        """)
        self.conn.commit()
        if not os.path.exists(self.json_path):
            with open(self.json_path, "w") as f:
                json.dump({}, f)

    def save(self, key, obj):
        import datetime
        stamp = datetime.datetime.utcnow().isoformat()
        raw = json.dumps(obj)
        cur = self.conn.cursor()
        cur.execute("REPLACE INTO scenarios VALUES (?,?,?,?)", (key, raw, stamp, stamp))
        self.conn.commit()
        if os.path.exists(self.json_path):
            with open(self.json_path, "r") as f:
                data = json.load(f)
        else:
            data = {}
        data[key] = obj
        with open(self.json_path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, key):
        cur = self.conn.cursor()
        cur.execute("SELECT payload FROM scenarios WHERE id=?", (key,))
        row = cur.fetchone()
        return json.loads(row[0]) if row else None


pm = PersistenceManager()

# -------------------------------------------------------------
# CAS UTILITIES  (core math functions)
# -------------------------------------------------------------
@dataclass
class UtilityParams:
    gamma: float = 0.85
    lambda_loss: float = 1.5

@dataclass
class DynamicParams:
    method: str = "smoothed_best_response"
    eta: float = 0.15
    beta: float = 3.0

def utility_transform(x, u):
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = np.power(x[pos], u.gamma)
    out[~pos] = -u.lambda_loss * np.power(-x[~pos], u.gamma)
    return out

def expected_utilities_matrix(payoff, opp_mix, u):
    S = payoff.shape[0]
    EU = np.zeros(S)
    trans = utility_transform(payoff, u)
    for i in range(S):
        EU[i] = float(np.dot(trans[i,:], opp_mix))
    return EU

def softmax(x, beta):
    z = x - np.max(x)
    e = np.exp(beta * z)
    return e / np.clip(np.sum(e), 1e-12, None)

def normalized(v):
    v = np.clip(v, 1e-12, None)
    return v / np.sum(v)

def apply_misperception(true_mix, C, confidence):
    Cn = C / np.clip(np.sum(C,axis=0),1e-12,None)
    perceived = Cn @ true_mix
    uniform = np.ones_like(true_mix)/len(true_mix)
    return normalized(confidence*perceived + (1-confidence)*uniform)

def update_mix(x, EU, dyn):
    if dyn.method == "replicator":
        avg = float(np.dot(x,EU))
        nxt = x*(1 + dyn.eta*(EU-avg))
        return normalized(nxt)
    br = softmax(EU, dyn.beta)
    nxt = (1-dyn.eta)*x + dyn.eta*br
    return normalized(nxt)

# -------------------------------------------------------------
# Coalition Temporal Dynamics α(t), β(t)
# -------------------------------------------------------------
st.sidebar.subheader("Temporal Coalition Dynamics")

alpha_0 = st.sidebar.slider("Initial α₀", 0.0, 3.0, 1.0, 0.1)
alpha_slope = st.sidebar.slider("α slope per timestep", -0.1, 0.1, 0.0, 0.01)

beta_0 = st.sidebar.slider("Initial β₀", 1.0, 5.0, 2.0, 0.1)
beta_slope = st.sidebar.slider("β slope per timestep", -0.1, 0.1, 0.0, 0.01)

def compute_alpha_beta(t):
    a = max(alpha_0 + alpha_slope * t, 0.0)
    b = max(beta_0 + beta_slope * t, 1e-3)
    return a, b

def adjust_weights_for_coalitions(W, coalitions, beta):
    W2 = W.copy()
    for cname, members in coalitions.items():
        for p in members:
            for q in members:
                if p != q:
                    W2[(p,q)] *= beta
    return W2

# -------------------------------------------------------------
# Coalition presets
# -------------------------------------------------------------
coalition_presets = {
    "No Coalitions": {},
    "Blue vs Red": {"Blue":["US","China"], "Red":["Iran","Russia"]},
    "Tripolar": {"BlocA":["US"], "BlocB":["China"], "BlocC":["Iran","Russia"]},
    "Bridge Scenario": {"Deal":["US","Iran"], "Outside":["Russia","China"]}
}

st.sidebar.subheader("Coalition Designer")
preset_choice = st.sidebar.selectbox("Preset", list(coalition_presets.keys()))
coalitions = coalition_presets[preset_choice]
custom_json = st.sidebar.text_area("Custom coalition JSON (optional)", key="custom_coal_json")
if custom_json.strip():
    try:
        coalitions = json.loads(custom_json)
        st.sidebar.success("Loaded custom coalition")
    except:
        st.sidebar.error("Invalid JSON")
st.session_state["coalitions"] = coalitions

# -------------------------------------------------------------
# Scenario Loader (Option A)
# -------------------------------------------------------------
scenario_presets = {
    "None": {},
    "Scenario: LNG Stress": {
        "payoff_shift": 0.3,
        "confidence": 0.7,
        "loss_aversion": 1.8,
        "coalition": "Blue vs Red"
    },
    "Scenario: Hormuz Shock": {
        "payoff_shift": 0.5,
        "confidence": 0.6,
        "loss_aversion": 2.0,
        "coalition": "Tripolar"
    },
    "Scenario: Industrial Fragility": {
        "payoff_shift": 0.2,
        "confidence": 0.9,
        "loss_aversion": 2.5,
        "coalition": "Bridge Scenario"
    },
    "Scenario: Global Tightness": {
        "payoff_shift": 0.4,
        "confidence": 0.8,
        "loss_aversion": 1.6,
        "coalition": "Blue vs Red"
    }
}

st.sidebar.subheader("Load Scenario Preset")
sel_scenario = st.sidebar.selectbox("Simulation Scenario", list(scenario_presets.keys()))

def apply_scenario(preset_name):
    if preset_name == "None":
        return
    sc = scenario_presets[preset_name]
    if "payoff" in st.session_state:
        for k, M in st.session_state["payoff"].items():
            st.session_state["payoff"][k] = M + sc["payoff_shift"]
    st.session_state["confidence_override"] = sc["confidence"]
    st.session_state["loss_aversion_override"] = sc["loss_aversion"]
    sc_coal = sc.get("coalition")
    if sc_coal in coalition_presets:
        st.session_state["coalitions"] = coalition_presets[sc_coal]

if st.sidebar.button("Apply Scenario"):
    apply_scenario(sel_scenario)
    st.sidebar.success(f"Scenario '{sel_scenario}' applied.")

# -------------------------------------------------------------
# Navigation
# -------------------------------------------------------------
page = st.sidebar.selectbox(
    "Module",
    [
        "Simulation Engine",
        "Coalition Visualization",
        "ABM Visualization",
        "Evolutionary Winners",
        "DST Fusion",
        "Info-Gap",
        "Network Control",
        "Market Impact",
        "OSINT Monitor",
        "Scenario Persistence",
        "User Guide"
    ]
)

# --- Sidebar Explainer: Simulation Parameters & Results ---
st.sidebar.divider()
with st.sidebar.expander("🧭 Simulation Explainer — Parameters & Results", expanded=False):
    st.markdown(
        """
**What you configure (inputs)**

- **Players & Strategies**  
  Choose 3 or 4 strategies (DE, DT, ES, HY). Each player (US, Iran, Russia, China) plays a *mixed strategy* over these actions.

- **Payoff Matrices** *(per pair p vs q)*  
  Cells represent expected utility for the **row player** against the **column player**.  
  Higher values mean the row strategy benefits more versus the column strategy.

- **Information Asymmetry (C‑matrix)**  
  Captures how accurately a player **perceives** an opponent’s strategy mix.  
  *Diagonal high → good classification; off-diagonal high → confusion/misperception.*  
  The **Confidence** slider blends perceived mix with uniform noise.

- **Influence Weights (wₚ→q)**  
  How strongly player **p** takes **q** into account when updating.  
  Large values mean p is very sensitive to q’s behavior.

- **Temporal Coalitions α(t), β(t)**  
  - **α(t)**: *Cohesion bonus* added to payoffs for **intra‑coalition** pairs.  
  - **β(t)**: Multiplier on **intra‑coalition influence weights** (wₚ→q).  
  Both evolve linearly with time using initial values and slopes you set in the sidebar.

- **Dynamics & Utility**  
  - **Update rule**:  
    - *Smoothed best response*: stochastic choice via logit/softmax  
    - *Replicator*: strategies that outperform the average grow in share  
  - **η (learning rate)**: how much the mix moves each step  
  - **β (logit sensitivity)**: higher = more “rational” best‑response  
  - **γ (utility curvature)**: risk/curvature of utility; <1 → diminishing sensitivity  
  - **λ (loss aversion)**: losses weigh more than gains when >1

- **Simulation Controls**  
  - **T**: number of timesteps  
  - **Init mixes**: uniform or random  
  - **Noise**: payoff perturbation for stress‑testing  
  - **Monte Carlo**: run multiple perturbed simulations to see fixation/variance

---

**What you see (outputs)**

- **Time Series Plots**  
  Per player, the share of each strategy across time.

- **Final Strategy Distribution**  
  The last‑step mixed strategy for each player (a point estimate or MC mean).

- **Monte‑Carlo Bundles (if enabled)**  
  Shaded bands show uncertainty ranges (e.g., 5–95th percentiles) over time.

- **ABM Visualization**  
  Agent dots move, interact, and imitate; clusters indicate emergent dominance.

- **Evolutionary Winners**  
  Counts how often each strategy **fixates** (dominates) across MC runs;  
  shows **coalition dominance** (which bloc achieves higher final strengths).

- **Downstream Modules**  
  - **DST Fusion** → informs risk signals (used by Market Impact)  
  - **Info‑Gap** → robustness to uncertainty via α grid  
  - **Network Control** → centralities + **minimum driver nodes**  
  - **Market Impact** → proxy indicators (energy/industry/defense/trade)

---

**Practical tips**

- Start with **uniform** mixes and modest **η** (0.1–0.2).  
- Use the **Scenario Loader** to quickly stress‑test settings.  
- If the system looks “jittery,” lower **η** or **β**, or reduce **noise**.  
- Turn on **Monte Carlo** to understand stability and fixation tendencies.  
- Use **Reset Simulation State** to clear everything cleanly.
        """
    )


# -------------------------------------------------------------
# SIMULATION ENGINE (BEGIN)
# -------------------------------------------------------------
if page == "Simulation Engine":
    st.header("4‑Player CAS Simulation Engine")

    # Reset Button
    if st.button("🔄 Reset Simulation State"):
        for key in ["traj_to_plot","mc_bundle","payoff","C","W"]:
            if key in st.session_state:
                del st.session_state[key]
        st.success("Simulation state reset.")
        st.stop()

    players = ["US","Iran","Russia","China"]
    n_strats = st.selectbox("Number of strategies", [3,4], index=1)
    strategies = ["DE","DT","ES","HY"][:n_strats]
    S = n_strats

    # Payoff initialization
    if "payoff" not in st.session_state:
        base4 = np.array([[6,3,-4,1],[3,2,-2,0],[-5,-2,-8,-1],[1,-1,0,3]],float)
        base3 = np.array([[6,3,-4],[3,2,-2],[-5,-2,-8]],float)
        base = base4[:S,:S] if S==4 else base3
        st.session_state["payoff"] = {(p,q): base.copy() for p in players for q in players if p!=q}

    # Ensure shape if S changed
    for k in st.session_state["payoff"]:
        M = st.session_state["payoff"][k]
        if M.shape != (S,S):
            M2 = np.tile(M, ((S+M.shape[0]-1)//M.shape[0], (S+M.shape[1]-1)//M.shape[1]))
            st.session_state["payoff"][k] = M2[:S,:S]

    # Edit payoff matrix
    col1,col2 = st.columns(2)
    with col1:
        rowP = st.selectbox("Row player", players)
    with col2:
        colQ = st.selectbox("Column player", [x for x in players if x!=rowP])

    df_edit = st.data_editor(
        pd.DataFrame(st.session_state["payoff"][(rowP,colQ)], index=strategies, columns=strategies),
        key=f"payoff_editor_{rowP}_{colQ}"
    )
    st.session_state["payoff"][(rowP,colQ)] = df_edit.values.astype(float)

    # Information asymmetry
    st.subheader("Information Asymmetry")
    confidence = st.slider(
        "Belief confidence",
        0.0,1.0,
        st.session_state.get("confidence_override",0.8),
        0.05
    )
    
    
    # Information asymmetry matrices
    def default_C(n):
        if n == 1:
            return np.ones((1,1))
        return 0.9*np.eye(n) + 0.1*(np.ones((n,n))-np.eye(n))/(n-1)

    if "C" not in st.session_state:
        st.session_state["C"] = {p: default_C(S) for p in players}

    for p in players:
        with st.expander(f"C-matrix for {p}"):
            dfC = st.data_editor(
                pd.DataFrame(st.session_state["C"][p], index=strategies, columns=strategies),
                key=f"C_matrix_editor_{p}"
            )
            st.session_state["C"][p] = dfC.values.astype(float)

    # Influence weights
    st.subheader("Influence Weights")
    if "W" not in st.session_state:
        st.session_state["W"] = {(p,q):1.0 for p in players for q in players if p!=q}

    with st.expander("Edit w_pq"):
        for p in players:
            cols = st.columns(len(players)-1)
            idx = 0
            for q in players:
                if p != q:
                    st.session_state["W"][(p,q)] = cols[idx].slider(
                        f"{p}->{q}", 0.0,3.0, st.session_state["W"][(p,q)],0.1
                    )
                    idx += 1

    # Dynamics & utility
    st.subheader("Dynamics & Utility")
    method = st.selectbox("Update rule", ["smoothed_best_response","replicator"])
    eta = st.slider("Learning rate η", 0.01,0.5,0.15,0.01)
    beta_param = st.slider("Logit sensitivity β",0.5,10.0,3.0,0.5)

    gamma = st.slider("Utility curvature γ",0.25,1.5,
                      st.session_state.get("gamma_override",0.85),0.05)
    lam = st.slider("Loss aversion λ",1.0,3.0,
                    st.session_state.get("loss_aversion_override",1.5),0.1)

    dyn = DynamicParams(method=method, eta=eta, beta=beta_param)
    util = {p: UtilityParams(gamma=gamma, lambda_loss=lam) for p in players}

    # Simulation parameters
    st.subheader("Simulation Controls")
    T = st.number_input("Time steps T",10,5000,300,10)
    init_mode = st.selectbox("Initial mixes", ["uniform","random"])
    seed = st.number_input("Random seed",0,10**6,123)
    use_mc = st.checkbox("Enable Monte Carlo")
    runs = st.number_input("MC runs",1,500,30)
    noise = st.slider("Payoff noise std",0.0,5.0,0.3,0.1)

    coalitions = st.session_state["coalitions"]

    # ----------- Simulation Core -----------
    def simulate_once(payoff_dict, C_dict, W0, confidence, dyn, util,
                      T, init_mode, seed):
        rng = np.random.default_rng(seed)

        mixes = {
            p: (np.ones(S)/S if init_mode=="uniform" else normalized(rng.random(S)))
            for p in players
        }
        traj = {p: np.zeros((T,S)) for p in players}

        for t in range(int(T)):
            for p in players:
                traj[p][t] = mixes[p]

            alpha_t, beta_t = compute_alpha_beta(t)
            W_dynamic = adjust_weights_for_coalitions(W0, coalitions, beta_t)

            next_mixes = {}
            for p in players:
                EU_total = np.zeros(S)
                for q in players:
                    if p == q:
                        continue
                    Ppq = payoff_dict[(p,q)].copy()
                    for cname, members in coalitions.items():
                        if p in members and q in members:
                            Ppq = Ppq + alpha_t
                    perceived_q = apply_misperception(mixes[q], C_dict[p], confidence)
                    EU_pq = expected_utilities_matrix(Ppq, perceived_q, util[p])
                    EU_total += W_dynamic[(p,q)] * EU_pq

                next_mixes[p] = update_mix(mixes[p], EU_total, dyn)
            mixes = next_mixes

        return traj

    def simulate_mc(payoff_dict, C_dict, W0, confidence, dyn, util,
                    T, init_mode, seed, runs, noise):
        rng = np.random.default_rng(seed)
        all_traj = {p: [] for p in players}
        for r in range(runs):
            payoff_pert = {
                k: v + rng.normal(0, noise, v.shape)
                for k,v in payoff_dict.items()
            }
            traj = simulate_once(payoff_pert, C_dict, W0, confidence,
                                 dyn, util, T, init_mode, seed+31*r)
            for p in players:
                all_traj[p].append(traj[p])
        mean_traj = {
            p: np.mean(np.stack(all_traj[p],axis=0),axis=0)
            for p in players
        }
        return mean_traj, all_traj

    # ----------- Run Simulation Button -----------
    if st.button("Run Simulation"):
        payoff_dict = st.session_state["payoff"]
        C_dict = st.session_state["C"]
        W0 = st.session_state["W"]

        if use_mc:
            mean_traj, all_traj = simulate_mc(
                payoff_dict, C_dict, W0, confidence, dyn, util,
                T, init_mode, seed, runs, noise
            )
            st.session_state["traj_to_plot"] = mean_traj
            st.session_state["mc_bundle"] = all_traj
            st.success(f"Monte Carlo completed ({runs} runs).")
        else:
            traj = simulate_once(
                payoff_dict, C_dict, W0, confidence, dyn, util,
                T, init_mode, seed
            )
            st.session_state["traj_to_plot"] = traj
            st.session_state["mc_bundle"] = None
            st.success("Simulation complete.")

        # Plot time-series
        t = np.arange(int(T))
        traj_to_plot = st.session_state["traj_to_plot"]
        mc_bundle = st.session_state["mc_bundle"]

        for p in players:
            st.subheader(f"{p} Strategy Distribution Over Time")
            fig, ax = plt.subplots(figsize=(7,4))
            if mc_bundle is not None:
                mean = traj_to_plot[p]
                low = np.percentile(np.stack(mc_bundle[p],axis=0),5,axis=0)
                high = np.percentile(np.stack(mc_bundle[p],axis=0),95,axis=0)
                for i in range(S):
                    ax.plot(t, mean[:,i], label=strategies[i])
                    ax.fill_between(t, low[:,i], high[:,i], alpha=0.15)
            else:
                for i in range(S):
                    ax.plot(t, traj_to_plot[p][:,i], label=strategies[i])
            ax.set_ylim(0,1)
            ax.grid(True, alpha=0.3)
            ax.legend()
            st.pyplot(fig)

        # Final distribution
        final_df = pd.DataFrame(
            {p: traj_to_plot[p][-1] for p in players},
            index=strategies
        )
        st.subheader("Final Strategy Distribution")
        st.dataframe(final_df.style.format("{:.3f}"))

# -------------------------------------------------------------
# COALITION VISUALIZATION
# -------------------------------------------------------------
elif page == "Coalition Visualization":
    st.header("Coalition Visualization (Graph-Based)")
    players = ["US","Iran","Russia","China"]
    coalitions = st.session_state.get("coalitions",{})

    if not PYVIS_AVAILABLE:
        st.error("PyVis not installed. Run 'pip install pyvis'.")
    else:
        net = Network(height="500px", width="100%", bgcolor="#222", font_color="white")
        palette = ["#1abc9c","#3498db","#e74c3c","#9b59b6","#f1c40f"]
        color_map = {}
        idx = 0
        for cname,members in coalitions.items():
            for m in members:
                color_map[m] = palette[idx % len(palette)]
            idx += 1
        for p in players:
            net.add_node(p, label=p, color=color_map.get(p,"#95a5a6"))
        if "W" in st.session_state:
            W0 = st.session_state["W"]
            _, beta_init = compute_alpha_beta(0)
            W_adj = adjust_weights_for_coalitions(W0, coalitions, beta_init)
            for (a,b), w in W_adj.items():
                if a!=b:
                    net.add_edge(a,b, value=w, title=f"{a}->{b}: {w:.2f}")
        html = net.generate_html(notebook=False)
        st.components.v1.html(html, height=530)

# -------------------------------------------------------------
# ABM VISUALIZATION
# -------------------------------------------------------------
elif page == "ABM Visualization":
    st.header("Agent‑Based Model (ABM) Visualization")

    if "traj_to_plot" not in st.session_state:
        st.warning("Run a simulation first.")
    else:
        traj = st.session_state["traj_to_plot"]
        base_mix = traj["US"][-1]
        S = len(base_mix)
        strategies = ["DE","DT","ES","HY"][:S]
        colors = {"DE":"#e74c3c","DT":"#3498db","ES":"#2ecc71","HY":"#f1c40f"}

        N_agents = st.slider("Agents",20,500,120,10)
        speed = st.slider("Agent speed", 0.05,4.0,1.0,0.05)
        radius = st.slider("Interaction radius",0.01,0.5,0.08)
        steps = st.slider("Steps",10,300,80)

        rng = np.random.default_rng(2026)
        agents = [{
            "x":rng.random(),
            "y":rng.random(),
            "s":np.random.choice(strategies,p=base_mix)
        } for _ in range(N_agents)]

        def abm_step():
            for a in agents:
                a["x"] += (rng.random()-0.5)*speed*0.01
                a["y"] += (rng.random()-0.5)*speed*0.01
                a["x"] = min(max(a["x"],0),1)
                a["y"] = min(max(a["y"],0),1)
                for b in agents:
                    dx=a["x"]-b["x"]; dy=a["y"]-b["y"]
                    if dx*dx+dy*dy < radius*radius:
                        if rng.random()<0.05:
                            a["s"] = b["s"]

        placeholder = st.empty()
        if st.button("Run ABM"):
            for t in range(steps):
                abm_step()
                fig,ax = plt.subplots(figsize=(6,5))
                for s in strategies:
                    xs=[ag["x"] for ag in agents if ag["s"]==s]
                    ys=[ag["y"] for ag in agents if ag["s"]==s]
                    ax.scatter(xs,ys,c=colors[s],s=20,label=s)
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_xlim(0,1); ax.set_ylim(0,1)
                ax.set_title(f"ABM t={t}")
                ax.legend()
                placeholder.pyplot(fig)
                plt.close(fig)
                time.sleep(0.03)
                
                
     
 # -------------------------------------------------------------
# NETWORK CONTROL
# -------------------------------------------------------------
elif page == "Network Control":
    st.header("Network Controllability & Centrality")

    default_nodes = [
        "US","Iran","Russia","China","Hezbollah","Houthis","Iraqi_Militias",
        "Insurers","Reinsurers","Shippers","Escorts","BMD",
        "OPECplus","Qatar_LNG","EU","Asia","Maritime_Trade"
    ]

    nodes_txt = st.text_area("Nodes (comma-separated)", value=",".join(default_nodes))
    edges_txt = st.text_area("Edges (u->v per line)", value="\n".join([
        "Iran->Hezbollah","Iran->Houthis","Iran->Iraqi_Militias",
        "US->BMD","US->Escorts","US->Insurers","US->Reinsurers",
        "Russia->Iran","Russia->OPECplus",
        "China->Insurers","China->Shippers",
        "Insurers->Maritime_Trade","Shippers->Maritime_Trade",
        "Escorts->Maritime_Trade",
        "Qatar_LNG->Maritime_Trade","OPECplus->Maritime_Trade",
        "EU->Shippers","Asia->Shippers"
    ]))

    nodes = [n.strip() for n in nodes_txt.split(",") if n.strip()]
    edges = []
    for line in edges_txt.splitlines():
        if "->" in line:
            u,v = [x.strip() for x in line.split("->",1)]
            edges.append((u,v))

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # centralities
    pr = nx.pagerank(G, alpha=0.85)
    btw = nx.betweenness_centrality(G, normalized=True)
    deg_in = dict(G.in_degree())
    deg_out = dict(G.out_degree())

    # minimum driver nodes
    def minimum_driver_nodes(G: nx.DiGraph):
        B = nx.Graph()
        L = {f"{u}_out" for u in G.nodes()}
        R = {f"{v}_in" for v in G.nodes()}

        B.add_nodes_from(L, bipartite=0)
        B.add_nodes_from(R, bipartite=1)

        for u,v in G.edges():
            B.add_edge(f"{u}_out", f"{v}_in")

        match = nx.algorithms.bipartite.maximum_matching(B, top_nodes=L)
        matched_R = {b for a,b in match.items() if b in R}
        unmatched_R = [r for r in R if r not in matched_R]
        return [r[:-3] for r in unmatched_R]

    drivers = minimum_driver_nodes(G)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("PageRank")
        df_pr = pd.DataFrame(sorted(pr.items(), key=lambda x:-x[1]), columns=["node","pagerank"])
        st.dataframe(df_pr.style.format({"pagerank": "{:.4f}"}))
    with col2:
        st.subheader("Betweenness")
        df_bw = pd.DataFrame(sorted(btw.items(), key=lambda x:-x[1]), columns=["node","betweenness"])
        st.dataframe(df_bw.style.format({"betweenness":"{:.4f}"}))

    st.subheader("Minimum Driver Nodes")
    st.write(drivers)

# -------------------------------------------------------------
# MARKET IMPACT
# -------------------------------------------------------------
elif page == "Market Impact":
    st.header("Market Impact / Sector Stress — Proxy Model")

    # Pull DST fused evidence if exists
    fused = st.session_state.get("dst_fused", None)

    def proxy_energy_price(fused_dict):
        if fused_dict is None:
            return 70.0
        sustained = fused_dict.get(frozenset({"sustained"}), 0.0)
        managed   = fused_dict.get(frozenset({"managed"}), 0.0)
        rapid     = fused_dict.get(frozenset({"rapid"}), 0.0)
        return 70 + 35*sustained + 15*managed - 10*rapid

    oil_price = proxy_energy_price(fused)
    st.metric("Projected Oil Price (proxy)", f"{oil_price:.1f} USD")

    industry_stress = min(3.0, max(0.0, (oil_price/100)*2.0))
    st.metric("Industry Stress Index (0..3)", f"{industry_stress:.2f}")

    defense_pressure = industry_stress/2 + (oil_price/100)
    st.metric("Defense Pressure Score", f"{defense_pressure:.2f}")

    launch_mult = 1.0 + oil_price/200
    st.metric("Space Launch Cost Multiplier", f"{launch_mult:.2f}")

    trade_risk = min(1.0, max(0.0, 0.3 + industry_stress*0.2))
    st.metric("Trade War Probability (proxy)", f"{trade_risk:.2f}")

# -------------------------------------------------------------
# OSINT MONITOR
# -------------------------------------------------------------
elif page == "OSINT Monitor":
    st.header("OSINT Monitor — Mock Evidence Injection")

    FRAME = frozenset({"sustained","managed","rapid"})

    def parse_mass_local(obj):
        if isinstance(obj,str):
            data = json.loads(obj)
        else:
            data = obj
        m={}
        for k,v in data.items():
            key = str(k).strip().strip("{}")
            if key.lower()=="theta":
                A = FRAME
            else:
                if key=="":
                    A = FRAME
                else:
                    A = frozenset(x.strip() for x in key.split(",") if x.strip())
            m[A] = float(v)
        tot = sum(m.values())
        return {A:v/tot for A,v in m.items()} if tot>0 else {FRAME:1.0}

    def combine(m1,m2):
        K=0; m={}
        for A,vA in m1.items():
            for B,vB in m2.items():
                inter = A & B
                if len(inter)==0:
                    K += vA*vB
                else:
                    m[inter] = m.get(inter,0) + vA*vB
        if K>=1:
            return {FRAME:1.0}
        s=1/(1-K)
        for A in m:
            m[A] *= s
        tot = sum(m.values())
        return {A:v/tot for A,v in m.items()} if tot>0 else {FRAME:1.0}

    if "dst_fused" not in st.session_state:
        st.session_state["dst_fused"] = {FRAME:1.0}

    MOCK_ITEMS = [
        {"label":"Insurance withdrawal", "mass":{"{sustained}":0.45,"Theta":0.55}},
        {"label":"Convoy escalation",    "mass":{"{managed}":0.40,"Theta":0.60}},
        {"label":"LNG slowdown",         "mass":{"{sustained}":0.50,"Theta":0.50}},
    ]

    cols = st.columns(len(MOCK_ITEMS))
    for i,item in enumerate(MOCK_ITEMS):
        with cols[i]:
            st.write(f"**{item['label']}**")
            if st.button(f"Inject {i+1}"):
                st.session_state["dst_fused"] = combine(
                    st.session_state["dst_fused"],
                    parse_mass_local(item["mass"])
                )
                st.success("Evidence fused.")

    st.subheader("Inject LNG Shock")
    shock = {"{sustained}":0.55,"{managed,rapid}":0.30,"Theta":0.15}
    if st.button("Fuse LNG Shock"):
        st.session_state["dst_fused"] = combine(
            st.session_state["dst_fused"],
            parse_mass_local(shock)
        )
        st.success("LNG event fused.")

    # Display fused result
    fused = st.session_state["dst_fused"]
    Bel,Pl = {},{}
    for A in [frozenset({"sustained"}),frozenset({"managed"}),frozenset({"rapid"}),FRAME]:
        Bel[A] = sum(v for B,v in fused.items() if B.issubset(A))
        Pl[A]  = sum(v for B,v in fused.items() if len(B&A)>0)

    def fmt(A):
        return "{" + ",".join(sorted(A)) + "}" if A!=FRAME else "Theta"

    df = pd.DataFrame({
        "Set":[fmt(A) for A in Bel],
        "Belief":[Bel[A] for A in Bel],
        "Plausibility":[Pl[A] for A in Bel]
    })
    st.dataframe(df.style.format({"Belief":"{:.3f}","Plausibility":"{:.3f}"}))

# -------------------------------------------------------------
# SCENARIO PERSISTENCE
# -------------------------------------------------------------
elif page == "Scenario Persistence":
    st.header("Scenario Persistence")
    scen_id = st.text_input("Scenario ID")
    scen_raw = st.text_area("Scenario JSON")

    col1,col2 = st.columns(2)
    with col1:
        if st.button("Save Scenario"):
            try:
                obj = json.loads(scen_raw)
                pm.save(scen_id, obj)
                st.success("Saved.")
            except Exception as e:
                st.error(str(e))
    with col2:
        if st.button("Load Scenario"):
            obj = pm.load(scen_id)
            if obj: st.json(obj)
            else: st.warning("Not found")

# -------------------------------------------------------------
# USER GUIDE
# -------------------------------------------------------------
elif page == "User Guide":
    st.header("📘 User Guide — Unified CAS Dashboard")

    st.markdown("""
    ## Overview
    This dashboard integrates:
    - 4-player evolutionary-game simulation engine  
    - Temporal coalition dynamics α(t), β(t)  
    - Agent-based modeling (ABM)  
    - Evolutionary winners analysis  
    - DST evidence fusion  
    - Info-Gap robustness  
    - Network controllability  
    - Market-impact proxies  
    - OSINT event injection  
    - Scenario persistence

    ## 1) Run a Simulation
    - Open **Simulation Engine**
    - Configure payoffs, C-matrices, and w_pq influence weights
    - Adjust α(t), β(t) in the sidebar
    - Click **Run Simulation**
    - Results are stored for other modules (ABM, Winners, Market Impact)

    ## 2) Scenarios
    - Use sidebar **Scenario Loader** to preload parameters
    - Applies payoff shifts, confidence, loss aversion
    - Updates coalition presets

    ## 3) ABM Visualization
    - Uses the final strategy distribution
    - Click **Run ABM** to animate diffusion & imitation

    ## 4) Evolutionary Winners
    - Requires Monte Carlo runs
    - Displays fixation counts and coalition dominance

    ## 5) DST Fusion
    - Fuse evidence from sliders or JSON
    - View Belief & Plausibility per focal set

    ## 6) Info-Gap Robustness
    - Explore robustness across α-grid
    - Identify acceptable regions vs KPI thresholds

    ## 7) Network Control
    - Build directed graph
    - Inspect PageRank, betweenness, driver nodes

    ## 8) Market Impact
    - Proxy indicators derived from fused DST mass

    ## 9) OSINT Monitor
    - Inject mock events and fuse LNG shock
    - Optionally perturb CAS payoffs

    ## 10) Persistence
    - Save or Load scenario JSON
    """)

# END OF FILE
               
               