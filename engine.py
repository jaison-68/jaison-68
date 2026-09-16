"""
engine.py — all the maths for the SCMS delivery app.
No Streamlit imports here on purpose, so you can unit-test or reuse it.
"""

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# 1. DATA
# ----------------------------------------------------------------------

STAGES = {
    "S1_quote_to_po":        "S1  Quote -> PO issued",
    "S2_po_to_plan":         "S2  PO -> scheduled delivery",
    "S3_plan_to_actual":     "S3  Schedule slip at delivery",
    "S4_actual_to_recorded": "S4  Delivery -> recorded",
}

# the date column that marks a shipment ENTERING each stage (needed for arrival rate)
STAGE_ENTRY_DATE = {
    "S1_quote_to_po":        "PQ First Sent to Client Date_dt",
    "S2_po_to_plan":         "PO Sent to Vendor Date_dt",
    "S3_plan_to_actual":     "Scheduled Delivery Date_dt",
    "S4_actual_to_recorded": "Delivered to Client Date_dt",
}

DATE_COLS = list(STAGE_ENTRY_DATE.values()) + ["Delivery Recorded Date_dt"]


def load_shipments(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in DATE_COLS:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    # S3 is a signed deviation (negative = early). Split it into the two ideas.
    df["S3_delay_only"] = df["S3_plan_to_actual"].clip(lower=0)
    df["S3_early_only"] = (-df["S3_plan_to_actual"]).clip(lower=0)
    return df


def apply_filters(df, years=None, channels=None, modes=None,
                  countries=None, vendors=None):
    out = df
    if years:
        out = out[out["year"].between(years[0], years[1])]
    for col, sel in [("channel", channels), ("Shipment Mode", modes),
                     ("Country", countries), ("Vendor", vendors)]:
        if sel:
            out = out[out[col].isin(sel)]
    return out


def stage_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Mean / median / sd / CV / n for every stage, ignoring missing values."""
    rows = []
    for col, label in STAGES.items():
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if col == "S3_plan_to_actual":           # duration cannot be negative
            s = s.clip(lower=0)
        mean = s.mean() if len(s) else np.nan
        sd = s.std(ddof=1) if len(s) > 1 else np.nan
        rows.append({
            "stage": col,
            "label": label,
            "n": int(len(s)),
            "mean_days": mean,
            "median_days": s.median() if len(s) else np.nan,
            "p90_days": s.quantile(0.90) if len(s) else np.nan,
            "sd_days": sd,
            "cv": (sd / mean) if (mean and mean > 0) else np.nan,
            "coverage_pct": 100 * len(s) / len(df) if len(df) else 0.0,
        })
    return pd.DataFrame(rows)


def arrival_rate(df: pd.DataFrame, stage: str) -> tuple:
    """lambda = shipments entering the stage per calendar day, over the observed span."""
    col = STAGE_ENTRY_DATE[stage]
    d = df[col].dropna()
    if len(d) < 2:
        return np.nan, 0, np.nan
    span = (d.max() - d.min()).days
    span = max(span, 1)
    return len(d) / span, len(d), span


# ----------------------------------------------------------------------
# 2. CPM  (Critical Path Method)
# ----------------------------------------------------------------------

def build_network(dur, split):
    """
    Turn the 4 measured stage durations into a 7-activity project network
    that has genuine parallel paths (CPM on a pure chain is just a sum).

    dur   : {'S1':.., 'S2':.., 'S3':.., 'S4':..} in days
    split : {'mfg':.., 'qa':.., 'book':..} fractions of S2.
            Transit = S2 * (1 - mfg - qa).  'book' runs in PARALLEL with mfg+qa.
    """
    s2 = dur["S2"]
    mfg = s2 * split["mfg"]
    qa = s2 * split["qa"]
    book = s2 * split["book"]
    transit = max(s2 * (1 - split["mfg"] - split["qa"]), 0.0)

    return [
        dict(id="A", name="Quote approval (PQ -> PO)",       dur=dur["S1"], preds=[]),
        dict(id="B", name="Vendor manufacturing",            dur=mfg,       preds=["A"]),
        dict(id="C", name="Freight booking / carrier slot",  dur=book,      preds=["A"]),
        dict(id="D", name="QA release & packing",            dur=qa,        preds=["B"]),
        dict(id="E", name="Transit to destination",          dur=transit,   preds=["C", "D"]),
        dict(id="F", name="Clearance / last-mile slip",      dur=dur["S3"], preds=["E"]),
        dict(id="G", name="POD & delivery recording",        dur=dur["S4"], preds=["F"]),
    ]


def cpm(activities):
    """Forward pass, backward pass, float, critical path. Returns (DataFrame, duration, path)."""
    acts = {a["id"]: dict(a) for a in activities}

    # --- topological order (Kahn) -------------------------------------
    succ = {i: [] for i in acts}
    indeg = {i: 0 for i in acts}
    for i, a in acts.items():
        for p in a["preds"]:
            if p not in acts:
                raise ValueError(f"activity {i} has unknown predecessor {p}")
            succ[p].append(i)
            indeg[i] += 1
    queue = [i for i in acts if indeg[i] == 0]
    order = []
    while queue:
        i = queue.pop(0)
        order.append(i)
        for j in succ[i]:
            indeg[j] -= 1
            if indeg[j] == 0:
                queue.append(j)
    if len(order) != len(acts):
        raise ValueError("network has a cycle")

    # --- forward pass: ES / EF ----------------------------------------
    for i in order:
        a = acts[i]
        a["ES"] = max([acts[p]["EF"] for p in a["preds"]], default=0.0)
        a["EF"] = a["ES"] + a["dur"]
    project = max(a["EF"] for a in acts.values())

    # --- backward pass: LF / LS ---------------------------------------
    for i in reversed(order):
        a = acts[i]
        a["LF"] = min([acts[s]["LS"] for s in succ[i]], default=project)
        a["LS"] = a["LF"] - a["dur"]
        a["TF"] = a["LS"] - a["ES"]                       # total float
    for i in order:                                        # free float
        a = acts[i]
        nxt = [acts[s]["ES"] for s in succ[i]]
        a["FF"] = (min(nxt) - a["EF"]) if nxt else (project - a["EF"])

    eps = 1e-9
    tbl = pd.DataFrame([
        {"ID": i, "Activity": acts[i]["name"],
         "Predecessors": ",".join(acts[i]["preds"]) or "-",
         "Duration": round(acts[i]["dur"], 2),
         "ES": round(acts[i]["ES"], 2), "EF": round(acts[i]["EF"], 2),
         "LS": round(acts[i]["LS"], 2), "LF": round(acts[i]["LF"], 2),
         "Total float": round(acts[i]["TF"], 2),
         "Free float": round(acts[i]["FF"], 2),
         "Critical": acts[i]["TF"] <= eps}
        for i in order
    ])
    path = [i for i in order if acts[i]["TF"] <= eps]
    return tbl, project, path


# ----------------------------------------------------------------------
# 3. QUEUEING THEORY
# ----------------------------------------------------------------------

def erlang_b(c: int, a: float) -> float:
    """Numerically stable recursion. a = offered load = lambda/mu."""
    b = 1.0
    for n in range(1, int(c) + 1):
        b = (a * b) / (n + a * b)
    return b


def erlang_c(c: int, a: float) -> float:
    """P(wait) for M/M/c. Derived from Erlang B so it stays stable at large c."""
    rho = a / c
    if rho >= 1:
        return 1.0
    b = erlang_b(c, a)
    return b / (1 - rho * (1 - b))


def mmc_metrics(lam: float, mu: float, c: int):
    """Classic M/M/c. Times come back in the same unit as 1/mu (days)."""
    if not np.isfinite(lam) or not np.isfinite(mu) or mu <= 0 or c < 1:
        return {k: np.nan for k in
                ["a", "rho", "Pw", "Wq", "Lq", "W", "L", "stable"]}
    a = lam / mu
    rho = a / c
    if rho >= 1:
        return dict(a=a, rho=rho, Pw=1.0, Wq=np.inf, Lq=np.inf,
                    W=np.inf, L=np.inf, stable=False)
    pw = erlang_c(c, a)
    wq = pw / (c * mu - lam)
    return dict(a=a, rho=rho, Pw=pw, Wq=wq, Lq=lam * wq,
                W=wq + 1 / mu, L=lam * (wq + 1 / mu), stable=True)


def kingman_wq(wq_mmc: float, ca2: float, cs2: float) -> float:
    """G/G/c approximation: scale the M/M/c wait by the variability ratio."""
    if not np.isfinite(wq_mmc):
        return np.inf
    return wq_mmc * (ca2 + cs2) / 2.0


def min_servers(lam: float, mu: float, target_rho: float = 0.95) -> int:
    """Smallest c that keeps utilisation under target_rho."""
    if not np.isfinite(lam) or not np.isfinite(mu) or mu <= 0:
        return 1
    return max(1, int(np.ceil(lam / (mu * target_rho))))


def queue_table(df: pd.DataFrame, servers: dict, ca2: float = 1.0):
    """One row per stage: lambda, mu, rho, Wq (M/M/c and G/G/c)."""
    stats = stage_stats(df).set_index("stage")
    rows = []
    for stage, label in STAGES.items():
        lam, n_arr, span = arrival_rate(df, stage)
        mean = stats.loc[stage, "mean_days"]
        cv = stats.loc[stage, "cv"]
        mu = 1.0 / mean if (mean and mean > 0) else np.nan
        c = int(servers.get(stage, min_servers(lam, mu)))
        m = mmc_metrics(lam, mu, c)
        cs2 = cv ** 2 if np.isfinite(cv) else 1.0
        rows.append({
            "Stage": label,
            "lambda (ship/day)": round(lam, 3) if np.isfinite(lam) else np.nan,
            "Mean service (days)": round(mean, 2) if np.isfinite(mean) else np.nan,
            "mu (ship/day/server)": round(mu, 4) if np.isfinite(mu) else np.nan,
            "Servers c": c,
            "Utilisation rho": round(m["rho"], 3) if np.isfinite(m["rho"]) else np.nan,
            "P(wait)": round(m["Pw"], 3) if np.isfinite(m["Pw"]) else np.nan,
            "Wq M/M/c (days)": round(m["Wq"], 2) if np.isfinite(m["Wq"]) else np.inf,
            "Wq G/G/c (days)": round(kingman_wq(m["Wq"], ca2, cs2), 2),
            "Lq (shipments)": round(m["Lq"], 1) if np.isfinite(m["Lq"]) else np.inf,
            "W total (days)": round(m["W"], 2) if np.isfinite(m["W"]) else np.inf,
            "Service CV": round(cv, 2) if np.isfinite(cv) else np.nan,
            "Stable": m["stable"],
            "_stage": stage,
        })
    return pd.DataFrame(rows)
