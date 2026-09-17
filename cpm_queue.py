"""
cpm_queue.py — analytical engine for the CPM + queueing extension to
24PME2072 "End-to-End Supply Chain Performance" (Jaison A, DM25030).

Deliberately contains no Streamlit imports so it can be unit-tested, imported
by app.py, or run headless to write results/t13 and t14 for the appendix.

CONVENTIONS INHERITED FROM src/03_stages.py — do not change these silently:
  * the stage profile is computed on the FULLY OBSERVED CHAIN,
    i.e. dropna(subset=[S1,S2,S3,S4]) -> n = 2,857, all Direct Drop.
  * S3 is a signed deviation (negative = early). It is NOT clipped when
    reproducing Table 4.1. It IS clipped to >= 0 when used as a CPM activity
    duration, because an activity cannot take negative time. Both are shown.
"""

import json
import numpy as np
import pandas as pd

STAGES = ["S1_quote_to_po", "S2_po_to_plan", "S3_plan_to_actual", "S4_actual_to_recorded"]

LABEL = {
    "S1_quote_to_po":        "S1 Quotation to PO release",
    "S2_po_to_plan":         "S2 PO release to scheduled delivery",
    "S3_plan_to_actual":     "S3 Schedule adherence",
    "S4_actual_to_recorded": "S4 Delivery to record closure",
}

# date column marking entry into each stage — needed for the arrival rate
ENTRY_DATE = {
    "S1_quote_to_po":        "PQ First Sent to Client Date_dt",
    "S2_po_to_plan":         "PO Sent to Vendor Date_dt",
    "S3_plan_to_actual":     "Scheduled Delivery Date_dt",
    "S4_actual_to_recorded": "Delivered to Client Date_dt",
}

DATE_COLS = list(ENTRY_DATE.values()) + ["Delivery Recorded Date_dt"]

# Canonical published values (report Table 4.1 / results/stage_results.json).
# The app displays these as fixed reference figures and checks itself against
# them. Never overwrite them with a recomputation.
PUBLISHED = {
    "n_full_chain": 2857,
    "S1_quote_to_po": 35.77, "S2_po_to_plan": 117.67,
    "S3_plan_to_actual": 1.69, "S4_actual_to_recorded": 2.04,
    "E2E": 157.17,
    "kpi_watches_mean_pct": 1.08, "kpi_watches_var_pct": 1.78,
    "mean_share": {"S1_quote_to_po": 22.76, "S2_po_to_plan": 74.87,
                   "S3_plan_to_actual": 1.08, "S4_actual_to_recorded": 1.30},
    "var_share": {"S1_quote_to_po": 26.65, "S2_po_to_plan": 70.80,
                  "S3_plan_to_actual": 1.78, "S4_actual_to_recorded": 0.76},
    "sd_dev_direct": 18.06, "sd_dev_rdc": 25.58,
    "abs_dev_direct": 4.09, "abs_dev_rdc": 14.52,
    "rdc_daily_flow_usd": 297362.0, "z95": 1.6449,
    "excess_safety_stock_usd": 3678263.0,
}

DIRECT = "Direct Drop (order-driven)"
RDC = "RDC (stock-held)"


# ======================================================================
# 1. DATA
# ======================================================================

def load(path="shipments.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in DATE_COLS:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    # PQ -> scheduled date. For Direct Drop this equals S1 + S2 exactly; for
    # RDC it is the only visible upstream block, because the PO date is never
    # recorded (po_absent_structural is True for every RDC shipment).
    df["pq_to_sched"] = (df["Scheduled Delivery Date_dt"]
                         - df["PQ First Sent to Client Date_dt"]).dt.days
    df["S3_delay"] = df["S3_plan_to_actual"].clip(lower=0)   # CPM duration
    df["S3_early"] = (-df["S3_plan_to_actual"]).clip(lower=0)
    return df


def full_chain(df: pd.DataFrame) -> pd.DataFrame:
    """The n = 2,857 basis used for Table 4.1."""
    return df.dropna(subset=STAGES).copy()


def apply_filters(df, years=None, modes=None, countries=None,
                  vendors=None, groups=None):
    out = df
    if years:
        out = out[out["year"].between(years[0], years[1])]
    for col, sel in [("Shipment Mode", modes), ("Country", countries),
                     ("Vendor", vendors), ("Product Group", groups)]:
        if sel:
            out = out[out[col].isin(sel)]
    return out


def stage_profile(chain: pd.DataFrame) -> pd.DataFrame:
    """Reproduces src/03_stages.py Table 4.1 on whatever subset is passed."""
    c = chain.copy()
    c["T"] = c[STAGES].sum(axis=1)
    rows = []
    for x in STAGES:
        v = c[x]
        rows.append({
            "stage": x, "Stage": LABEL[x], "n": int(len(v)),
            "Mean": v.mean(), "SD": v.std(), "P50": v.median(),
            "P95": v.quantile(.95),
            "Mean share %": 100 * v.mean() / c["T"].mean() if c["T"].mean() else np.nan,
            "Variance share %": 100 * np.cov(v, c["T"])[0, 1] / c["T"].var()
                                if len(c) > 1 and c["T"].var() else np.nan,
            "CV": v.std() / v.mean() if v.mean() not in (0, np.nan) else np.nan,
        })
    rows.append({"stage": "E2E", "Stage": "End-to-end", "n": int(len(c)),
                 "Mean": c["T"].mean(), "SD": c["T"].std(), "P50": c["T"].median(),
                 "P95": c["T"].quantile(.95), "Mean share %": 100.0,
                 "Variance share %": 100.0,
                 "CV": c["T"].std() / c["T"].mean() if c["T"].mean() else np.nan})
    return pd.DataFrame(rows)


def reproduction_check(df: pd.DataFrame) -> pd.DataFrame:
    """Confirms the loaded file still yields the published Table 4.1 figures."""
    p = stage_profile(full_chain(df)).set_index("stage")
    rows = [{"Quantity": "n (full chain)", "Published": PUBLISHED["n_full_chain"],
             "Recomputed": int(p.loc["S1_quote_to_po", "n"])}]
    for s in STAGES:
        rows.append({"Quantity": f"{LABEL[s]} mean", "Published": PUBLISHED[s],
                     "Recomputed": round(p.loc[s, "Mean"], 2)})
    rows.append({"Quantity": "End-to-end mean", "Published": PUBLISHED["E2E"],
                 "Recomputed": round(p.loc["E2E", "Mean"], 2)})
    out = pd.DataFrame(rows)
    out["Match"] = np.isclose(out["Published"].astype(float),
                              out["Recomputed"].astype(float), atol=0.02)
    return out


# ======================================================================
# 2. CPM
# ======================================================================

def direct_network(dur: dict, split: dict):
    """
    Direct Drop chain, expanded to 7 activities so the network has genuine
    parallel paths. CPM on a four-link chain is only a sum; the parallelism is
    where float, and therefore the method, comes from.

    dur   : {'S1','S2','S3','S4'} in days (S3 already clipped at 0)
    split : fractions of S2 -> {'mfg','qa','book'}; 'book' runs in parallel
            with mfg + qa, transit is the remainder of S2.
    """
    s2 = dur["S2"]
    mfg, qa, book = s2 * split["mfg"], s2 * split["qa"], s2 * split["book"]
    transit = max(s2 * (1 - split["mfg"] - split["qa"]), 0.0)
    return [
        dict(id="A", name="Quotation to PO release (S1)",      dur=dur["S1"], preds=[],         stage="S1_quote_to_po"),
        dict(id="B", name="Vendor manufacturing (S2a)",        dur=mfg,       preds=["A"],      stage="S2_po_to_plan"),
        dict(id="C", name="Freight booking / slot (S2b, ∥)",   dur=book,      preds=["A"],      stage="S2_po_to_plan"),
        dict(id="D", name="QA release and packing (S2c)",      dur=qa,        preds=["B"],      stage="S2_po_to_plan"),
        dict(id="E", name="Transit to consignee (S2d)",        dur=transit,   preds=["C", "D"], stage="S2_po_to_plan"),
        dict(id="F", name="Schedule slip at delivery (S3)",    dur=dur["S3"], preds=["E"],      stage="S3_plan_to_actual"),
        dict(id="G", name="Delivery to record closure (S4)",   dur=dur["S4"], preds=["F"],      stage="S4_actual_to_recorded"),
    ]


def rdc_network(dur: dict):
    """
    Stock-held chain. The PO date is never recorded, so S1 and S2 cannot be
    separated: everything from quotation to the scheduled date is one opaque
    block. There is nothing to run in parallel and therefore no float — which
    is the finding, not a modelling shortcut.
    """
    return [
        dict(id="R", name="Quotation to scheduled date (PO date not recorded)",
             dur=dur["R1"], preds=[], stage="pq_to_sched"),
        dict(id="F", name="Schedule slip at delivery (S3)", dur=dur["S3"], preds=["R"],
             stage="S3_plan_to_actual"),
        dict(id="G", name="Delivery to record closure (S4)", dur=dur["S4"], preds=["F"],
             stage="S4_actual_to_recorded"),
    ]


def cpm(activities):
    """Forward pass, backward pass, total and free float, critical path."""
    acts = {a["id"]: dict(a) for a in activities}
    succ = {i: [] for i in acts}
    indeg = {i: 0 for i in acts}
    for i, a in acts.items():
        for p in a["preds"]:
            if p not in acts:
                raise ValueError(f"{i} has unknown predecessor {p}")
            succ[p].append(i)
            indeg[i] += 1

    q = [i for i in acts if indeg[i] == 0]
    order = []
    while q:
        i = q.pop(0)
        order.append(i)
        for j in succ[i]:
            indeg[j] -= 1
            if indeg[j] == 0:
                q.append(j)
    if len(order) != len(acts):
        raise ValueError("cycle in network")

    for i in order:                                   # forward
        a = acts[i]
        a["ES"] = max([acts[p]["EF"] for p in a["preds"]], default=0.0)
        a["EF"] = a["ES"] + a["dur"]
    project = max(a["EF"] for a in acts.values())

    for i in reversed(order):                         # backward
        a = acts[i]
        a["LF"] = min([acts[s]["LS"] for s in succ[i]], default=project)
        a["LS"] = a["LF"] - a["dur"]
        a["TF"] = a["LS"] - a["ES"]
    for i in order:                                   # free float
        nxt = [acts[s]["ES"] for s in succ[i]]
        acts[i]["FF"] = (min(nxt) - acts[i]["EF"]) if nxt else (project - acts[i]["EF"])

    eps = 1e-9
    tbl = pd.DataFrame([{
        "ID": i, "Activity": acts[i]["name"],
        "Pred": ",".join(acts[i]["preds"]) or "—",
        "Duration": round(acts[i]["dur"], 2),
        "ES": round(acts[i]["ES"], 2), "EF": round(acts[i]["EF"], 2),
        "LS": round(acts[i]["LS"], 2), "LF": round(acts[i]["LF"], 2),
        "Total float": round(acts[i]["TF"], 2),
        "Free float": round(acts[i]["FF"], 2),
        "Critical": bool(acts[i]["TF"] <= eps),
        "stage": acts[i].get("stage", ""),
    } for i in order])
    return tbl, project, [i for i in order if acts[i]["TF"] <= eps]


def kpi_visibility(tbl: pd.DataFrame, project: float) -> dict:
    """
    How much of the critical path the reported on-time metric can see.
    The KPI inspects S3 alone; this is the CPM restatement of the report's
    1.08% of elapsed time.
    """
    s3 = tbl.loc[tbl["stage"] == "S3_plan_to_actual", "Duration"].sum()
    crit = tbl.loc[tbl["Critical"], "Duration"].sum()
    return {"s3_days": s3, "project_days": project,
            "pct_of_project": 100 * s3 / project if project else np.nan,
            "critical_days": crit,
            "unobserved_days": project - s3}


# ======================================================================
# 3. QUEUEING
# ======================================================================

def erlang_b(c: int, a: float) -> float:
    """Stable recursion; a = offered load = lambda/mu."""
    b = 1.0
    for n in range(1, int(c) + 1):
        b = (a * b) / (n + a * b)
    return b


def erlang_c(c: int, a: float) -> float:
    """P(wait) for M/M/c, derived from Erlang B so it holds at large c."""
    rho = a / c
    if rho >= 1:
        return 1.0
    b = erlang_b(c, a)
    return b / (1 - rho * (1 - b))


def mmc(lam: float, mu: float, c: int) -> dict:
    if not (np.isfinite(lam) and np.isfinite(mu)) or mu <= 0 or c < 1:
        return dict(a=np.nan, rho=np.nan, Pw=np.nan, Wq=np.nan,
                    Lq=np.nan, W=np.nan, L=np.nan, stable=False)
    a = lam / mu
    rho = a / c
    if rho >= 1:
        return dict(a=a, rho=rho, Pw=1.0, Wq=np.inf, Lq=np.inf,
                    W=np.inf, L=np.inf, stable=False)
    pw = erlang_c(c, a)
    wq = pw / (c * mu - lam)
    return dict(a=a, rho=rho, Pw=pw, Wq=wq, Lq=lam * wq,          # Lq = λWq, Little
                W=wq + 1 / mu, L=lam * (wq + 1 / mu), stable=True)


def kingman(wq_mmc: float, ca2: float, cs2: float) -> float:
    """G/G/c approximation: variability ratio x the M/M/c wait."""
    return np.inf if not np.isfinite(wq_mmc) else wq_mmc * (ca2 + cs2) / 2.0


def min_servers(lam: float, mu: float, target_rho: float = 0.85) -> int:
    if not (np.isfinite(lam) and np.isfinite(mu)) or mu <= 0:
        return 1
    return max(1, int(np.ceil(lam / (mu * target_rho))))


def arrival_rate(df: pd.DataFrame, date_col: str):
    """λ = records entering per calendar day across the observed span."""
    d = df[date_col].dropna()
    if len(d) < 2:
        return np.nan, 0, np.nan
    span = max((d.max() - d.min()).days, 1)
    return len(d) / span, int(len(d)), span


def queue_table(df: pd.DataFrame, stage_cols: dict, servers: dict,
                ca2: float = 1.0, clip_s3: bool = True) -> pd.DataFrame:
    """
    One row per station. stage_cols maps stage key -> (label, duration column,
    entry-date column).
    """
    rows = []
    for key, (label, dur_col, date_col) in stage_cols.items():
        v = pd.to_numeric(df[dur_col], errors="coerce").dropna()
        if clip_s3 and dur_col == "S3_plan_to_actual":
            v = v.clip(lower=0)
        mean = v.mean() if len(v) else np.nan
        sd = v.std(ddof=1) if len(v) > 1 else np.nan
        cv = sd / mean if (mean and mean > 0) else np.nan
        mu = 1 / mean if (mean and mean > 0) else np.nan
        lam, n_arr, span = arrival_rate(df, date_col)
        c = int(servers.get(key, min_servers(lam, mu)))
        m = mmc(lam, mu, c)
        cs2 = cv ** 2 if np.isfinite(cv) else 1.0
        wq_gg = kingman(m["Wq"], ca2, cs2)
        rows.append({
            "Station": label, "n": int(len(v)),
            "λ /day": round(lam, 3) if np.isfinite(lam) else np.nan,
            "Mean service (d)": round(mean, 2) if np.isfinite(mean) else np.nan,
            "SD (d)": round(sd, 2) if np.isfinite(sd) else np.nan,
            "Cs²": round(cs2, 2),
            "μ /day/server": round(mu, 4) if np.isfinite(mu) else np.nan,
            "c": c,
            "ρ": round(m["rho"], 3) if np.isfinite(m["rho"]) else np.nan,
            "P(wait)": round(m["Pw"], 3) if np.isfinite(m["Pw"]) else np.nan,
            "Wq M/M/c (d)": round(m["Wq"], 2) if np.isfinite(m["Wq"]) else np.inf,
            "Wq G/G/c (d)": round(wq_gg, 2) if np.isfinite(wq_gg) else np.inf,
            "Lq (Little)": round(m["Lq"], 1) if np.isfinite(m["Lq"]) else np.inf,
            "W = Wq + 1/μ (d)": round(m["W"], 2) if np.isfinite(m["W"]) else np.inf,
            "Stable": m["stable"], "_key": key,
        })
    return pd.DataFrame(rows)


# How S3 enters a queueing station. A service time cannot be negative, so the
# signed deviation has to be transformed. The two defensible choices disagree,
# and the app shows both rather than picking one silently:
#   "delay"  -> max(S3, 0): counts only lateness. Discards earliness, which is
#               precisely where the stock-held channel's problem lies, so it
#               UNDERSTATES the RDC.
#   "abs"    -> |S3|: schedule error in either direction. Reproduces the
#               report's mean absolute deviation (4.09 direct vs 14.52 RDC).
S3_MEASURE = {"delay": ("S3_delay", "S3 delay at delivery, max(S3,0)"),
              "abs":   ("abs_dev", "S3 absolute schedule error, |S3|")}


def direct_stations(s3_measure: str = "delay"):
    col, lab = S3_MEASURE[s3_measure]
    st = {}
    for s in STAGES:
        if s == "S3_plan_to_actual":
            st[s] = (lab, col, ENTRY_DATE[s])
        else:
            st[s] = (LABEL[s], s, ENTRY_DATE[s])
    return st


def rdc_stations(s3_measure: str = "delay"):
    col, lab = S3_MEASURE[s3_measure]
    return {
        "pq_to_sched": ("R Quotation to scheduled date (opaque block)", "pq_to_sched",
                        "PQ First Sent to Client Date_dt"),
        "S3_plan_to_actual": (lab, col, ENTRY_DATE["S3_plan_to_actual"]),
        "S4_actual_to_recorded": (LABEL["S4_actual_to_recorded"], "S4_actual_to_recorded",
                                  ENTRY_DATE["S4_actual_to_recorded"]),
    }


def variability_lever(wq_now: float, cs2_now: float, ca2: float,
                      gap_closed: float, cs2_target: float) -> dict:
    """
    Kingman is linear in Cs², so closing part of the variability gap scales the
    wait directly. gap_closed in [0,1] moves Cs² from its current value towards
    cs2_target. Mirrors the report's sensitivity sweep on the RDC spread gap.
    """
    cs2_new = cs2_now - gap_closed * (cs2_now - cs2_target)
    if not np.isfinite(wq_now) or (ca2 + cs2_now) == 0:
        return {"cs2_new": cs2_new, "wq_new": np.nan, "days_saved": np.nan}
    wq_new = wq_now * (ca2 + cs2_new) / (ca2 + cs2_now)
    return {"cs2_new": cs2_new, "wq_new": wq_new, "days_saved": wq_now - wq_new}


# ======================================================================
# 4. HEADLESS RUN -> appendix tables
# ======================================================================

if __name__ == "__main__":
    import pathlib
    root = pathlib.Path(__file__).resolve().parent
    data = root / "shipments.csv"
    if not data.exists():
        data = root / "data" / "shipments.csv"
    df = load(str(data))

    print(reproduction_check(df).to_string(index=False), "\n")

    prof = stage_profile(full_chain(df)).set_index("stage")
    dur = {"S1": prof.loc["S1_quote_to_po", "Mean"],
           "S2": prof.loc["S2_po_to_plan", "Mean"],
           "S3": max(prof.loc["S3_plan_to_actual", "Mean"], 0.0),
           "S4": prof.loc["S4_actual_to_recorded", "Mean"]}
    tbl, project, path = cpm(direct_network(dur, {"mfg": .45, "qa": .15, "book": .30}))
    print(tbl.drop(columns=["stage"]).to_string(index=False))
    print(f"\nProject duration {project:.1f} d   critical path {' → '.join(path)}")
    print("KPI visibility:", {k: round(v, 2) for k, v in
                              kpi_visibility(tbl, project).items()}, "\n")

    q = queue_table(df[df.channel == DIRECT], direct_stations("delay"), {}, 1.0)
    print(q.drop(columns=["_key"]).to_string(index=False))

    out = root / "results"
    out.mkdir(exist_ok=True)
    tbl.drop(columns=["stage"]).to_csv(out / "t13_cpm_schedule.csv", index=False)
    q.drop(columns=["_key"]).to_csv(out / "t14_queue_metrics.csv", index=False)
    print(f"\nwrote {out/'t13_cpm_schedule.csv'} and {out/'t14_queue_metrics.csv'}")
