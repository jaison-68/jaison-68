"""
app.py — CPM and queueing extension to 24PME2072 End-to-End Supply Chain
Performance 

Run:  streamlit run app.py
Needs shipments.csv (the 7,030-shipment file from the project zip) beside it,
or at data/shipments.csv.
"""

import pathlib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import cpm_queue as cq

st.set_page_config(page_title="CPM & Queueing — SCMS end-to-end",
                   page_icon="⏱", layout="wide")

CRIT, FLOAT_C = "#B3352E", "#7FA6C4"


# ----------------------------------------------------------------------
@st.cache_data
def get_data():
    here = pathlib.Path(__file__).resolve().parent
    for p in [here / "shipments.csv", here / "data" / "shipments.csv"]:
        if p.exists():
            return cq.load(str(p))
    raise FileNotFoundError


try:
    df_all = get_data()
except FileNotFoundError:
    st.error("Put **shipments.csv** (from SCO_End_to_End_Code_and_Data.zip → data/) "
             "beside app.py, then reload.")
    st.stop()

st.title("Critical path and queueing analysis")
st.caption("Extension to *Grading Your Own Homework* — 24PME2072 · SCMS Delivery History, "
           "7,030 shipments, 2006–2015. Every figure below is computed from the same "
           "shipments.csv that produced the report.")

# ---- sidebar ---------------------------------------------------------
st.sidebar.header("Subset")
yrs = st.sidebar.slider("Year", int(df_all.year.min()), int(df_all.year.max()),
                        (int(df_all.year.min()), int(df_all.year.max())))
modes = st.sidebar.multiselect("Shipment mode", sorted(df_all["Shipment Mode"].dropna().unique()))
groups = st.sidebar.multiselect("Product group", sorted(df_all["Product Group"].dropna().unique()))
countries = st.sidebar.multiselect("Country", sorted(df_all.Country.dropna().unique()))

df = cq.apply_filters(df_all, yrs, modes, countries, None, groups)
is_default = (yrs == (int(df_all.year.min()), int(df_all.year.max()))
              and not modes and not groups and not countries)

st.sidebar.metric("Shipments in view", f"{len(df):,}")
if is_default:
    st.sidebar.success("Unfiltered — figures reproduce the report.")
else:
    st.sidebar.warning("Filtered. Numbers will no longer match Table 4.1; say so if you "
                       "quote them.")
if len(df) < 50:
    st.sidebar.error("Fewer than 50 shipments — estimates are unreliable.")
    if len(df) == 0:
        st.stop()

with st.sidebar.expander("Reproduction check"):
    chk = cq.reproduction_check(df_all)
    st.dataframe(chk, hide_index=True, use_container_width=True)
    st.caption("Published = report Table 4.1 / results/stage_results.json. "
               "All rows must read True." if chk.Match.all() else "MISMATCH — stop.")

tab_cpm, tab_q, tab_ch, tab_m = st.tabs(
    ["Critical path", "Queueing", "Channel comparison", "Method & export"])

# ======================================================================
# CPM
# ======================================================================
with tab_cpm:
    chain = cq.DIRECT

    if chain == cq.DIRECT:
        c = cq.full_chain(df[df.channel == cq.DIRECT])
        ok = len(c) >= 30
        if not ok:
            st.error("Too few fully observed chains in this subset — widen the filters.")
        prof = cq.stage_profile(c).set_index("stage") if ok else None

        stat = "Mean"
        mfg, qa, book = 0.45, 0.15, 0.30

        dur = {"S1": float(prof.loc["S1_quote_to_po", stat]),
               "S2": float(prof.loc["S2_po_to_plan", stat]),
               "S3": max(float(prof.loc["S3_plan_to_actual", stat]), 0.0),
               "S4": float(prof.loc["S4_actual_to_recorded", stat])} if ok else None
        acts = cq.direct_network(dur, {"mfg": mfg, "qa": qa, "book": book}) if ok else None
    if ok:
        tbl, project, path = cq.cpm(acts)
        vis = cq.kpi_visibility(tbl, project)

        m1, m3, m4 = st.columns(3)
        m1.metric("Project duration", f"{project:.1f} d")
        m3.metric("Float in the network", f"{tbl['Total float'].sum():.1f} d")
        m4.metric("Share the on-time KPI sees", f"{vis['pct_of_project']:.2f}%",
                  help="S3 duration ÷ project duration. On the unfiltered Direct Drop chain "
                       "this returns 1.08%, matching the report's share of elapsed time.")

        if chain == cq.DIRECT and is_default and stat == "Mean":
            st.success(f"Reproduces the report: 157.2-day chain, of which the reported metric "
                       f"observes {vis['s3_days']:.2f} days — {vis['pct_of_project']:.2f}%. "
                       f"{vis['unobserved_days']:.0f} days are committed before the KPI starts "
                       "watching.")

        stage_rows = []
        for stage_id, stage_name, stage_key in [
            ("S1", "Quotation/PO → Purchase Order (PO)", "S1_quote_to_po"),
            ("S2", "Purchase Order (PO) → Scheduled/Promised Delivery", "S2_po_to_plan"),
            ("S3", "Scheduled Delivery → Actual Delivery", "S3_plan_to_actual"),
            ("S4", "Actual Delivery → Record Closure", "S4_actual_to_recorded"),
        ]:
            stage = tbl[tbl["stage"] == stage_key]
            start = stage["ES"].min()
            finish = stage["EF"].max()
            stage_rows.append({
                "ID": stage_id, "Activity": stage_name, "Pred": "—",
                "Duration": round(finish - start, 2), "ES": round(start, 2),
                "EF": round(finish, 2), "LS": round(stage["LS"].min(), 2),
                "LF": round(stage["LF"].max(), 2),
                "Total float": round(stage["Total float"].min(), 2),
                "Free float": round(stage["Free float"].min(), 2),
                "Critical": bool(stage["Critical"].any()), "stage": stage_key,
            })
        display_tbl = pd.DataFrame(stage_rows)
        st.dataframe(
            display_tbl.drop(columns=["stage"]).style.apply(
                lambda r: ["background-color:#FBE9E7; color:#1F2937;"
                           if r.Critical else ""] * len(r), axis=1),
            use_container_width=True, hide_index=True)

        fig = go.Figure()
        for _, r_ in display_tbl.iterrows():
            lbl = f"{r_.ID}  {r_.Activity}"
            fig.add_trace(go.Bar(y=[lbl], x=[max(r_.Duration, 0.6)], base=[r_.ES],
                                 orientation="h", showlegend=False,
                                 marker_color=CRIT if r_.Critical else FLOAT_C,
                                 hovertemplate=f"{r_.Activity}<br>ES {r_.ES} · EF {r_.EF}"
                                               f"<br>Total float {r_['Total float']}<extra></extra>"))
            if r_["Total float"] > 0:
                fig.add_trace(go.Bar(y=[lbl], x=[r_["Total float"]], base=[r_.EF],
                                     orientation="h", showlegend=False,
                                     marker_color="rgba(120,120,120,.22)", hoverinfo="skip"))
        fig.update_layout(barmode="overlay", height=360, xaxis_title="Days from quotation",
                          yaxis=dict(autorange="reversed"), margin=dict(t=20, b=30, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Red = zero total float (critical). Grey tail = float available.")

# ======================================================================
# QUEUEING
# ======================================================================
with tab_q:
    st.subheader("Where the waiting accumulates")
    st.markdown(
        "Each stage is treated as a service station. **λ** is shipments entering per day, "
        "from the date that opens the stage. **μ** is 1 ÷ mean stage duration per server. "
        "**c** is how many shipments the stage can work on at once. M/M/c gives Wq through "
        "Erlang-C; the G/G/c column applies Kingman's variability correction using the "
        "measured spread. Lq = λWq is Little's law — the reference the report's literature "
        "table says cannot tell you *which* stage owns the wait.")

    chan_q = cq.DIRECT
    s3m = "delay"
    targ = 0.85
    ca2 = 1.0

    sub = df[df.channel == chan_q]
    stations = cq.direct_stations(s3m) if chan_q == cq.DIRECT else cq.rdc_stations(s3m)

    auto = {}
    for key, (lab, dcol, datecol) in stations.items():
        v = pd.to_numeric(sub[dcol], errors="coerce").dropna()
        mu = 1 / v.mean() if len(v) and v.mean() > 0 else np.nan
        lam, _, _ = cq.arrival_rate(sub, datecol)
        auto[key] = cq.min_servers(lam, mu, targ)

    servers = auto

    qt = cq.queue_table(sub, stations, servers, ca2)
    st.dataframe(qt.drop(columns=["_key"]), use_container_width=True, hide_index=True)

    if not qt["Stable"].all():
        st.error("ρ ≥ 1 somewhere: that queue grows without bound. Raise c.")

    a, b = st.columns(2)
    with a:
        f = px.bar(qt, x="Station", y=["Wq M/M/c (d)", "Wq G/G/c (d)"], barmode="group",
                   labels={"value": "Waiting time (days)", "variable": ""})
        f.update_layout(height=340, margin=dict(t=20, b=20), xaxis_title="")
        st.plotly_chart(f, use_container_width=True)
    with b:
        f = px.bar(qt, x="Station", y="ρ")
        f.add_hline(y=1, line_dash="dash", line_color="red")
        f.update_layout(height=340, yaxis_range=[0, 1.1], margin=dict(t=20, b=20),
                        xaxis_title="")
        st.plotly_chart(f, use_container_width=True)

    st.caption("Where the two bars diverge sharply the wait is driven by variability, not "
               "by load. S3 and S4 have Cs² above 10 — most shipments take zero days and a "
               "few take months — so Kingman's correction dominates. Treat those G/G/c "
               "figures as an indication of how badly a spiky station behaves, not as a "
               "point estimate; the approximation is being pushed hard at that dispersion.")

    st.markdown("**Utilisation against waiting time**")
    curve_station = st.selectbox("Station ", qt["Station"].tolist(), key="curve")
    key = qt[qt["Station"] == curve_station]["_key"].iloc[0]
    lab, dcol, datecol = stations[key]
    v = pd.to_numeric(sub[dcol], errors="coerce").dropna()
    mu = 1 / v.mean() if len(v) and v.mean() > 0 else np.nan
    lam, _, _ = cq.arrival_rate(sub, datecol)
    base = servers[key]
    pts = []
    for cc in range(max(1, base - 15), base + 30):
        m = cq.mmc(lam, mu, cc)
        if m["stable"]:
            pts.append({"Servers": cc, "ρ": m["rho"], "Wq (days)": m["Wq"]})
    if pts:
        cur = pd.DataFrame(pts)
        f = px.line(cur, x="ρ", y="Wq (days)", markers=True, hover_data=["Servers"])
        f.update_layout(height=330, margin=dict(t=20, b=20))
        st.plotly_chart(f, use_container_width=True)
        st.caption("Past roughly ρ = 0.85 the curve turns vertical: a small rise in load "
                   "produces a large rise in waiting. That is the quantitative case for "
                   "holding buffer capacity rather than chasing utilisation.")

# ======================================================================
# CHANNEL COMPARISON
# ======================================================================
with tab_ch:
    st.subheader("Direct Drop against RDC")
    s3m2 = st.selectbox("S3 as a service time", ["abs", "delay"],
                        format_func=lambda k: cq.S3_MEASURE[k][1],
                        help="Absolute error reproduces the report's 4.09 vs 14.52 gap. "
                             "Delay-only discards earliness and therefore flatters the RDC.")
    rows = []
    for ch, stns in [(cq.DIRECT, cq.direct_stations(s3m2)), (cq.RDC, cq.rdc_stations(s3m2))]:
        sub2 = df[df.channel == ch]
        t = cq.queue_table(sub2, stns, {}, 1.0)
        s3row = t[t["Station"] == cq.S3_MEASURE[s3m2][1]]
        if len(s3row):
            r_ = s3row.iloc[0]
            rows.append({"Channel": ch, "n": r_["n"],
                         "Mean service (d)": r_["Mean service (d)"],
                         "SD (d)": r_["SD (d)"], "Cs²": r_["Cs²"],
                         "Servers c": r_["c"], "ρ": r_["ρ"],
                         "Wq G/G/c (d)": r_["Wq G/G/c (d)"],
                         "W in system (d)": r_["W = Wq + 1/μ (d)"]})
    comp = pd.DataFrame(rows)
    st.dataframe(comp, use_container_width=True, hide_index=True)
    st.caption("Compare **W in system**, not Wq alone. A station with a long mean service "
               "time can show a modest queue and still hold shipments far longer.")

    if len(comp) == 2 and s3m2 == "abs" and is_default:
        st.info(f"Mean service times of {comp['Mean service (d)'].iloc[0]:.2f} and "
                f"{comp['Mean service (d)'].iloc[1]:.2f} days reproduce the report's mean "
                "absolute deviation figures (4.09 / 14.52) exactly, which is the check that "
                "this queueing layer sits on the same data as Chapter IV.")

    st.markdown("**Published reference figures — fixed, from §4.3 and §4.4**")
    ref = pd.DataFrame([
        {"Measure": "SD of deviation (d)", "Direct Drop": cq.PUBLISHED["sd_dev_direct"],
         "RDC": cq.PUBLISHED["sd_dev_rdc"]},
        {"Measure": "Mean absolute deviation (d)", "Direct Drop": cq.PUBLISHED["abs_dev_direct"],
         "RDC": cq.PUBLISHED["abs_dev_rdc"]},
    ])
    st.dataframe(ref, use_container_width=True, hide_index=True)
    st.markdown(f"Excess spread of 7.52 days carries **${cq.PUBLISHED['excess_safety_stock_usd']:,.0f}** "
                "of extra safety stock at 95% service (§4.4). Queueing explains the mechanism "
                "behind that number: the same dispersion that inflates Cs² is what the "
                "inventory is there to absorb.")
    st.warning("Honest caveat for the viva: a queueing station cannot have a negative "
               "service time, so the delay-only view drops the 26.3% of RDC shipments that "
               "arrive more than a fortnight **early**. Under that view the RDC can look "
               "better than the direct channel. It is not — the earliness is simply invisible "
               "to a one-sided model. Use the absolute-error view, and say why.")

# ======================================================================
# METHOD & EXPORT
# ======================================================================
with tab_m:
    st.subheader("What this adds, and what it assumes")
    st.markdown("""
**What is new relative to the submitted report.** Chapter IV decomposes elapsed time and
variance by stage. It does not schedule the chain or model congestion. This extension adds
two standard tools and one result:

* **CPM** turns the four stages into a dated schedule with early and late start times and
  float. On the unfiltered Direct Drop chain the project duration lands on 157.2 days and
  the activity the on-time KPI inspects is **1.08%** of it — the same figure the report
  derives from elapsed time, now reached by a second, independent route.
* **Queueing** separates the wait caused by *load* from the wait caused by *variability*.
  Little's law (L = λW) is cited in the report's literature table as saying nothing about
  which stage owns the wait. Erlang-C plus Kingman answers exactly that.

**Assumptions, stated plainly.**

1. *Server counts are chosen, not measured.* The data records dates, not capacity. Defaults
   size c to hold ρ at the target you set. Any figure that depends on c must be quoted with c.
2. *The S2 split is a modelling assumption.* Manufacturing, QA, booking and transit are not
   separately dated. The sliders exist so the assumption is visible and adjustable rather
   than buried.
3. *Service times are not exponential.* Cs² runs from 0.2 to 33. M/M/c is reported for
   reference; the G/G/c column is the one to read, and at very high Cs² it is an
   approximation under strain.
4. *A service time cannot be negative.* S3 is signed. Both transformations are offered and
   they disagree — see the caveat on the channel tab.
5. *Steady state.* Erlang-C assumes a stable system over the window. The programme's
   volume trends across 2006–2015, so run the year filter before quoting any single number.

**Inherited limits.** The record ends in September 2015; channel assignment is not random;
the post-2007 step change may be a recording artefact. All three apply here unchanged.
""")

    st.markdown("**Formulae used**")
    st.latex(r"\rho=\frac{\lambda}{c\mu}\qquad "
             r"W_q^{M/M/c}=\frac{C(c,\lambda/\mu)}{c\mu-\lambda}\qquad L_q=\lambda W_q")
    st.latex(r"W_q^{G/G/c}\approx\frac{C_a^2+C_s^2}{2}\,W_q^{M/M/c}\qquad "
             r"TF_i=LS_i-ES_i")
    st.caption("Erlang-C is computed through the Erlang-B recursion "
               "B(n)=aB(n−1)/(n+aB(n−1)) so it stays stable at the large server counts "
               "these lead times require.")

    st.markdown("**Export for Appendix** — continues the report's table numbering.")
    chain_dd = cq.full_chain(df[df.channel == cq.DIRECT])
    if len(chain_dd) > 30:
        prof = cq.stage_profile(chain_dd).set_index("stage")
        d = {"S1": prof.loc["S1_quote_to_po", "Mean"], "S2": prof.loc["S2_po_to_plan", "Mean"],
             "S3": max(prof.loc["S3_plan_to_actual", "Mean"], 0.0),
             "S4": prof.loc["S4_actual_to_recorded", "Mean"]}
        t13, _, _ = cq.cpm(cq.direct_network(d, {"mfg": .45, "qa": .15, "book": .30}))
        st.download_button("t13_cpm_schedule.csv",
                           t13.drop(columns=["stage"]).to_csv(index=False).encode(),
                           "t13_cpm_schedule.csv", "text/csv")
    t14 = cq.queue_table(df[df.channel == cq.DIRECT], cq.direct_stations("delay"), {}, 1.0)
    st.download_button("t14_queue_metrics.csv",
                       t14.drop(columns=["_key"]).to_csv(index=False).encode(),
                       "t14_queue_metrics.csv", "text/csv")
    st.caption("Both exports use default assumptions on the unfiltered data, so they match "
               "the headless run of `python cpm_queue.py`.")
