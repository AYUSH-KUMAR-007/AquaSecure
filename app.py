"""AquaSecure — A Simulation Framework for Security-Aware Underwater
Quantum Key Distribution in Turbid Water.

Run:  streamlit run app.py
"""

import math
import dataclasses

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from aquasecure.waters import WATER_TYPES, BACKGROUND_LEVELS, BACKGROUND_DESCRIPTIONS
from aquasecure.channel import ChannelInput, evaluate_channel
from aquasecure.qkd import (
    QKDConfig, evaluate_qkd, evaluate_at_distance, sweep_distance,
    max_secure_distance, optimal_mu, QBER_SECURE_MAX,
)
from aquasecure.mc import run_bb84_mc
from aquasecure.recommend import build_recommendations
from aquasecure.figures import (
    fig_rate_qber_vs_distance, fig_rate_vs_ntu, fig_transmittance_waterfall,
    fig_mu_sweep, fig_eve_analysis, fig_secret_fraction, fig_to_png_bytes,
    NAVY, TEAL, CYAN, GOLD, RED, GREEN, GREY,
)
from aquasecure.export import (
    scenario_snapshot, snapshot_csv, snapshot_json, snapshot_latex_row,
    LATEX_TABLE_PREAMBLE,
)
import pandas as pd
from aquasecure.source import evaluate_source, decomposition_errors
from aquasecure.nanophotonics import NANOPHOTONIC_STRUCTURES
from aquasecure.background import BACKGROUND_FLUX
from aquasecure.circuit import qiskit_qber, statevector_qber
from aquasecure.report_pdf import build_pdf
from aquasecure.chat import generate_chat_key, encrypt_chat_packet, aes_gcm_integrity_tests, packet_log_json
from aquasecure.figures import fig_lines

YEAR = 2026

st.set_page_config(
    page_title="AquaSecure — Underwater QKD Simulator",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------- CSS
st.markdown("""
<style>
.hero {
  background: linear-gradient(135deg, #062a3d 0%, #0b3c5d 55%, #0e6f8e 100%);
  border-radius: 14px; padding: 18px 24px; color: #eaf6fb;
  margin-bottom: 14px; border: 1px solid rgba(255,255,255,.08);
}
.hero h1 { font-size: 1.55rem; margin: 0 0 4px 0; letter-spacing: .3px; color:#ffffff; }
.hero p  { font-size: .82rem; margin: 0; color: #bfe3ee; }
.hero .badge {
  display:inline-block; margin-top:8px; font-size:.68rem; font-weight:600;
  color:#062a3d; background:#7fd1e8; border-radius:999px; padding:2px 12px;
  letter-spacing:.6px; text-transform:uppercase;
}
.banner { padding: 12px 18px; border-radius: 12px; text-align:center;
          font-size: 1.12rem; font-weight: 800; letter-spacing: 1px;
          color:#fff; margin: 6px 0 14px 0; }
.banner small { display:block; font-size:.75rem; font-weight:500; letter-spacing:.2px;
                opacity:.95; margin-top:2px; }
.secure   { background: linear-gradient(135deg,#0f7b5f,#149a76); box-shadow:0 2px 12px rgba(15,123,95,.4); }
.marginal { background: linear-gradient(135deg,#b45309,#d97706); box-shadow:0 2px 12px rgba(217,119,6,.4); }
.insecure { background: linear-gradient(135deg,#b91c1c,#dc2626); box-shadow:0 2px 12px rgba(185,28,28,.4); }
.kpi { background:#f2f8fa; border:1px solid #d4e6ec; border-radius:12px;
       padding:10px 14px; height:100%; }
.kpi .lab { font-size:.68rem; text-transform:uppercase; letter-spacing:.7px;
            color:#5b6b7a; font-weight:700; }
.kpi .val { font-size:1.32rem; font-weight:800; color:#0b3c5d; line-height:1.25; }
.kpi .sub { font-size:.70rem; color:#5b6b7a; }
.rec { border-left: 5px solid; border-radius:8px; background:#fbfdfd;
       padding:8px 12px; margin:6px 0; border:1px solid #e3ecef; border-left-width:5px;}
.rec.critical { border-left-color:#b91c1c; }
.rec.high     { border-left-color:#d97706; }
.rec.medium   { border-left-color:#2ea3a0; }
.rec.info     { border-left-color:#0e6f8e; }
.rec b { color:#0b3c5d; }
.rec .tag { font-size:.62rem; font-weight:800; letter-spacing:.8px; text-transform:uppercase; }
.tag.critical{color:#b91c1c}.tag.high{color:#d97706}.tag.medium{color:#2ea3a0}.tag.info{color:#0e6f8e}
.small { font-size:.75rem; color:#5b6b7a; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------- Sidebar
with st.sidebar:
    st.markdown("## 🌊 AquaSecure\n**Control panel**")

    st.markdown("### 1 · Channel")
    water_type = st.selectbox("Water type", list(WATER_TYPES.keys()), index=1,
                              help="Optical water type — sets absorption a and scattering b at 532 nm.")
    st.caption(f"{WATER_TYPES[water_type]['description']}  Typical: {WATER_TYPES[water_type]['typical_ntu']} NTU.")
    distance_m = st.slider("Distance (m)", 1.0, 20.0, 5.0, 0.1)
    ntu = st.slider("Turbidity (NTU)", 0.0, 100.0, 20.0, 1.0)
    wavelength = st.selectbox("Wavelength λ (nm)", [480, 510, 532, 550, 580, 650, 850, 1550], index=2,
                              help="532 nm is the reference (reproduces v1.0 bit-identically). b(λ) = b₅₃₂·(532/λ)^1.14; NTU adder ∝ (532/λ)^0.8. Absorption is NOT re-modelled.")
    background = st.selectbox("Background light", list(BACKGROUND_LEVELS.keys()), index=0,
                              help="Sets the mean background-photon detection rate at the receiver.")
    st.caption(BACKGROUND_DESCRIPTIONS[background])
    with st.expander("v1.1 · physics-based background model"):
        background_model = st.radio("Model", ["Preset (cps, v1.0)", "Physics-based (radiance)"], index=0,
                                    help="Physics mode: R_bg = Φ·πr²·π·FOV²·BW·exp(−a·min(L,5 m)); Y₀ uses gate window.")
        if background_model.startswith("Physics"):
            bg_level = st.selectbox("Irradiance class", list(BACKGROUND_FLUX.keys()), index=1)
            filter_bw = st.slider("Filter bandwidth (nm)", 0.5, 50.0, 10.0, 0.5)
            gate_ns = st.slider("Gate window (ns)", 0.1, 10.0, 1.0, 0.1)
            fov_mrad = st.slider("Receiver FOV (full angle, mrad)", 5.0, 200.0, 50.0, 5.0)
        else:
            bg_level, filter_bw, gate_ns, fov_mrad = "Low ambient", 10.0, 1.0, 50.0

    st.markdown("### 2 · Detector")
    det_eff = st.slider("Detector efficiency", 0.01, 0.40, 0.15, 0.01,
                        format="%.2f", help="Quantum efficiency of the single-photon detector (InGaAs: typically 0.05–0.25).")
    dark_cps = st.number_input("Dark-count rate (counts/s)", 1.0, 1.0e5, 100.0, 10.0)
    clock_hz = st.selectbox("System clock (pulses/s)",
                            [1e6, 5e6, 1e7, 5e7, 1e8], index=2,
                            format_func=lambda v: f"{v/1e6:.0f} MHz")

    st.markdown("### 3 · Alignment & optics")
    misalignment = st.slider("Pointing misalignment (mrad)", 0.0, 10.0, 0.5, 0.1)
    divergence = st.slider("Beam divergence, full angle (mrad)", 1.0, 50.0, 10.0, 0.5)
    with st.expander("Advanced optics"):
        tx_ap = st.slider("Transmitter aperture radius (cm)", 0.5, 10.0, 2.0, 0.5) / 100.0
        rx_ap = st.slider("Receiver aperture radius (cm)", 1.0, 25.0, 5.0, 0.5) / 100.0
        k_ntu = st.slider("Turbidity scattering coefficient k (m⁻¹/NTU)", 0.0, 0.15, 0.05, 0.005)

    st.markdown("### 4 · QKD protocol")
    protocol = st.radio("Protocol", ["Decoy-state BB84", "BB84 (ideal source, simplified)"], index=0)
    mu = st.slider("Mean photon number μ", 0.05, 0.8, 0.5, 0.01)
    if protocol.startswith("Decoy"):
        decoy_nu = st.slider("Decoy intensity ν", 0.02, 0.3, 0.1, 0.01)
    else:
        decoy_nu = 0.1

    st.markdown("### 5 · Eavesdropper")
    eve = st.radio("Eve", ["None", "Intercept–resend"], index=0, horizontal=True)
    eve_fraction = 0.0
    if eve == "Intercept–resend":
        eve_fraction = st.slider("Intercepted fraction of pulses", 0.0, 1.0, 1.0, 0.05)

    st.markdown("### 6 · Source (v1.1)")
    source_mode = st.radio("Pulse source", ["Weak coherent pulse (v1.0)", "Quantum-dot-inspired (QD)"],
                           index=0, help="QD: quasi-single-photon source; rate uses source-aware GLLP form.")
    qd_on = source_mode.startswith("Quantum")
    with st.expander("QD source parameters", expanded=qd_on):
        qd_yield = st.slider("Quantum yield", 0.2, 1.0, 0.85, 0.01)
        qd_blink = st.slider("Blinking factor (on-time)", 0.3, 1.0, 0.92, 0.01)
        qd_g2 = st.slider("g²(0)", 0.0, 0.30, 0.03, 0.005,
                          help="Second-order coherence at zero delay. p_multi = min(½, ½·η_src²·g²(0)).")
        qd_lw = st.slider("Emission linewidth (nm)", 2.0, 80.0, 15.0, 1.0)
        qd_bw = st.slider("QD filter bandwidth (nm)", 1.0, 50.0, 10.0, 1.0)
        qd_temp = st.slider("Emitter temperature (°C)", 4.0, 150.0, 20.0, 1.0)

    st.markdown("### 7 · Nanophotonic transmitter (v1.1)")
    nano_structure = st.selectbox("Structure", list(NANOPHOTONIC_STRUCTURES.keys()), index=0,
                                  help="None = bare source (v1.0). Strcutures are literature-inspired engineering values.")
    with st.expander("Structure parameters", expanded=(nano_structure != "None")):
        nano_specs = NANOPHOTONIC_STRUCTURES[nano_structure]
        st.caption("Defaults from the digital-twin lookup table; adjust to your device model.")
        nano_eta = st.slider("Extraction efficiency η_ext", 0.1, 0.9, float(nano_specs["eta_ext_base"]), 0.05)

    with st.expander("Advanced security parameters"):
        f_ec = st.slider("Error-correction efficiency f", 1.0, 1.4, 1.16, 0.01)
        e_intrinsic = st.slider("Intrinsic error rate e_d (%)", 0.0, 5.0, 1.5, 0.1) / 100.0
        decomposed_error = st.checkbox("Physics-decomposed intrinsic error (v1.1)", value=False,
                                       help="e_d = pol(nano) + ½sin²(1.8·φ) + depolarisation(ρ, L, NTU) + 0.4 % jitter. Overrides e_d user setting.")
        seed = st.number_input("Random seed", 0, 99999, 42, 1)

    st.markdown("---")
    st.caption("AquaSecure v1.1 · simplified asymptotic model · not a security proof. See Methodology tab.")

# ---------------------------------------------------------------- Compute
bg_mode = "physics" if background_model.startswith("Physics") else "preset"
ci = ChannelInput(
    water_type=water_type, distance_m=distance_m, ntu=ntu, k_ntu=k_ntu,
    background=background, detector_efficiency=det_eff,
    dark_count_cps=float(dark_cps), clock_hz=float(clock_hz),
    misalignment_mrad=misalignment, divergence_mrad=divergence,
    tx_aperture_radius_m=tx_ap, rx_aperture_radius_m=rx_ap,
    wavelength_nm=float(wavelength),
    background_model=bg_mode,
    background_physics_level=bg_level,
    filter_bw_nm=float(filter_bw),
    gate_window_ns=float(gate_ns),
    receiver_fov_mrad=float(fov_mrad),
    nanophotonic_structure=nano_structure,
    nanophotonic_eta_ext=(float(nano_eta) if nano_structure != "None" else None),
)
src = evaluate_source("QD" if qd_on else "WCP", wavelength_nm=float(wavelength),
                      quantum_yield=qd_yield, linewidth_nm=qd_lw, g2_zero=qd_g2,
                      blinking_factor=qd_blink, temp_c=qd_temp, filter_bw_nm=qd_bw,
                      rep_rate_hz=float(clock_hz))
cfg = QKDConfig(protocol=protocol, mu=mu, decoy_nu=decoy_nu, f_ec=f_ec,
                e_intrinsic=e_intrinsic, eve_fraction=eve_fraction,
                source_mode=("QD" if qd_on else "WCP"),
                source_eff=src["source_efficiency"],
                p_multi=src["multi_photon_prob"],
                g2_zero=(qd_g2 if qd_on else None),
                qd_quality=src["qd_quality_score"],
                decomposed_error=decomposed_error)

ch = evaluate_channel(ci)
ev = evaluate_qkd(ch, cfg)
l_max = max_secure_distance(ci, cfg)
opt_mu, opt_r = optimal_mu(ci, cfg)
recs = build_recommendations(ch, ev, cfg, opt_mu, l_max)

# ---------------------------------------------------------------- Header
st.markdown("""
<div class="hero">
  <h1>AquaSecure — Security-Aware Underwater Quantum Key Distribution in Turbid Water</h1>
  <p>Interactive simulation framework for a BB84 / decoy-state QKD link over a realistically lossy
     underwater channel: water attenuation, turbidity scattering, geometric coupling, pointing error,
     background light, detector imperfections and intercept–resend eavesdropping.</p>
  <span class="badge">Asymptotic · Simplified · Simulation-only</span>
</div>
""", unsafe_allow_html=True)

status = ev["security_status"]
cls = status.lower()
reason = (f"QBER {ev['qber_percent']:.2f} % · "
          f"secure rate {ev['secure_rate_bps']/1e3:+.2f} kbit/s · "
          f"{'Eve present' if cfg.eve_fraction else 'no Eve'}")
st.markdown(f'<div class="banner {cls}">{status.upper()}<small>{reason}</small></div>',
            unsafe_allow_html=True)

def kpi(col, label, value, sub=""):
    col.markdown(f'<div class="kpi"><div class="lab">{label}</div>'
                 f'<div class="val">{value}</div><div class="sub">{sub}</div></div>',
                 unsafe_allow_html=True)

k1, k2, k3, k4, k5, k6 = st.columns(6)
kpi(k1, "Channel transmittance", f"{ch['eta_channel']*100:.2e}" if ch['eta_channel'] < 1e-2 else f"{ch['eta_channel']:.3f}",
    f"{ch['channel_loss_db']:.1f} dB channel · {ch['total_loss_db']:.1f} dB incl. detector")
kpi(k2, "Received photon rate", f"{ev['received_rate_cps']/1e3:.2f} k/s",
    f"yield Q_μ = {ev['q_mu']:.2e} / pulse")
kpi(k3, "QBER", f"{ev['qber_percent']:.2f} %",
    f"threshold 11 % · intrinsic {100*ev['e_d_effective']:.1f} %{' + Eve' if cfg.eve_fraction else ''}")
kpi(k4, "Sifted-key rate", f"{ev['sifted_rate_bps']/1e3:.2f} kbit/s",
    "basis reconciliation (q = 1/2)")
kpi(k5, "Secure-key rate", f"{max(ev['secure_rate_bps'], 0.0)/1e3:.2f} kbit/s",
    (f"{ev['secure_bits_per_pulse']:.2e} bits/pulse" + ("" if ev["secure_bits_per_pulse"] > 0 else " · negative → abort")))
kpi(k6, "Max secure distance", f"{l_max:.2f} m" if l_max else "< 1 m",
    "QBER < 11 % and R > 0")
st.write("")

# ---------------------------------------------------------------- Tabs
tab_dash, tab_lab, tab_sweep, tab_max, tab_mc, tab_eve, tab_chat, tab_export, tab_method = st.tabs([
    "📊 Results", "🔬 v1.1 Physics lab", "📈 Sweeps & curves", "🎯 Max distance & settings",
    "🎲 Monte Carlo check", "🕵️ Security & Eve", "💬 Live Secure Chat", "⬇️ Export", "📖 Methodology",
])

# ================= Results dashboard =================
with tab_dash:
    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("Transmittance budget")
        fig = fig_transmittance_waterfall(ch)
        st.pyplot(fig, width="stretch")
        st.download_button("Download figure (PNG)", fig_to_png_bytes(fig),
                           "aquasecure_budget.png", "image/png")
    with right:
        st.subheader("Reading the link")
        _wl_note = (" (reference \u2014 v1.0 bit-identical)" if wavelength == 532 else
                    " (b(\u03bb) \u00d7{:.2f}, NTU adder \u00d7{:.2f})".format((532 / wavelength) ** 1.14, (532 / wavelength) ** 0.8))
        _src_note = ("" if cfg.source_mode == "WCP" else
                     " \u2014 \u03b7_src {:.2f}, g\u00b2(0) {:.3f}, QD score {:.0f}/100".format(src["source_efficiency"], qd_g2, src["qd_quality_score"]))
        _nano_note = ("" if ci.nanophotonic_structure == "None" else
                      " (eff. divergence {:.1f} mrad, \u03b7_nano {:.2f})".format(ch["effective_divergence_mrad"], ch["eta_nano"]))
        st.markdown(f"""
- **What dominates the loss?** Expected loss = `c·L` = **{ch['attenuation_c']:.2f} m⁻¹ × {distance_m:.1f} m =
  {ch['beer_loss_db']:.1f} dB** from Beer–Lambert (absorption {ch['beer_loss_absorption_db']:.1f} dB +
  scattering {ch['beer_loss_scattering_db']:.1f} dB, of which NTU adds {ch['scattering_ntu_b']:.2f} m⁻¹).
- Geometric collection efficiency: **{100*ch['eta_geo']:.1f} %** (beam radius at receiver ≈ {ch['beam_radius_m']*100:.1f} cm).
- Pointing coupling: **{100*ch['eta_mis']:.1f} %** (offset ≈ {1e3*ch['misalign_disp_m']:.1f} mm).
- Background+dark yield per gate: **Y₀ = {ch['y0']:.2e}** ({BACKGROUND_LEVELS[background]:.0e} cps background + {dark_cps:.0f} cps dark).
- Effective signal error incl. Eve: **e_d = {100*ev['e_d_effective']:.2f} %**.
- Wavelength **{wavelength} nm**{_wl_note}.
- Source **{cfg.source_mode}**{_src_note}; transmitter **{ci.nanophotonic_structure}**{_nano_note}.
""")
        if ev["security_status"] == "Secure":
            st.success(f"Link is secure: {ev['secure_rate_bps']/1e3:.2f} kbit/s of secret key under asymptotic collective-attack assumptions.")
        elif ev["security_status"] == "Marginal":
            st.warning("Link is marginal — key can be extracted, but margin is thin. Review recommendations.")
        else:
            st.error("Link is insecure — do NOT use its output as key. See 🎯 Max distance & settings for recovery actions.")

        st.subheader("Full parameter audit")
        with st.expander("Show channel & protocol internals"):
            audit = {
                "water_type": ch["water_type"],
                "a (m⁻¹)": ch["absorption_a"], "b base (m⁻¹)": ch["scattering_base_b"],
                "b NTU (m⁻¹)": ch["scattering_ntu_b"], "c total (m⁻¹)": ch["attenuation_c"],
                "η_Beer–Lambert": ch["eta_beer"], "η_geometric": ch["eta_geo"],
                "η_misalignment": ch["eta_mis"], "η_channel": ch["eta_channel"],
                "η_total (incl. detector)": ch["eta_total"],
                "Y₀": ch["y0"], "Q_μ": ev["q_mu"], "E_μ (QBER)": ev["e_mu"],
                "H₂(QBER)": ev["h2_qber"],
            }
            if protocol.startswith("Decoy") and cfg.source_mode != "QD":
                audit.update({
                    "ν (weak decoy)": ev.get("nu"), "Q_ν": ev.get("q_nu"),
                    "Y₁ lower bound": ev.get("y1_lower_bound"),
                    "Q₁ lower bound": ev.get("q1_lower_bound"),
                    "e₁ upper bound": ev.get("e1_upper_bound"),
                })
            elif cfg.source_mode == "QD":
                audit.update({
                    "η_src": ev.get("qd_source_eff"), "p_multi": ev.get("qd_p_multi"),
                    "g²(0)": ev.get("qd_g2"), "QD quality (/100)": ev.get("qd_quality"),
                })
            st.json(audit, expanded=False)

# ================= v1.1 Physics Lab =================
with tab_lab:
    st.subheader("v1.1 physics bench — transmitter, source, wavelength & circuit check")
    lab_on = []
    if cfg.source_mode == "QD":
        lab_on.append("QD source")
    if ci.nanophotonic_structure != "None":
        lab_on.append(f"nanophotonics · {ci.nanophotonic_structure}")
    if ci.background_model == "physics":
        lab_on.append("physics background")
    if cfg.decomposed_error:
        lab_on.append("decomposed e_d")
    if lab_on:
        st.caption("Active v1.1 models: " + " · ".join(lab_on))
    else:
        st.info("Defaults reproduce v1.0 exactly. Switch on any of: Source → Quantum-dot-inspired, "
                "a nanophotonic structure, the physics-based background model, or decomposed intrinsic error "
                "to see this bench move. All benchmarks below run regardless.")

    st.markdown("#### Transmitter benchmark — structure ladder (current water, distance, Eve & protocol)")
    bench_rows = []
    import dataclasses as _dc
    for name in NANOPHOTONIC_STRUCTURES:
        ci_b = _dc.replace(ci, nanophotonic_structure=name, nanophotonic_eta_ext=None)
        ch_b = evaluate_channel(ci_b)
        ev_b = evaluate_qkd(ch_b, cfg)
        bench_rows.append({
            "Structure": name,
            "Div. (mrad)": round(ch_b["effective_divergence_mrad"], 2),
            "η_ext·coupling": round((ch_b["nanophotonic"]["net_tx_efficiency"] if ch_b["nanophotonic"] else 1.0), 3),
            "η_geo": round(ch_b["eta_geo"], 4),
            "Rate (kbit/s)": round(max(ev_b["secure_rate_bps"], 0.0) / 1e3, 2),
            "Status": ev_b["security_status"],
        })
    bench_df = pd.DataFrame(bench_rows)
    st.dataframe(bench_df, width="stretch", hide_index=True)
    fig_bench = fig_lines([r["Structure"] for r in bench_rows],
                          {"Secure rate (kbit/s)": [r["Rate (kbit/s)"] for r in bench_rows]},
                          "Structure", "Secure rate (kbit/s)",
                          "Transmitter ladder at current operating point", ymin0=True)
    st.pyplot(fig_bench, width="stretch")
    st.caption("Structure defaults (η_ext, div_factor, immersion loss) from the digital-twin engineering lookup "
               "table — not measured devices. Baseline ‘None’ reproduces the v1.0 transmitter.")

    wl_col, div_col = st.columns(2)
    with wl_col:
        st.markdown("#### Secure rate vs wavelength")
        wl_list = [450, 480, 510, 532, 550, 600, 650, 850, 1310, 1550]
        wl_rates = []
        for wl in wl_list:
            ci_w = _dc.replace(ci, wavelength_nm=float(wl))
            ev_w = evaluate_qkd(evaluate_channel(ci_w), cfg)
            wl_rates.append(max(ev_w["secure_rate_bps"], 0.0) / 1e3)
        fig_wl = fig_lines(wl_list, {"Secure rate (kbit/s)": wl_rates},
                           "Wavelength (nm)", "Secure rate (kbit/s)",
                           "λ-scaled scattering model (b ∝ λ^-1.14, absorption fixed)")
        st.pyplot(fig_wl, width="stretch")
        st.caption("Absorption a is held at its 532 nm value — the clear-ocean blue advantage is NOT captured. "
                   "Treat the ordering, not the absolute gain, as indicative.")
    with div_col:
        st.markdown("#### Secure rate vs divergence (1–30 mrad)")
        divs = list(range(1, 31, 2))
        series = {}
        for name in (["None"] + ([ci.nanophotonic_structure] if ci.nanophotonic_structure != "None" else ["Dielectric metasurface"])):
            rates = []
            for d_ in divs:
                ci_d = _dc.replace(ci, divergence_mrad=float(d_),
                                   nanophotonic_structure=name, nanophotonic_eta_ext=None)
                ev_d = evaluate_qkd(evaluate_channel(ci_d), cfg)
                rates.append(max(ev_d["secure_rate_bps"], 0.0) / 1e3)
            series[f"{name}"] = rates
        fig_div = fig_lines(divs, series, "Full-angle divergence (mrad)",
                            "Secure rate (kbit/s)", "Beam divergence vs secure rate")
        st.pyplot(fig_div, width="stretch")
        st.caption("Collimating the beam raises the geometric capture term quadratically (until diffraction/aperture limits).")

    qd_col = st.columns(1)[0]
    with qd_col:
        st.markdown("#### QD emitter temperature sweep (4–150 °C)")
        temps = list(range(4, 151, 10))
        t_rates = []
        for T in temps:
            src_t = evaluate_source("QD", wavelength_nm=float(wavelength), quantum_yield=qd_yield,
                                    linewidth_nm=qd_lw, g2_zero=qd_g2, blinking_factor=qd_blink,
                                    temp_c=float(T), filter_bw_nm=qd_bw, rep_rate_hz=float(clock_hz))
            cfg_t = _dc.replace(cfg, source_eff=src_t["source_efficiency"],
                                p_multi=src_t["multi_photon_prob"],
                                qd_quality=src_t["qd_quality_score"])
            ev_t = evaluate_qkd(ch, cfg_t)   # channel unchanged; source-side only
            t_rates.append(max(ev_t["secure_rate_bps"], 0.0) / 1e3)
        fig_t = fig_lines(temps, {"QD secure rate (kbit/s)": t_rates},
                          "Emitter temperature (°C)", "Secure rate (kbit/s)",
                          "Thermal penalty e^(−0.008·max(0,T−20))")
        st.pyplot(fig_t, width="stretch")
        st.caption("Applies only when Source = Quantum-dot-inspired; with WCP selected the sweep is illustrative of the QD branch.") if not qd_on else None

    st.markdown("#### Circuit-level cross-check — BB84 on a quantum simulator")
    st.caption("Ported from the digital-twin notebook. Prepares |0⟩/|+⟩ states, applies an Ry error rotation "
               "(sin²(θ/2) = e_opt), optionally inserts an Eve intercept–resend, measures. Not experimental validation.")
    cc1, cc2, cc3 = st.columns(3)
    n_circ = cc1.selectbox("Circuits", [200, 600, 1200, 2400], index=1)
    eve_flag = cc2.checkbox("Include Eve intercept–resend", value=(cfg.eve_fraction > 0))
    e_opt_cfg = cfg.e_intrinsic if not cfg.decomposed_error else ev["e_decomposition"]["e_intrinsic_decomposed"]
    cc3_run = cc3.button("▶ Run circuit check", type="primary")
    if cc3_run:
        with st.spinner(f"Running {n_circ} circuits on Aer/statevector…"):
            pref = qiskit_qber(n_circ, e_opt_cfg, 0.01, eve_flag, int(seed))
            if pref.get("engine") == "qiskit-unavailable" or "error" in pref:
                pref = statevector_qber(n_circ, e_opt_cfg, 0.01, eve_flag, int(seed))
        c1_, c2_, c3_ = st.columns(3)
        c1_.metric("Engine", pref["engine"])
        c2_.metric("Sifted bits", f"{pref['sifted_bits']}")
        c3_.metric("QBER (circuit)", f"{pref['qber_percent']:.2f} %",
                   f"analytic e_eff {100 * e_opt_cfg:.2f} %")

    if ev.get("e_decomposition"):
        st.markdown("#### Decomposed intrinsic error breakdown")
        dec = ev["e_decomposition"]
        fig_dec, ax = plt.subplots(figsize=(6.6, 2.6))
        labels_ = ["Polarisation (nano)", "Misalignment", "Depolarisation", "Timing jitter"]
        vals_ = [100 * dec[k] for k in ["pol_error", "e_misalign_decomposed", "e_depolarisation", "timing_jitter"]]
        ax.barh(labels_, vals_, color=[TEAL, GOLD, CYAN, GREY])
        ax.set_xlabel("Error probability (%)")
        ax.invert_yaxis(); ax.grid(axis="x", ls=":", alpha=0.4)
        ax.set_title(f"e_opt = {100 * dec['e_intrinsic_decomposed']:.2f} % (sum before Eve)")
        fig_dec.tight_layout()
        st.pyplot(fig_dec, width="stretch")

    st.markdown("#### Physics-based background readout")
    if ci.background_model == "physics":
        st.markdown(f"""
- Apparent irradiance class **{ci.background_physics_level}**: flux Φ = {BACKGROUND_FLUX[ci.background_physics_level]:.1e} ph/s/m²/sr/nm
- R_bg = **{ch['background_cps']:.3e} cps** → n_bg per {ci.gate_window_ns:.1f} ns gate = **{ch['bg_photons_per_gate']:.2e}** photons
- Dark counts per gate = {ch['dark_counts_per_gate']:.2e} → **Y₀ = {ch['y0']:.3e}**
- Reference preset Y₀ (Night) = {1 - math.exp(-(100.0 / ci.clock_hz + 0.0)):.2e} — presets and physics mode are NOT interchangeable
""")
    else:
        st.markdown(f"Preset mode (v1.0): Y₀ = **{ch['y0']:.2e}** at clock {ci.clock_hz/1e6:.0f} MHz "
                    f"({BACKGROUND_LEVELS[background]:.0e} cps background). Switch to physics mode in the Channel panel to compare.")

# ================= Sweeps =================
with tab_sweep:
    st.subheader("How the link behaves across its operating envelope")
    c1, c2 = st.columns(2)
    with c1:
        fig1 = fig_rate_qber_vs_distance(ci, cfg)
        st.pyplot(fig1, width="stretch")
        st.download_button("Download PNG", fig_to_png_bytes(fig1), "rate_vs_distance.png", "image/png")
    with c2:
        fig2 = fig_rate_vs_ntu(ci, cfg)
        st.pyplot(fig2, width="stretch")
        st.download_button("Download PNG", fig_to_png_bytes(fig2), "rate_vs_ntu.png", "image/png")

    c3, c4 = st.columns(2)
    with c3:
        fig3 = fig_mu_sweep(ci, cfg)
        st.pyplot(fig3, width="stretch")
        st.download_button("Download PNG", fig_to_png_bytes(fig3), "rate_vs_mu.png", "image/png")
        st.caption(f"Grid optimum at current distance & water: μ ≈ {opt_mu:.2f}")
    with c4:
        st.subheader("Turbulence (optional overlay)")
        st.caption("Log-normal fading of η_Beer–Lambert under weak turbulence. "
                   "Reports the outage probability: P(secure rate = 0).")
        cn2 = st.selectbox("Turbulence strength Cₙ² (m⁻²ᐟ³)",
                           [1e-13, 1e-12, 1e-11], index=1,
                           format_func=lambda v: f"{v:.0e}")
        n_fades = st.select_slider("Fade samples", [200, 500, 1000, 2000], value=500)
        if st.button("Run fade analysis"):
            k = 2 * np.pi / (ci.wavelength_nm * 1e-9)
            L = ci.distance_m
            rytov = 1.23 * cn2 * k ** (7 / 6) * L ** (11 / 6)
            sigma_ln = math.sqrt(max(rytov, 1e-12))     # log-amplitude std (weak Rytov)
            rng = np.random.default_rng(int(seed))
            gains_ = rng.lognormal(mean=math.log(max(ch["eta_beer"], 1e-12)) - sigma_ln**2,
                                   sigma=2 * sigma_ln, size=n_fades)
            outage = 0
            rates = []
            for g in gains_:
                ci2 = dataclasses.replace(ci)
                ci2 = dataclasses.replace(ci2, k_ntu=ci.k_ntu)  # unchanged; fade applied via eta directly below
                eta_tot_fade = g * ch["eta_geo"] * ch["eta_mis"] * ci.detector_efficiency
                ch2 = dict(ch); ch2["eta_total"] = eta_tot_fade
                r_ = evaluate_qkd(ch2, cfg)
                rates.append(r_["secure_rate_bps"] / 1e3)
                outage += (r_["secure_rate_bps"] <= 0)
            rates = np.array(rates)
            st.metric("Rytov variance σ_R²", f"{rytov:.2e}")
            st.metric("Outage probability (P secure = 0)", f"{outage / n_fades * 100:.1f} %")
            st.metric("Faded secure rate — median", f"{np.median(rates):.2f} kbit/s")
            st.caption("Overlay only — baseline QBER/rates ignore turbulence. Log-amplitude log-normal "
                       "approximation; not a validated ocean turbulence model.")

# ================= Max distance & settings =================
with tab_max:
    m1, m2 = st.columns([1, 1.25])
    with m1:
        st.subheader("Maximum secure communication distance")
        if l_max:
            st.markdown(f'<div class="kpi"><div class="lab">L<sub>max</sub> (current settings)</div>'
                        f'<div class="val">{l_max:.2f} m</div>'
                        f'<div class="sub">Largest L with QBER &lt; {100*QBER_SECURE_MAX:.0f} % AND R &gt; 0</div></div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="kpi"><div class="lab">L<sub>max</sub></div>'
                        '<div class="val">&lt; 1 m</div>'
                        '<div class="sub">No secure operating point at any distance 1–20 m</div></div>',
                        unsafe_allow_html=True)
        st.write("")
        st.markdown(f"""
**Operating point**
- Current: L = {distance_m:.1f} m · μ = {mu:.2f} · {background}
- At L_max the QBER approaches the 11 % ceiling; in practice operate at **≤ {max(0.0, (l_max or 0) * 0.75):.1f} m** (¾ of L_max) for margin.
- Optimal pump intensity (grid search): **μ*\u2009≈ {opt_mu:.2f}** → {opt_r*ch['clock_hz']/1e3:.2f} kbit/s.
""")
        st.subheader("Recommended protocol")
        st.info("**Decoy-state BB84** for any weak-coherent-pulse source (closes the PNS attack). "
                "Idealised BB84 is shown only as a benchmark — do not quote it as achievable security.")
    with m2:
        st.subheader("Recommended operating settings")
        icon = {"critical": "🛑", "high": "⚠️", "medium": "🔧", "info": "✅"}
        for r in recs:
            st.markdown(
                f'<div class="rec {r["priority"]}"><span class="tag {r["priority"]}">'
                f'{icon[r["priority"]]} {r["priority"]} · {r["area"]}</span><br>'
                f'<b>Finding.</b> {r["finding"]}<br>'
                f'<b>Action.</b> {r["action"]}<br>'
                f'<b>Expected effect.</b> {r["expected"]}</div>',
                unsafe_allow_html=True)

# ================= Monte Carlo =================
with tab_mc:
    st.subheader("Independent stochastic cross-check (NumPy, vectorised)")
    st.caption("Vectorised BB84 Monte Carlo vs. the analytic model. "
               "Sampling noise σ(QBER) ≈ √(p(1−p)/N_sifted); agreement within a few σ validates the pipeline. "
               "This is a software-level check, not experimental validation.")
    mc1, mc2, mc3 = st.columns(3)
    n_pulses = mc1.selectbox("Pulses simulated", [100_000, 250_000, 500_000, 1_000_000], index=1)
    mc_seed = mc2.number_input("Seed", 0, 99999, int(seed), 1)
    run = mc3.button("▶ Run Monte Carlo", type="primary")
    if "mc_history" not in st.session_state:
        st.session_state.mc_history = []
    if run:
        with st.spinner(f"Simulating {n_pulses:,} pulses…"):
            mc_res = run_bb84_mc(n_pulses, cfg.mu, ch["eta_total"], ch["y0"],
                                 ev["e_d_effective"], cfg.eve_fraction, int(mc_seed))
            st.session_state.mc_history.insert(0, mc_res)

    if st.session_state.mc_history:
        res = st.session_state.mc_history[0]
        a, b, c_ = st.columns(3)
        a.metric("Detection yield (MC)", f"{res['detection_yield']:.3e}",
                 f"analytic {ev['q_mu']:.3e}")
        b.metric("QBER (MC)", f"{res['qber_percent']:.2f} % ± {100*res['qber_stderr']:.2f}",
                 f"analytic {ev['qber_percent']:.2f} %")
        c_.metric("Sifted bits", f"{res['sifted_bits']:,}",
                  f"errors {res['errors']:,}")
        fig, ax = plt.subplots(figsize=(7.4, 3.2))
        labels = ["Detection yield (×10³)", "QBER (%)", "Sifted / pulse (×10³)"]
        analytic = [1e3 * ev["q_mu"], ev["qber_percent"], 1e3 * ev["sifted_bits_per_pulse"]]
        mc = [1e3 * res["detection_yield"], res["qber_percent"], 1e3 * res["sifted_fraction"]]
        x = np.arange(3)
        ax.bar(x - 0.2, analytic, 0.36, label="Analytic", color=TEAL)
        ax.bar(x + 0.2, mc, 0.36, label="Monte Carlo", color=GOLD)
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_title("Analytic vs. Monte Carlo")
        ax.legend(frameon=False); ax.grid(axis="y", ls=":", alpha=.4)
        fig.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)
        st.subheader("Run history")
        st.dataframe(st.session_state.mc_history, width="stretch")

# ================= Security & Eve =================
with tab_eve:
    st.subheader("Intercept–resend eavesdropping")
    st.markdown("""
In an intercept–resend attack Eve measures each intercepted pulse in a randomly chosen basis and resends
the result. Alice and Bob keep only matched bases, but Eve's resend randomises one bit in four:
she introduces **+25 % error per intercepted pulse** → effective QBER `e_d + 0.25·f_eve`.
The sifted key must then be considered **compromised**; discard it.
""")
    c1, c2 = st.columns(2)
    with c1:
        fig = fig_eve_analysis(ch, cfg)
        st.pyplot(fig, width="stretch")
        st.download_button("Download PNG", fig_to_png_bytes(fig), "eve_qber.png", "image/png")
    with c2:
        fig = fig_secret_fraction()
        st.pyplot(fig, width="stretch")
        st.download_button("Download PNG", fig_to_png_bytes(fig), "secret_fraction.png", "image/png")
    st.subheader("What AquaSecure does and does NOT cover")
    na, nb = st.columns(2)
    with na:
        st.markdown("**Covered**\n- Intercept–resend (fraction adjustable)\n- Optical loss & noise (Beer–Lambert, NTU scattering, coupling, pointing)\n- Detector dark counts & efficiency\n- Weak-decoy parameter estimation (simplified)\n- QBER-threshold security verdict")
    with nb:
        st.markdown("**Not covered (see Methodology)**\n- Collective/coherent general attacks beyond the simplified decoy bound\n- Finite-key effects; no composable ε-security\n- PNS attack against plain BB84 (numerical only)\n- Side channels: detector blinding, Trojan-horse, timing\n- Turbulence in the baseline verdict (optional overlay only)")

# ================= Live Secure Chat (Alice <-> Bob) =================
with tab_chat:
    st.subheader("Live Secure Chat — Alice ↔ Bob (simulated BB84 key + AES-256-GCM)")
    st.caption("Separate educational bit-exchange simulation parameterised by this run's verified "
               "QBER, click probabilities and verdict. Idealised reconciliation; Toeplitz privacy "
               "amplification is illustrative, not a finite-key proof.")
    # scenario fingerprint: a chat key is only valid for the scenario that created it
    import hashlib as _hl, json as _json
    _fp_src = _json.dumps({"ci": ci.__dict__, "cfg": cfg.as_dict(),
                           "status": ev["security_status"], "qber": round(ev["qber_percent"], 4)}, sort_keys=True, default=str)
    _scenario_fp = _hl.sha256(_fp_src.encode()).hexdigest()[:12]
    _stored_fp = st.session_state.get("chat_scenario_fp")
    if st.session_state.get("chat_key") and _stored_fp != _scenario_fp:
        st.warning("Scenario changed since the chat key was generated — the old key is stale. Generate a new key for this scenario.")
    if ev["security_status"] != "Secure":
        st.error(f"Current scenario is {ev['security_status']} (QBER {ev['qber_percent']:.2f}%). "
                 "Chat is blocked: fix the link first (shorter distance, lower NTU, night, Eve off), then generate a key.")
    if st.button("🔑 Generate / rotate simulated BB84 chat key", type="primary"):
        try:
            _stats = generate_chat_key(ch, ev, cfg)
            st.session_state["chat_key"] = _stats["key"]
            st.session_state["chat_key_stats"] = {k: v for k, v in _stats.items() if k != "key"}
            st.session_state["chat_scenario_fp"] = _scenario_fp
            st.session_state["chat_messages"] = []
            st.success("Simulated shared key established. Alice and Bob can now exchange encrypted messages.")
        except Exception as _e:
            st.session_state.pop("chat_key", None)
            st.error(str(_e))
    _key = st.session_state.get("chat_key")
    _stats = st.session_state.get("chat_key_stats")
    if _key and _stats and _stored_fp == _scenario_fp:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Key status", "Established"); k2.metric("AES key", "256-bit")
        k3.metric("Sifted detections", f"{_stats['sifted']:,}"); k4.metric("Sampled QBER", f"{100*_stats['qber_sample']:.2f} %")
        st.caption(f"Key fingerprint `{_stats['key_fingerprint']}` • 256-bit Toeplitz privacy amplification (educational)")
        with st.expander("How this key was made (step-by-step)"):
            st.write({
                "1. Pulses simulated": _stats["n_pulses"],
                "2. Bob detections": _stats["detected"],
                "3. After basis sifting": _stats["sifted"],
                "4. Bits sacrificed for QBER sample": _stats["sampled"],
                "5. Key bits before privacy amplification": _stats["key_bits_before_pa"],
                "6. Estimated error-correction leakage (bits)": _stats["estimated_ec_leakage_bits"],
                "7. Estimated secure bits remaining": _stats["estimated_secure_bits"],
            })
        c1, c2, c3 = st.columns(3)
        _loss = c1.slider("Packet loss (%)", 0.0, 30.0, 0.0, 0.5) / 100.0
        _corr = c2.slider("Packet corruption (%)", 0.0, 10.0, 0.0, 0.1) / 100.0
        _lat = c3.slider("Simulated latency (ms)", 1, 500, 40, 1)
        if st.button("🛡️ Run AES-GCM tampering tests"):
            st.session_state["chat_integrity"] = aes_gcm_integrity_tests(_key)
        if st.session_state.get("chat_integrity"):
            st.dataframe(pd.DataFrame(st.session_state["chat_integrity"]), width="stretch", hide_index=True)
        for _m in st.session_state.get("chat_messages", []):
            with st.chat_message("user" if _m["sender"] == "Alice" else "assistant",
                                 avatar="👩\u200d💻" if _m["sender"] == "Alice" else "🧑\u200d💻"):
                st.caption(f"{_m['sender']} → {_m['receiver']} · packet {_m['packet_id']}")
                if _m["delivered"]:
                    st.write(_m["received_text"]); st.caption(f"{_m['status']} · {_m['simulated_latency_ms']} ms")
                else:
                    st.error(_m["status"])
                with st.expander("Encrypted packet details"):
                    st.code(_m["ciphertext_and_tag_b64"]); st.caption(f"Nonce {_m['nonce_b64']} · AES-GCM tag appended")
        _sender = st.selectbox("Send message as", ["Alice", "Bob"])
        _msg = st.chat_input("Write a message to the other endpoint…")
        if _msg:
            st.session_state.setdefault("chat_messages", []).append(
                encrypt_chat_packet(_msg, _sender, _key, _loss, _corr, _lat))
            st.rerun()
        if st.session_state.get("chat_messages"):
            st.download_button("Download encrypted packet log (JSON)", packet_log_json(st.session_state["chat_messages"]),
                               "aquasecure_encrypted_chat_packets.json", "application/json")
            if st.button("Clear chat history"):
                st.session_state["chat_messages"] = []; st.rerun()
    elif ev["security_status"] == "Secure":
        st.info("No chat key yet for this scenario. Generate one above to start messaging.")

# ================= Export =================
with tab_export:
    st.subheader("Export the current scenario")
    snap = scenario_snapshot(ch, cfg, ev, l_max)
    st.markdown("**One-page PDF report** — scenario, key results, recommendations, claim boundary:")
    import tempfile, os
    from aquasecure.figures import fig_transmittance_waterfall as _fw, fig_mu_sweep as _fm
    with st.spinner("Rendering PDF…"):
        _tmpdir = tempfile.mkdtemp(prefix="aquasecure_")
        _fp = []
        try:
            _fig = _fw(ch); _p = os.path.join(_tmpdir, "budget.png"); _fig.savefig(_p, dpi=110); plt.close(_fig); _fp.append(_p)
        except Exception:
            pass
        try:
            _fig = _fm(ci, cfg); _p = os.path.join(_tmpdir, "mu.png"); _fig.savefig(_p, dpi=110); plt.close(_fig); _fp.append(_p)
        except Exception:
            pass
        _pdf = build_pdf(snap, recs, _fp or None)
        st.download_button("📄 Download PDF report", _pdf, "aquasecure_report.pdf", "application/pdf", type="primary")
    c1, c2, c3 = st.columns(3)
    c1.download_button("⬇️ Results (CSV)", snapshot_csv(snap),
                       "aquasecure_scenario.csv", "text/csv", width="stretch")
    c2.download_button("⬇️ Scenario (JSON)", snapshot_json(snap),
                       "aquasecure_scenario.json", "application/json", width="stretch")
    c3.download_button("⬇️ Parameters (LaTeX row)", snapshot_latex_row(snap),
                       "aquasecure_row.tex", "text/plain", width="stretch")
    st.subheader("LaTeX table (paste-ready)")
    st.code(LATEX_TABLE_PREAMBLE + "\n% Example row for current settings:\n" + snapshot_latex_row(snap),
            language="latex")
    st.subheader("Scenario snapshot")
    st.json(snap)

# ================= Methodology =================
with tab_method:
    st.subheader("Physical model")
    m1, m2 = st.columns(2)
    with m1:
        st.markdown("**Channel**")
        st.latex(r"\eta_{BL} = 10^{-(a + b + k\cdot NTU)\,L / 10}")
        st.latex(r"w(L) = w_0 + L\tan(\theta/2), \qquad \eta_{geo} = \min\!\left[1, \left(\frac{r_{rx}}{w(L)}\right)^2\right]")
        st.latex(r"\eta_{mis} = \exp\!\left[-2\left(\frac{L\tan\varphi}{w(L)}\right)^2\right]")
        st.latex(r"\eta_{ch} = \eta_{BL}\,\eta_{geo}\,\eta_{mis}, \qquad \eta = \eta_{ch}\,\eta_{det}")
        st.latex(r"Y_0 = 1 - e^{-(R_{dc} + R_{bg})/f_{clock}}")
    with m2:
        st.markdown("**Protocol (asymptotic)**")
        st.latex(r"Q_\mu = Y_0 + 1 - e^{-\eta\mu}, \qquad E_\mu = \frac{e_d\,\eta\mu + \tfrac12 Y_0}{Q_\mu}")
        st.latex(r"R_{BB84} = q\,Q_\mu\left[1 - H_2(E_\mu) - f\,H_2(E_\mu)\right]")
        st.latex(r"Y_1^{L} = \frac{Q_\mu - Y_0 - \frac{\mu^2}{\nu^2} e^{\nu-\mu}(Q_\nu - Y_0)}{\mu e^{-\mu}\left(1 - \mu/\nu\right)},\quad R_{decoy} = q\left[Q_1^L(1-H_2(e_1^U)) - Q_\mu f H_2(E_\mu)\right]")
        st.latex(r"H_2(x) = -x\log_2 x - (1-x)\log_2(1-x), \qquad q = 1/2")

    st.subheader("Water-type parameter table (532 nm)")
    st.dataframe([
        {"Water type": k, "a (m⁻¹)": v["absorption"], "b (m⁻¹)": v["scattering"],
         "Typical NTU": v["typical_ntu"]} for k, v in WATER_TYPES.items()
    ], width="stretch", hide_index=True)
    st.caption("Literature-typical values in the style of Mobley, Light and Water (1994) and UOWC reviews. "
               "Verify against the primary source and state the convention before submission.")

    st.subheader("Security status thresholds")
    st.markdown("- **Secure**: QBER < 8 % **and** secure-key rate > 0 with margin\n"
                "- **Marginal**: 8 % ≤ QBER < 11 % **or** very small positive rate\n"
                "- **Insecure**: QBER ≥ 11 % (one-way BB84 limit) **or** non-positive rate — discard the key")

    st.subheader("Source & transmitter models (v1.1)")
    st.markdown("**Quantum-dot-inspired source** (opt-in): spectral efficiency \\(\\min\\!\\left(1, \\frac{BW}{0.85\\,\\lambda_{lw}}\\right)\\); emitted rate = clock × η_src; multi-photon \\(p_{multi} = \\min(\\tfrac12, \\tfrac12 \\eta_{src}^2 g^{(2)}(0))\\). QD rate uses a source-aware GLLP form:")
    st.latex(r"\eta_{src} = q_y\, B\, e^{-0.008\,\max(0,\,T-20)} \cdot \min\!\left(1, \frac{BW}{0.85\,\lambda_{lw}}\right)")
    st.latex(r"R_{QD} = q\left[\eta_{src}\,\eta\,(1 - p_{multi})\,(1 - H_2(E)) - \,Q_\mu\, f H_2(E)\right]")
    st.markdown("**Nanophotonic transmitter** (dielectric metasurface · plasmonic antenna · nanowire): net extraction η_ext·(1 − immersion loss); divergence × div_factor (metasurface 0.28); PER injects 1/(1+10^(PER/10)) polarisation error into the decomposed model; adds a nanophotonic dB line to the budget.")
    st.markdown("**Physics-based background** (opt-in): per-gate yield n_bg = R_bg·t_gate, with")
    st.latex(r"R_{bg} = \Phi \cdot \pi r_{rx}^2 \cdot \pi\,FOV^2 \cdot BW \cdot e^{-a \min(L,\,5)}, \qquad Y_0 = 1 - e^{-(R_{dc}+R_{bg})\,t_{gate}}")
    st.markdown("**Wavelength scaling**: b(λ) = b₅₃₂·(532/λ)^1.14; NTU adder ∝ (532/λ)^0.8; absorption held constant (clear-ocean blue advantage NOT modelled). Reference 532 nm reproduces v1.0 bit-identically.")
    st.markdown("**Decomposed intrinsic error** (opt-in): e_opt = pol_error(nano) + ½sin²(1.8·φ) + ½(1 − e^(−ρ(1+0.02·NTU)L)) + 0.004, ρ = 0.002 (clear) … 0.035 (turbid).")
    st.markdown("Simulation core and these models are ported from the author's earlier **AquaSecure digital-twin** Colab notebook with attribution; Qiskit Aer is used when installed, otherwise an identical NumPy statevector circuit.")

    st.subheader("Claim boundaries (read before publishing)")
    st.warning("""
AquaSecure is a **computational simulation framework**. It does **not** claim: a new QKD protocol; an experimentally
validated link; composable finite-key security (all results are asymptotic); security against arbitrary attacks (only
intercept–resend is modelled); or a globally optimal adaptive controller (recommendations are heuristic rules).
Decoy-state estimation is the simplified two-decoy (vacuum + weak) closed form, truncated after two-photon terms.
Turbidity is folded in as a linear scattering adder k·NTU — a defensible engineering simplification, not a full
oceanographic model. Any journal manuscript should repeat these boundaries verbatim in its Limitations section.
""")

    st.subheader("Open research items (from the comparative analysis)")
    st.markdown("""
1. Finite-key statistical bounds with explicit ε-composable security parameters
2. Time-varying channel (dynamic turbidity / pointing jitter) and controller evaluation under disturbance
3. Fair benchmark of rule-based control vs. GA / PSO / Bayesian optimisation under identical budgets
4. Validation against published field data or a testbed; larger-sample Monte Carlo; Qiskit Aer cross-check
5. Sensitivity/uncertainty analysis to identify dominant loss & error terms per water type
""")

    st.subheader("Key assumptions")
    st.markdown("""
- Wavelength fixed at 532 nm (green window); blue (≈480 nm) may suit clear ocean better — candidate extension.
- q = 1/2 sifting; one-way classical post-processing; collective attacks; asymptotic limit.
- Plain BB84 rate assumes an idealised single-photon source (upper bound for WCP without decoys).
- Error-correction efficiency f = 1.16 default; intrinsic error 1.5 % default.
- Background cps by lighting class; Y₀ = 1 − exp(−(R_dc + R_bg)/f_clock).
- Turbulence modelled (optionally) as log-normal fading; not part of the baseline verdict.
""")

st.markdown("---")
st.markdown(f'<p class="small">AquaSecure v1.1 · {YEAR} · Simulation-only; no hardware validation claimed. '
            f'Literature leads in the research-gap document must be verified against publishers before citation.</p>',
            unsafe_allow_html=True)
