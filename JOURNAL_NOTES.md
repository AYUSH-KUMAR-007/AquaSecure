# AquaSecure — journal companion notes

Working notes mapping the proposed app to the research-gap document
(`AquaSecure_Comparative_Analysis_Research_Gaps.pdf`) and to candidate claims.
**Not for submission as-is.** Verify every literature lead against the publisher
before citing; the PDF itself flags this.

## Contribution framing (honest)

AquaSecure is the **integrated comparative evaluation** of source, channel and
security parameters under common simulated conditions, *extended* by this version
with (i) an explicit turbidity-scattering term (k·NTU), (ii) selectable background
light levels, (iii) decoy-state BB84 parameter estimation (simplified), (iv) an
intercept–resend Eve model with adjustable interception fraction, (v) maximum
secure distance, (vi) rule-based operating recommendations, and (vii) seeded
Monte Carlo cross-checking.

It must NOT be framed as "the first underwater QKD simulator" or "the first
integrated source–channel–protocol model" — the gap document explicitly rejects
those claims. Frame it as a reproducible, open framework addressing the listed gaps.

## Gap → feature mapping

| Research gap (from the comparative analysis) | AquaSecure (this build) | Residual gap |
|---|---|---|
| More complete source-security treatment (decoy-state) | Simplified vacuum + weak-decoy Y1ᴸ estimation, Q1ˡ, e1ᵁ bounds | Full parameter estimation w/ vacuum + 2 decoys, finite key, optimised (μ, ν) |
| Security-aware adaptation rather than signal-only optimisation | Rule-based engine ranks actions by security impact; L_max computed; optimal μ grid | GA/PSO/BO comparison under identical budgets; RL longer-term |
| Joint source imperfections + underwater impairments | g²/blinking source model NOT ported — note as limitation | Port quantum-yield / blinking source model; sensitivity analysis |
| Dynamic underwater channel conditions | Optional log-normal turbulence overlay + outage probability | Time-varying turbidity/pointing; controller evaluation under disturbance |
| Validation against independent evidence | Seeded NumPy MC vs analytic; PNG/CSV export for paper figures | Published field datasets; testbed; larger samples; Qiskit Aer cross-check |
| Fair comparison of adaptive algorithms | Single baseline (rule-based) only | Implement GA/PSO/BO with shared objective, constraints, seeds, budget |
| Finite-key security | Not implemented (asymptotic only) | ε-composable finite-key bounds before any strong security claim |

## Candidate paper structure

1. Introduction — UQKD motivation; turbid-water challenge; why security-aware simulation
2. Related work — underwater BB84, decoy-state UQKD, UOWC channel models, adaptive optics (verify leads)
3. System & channel model — equations as in README; water table; assumptions
4. Security model — QBER, BB84/decoy rates, intercept–resend, thresholds
5. Implementation — the Streamlit framework; MC cross-check; export/reproducibility
6. Results — case matrix: {clear, coastal, harbor, turbid} × {1..20 m} × NTU sweep × lighting × Eve on/off
   - secure-rate maps (distance × NTU) per water type  ← headline figure
   - L_max vs NTU
   - Eve interception fraction vs QBER
   - analytic vs MC
7. Discussion — heuristic recommendations; k·NTU validity; noise dominance transitions
8. Limitations & future work — the claim boundaries (verbatim) + items 1–5 above
9. Conclusion

## Candidate venues (verify scope & recent UQKD/QKD simulation papers)

IEEE Access · Applied Optics · Optics Communications · Quantum Engineering (Wiley) ·
Journal of Optical Communications and Networking (JOCN) · Optics Express.
Check each journal's recent UQKD/underwater-QKD publications and author guidelines.

## Numbers worth generating for the results section

- η_channel, QBER, R_sec vs distance for each water type at NTU ∈ {0, 10, 25, 50, 100}
- L_max(NTU) curves per water type, day/night
- "Security margin" plot: QBER budget decomposition (e_d vs dark vs background vs Eve)
- Outage probability vs Rytov variance

## Before submission checklist

- [ ] Verify every cited reference against the publisher page (the gap PDF's leads are unverified)
- [ ] State the a/b parameter convention and wavelength explicitly
- [ ] Repeat claim boundaries in Limitations
- [ ] Disclose that the tool is simulation-only; no hardware validation
- [ ] Report all RNG seeds, toolkit versions, and the exact k·NTU assumption

## v1.1 port log (2026-11-22)

Ported from `uploads/aquasequre_qkd_digital_twin.py` (educational notebook
`underwater_bb84_qkd_colab.py` reviewed, not used as source):
new modules `source.py`, `nanophotonics.py`, `background.py`, `circuit.py`,
`report_pdf.py`; extended `channel.py` (λ scaling, physics background, nano
stage), `qkd.py` (QD/GLLP branch, decomposed e_d), `figures.py` (`fig_lines`,
waterfall nano line), `recommend.py` (4 opt-in rules), `export.py` (46-key
snapshot). New UI: sidebar sections 6–7, Physics-lab tab, PDF export.
Verified: default AppTest run QBER 1.87 %, +24.77 kbit/s (v1.0 identical);
all-v1.1-on run clean; Qiskit-Aer and NumPy circuit engines agree exactly at
600 circuits (2.24 % no-Eve, 25.37 % Eve); PDF renders Greek glyphs via DejaVu.
NOT ported: notebook `binary_entropy` bug, MC shot-noise blending,
ipywidgets/google.colab/`!pip` cells. Section 9 report untouched.
