# AquaSecure — Simulation Framework for Security-Aware Underwater QKD in Turbid Water

Interactive Streamlit framework that simulates an underwater BB84 / decoy-state
quantum key distribution (QKD) link and instantly shows whether the link is
**Secure, Marginal or Insecure** under different water types, distance,
turbidity, background light, detector settings, pointing error and
intercept–resend eavesdropping.

> **Simulation only.** No hardware is connected and no experimental result is
> claimed. Rates are asymptotic and simplified (collective attacks, one-way
> error correction). Water coefficients are literature-typical values at
> 532 nm — verify against primary sources before citing them.

## Run locally

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Features

- Sidebar controls: water type, distance (1–20 m), turbidity (0–100 NTU),
  background light, detector efficiency, dark counts, pointing misalignment,
  beam optics, protocol (decoy-state BB84 / idealised BB84), Eve fraction.
- Outputs: channel/total transmittance (dB loss-budget waterfall), received
  photon rate, QBER vs the 11% threshold, sifted- and secure-key rates,
  security status, maximum secure distance Lmax, recommended settings.
- Sweeps and heatmaps (distance × turbidity), Monte Carlo cross-check
  (seeded NumPy BB84), Eve analysis, and CSV / JSON / LaTeX / PDF export.
- **Live Secure Chat (Alice <-> Bob):** when the scenario verdict is Secure,
  a simulated BB84 bit exchange (sifting, QBER sample, idealised
  reconciliation, 256-bit Toeplitz privacy amplification) derives an
  AES-256-GCM key; Alice/Bob then exchange encrypted messages with packet
  loss/corruption simulation, deterministic AES-GCM tamper tests and an
  encrypted packet-log export. Chat is blocked for Marginal/Insecure
  scenarios. Educational demonstration - not finite-key QKD.
- **v1.1 Physics Lab (opt-in):** wavelength scaling, physics-based background
  radiance, quantum-dot source (GLLP rate branch), nanophotonic transmitter
  structures, decomposed intrinsic error, and a circuit-level BB84 check
  (Qiskit Aer when installed, pure-NumPy statevector fallback otherwise).
  All v1.1 models are opt-in; default settings reproduce v1.0 results
  bit-identically.

## Project report

`report/` contains the Section 9 chapter *Testing, Results and Discussion*
(Times New Roman, 12 pt justified, template format 9.1–9.5) with labelled
tables and figures, plus the scripts that generate every number and graph:

```bash
python report/generate_results.py   # writes report/results.json + figures
python report/build_report_v3.py    # rebuilds the .docx chapter
```

Headline verified result (baseline: coastal ocean, 5 m, 20 NTU, night):
attenuation 1.398 m⁻¹, total loss 15.26 dB, QBER 1.869%,
secure-key rate 24.77 kbit/s (Secure); Monte Carlo agrees within
0.049 percentage points of QBER over 2.5 × 10⁶ pulses.

## Layout

```
app.py                    Streamlit UI
aquasecure/               Physics & QKD engine (channel, qkd, mc, source,
                          nanophotonics, background, circuit, export, ...)
report/                   Section 9 report (.docx), results.json, figures,
                          and the scripts that generate them
requirements.txt          Python dependencies (qiskit optional)
```

## Claim boundaries

- Not a new protocol, not an experiment, not a composable finite-key proof.
- Idealised BB84 assumes a perfect single-photon source (upper bound);
  decoy-state results are the defensible headline numbers.
- Intercept–resend is the only modelled attack; no side channels.
- Negative secure rates mean "abort — no key" and are reported raw.
