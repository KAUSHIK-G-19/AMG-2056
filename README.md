# AMG-2056 — Autonomous Micro-Grid Governor

> **Neighbourhood Sub-Distributor Decentralised Feeder Control System**  
> Real-time renewable intermittency management using Physics-Informed Neural Networks (PINN) and Hamiltonian Neural Networks (HNN) at the transformer-secondary edge node.

---

## Overview

The **Autonomous Micro-Grid Governor (AMG-2056)** is an interactive operational dashboard prototype for a decentralised feeder-level energy management unit. It simulates the monitoring and autonomous control of a 100 kVA neighbourhood distribution transformer that serves residential and small commercial loads supplied in part by rooftop photovoltaic generation and a community Battery Energy Storage System (BESS).

The system is designed to operate fully **offline and edge-first**, maintaining grid stability and voltage regulation without a continuous SCADA uplink. When the substation radio link is lost, the embedded Hamiltonian Neural Network governor assumes full autonomous droop control.

---

## Problem Statement

Modern low-voltage distribution feeders are increasingly subject to **bidirectional power flow**, **solar intermittency**, and **unexpected demand surges** (community events, EV charging clusters). Traditional centralised SCADA systems cannot respond fast enough at the feeder edge, particularly under communication blackouts.

AMG-2056 addresses this by embedding a lightweight **Physics-Informed Neural Network (PINN)** for 5-minute-horizon load and solar forecasting, combined with a **Hamiltonian Neural Network (HNN)** that enforces the system's energy conservation law as a hard physical constraint:

```
dH/dt = (∂H/∂q)·q̇ + (∂H/∂p)·ṗ = 0   [Symplectic Invariant]
```

This ensures the neural governor never dispatches a control action that violates energy balance — a property that standard neural networks cannot guarantee.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│              11 kV / 415 V Distribution Transformer     │
│                   Dyn11 · 100 kVA · R/X = 1.85         │
└────────────────────────┬────────────────────────────────┘
                         │
          ┌──────────────┼──────────────┐
          │              │              │
    ┌─────▼──────┐ ┌─────▼──────┐ ┌───▼────────┐
    │ Rooftop PV │ │  Community │ │ Local Load │
    │  35 kWp    │ │  BESS      │ │ Residential│
    │  Array     │ │  50 kWh    │ │ + SME      │
    └─────┬──────┘ │  LiFePO₄  │ └───┬────────┘
          │        │  25 kW     │     │
          └────────┤ Bidirec.   ├─────┘
                   │ Inverter   │
                   └─────┬──────┘
                         │
          ┌──────────────▼──────────────┐
          │     AMG-2056 Edge Governor  │
          │  ┌──────────┐ ┌──────────┐ │
          │  │   PINN   │ │   HNN    │ │
          │  │ Forecast │ │ Dispatch │ │
          │  │ Engine   │ │ Engine   │ │
          │  └──────────┘ └──────────┘ │
          │     EN 50160 · 230 V ±5%   │
          └──────────────┬──────────────┘
                         │ (Optional)
                   SCADA Radio Link
                   to 11 kV Substation
```

---

## Dashboard Features

### 1. Real-Time HMI Metric Cards
| Card | Description |
|------|-------------|
| **Feeder Voltage** | Live secondary voltage (V) with EN 50160 ±5% compliance indicator |
| **Solar Generation** | Active rooftop PV output (kW) from 35 kWp array |
| **Hamiltonian State** | `\|dH/dt\|` symplectic energy invariant (pu) — HNN health metric |
| **BESS State of Charge** | Community battery SoC (%) with kWh reservation and charge state |

### 2. Interactive Plotly Charts
- **Predicted vs. Actual Load & Solar Generation** — Dual-axis 24-hour time-series with HNN forecast confidence bands. Secondary Y-axis for solar, primary for demand.
- **Hamiltonian Convergence & Loss** — Log-scale training loss chart showing Total Symplectic Loss, Conservation Error `|dH/dt|`, and Data MSE Loss across 160 training epochs.

### 3. Community BESS Reserve Monitor
- **Plotly Indicator Gauge** with four colour-coded SoC zones:
  - 🔴 0–20 % — Critical / Cutoff
  - 🟠 20–50 % — Buffer / Caution
  - 🔵 50–70 % — Nominal Operation
  - 🟢 70–95 % — Optimal Reserve
- **Diagnostic Tile Grid** — DC Bus Voltage, Internal Temperature, Inverter State of Health (SOH), C-Rate

### 4. Micro-Controller Autonomous Event Stream
- Scrollable live terminal feed showing timestamped HNN governor decisions, BESS dispatch events, solar intermittency warnings, and manual override alerts

### 5. Control Sidebar
| Control | Function |
|---------|----------|
| **Dark / Light Theme** | SCADA night-shift dark mode or daylight engineering light mode |
| **SCADA Radio Toggle** | Simulate online / offline communication link |
| **Scenario Selector** | Normal · Sudden Cloud Cover · Ceremony/Event Load Spike |
| **Manual Override** | Force BESS discharge + non-critical load shedding |
| **Step / Reset** | Single-tick simulation advance or full state reset |
| **Live Auto-Run** | Configurable 2–10 s auto-refresh for real-time simulation |

---

## Simulation Scenarios

| Scenario | Behaviour |
|----------|-----------|
| **Normal** | Typical residential diurnal load profile with standard solar irradiance |
| **Sudden Cloud Cover** | Solar PV drops 65–85%. HNN dispatches BESS and throttles non-critical loads |
| **Ceremony / Event Surge** | Unexpected +28 kW demand spike. HNN maximises BESS discharge to hold 230 V nominal |

---

## Technical Specifications

| Parameter | Value |
|-----------|-------|
| Transformer Rating | 100 kVA, 11 kV / 415 V, Dyn11 |
| Feeder Impedance | R/X = 1.85, 95 mm² Al XLPE cable |
| Community BESS Capacity | 50 kWh LiFePO₄, 25 kW bidirectional |
| Inverter Type | 4-quadrant bidirectional converter |
| PV Array Capacity | 35 kWp rooftop distributed |
| Voltage Standard | EN 50160 — 230 V ±5% (218.5 V – 241.5 V) |
| BESS Minimum Reserve | 20% SoC (10 kWh) hard cutoff |
| HNN Convergence Threshold | `\|dH/dt\|` ≤ 0.0025 pu |
| Forecast Horizon | 5-minute intervals over 24-hour window (288 steps) |

---

## Technology Stack

| Layer | Technology |
|-------|------------|
| **Dashboard** | Python 3.10+ · Streamlit |
| **Visualisation** | Plotly Graph Objects · Plotly Subplots |
| **Simulation Engine** | NumPy · Pandas |
| **Physics Models** | Hamiltonian Mechanics · PINN loss formulation |
| **Styling** | Vanilla CSS · Google Fonts (Roboto / Roboto Mono) |
| **Theme** | Dark SCADA / Light Engineering — toggle at runtime |

---

## Installation & Running

### Prerequisites
```bash
Python >= 3.10
pip install streamlit plotly numpy pandas
```

### Run Locally
```bash
git clone https://github.com/<your-username>/AMG-2056.git
cd AMG-2056
pip install -r requirements.txt
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## Project Structure

```
AMG-2056/
├── app.py                  # Main dashboard application
├── requirements.txt        # Python dependencies
├── README.md               # System documentation & engineering specification
├── LICENSE                 # MIT License
└── .streamlit/
    └── config.toml         # Streamlit theme configuration
```

---

## Key Engineering Concepts

### Physics-Informed Neural Network (PINN)
The forecasting engine incorporates the physical constraint that net power balance must be maintained:

```
P_solar(t) + P_BESS(t) = P_load(t) + P_losses(t)
```

The PINN loss function penalises predictions that violate this constraint, in addition to standard data-fit loss (MSE).

### Hamiltonian Neural Network (HNN) Dispatch
The dispatch controller uses a Hamiltonian formulation where the system state `(q, p)` represents generalised coordinates (voltage phasors) and momenta (reactive power flows). The Hamiltonian:

```
H(q, p) = T(p) + V(q)   →   dH/dt = 0 (conservative system)
```

guarantees that every dispatch decision preserves system energy balance. The `|dH/dt|` metric on the dashboard is the real-time health indicator — values below 0.0025 pu indicate a stable, converged system.

### Autonomous Islanded Operation
When the SCADA radio link is offline, the governor transitions to **autonomous droop mode**: voltage and frequency regulation are handled entirely at the edge node using the HNN dispatch engine, without any upstream communication.

---

## Roadmap

- [ ] Integrate real-time hardware telemetry via Modbus TCP / IEC 61850
- [ ] Deploy on Raspberry Pi 4 / NVIDIA Jetson for edge inference
- [ ] Add three-phase unbalance detection and per-phase voltage monitoring
- [ ] Export event logs to CSV / InfluxDB for post-incident analysis
- [ ] REST API endpoint for upstream SCADA integration
- [ ] Mobile-responsive HMI layout for tablet field access

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- **EN 50160** — European standard for voltage characteristics in public distribution systems
- **IEC 61850** — Communication networks and systems for power utility automation
- **Hamiltonian Neural Networks** — Greydanus et al., NeurIPS 2019
- **Physics-Informed Neural Networks** — Raissi, Perdikaris & Karniadakis, Journal of Computational Physics 2019
