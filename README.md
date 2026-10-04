# AMG-2056 — Autonomous Micro-Grid Governor

> **Neighbourhood Sub-Distributor Decentralised Feeder Control System**  
> Real-time renewable intermittency management using Physics-Informed Neural Networks (PINN) and Hamiltonian Neural Networks (HNN) at the transformer-secondary edge node.

---

## Overview

The **Autonomous Micro-Grid Governor (AMG-2056)** is an interactive operational dashboard prototype for a decentralised feeder-level energy management unit. It simulates the monitoring and autonomous control of a 100 kVA neighbourhood distribution transformer that serves residential and small commercial loads supplied in part by rooftop photovoltaic generation and a community Battery Energy Storage System (BESS).

The dashboard is a **self-contained simulation prototype**. It generates synthetic feeder, solar, load, and battery values for demonstration and does not require physical hardware or a SCADA connection. The radio toggle and autonomous-control states are simulated.

---

## Problem Statement

Modern low-voltage distribution feeders are increasingly subject to **bidirectional power flow**, **solar intermittency**, and **unexpected demand surges** (community events, EV charging clusters). Traditional centralised SCADA systems cannot respond fast enough at the feeder edge, particularly under communication blackouts.

This project illustrates PINN and HNN concepts through simulated forecasts, dispatch states, and energy-invariant metrics. It does **not** train or run neural-network models, connect to real sensors, or guarantee physical grid stability. The following equation describes the concept represented by the dashboard:

```
dH/dt = (∂H/∂q)·q̇ + (∂H/∂p)·ṗ = 0   [Symplectic Invariant]
```

The dashboard's values are demonstration data, not the output of a validated grid controller.

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

### 5. Optional Prototype Panels
- **3D Hardware Prototype** — interactive schematic component layout, GPIO connection traces, and a file-upload control for STL, OBJ, or STEP files. Uploaded models are not rendered by the current app.
- **Hardware Connections** — ESP32 pin-mapping reference for the voltage/current sensors, relays, and NRF24L01 module
- **Maintenance & Fixes** — calibration guidance, team practices, common faults, and suggested remedies

### 6. Control Sidebar
| Control | Function |
|---------|----------|
| **Dark / Light Theme** | SCADA night-shift dark mode or daylight engineering light mode |
| **SCADA Radio Toggle** | Simulate online / offline communication link |
| **Scenario Selector** | Normal · Sudden Cloud Cover · Ceremony/Event Load Spike |
| **Manual Override** | Force BESS discharge + non-critical load shedding |
| **Step / Reset** | Single-tick simulation advance or full state reset |
| **Live Auto-Run** | Configurable 2–10 s auto-refresh for real-time simulation |
| **Panel Navigator** | Show or hide the 3D prototype, hardware connections, and maintenance panels |

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
| **Simulation** | Synthetic data and PINN/HNN-inspired metrics; no model training |
| **Styling** | Theme-aware CSS · Google Fonts (Barlow, JetBrains Mono) |
| **Theme** | Dark SCADA / Light Engineering — toggle at runtime |

---

## Installation & Running

### Prerequisites
- Python 3.10 or newer
- Git (optional, for cloning)

### Run Locally
Clone the repository, then create an isolated environment and install every dependency from `requirements.txt`.

**Windows PowerShell**
```powershell
git clone https://github.com/KAUSHIK-G-19/AMG-2056.git
cd AMG-2056
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

**macOS / Linux**
```bash
git clone https://github.com/KAUSHIK-G-19/AMG-2056.git
cd AMG-2056
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Streamlit prints the local URL when the server starts (typically **http://localhost:8501**). Stop the server with `Ctrl+C`.

---

## Project Structure

```
AMG-2056/
├── app.py                  # Main dashboard application
├── requirements.txt        # Python dependencies
├── README.md               # Project overview, limitations, and run instructions
├── LICENSE                 # MIT License
└── .streamlit/
    └── config.toml         # Streamlit theme configuration
```

---

## Key Engineering Concepts

### Physics-Informed Neural Network (PINN)
The demo presents the physical balance as a conceptual constraint:

```
P_solar(t) + P_BESS(t) = P_load(t) + P_losses(t)
```

The app visualizes synthetic forecast and loss curves; it does not train a PINN or calculate this loss from model predictions.

### Hamiltonian Neural Network (HNN) Dispatch
The dashboard uses a Hamiltonian formulation for explanatory context. The state `(q, p)` and Hamiltonian below are illustrative and are not connected to a physical grid solver:

```
H(q, p) = T(p) + V(q)   →   dH/dt = 0 (conservative system)
```

The displayed `|dH/dt|` value is simulated and should not be treated as a safety or convergence guarantee.

### Autonomous Islanded Operation
When the simulated SCADA radio link is switched off, the dashboard displays an islanded/autonomous status. No actual voltage or frequency regulation occurs.

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
