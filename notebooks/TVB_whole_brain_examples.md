# Whole-brain simulation with TVB in NeuroWorkflow — one workflow, eight examples

This project contains **one** workflow built from the generic `NW_TVB_*` nodes.
As delivered it runs **Example 1** (Generic2dOscillator resting state).
Examples 2–8 use the **same graph**: you only change parameters (by hand or by asking the
assistant), never nodes or connections. Together they cover 8 neural mass models, 5 integrators
and 5 monitor types.

All examples were run and checked on this graph (human 76-region connectome, TVB 2.10) —
results and run times are reported per example.

---

## 1. The workflow

```
                                   ┌──────────► NW_TVB_Model ── tvb_model_info ──► NW_TVB_Integrator
                                   │                  │                                  │
 NW_TVB_Connectivity ──────────────┤                  ▼                                  ▼
 (data/connectivity_76.zip)        ├────────────► NW_TVB_Simulator ◄────────────── (tvb_integrator)
                                   │               ▲    ▲     ▲
                                   │               │    │     └──── NW_TVB_Coupling
                                   │   NW_TVB_Monitor_fast  NW_TVB_Monitor_bold   (fan-in port tvb_monitors)
                                   │                  │
                                   │                  ├──► NW_TVB_TimeSeriesPlot_fast   (fast activity + spectrum)
                                   │                  ├──► NW_TVB_TimeSeriesPlot_bold   (BOLD signal)
                                   └──────────────────┴──► NW_TVB_FunctionalConnectivity (FC, FCD)
```

| Node (instance name) | Class | Role |
|---|---|---|
| NW_TVB_Connectivity | `NW_TVB_Connectivity` | Structural connectome (76 human regions), weights normalised to max = 1 |
| NW_TVB_Model | `NW_TVB_Model` | Neural mass model of each region; loads a tested preset for the chosen model |
| NW_TVB_Coupling | `NW_TVB_Coupling` | How regions drive each other through the connectome; `a` = global coupling G |
| NW_TVB_Integrator | `NW_TVB_Integrator` | Numerical scheme and noise. `dt = 0` and `nsig = 'auto'` take the model preset's values |
| NW_TVB_Monitor_fast | `NW_TVB_Monitor` | Fast recording (label `fast`): neural activity at 1 kHz |
| NW_TVB_Monitor_bold | `NW_TVB_Monitor` | BOLD fMRI recording (label `bold`), TR = 1 s |
| NW_TVB_Simulator | `NW_TVB_Simulator` | Runs TVB; drops the first `transient` ms; returns results by monitor label |
| NW_TVB_TimeSeriesPlot_fast | `NW_TVB_TimeSeriesPlot` | Traces of the `fast` monitor + power spectrum; saves `results/fast.npz` |
| NW_TVB_TimeSeriesPlot_bold | `NW_TVB_TimeSeriesPlot` | Traces of the `bold` monitor; saves `results/bold.npz` |
| NW_TVB_FunctionalConnectivity | `NW_TVB_FunctionalConnectivity` | FC, FCD (edge co-activation), FC–SC correlation; saves `results/fc.npz` |

**Node names.** Each node instance is named after its class (the two monitors and the two plots,
which share a class, carry the suffix `_fast` / `_bold`). In the generated `workflow.py` the code
generator renames instances whose name equals the class name (e.g. `instance_NW_TVB_Model_002`), so
the instance does not overwrite the class; the canvas keeps the names used here.

**Data.** All input data are read from this project folder: the connectome is
`data/connectivity_76.zip` (`NW_TVB_Connectivity.connectivity_file = './data/connectivity_76.zip'`).
To use another connectome, copy its TVB zip into `data/` and point `connectivity_file` to it
(e.g. `'./data/connectivity_marmoset.zip'`). Outputs go to `results/` (`fast.npz`, `bold.npz`, `fc.npz`).

Why examples switch by parameters only:

* **Integrator `dt = 0`, `nsig = 'auto'`** — the time step and the noise follow the model preset,
  so changing the model never leaves an unstable step or a noise vector of the wrong length.
* **Monitors are selected by label** (`fast`, `bold`), and variables by name, so plots and FC
  keep working whatever the model.
* The `.npz` files keep the `time` / `data` format read by the TVB brain viewers.

---

## 2. How to switch between examples

The table lists **every parameter that changes** between examples. To go to example *N*, set
every row to the value in column *N* (values in rows you did not touch are the same in all
examples). Do not skip rows: e.g. `NW_TVB_Monitor_bold.variables = ['V']` from Example 2 is invalid for
the ReducedWongWangExcInh model of Example 3.

Easiest: ask the assistant, e.g.

> *"Set up example 3 from TVB_whole_brain_examples.md in this workflow and run it."*
> *"Go back to example 1."*

| Node | Parameter | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| NW_TVB_Connectivity | conduction_speed | `0` | `0` | `0` | `3` | `0` | `0` | `0` | `0` |
| NW_TVB_Model | model_type | `'Generic2dOscillator'` | `'MontbrioPazoRoxin'` | `'ReducedWongWangExcInh'` | `'JansenRit'` | `'SupHopf'` | `'WilsonCowan'` | `'Epileptor'` | `'EpileptorRestingState'` |
| NW_TVB_Model | region_params | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | EZ (see §7) | EZ (see §8) |
| NW_TVB_Coupling | coupling_type | `'Scaling'` | `'Scaling'` | `'Linear'` | `'SigmoidalJansenRit'` | `'Difference'` | `'Linear'` | `'Difference'` | `'Difference'` |
| NW_TVB_Coupling | a | `0.0075` | `0.1` | `0.02` | `10` | `0.1` | `0.1` | `1` | `1` |
| NW_TVB_Integrator | integrator_type | `'HeunDeterministic'` | `'HeunStochastic'` | `'EulerStochastic'` | `'HeunStochastic'` | `'HeunStochastic'` | `'RungeKutta4thOrderDeterministic'` | `'HeunStochastic'` | `'Dopri5Stochastic'` |
| NW_TVB_Monitor_fast | monitor_type | `'TemporalAverage'` | `'TemporalAverage'` | `'TemporalAverage'` | `'TemporalAverage'` | `'GlobalAverage'` | `'SubSample'` | `'TemporalAverage'` | `'TemporalAverage'` |
| NW_TVB_Monitor_bold | variables | `[]` | `['V']` | `['S_e']` | `[]` | `[]` | `[]` | `[]` | `['x_rs']` |
| NW_TVB_Simulator | simulation_length (ms) | `30000` | `60000` | `60000` | `10000` | `60000` | `10000` | `20000` | `20000` |
| NW_TVB_Simulator | transient (ms) | `5000` | `10000` | `10000` | `2000` | `10000` | `1000` | `1000` | `2000` |
| NW_TVB_TimeSeriesPlot_fast | variable | `''` | `'r'` | `'S_e'` | `'y1-y2'` | `'x'` | `'E'` | `'x2 - x1'` | `'x_rs'` |
| NW_TVB_TimeSeriesPlot_fast | regions | `10` | `10` | `10` | `10` | `10` | `10` | EZ list (see §7) | `10` |
| NW_TVB_TimeSeriesPlot_fast | normalize | `True` | `True` | `True` | `True` | `True` | `True` | `False` | `True` |
| NW_TVB_FunctionalConnectivity | variable | `''` | `'V'` | `'S_e'` | `''` | `'x'` | `''` | `''` | `'x_rs'` |

Unchanged everywhere: `NW_TVB_Integrator.dt = 0`, `NW_TVB_Integrator.nsig = 'auto'`, `NW_TVB_Monitor_fast.period = 1`,
`NW_TVB_Monitor_bold.monitor_type = 'Bold'`, `NW_TVB_Monitor_bold.period = 1000`, connectome
`'./data/connectivity_76.zip'` (the copy inside this project folder) normalised `'max'`.

Run times below were measured on this JupyterHub with no other load; they can be 2–3× longer
on a busy machine.

---

## 3. The examples

### Example 1 — Generic2dOscillator: deterministic resting-state oscillations (the delivered workflow)

This is the configuration saved in the project. It is also the **reference** for the other
examples: every parameter not listed in the table of §2 keeps the value below.

* **Model:** generic 2-variable oscillator (V, W) in a limit cycle (`a = 1.74`, the same regime as
  the V2 human/marmoset workflows). **Coupling:** Scaling, G = 0.0075. **Integrator:**
  HeunDeterministic (no noise), dt = 0.1 ms from the preset. **Monitors:** TemporalAverage 1 ms
  (`fast`) + BOLD with TR = 1 s (`bold`). **Simulation:** 30 s, first 5 s dropped.
* **Full configuration** (non-default values in bold):

  | Node | Parameters |
  |---|---|
  | NW_TVB_Connectivity | **connectivity_file = `'./data/connectivity_76.zip'`**, normalize_weights = `'max'`, remove_self_connections = `True`, conduction_speed = `0` (no delays), show_plot = `True` |
  | NW_TVB_Model | model_type = `'Generic2dOscillator'`, use_preset = `True`, model_params = `{}`, region_params = `{}`, variables_of_interest = `[]` (preset: `['V']`), state_variable_range = `{}`, stimulus_variables = `[]` |
  | NW_TVB_Coupling | coupling_type = `'Scaling'`, a = `0.0075`, coupling_params = `{}` |
  | NW_TVB_Integrator | **integrator_type = `'HeunDeterministic'`**, **dt = `0`** (preset → 0.1 ms), **nsig = `'auto'`** (unused: deterministic), noise_type = `'Additive'`, noise_seed = `42` |
  | NW_TVB_Monitor_fast | monitor_type = `'TemporalAverage'`, period = `1.0` ms, variables = `[]`, **label = `'fast'`** |
  | NW_TVB_Monitor_bold | **monitor_type = `'Bold'`**, **period = `1000` ms**, variables = `[]`, hrf_kernel = `'FirstOrderVolterra'`, **label = `'bold'`** |
  | NW_TVB_Simulator | **simulation_length = `30000` ms**, **transient = `5000` ms**, random_seed = `42` |
  | NW_TVB_TimeSeriesPlot_fast | **monitor = `'fast'`**, variable = `''` (→ `V`), regions = `10`, normalize = `True`, **show_spectrum = `True`**, **title = `'Fast activity'`**, **save_to_file = `True`**, **output_path = `'./results/fast.npz'`** |
  | NW_TVB_TimeSeriesPlot_bold | **monitor = `'bold'`**, variable = `''` (→ `V`), regions = `10`, normalize = `True`, show_spectrum = `False`, **title = `'BOLD'`**, **save_to_file = `True`**, **output_path = `'./results/bold.npz'`** |
  | NW_TVB_FunctionalConnectivity | monitor = `'bold'`, variable = `''` (→ `V`), fcd_method = `'edge'`, empirical_fc_file = `''`, show_plot = `True`, **save_to_file = `True`**, output_path = `'./results/fc.npz'` |

* **What you get:**
  * `NW_TVB_Connectivity` — weight and tract-length matrices of the 76 regions.
  * `NW_TVB_TimeSeriesPlot_fast` — regular **~9 Hz** oscillations in every region (first 10 shown)
    and a power spectrum with a sharp peak.
  * `NW_TVB_TimeSeriesPlot_bold` — slow BOLD fluctuations (26 samples at TR = 1 s).
  * `NW_TVB_FunctionalConnectivity` — SC, BOLD FC and edge FCD. The network partly synchronises
    the oscillators, so FC is high (**mean FC ≈ 0.71**) and only loosely shaped by the connectome
    (**FC–SC ≈ 0.21**); fcd_mean ≈ 0.14, fcd_var ≈ 0.06.
  * Files: `results/fast.npz` (25 000 × 76), `results/bold.npz` (26 × 76), `results/fc.npz`.
* **Run time:** ~16–20 s.
* **Try next:** `NW_TVB_Integrator.integrator_type = 'HeunStochastic'` (adds the preset noise on V:
  less regular rhythm, lower FC); `NW_TVB_Coupling.a = 0` (regions decouple: FC near 0) or `0.05`
  (stronger synchrony); `NW_TVB_Model.model_params = {'a': -0.5}` (damped instead of oscillating —
  combine with HeunStochastic to get noise-driven activity).
* **Return to Example 1** from any other example: set column 1 of the table in §2
  (or ask the assistant *"go back to example 1"*).

### Example 2 — MontbrioPazoRoxin: bistable up/down states and resting-state FC (Rabuffo tutorial)

* **Model:** exact mean field of spiking QIF neurons — firing rate `r`, membrane potential `V`;
  parameters η = −4.6, Δ = 0.7, J = 14.5 put each region in a **bistable** regime (low "down" and
  high "up" activity). Noise (on V) makes regions jump between states.
* **Integrator:** HeunStochastic, dt = 0.025 ms. **Monitors:** firing rate at 1 kHz + BOLD of `V`.
* **What to look at:** in `NW_TVB_TimeSeriesPlot_fast` regions switch between low and high firing; in the test
  55 of 76 regions switched and ~70 % of the time was spent in the up state. `NW_TVB_FunctionalConnectivity` shows the
  BOLD FC and the edge-based FCD (moments of network-wide co-activation = bright FCD blocks).
* **Run time:** ~3.5 min for 60 s (the slowest model: small dt).
* **Try next:** `NW_TVB_Coupling.a` 0.05 (more down state) ↔ 0.2 (mostly up). Longer runs
  (`simulation_length = 300000`, 5 min) for a stable FC. `NW_TVB_FunctionalConnectivity.fcd_method = 'sliding_window'`.
* **Note vs the original tutorial:** the tutorial used G = 0.45 on a mouse connectome and noise on
  both r and V. On this human connectome G = 0.1 gives the switching regime, and noise on r drives
  r below 0 and diverges in TVB 2.10, so the preset puts noise on V only.

### Example 3 — ReducedWongWangExcInh: resting-state BOLD (Deco et al. 2014)

* **Model:** dynamic mean field of excitatory/inhibitory synaptic gating (`S_e`, `S_i`), the
  classic model for fitting empirical resting-state FC. **Integrator:** **EulerStochastic**
  (1st order). **Monitors:** S_e at 1 kHz + BOLD of `S_e`.
* **What to look at:** noisy fluctuations around a low-activity state (mean S_e ≈ 0.28);
  BOLD FC is weak (mean FC ≈ 0.03) with FC–SC ≈ 0.17 at this coupling.
* **Run time:** ~35 s.
* **Try next:** this is the model to sweep G: `NW_TVB_Coupling.a` 0 → 0.05 and watch `NW_TVB_FunctionalConnectivity.fc_sc_corr`
  and mean S_e (activity rises with G; this TVB version has no feedback inhibition control).
  Switch to `'HeunStochastic'` to compare integrators.

### Example 4 — JansenRit: alpha rhythm, EEG-like signal, conduction delays

* **Model:** cortical column with pyramidal, excitatory and inhibitory populations; the EEG-like
  signal is `y1 − y2`. Requires `SigmoidalJansenRit` coupling (the simulator warns otherwise).
* **Connectome:** `conduction_speed = 3 mm/ms` → real signal delays (up to tens of ms).
* **Integrator:** HeunStochastic. **Monitors:** TemporalAverage 1 ms (+ BOLD, not the focus).
* **What to look at:** `NW_TVB_TimeSeriesPlot_fast` (variable `'y1-y2'`) — spectrum peak at **~11 Hz** (alpha).
* **Run time:** ~20 s.
* **Try next:** `NW_TVB_Connectivity.conduction_speed` 1 vs 10 (delays shift the rhythm and
  synchrony); `NW_TVB_Coupling.a` 5 ↔ 20; `NW_TVB_Model.model_params = {'mu': 0.15}` (input level).

### Example 5 — SupHopf: noise-driven oscillations at a Hopf bifurcation (Deco et al. 2017)

* **Model:** Stuart-Landau oscillator per region, just below the bifurcation (`a = −0.01`), with
  a 10 Hz carrier; noise excites damped oscillations. Diffusive (`Difference`) coupling.
* **Integrator:** HeunStochastic. **Monitors:** **GlobalAverage** (one whole-brain trace, like a
  crude global EEG) + BOLD of `x`.
* **What to look at:** global signal peak ~9–10 Hz; the BOLD FC follows the connectome best of
  all examples (**FC–SC ≈ 0.64**, mean FC ≈ 0.39).
* **Run time:** ~40 s.
* **Try next:** `NW_TVB_Model.model_params = {'a': 0.02}` (above the bifurcation: self-sustained
  oscillations) vs `{'a': -0.1}` (strongly damped); `NW_TVB_Coupling.a` 0.05 ↔ 0.2 (FC rises with G).

### Example 6 — WilsonCowan: E/I oscillations with a deterministic 4th-order integrator

* **Model:** classic excitatory/inhibitory rate model (`E`, `I`) in an oscillatory regime.
* **Integrator:** **RungeKutta4thOrderDeterministic**. **Monitors:** **SubSample** (instantaneous
  samples every 1 ms, no averaging) + BOLD.
* **What to look at:** ~29 Hz (beta/gamma) oscillations; without noise and with identical regions
  the network **fully synchronises** (mean FC ≈ 1.0, FC unrelated to SC) — a useful contrast with
  the noisy examples.
* **Run time:** ~13 s.
* **Try next:** `NW_TVB_Coupling.a = 0.01` (weaker coupling); `'HeunStochastic'` integrator to break the
  synchrony with noise; `NW_TVB_Model.model_params = {'P': 1.0}` (less drive).

### Example 7 — Epileptor: seizures from an epileptogenic zone

* **Model:** 6-variable Epileptor (Jirsa et al. 2014); `x0` sets the excitability of each region.
  Epileptogenic zone (EZ) = right amygdala, hippocampus and parahippocampal cortex:

  ```python
  NW_TVB_Model.region_params = {"x0": {"all": -2.2, "regions": [2, 9, 24], "values": [-1.6, -1.6, -1.6]}}
  #                      healthy elsewhere      rAMYG rHC rPHC      epileptogenic
  NW_TVB_TimeSeriesPlot_fast.regions   = ['rAMYG', 'rHC', 'rPHC', 'rTCPOL', 'rTCI', 'rIA', 'lHC', 'lAMYG', 'rV1', 'rM1']
  NW_TVB_TimeSeriesPlot_fast.normalize = False      # keep true amplitudes: seizing vs non-seizing regions
  ```

* **Integrator:** HeunStochastic (dt 0.05). **Monitors:** TemporalAverage 1 ms (variable
  `x2 − x1`, the LFP-like signal) + BOLD.
* **What to look at:** the 3 EZ regions seize (signal std ≈ 0.97) while the others stay near
  baseline (std ≈ 0.20); seizure onsets start ~2 s into the run.
* **Run time:** ~35 s.
* **Try next:** move the EZ (other region indices — the labels are listed by the Connectivity
  plot or `connectivity_info`); raise `NW_TVB_Coupling.a` (e.g. 2–5) to see propagation to neighbouring
  regions; set `"all": -2.0` (closer to threshold) to recruit a propagation zone.

### Example 8 — EpileptorRestingState: resting oscillations + epileptic dynamics, adaptive integrator

* **Model:** Epileptor coupled to a resting-state oscillator (8 variables; Courtiol et al. 2020),
  same EZ as Example 7 but healthy regions at `x0 = −2.3`:

  ```python
  NW_TVB_Model.region_params = {"x0": {"all": -2.3, "regions": [2, 9, 24], "values": [-1.6, -1.6, -1.6]}}
  ```

* **Integrator:** **Dopri5Stochastic** (SciPy adaptive Runge-Kutta 4/5 — slower, but shows that
  any TVB integrator can be plugged in). **Monitors:** TemporalAverage + BOLD of `x_rs`.
* **What to look at:** `x_rs` (the resting-state subsystem) oscillates at ~12 Hz in every region;
  BOLD FC of `x_rs` is high (mean ≈ 0.93) with FC–SC ≈ 0.28. Set `NW_TVB_TimeSeriesPlot_fast.variable = 'x2 - x1'`
  to see the epileptic subsystem.
* **Run time:** ~60 s.
* **Try next:** compare with `'HeunStochastic'` (faster, same dynamics); change the EZ as in Example 7.

---

## 4. Summary of the tested runs

| # | Model | Integrator | Fast monitor | Dynamics found | Key numbers | Time |
|---|---|---|---|---|---|---|
| 1 | Generic2dOscillator | HeunDeterministic | TemporalAverage | ~9 Hz limit cycle | mean FC 0.71, FC–SC 0.21 | 16 s |
| 2 | MontbrioPazoRoxin | HeunStochastic | TemporalAverage | up/down switching | 55/76 regions switch; FC–SC 0.24 | ~3.5 min |
| 3 | ReducedWongWangExcInh | EulerStochastic | TemporalAverage | low-activity fluctuations | S_e 0.28; FC–SC 0.17 | 35 s |
| 4 | JansenRit (+ delays) | HeunStochastic | TemporalAverage | alpha rhythm | peak 11.3 Hz | 20 s |
| 5 | SupHopf | HeunStochastic | GlobalAverage | noisy 10 Hz oscillations | FC–SC **0.64** | 40 s |
| 6 | WilsonCowan | RK4 deterministic | SubSample | full synchrony | 28.6 Hz; mean FC 1.0 | 13 s |
| 7 | Epileptor | HeunStochastic | TemporalAverage | seizures in EZ only | EZ std 0.97 vs 0.20 | 35 s |
| 8 | EpileptorRestingState | Dopri5Stochastic | TemporalAverage | 12 Hz resting rhythm + EZ | mean FC 0.93, FC–SC 0.28 | 60 s |

The BOLD runs are short (20–50 BOLD samples), enough to see the dynamics but not for a robust FC;
for FC studies simulate ≥ 5 min (`simulation_length ≥ 300000`).

---

## 5. Tips and troubleshooting

* **"WARNING … non-finite values" (NaN):** the simulation diverged. Reduce `NW_TVB_Integrator.dt`
  (set an explicit value smaller than the preset), the noise (`nsig`, e.g. `{'V': 0.005}`) or
  `NW_TVB_Coupling.a`; for stiff models narrow `NW_TVB_Model.state_variable_range`.
* **"variable 'X' is not recorded":** a monitor/plot/FC variable from a previous example — reset
  `NW_TVB_Monitor_bold.variables`, `NW_TVB_TimeSeriesPlot_fast.variable`, `NW_TVB_FunctionalConnectivity.variable` (see §2).
* **"Only N samples …" in FC:** the BOLD signal is too short — increase `simulation_length` or
  decrease `transient`.
* **Model/coupling warning:** JansenRit needs `SigmoidalJansenRit`; Epileptor models need
  `Difference`.
* **Coupling values are connectome-specific.** Presets assume weights normalised to max = 1 on the
  76-region human connectome. The marmoset connectome has ~8× weaker average input per region
  after the same normalisation, so G must be larger (or use `normalize_weights = 'row_sum'`).
* **Reproducibility:** stochastic runs are reproducible with the same `NW_TVB_Integrator.noise_seed`;
  change it for independent realisations.

## 6. Going further (same nodes)

* **Other connectomes:** `NW_TVB_Connectivity.connectivity_file` →
  `'neuroworkflow/data/tvb_data/connectivity_marmoset.zip'`, `'connectivity_96.zip'`,
  `'connectivity_192.zip'`, `'neuroworkflow/data/tvb_data/connectivity_human.zip'`.
* **Stimulation:** add `NW_TVB_Stimulus` (regions by label, e.g. `['rV1', 'lV1']`; PulseTrain,
  Gaussian, Sinusoid …) → `NW_TVB_Simulator.tvb_stimulus`; choose the receiving variable with
  `NW_TVB_Model.stimulus_variables`.
* **3D visualisation:** connect `NW_TVB_TimeSeriesPlot_bold.saved_file_path` / `NW_TVB_TimeSeriesPlot_fast.saved_file_path` to the
  `bold_file` / `temporal_average_file` inputs of `TVBBrainViewerNode` or
  `TVBMarmosetBrainViewerNode`.
* **Fitting G:** `NW_TVB_Coupling.a` is optimizable; `NW_TVB_FunctionalConnectivity.fc_metrics` exposes `fc_sc_corr`,
  `fc_emp_corr` (with `NW_TVB_FunctionalConnectivity.empirical_fc_file`), `fcd_var` as targets for `NW_Optimization`.
* **Other parameters per model:** see the parameter descriptions in each node (all have examples),
  and `tvb_model_info` (output of the Model node) for the state variables and preset values.

### Available options (any combination works on this graph)

* **Models:** Generic2dOscillator, MontbrioPazoRoxin, ReducedWongWangExcInh, JansenRit, SupHopf,
  WilsonCowan, Epileptor, EpileptorRestingState.
* **Couplings:** Scaling, Linear, Difference, Sigmoidal, SigmoidalJansenRit, HyperbolicTangent,
  Kuramoto, PreSigmoidal.
* **Integrators:** Heun / Euler (Deterministic, Stochastic), RungeKutta4thOrderDeterministic,
  Dopri5, Dop853, VODE (+ Stochastic versions); noise Additive or Multiplicative.
* **Monitors:** Raw, SubSample, TemporalAverage, GlobalAverage, Bold (HRF: FirstOrderVolterra,
  Gamma, DoubleExponential, MixtureOfGammas), AfferentCouplingTemporalAverage.

## References

* Sanz-Leon et al. (2015) *NeuroImage* — The Virtual Brain (Generic2dOscillator).
* Montbrió, Pazó & Roxin (2015) *Phys Rev X*; Rabuffo et al. (2021) *eNeuro* — MPR model, bursts;
  tutorial: github.com/grabuffo/TVB-tutorial.
* Deco et al. (2014) *J Neurosci* — ReducedWongWangExcInh.
* Jansen & Rit (1995) *Biol Cybern*.
* Deco et al. (2017) *Sci Rep* — Hopf whole-brain model.
* Wilson & Cowan (1972) *Biophys J*.
* Jirsa et al. (2014) *Brain* — Epileptor; Courtiol et al. (2020) *J Neurosci* — Epileptor resting state.
* Faskowitz et al. (2020) *Nat Neurosci* — edge time series / edge FCD.
