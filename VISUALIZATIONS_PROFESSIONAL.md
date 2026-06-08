# Professional Research Visualizations

## Status: All benchmarks updated with publication-quality figures

All benchmark files now produce **professional, publication-ready figures** suitable for research papers and presentations.

---

## Available Professional Visualization Functions

Located in: `src/shadowvqe/visualization_research.py`

### 1. `plot_shadow_scaling_research()`
**Used in**: `benchmark_shadow_error_scaling.py`  
**Outputs**: `figures/research_shadow_scaling.png/pdf`  
**Shows**:
- Classical shadow error vs number of shadows (log-log)
- Theoretical 1/√N scaling validation
- Error bands (confidence intervals)
- Chemical accuracy threshold

### 2. `plot_vqe_shadow_comparison_research()`
**Used in**: `benchmark_molecular_vqe.py` (optional)  
**Outputs**: Side-by-side energy accuracy comparison  
**Shows**:
- Final ground state energy (bar chart)
- Absolute error in log scale
- Direct VQE vs Shadow-VQE on same molecule

### 3. `plot_shot_budget_research()`
**Used in**: `benchmark_shot_budget_fairness.py`  
**Outputs**: `figures/research_shot_budget.png/pdf`  
**Shows**:
- Error vs fixed budget (log-log)
- Crossover point where shadows win
- Chemical accuracy threshold
- Both methods with error bars

### 4. `plot_measurement_cost_research()`
**Used in**: `benchmark_heisenberg_measurement_cost.py`  
**Outputs**: `figures/research_measurement_cost.png/pdf`  
**Shows**:
- VQE vs Shadow measurement overhead
- Qubit requirement scaling
- Clear advantage visualization

### 5. `plot_variance_reduction_research()`
**Used in**: `benchmark_adaptive_shadow_variance.py`  
**Outputs**: `figures/research_variance_reduction.png/pdf`  
**Shows**:
- (A) Variance: Random vs Adaptive bases
- (B) Variance reduction factor
- Color-coded advantage zones

### 6. `plot_fmo_scaling_research()`
**Used in**: `benchmark_fmo_energy_assembly.py`  
**Outputs**: `figures/research_fmo_scaling.png/pdf`  
**Shows**:
- (A) Qubit requirement scaling (Direct vs FMO)
- (B) Number of fragments/subsystems
- (C) Energy accuracy comparison (if available)

---

## Visual Design Standards

All figures follow **publication-quality standards**:

| Property | Value |
|---|---|
| **DPI** | 150 (raster), vector (PDF) |
| **Colors** | Okabe-Ito colorblind-friendly palette |
| **Fonts** | sans-serif, 10-11pt |
| **Style** | Minimal, clean, no clutter |
| **Layout** | Proper spacing, labeled panels (A, B, C) |
| **Grid** | Light (alpha=0.25) on data area |
| **Legend** | Clear, positioned intelligently |
| **Error bands** | Filled regions for confidence |
| **Theory lines** | Dashed overlays for validation |

---

## How to Use

### In any benchmark file:

```python
from shadowvqe.visualization_research import plot_shadow_scaling_research

# After collecting data...
fig, ax = plot_shadow_scaling_research(
    n_shadows_list=my_n_list,
    mean_errors=my_errors,
    std_errors=my_stds,
    theory_slope=-0.5,
    savedir="figures",
)
```

### Running benchmarks:

```bash
# All produce professional figures automatically
python examples/benchmark_shadow_error_scaling.py
python examples/benchmark_shot_budget_fairness.py
python examples/benchmark_molecular_vqe.py
python examples/benchmark_heisenberg_measurement_cost.py
python examples/benchmark_pauli_term_scaling.py
python examples/benchmark_adaptive_shadow_variance.py
python examples/benchmark_fmo_energy_assembly.py
```

Figures appear in `figures/research_*.png` and `figures/research_*.pdf`

---

## Quick-Start: Add Professional Visualization to Your Benchmark

### Before (basic matplotlib):
```python
plt.plot(x, y)
plt.savefig("output.png")
```

### After (professional):
```python
from shadowvqe.visualization_research import plot_measurement_cost_research

fig, ax = plot_measurement_cost_research(
    system_sizes=[2, 4, 6],
    vqe_settings=[2, 6, 12],
    shadow_settings=[3, 4, 5],
    labels=["H2", "Heis-4q", "Heis-6q"],
    savedir="figures",
)
```

**Automatically handles:**
- Professional color scheme
- Proper typography & sizing
- Gridlines & spacing
- Legend positioning
- PNG + PDF export (150 DPI)
- Annotations & error bands

---

## Results

All benchmarks now produce figures that are:

✅ **Publication-ready** (suitable for papers, theses, presentations)  
✅ **Colorblind-friendly** (Okabe-Ito palette)  
✅ **High-contrast** (readable in print & screen)  
✅ **Consistent** (unified design across all figures)  
✅ **Vectorized** (PDF exports at any scale)  
✅ **Communicative** (clear, instant understanding)

---

## File Summary

- `src/shadowvqe/visualization_research.py` — 6 professional plot functions
- `examples/benchmark_*.py` — Updated to use professional visualizations
- `figures/research_*.png/pdf` — Output location for all professional figures

**Total benchmark files**: 7  
**Professional visualization functions**: 6  
**Status**: Complete & ready for publication
