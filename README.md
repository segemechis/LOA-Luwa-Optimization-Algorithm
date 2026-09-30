# LOA – Luwa Optimization Algorithm

MATLAB implementation of the **Luwa Optimization Algorithm (LOA)**, a
population-based metaheuristic inspired by the Sidama people's *Luwa*
age-grade governance system.

> **Paper:** *Luwa Optimization Algorithm: A Novel Population-Based
> Metaheuristic Inspired by Sidama Age-Grade Governance*  
> Submitted to *Scientific Reports* (2026).

---

## Repository Contents

| File | Description |
|---|---|
| `LOA.m` | Main LOA solver |
| `Run_Independent_Benchmark.m` | Benchmark runner (CEC-2017, CEC-2022) |
| `CEC17_*.m` / `CEC22_*.m` | CEC benchmark function definitions |
| `WOA.m`, `GWO.m`, `SCA.m`, … | Competitor algorithm implementations |
| *(all other `.m` files)* | Supporting utilities and helpers |

---

## Requirements

- MATLAB R2019b or later  
- No additional toolboxes required

---

## Quick Start

```matlab
% Run LOA on CEC-2017 F1, D=30, 30 independent runs
Run_Independent_Benchmark
```

Edit `Run_Independent_Benchmark.m` to change the function index,
dimension, number of runs, or population size.

---

## Algorithm Overview

LOA organises the population into **five rotating age grades**:

| Grade | Name | Mechanism |
|---|---|---|
| G1 | Darara | Lévy-flight exploration |
| G2 | Hirbora | Rank-competitive differential evolution |
| G3 | Mogissa | Archive-guided exploitation |
| G4 | Fulasa | Father-guidance + Yuba crossover |
| G5 | Wawasa | Opposition-based learning |

Two diversity-recovery mechanisms prevent premature convergence:

- **Yakka-lite** – reinitialises the bottom 50 % of a grade when
  within-grade diversity falls below threshold δ_YK  
- **Garo restart** – reinitialises the worst stagnating grade via
  Latin Hypercube Sampling after T_stag stagnation iterations

---


---

## Licence

MIT — see [LICENSE](LICENSE).
