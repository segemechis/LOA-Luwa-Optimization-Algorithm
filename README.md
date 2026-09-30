# Luwa Optimization Algorithm (LOA)

**Paper:** "Escaping Spatially Displaced Global Optima: The Luwa Optimization Algorithm and Mechanistic Ablation Evidence"  
**Author:** Sena Gemechis File  
**Affiliation:** Faculty of Electrical and Computer Engineering, Jimma Institute of Technology, Jimma University, Ethiopia  
**Status:** Submitted to *Scientific Reports* (Nature Publishing Group)

---

## About the Algorithm

The Luwa Optimization Algorithm (LOA) is a population-based metaheuristic inspired by the **Luwa age-grade governance system** of the Sidama people of Ethiopia. The population is divided into five rotating sub-populations (grades), each executing a distinct search operator:

| Grade | Name | Operator |
|---|---|---|
| G1 | Darara | Lévy-flight exploration |
| G2 | Hirbora | Rank-competitive DE mutation |
| G3 | Mogissa | Archive-guided crossover |
| G4 | Fulasa | Father-guidance toward global best + Yuba archive |
| G5 | Wawasa | Opposition-based learning |

Two mechanisms prevent premature convergence:

- **Yakka-lite** — reinitialises the bottom 50% of a grade when within-grade diversity collapses below threshold δ_YK *(primary escape mechanism)*
- **Garo restart** — reinitialises the worst stagnating grade via Latin Hypercube Sampling when fitness stagnates for T_stag iterations *(secondary recovery)*

---

## Repository Structure

LOA-Luwa-Optimization-Algorithm/
│
├── code/
│ ├── LOA.m # Main LOA implementation
│ ├── LOA_ablation.m # Ablation variant controller
│ ├── CCO.m / CMAES.m / DE.m / GA.m / GWO.m / HHO.m
│ ├── LSHADE.m / MVO.m / PSO.m / RIME.m / SA.m / SCA.m / SO.m / WOA.m
│ ├── wrap_LOA.m / wrap_CCO.m / ... # Unified wrappers for each algorithm
│ ├── Run_01_Schwefel_Low.m # Exp 1: Standard Schwefel D=10,30
│ ├── Run_02_Schwefel_High.m # Exp 2: Standard Schwefel D=50,100
│ ├── Run_04_Schwefel_Shifted.m # Exp 4: Shifted Schwefel D=10,30,50
│ ├── Run_08_Progressive_Shift_Severity.m
│ ├── Run_09_Partial_Dimension_Shift.m
│ ├── Run_10_Distant_Initialization.m
│ ├── Run_11_Local_Trap_Initialization.m
│ ├── Run_12_Wide_Gap_Severity.m
│ ├── Run_Ablation.m # Full ablation study runner
│ └── Run_Statistical_Analysis.m # Friedman + Wilcoxon tests
│
├── data/
│ ├── Ablation_summary.csv # Aggregated ablation results (1,560 runs)
│ ├── Exp01_Standard_D10_D30/ # Raw results: standard Schwefel D=10, D=30
│ ├── Exp02_Standard_D50_D100/ # Raw results: standard Schwefel D=50, D=100
│ ├── Exp04_Shifted_Schwefel/ # Raw results: shifted Schwefel D=10, D=30, D=50
│ ├── Exp08_Progressive_Shift/ # Progressive shift severity results
│ ├── Exp09_Partial_Dimension/ # Partial dimension shift results
│ ├── Exp10_Distant_Init/ # Distant initialisation results
│ ├── Exp11_Local_Trap/ # Local-trap initialisation results
│ ├── Exp12_Wide_Gap/ # Wide-gap severity results
│ └── Stats/ # Friedman ranks, Wilcoxon p-values, Holm correction
│
└── README.md


---

## How to Run

All scripts are written in **MATLAB**. No additional toolboxes are required.

1. Clone or download this repository
2. Open MATLAB and set the working directory to `code/`
3. Run any experiment script directly, for example:

```matlab
Run_04_Schwefel_Shifted   % Shifted Schwefel benchmark
Run_Ablation              % Full 1,560-run ablation study
```

Results are saved automatically as CSV files matching those in the `data/` folder.

---

## Benchmark Summary

LOA is evaluated against 14 competitor algorithms (CCO, CMA-ES, DE, GA, GWO, HHO, L-SHADE, MVO, PSO, RIME, SA, SCA, SO, WOA) over 30 independent runs per configuration.

| Benchmark | LOA Success Rate |
|---|---|
| Shifted Schwefel D=10 | **100%** |
| Shifted Schwefel D=30 | **100%** |
| Shifted Schwefel D=50 | **100%** |
| Standard Schwefel D=10 | Rank 1 (lowest mean error) |

LOA is the **only algorithm** among all 15 competitors to achieve 100% success rate on the shifted Schwefel at all three dimensions.

---

## Ablation Study

A controlled ablation across **1,560 runs** (4 variants × 13 sub-experiments × 30 seeds) tests:

| Variant | Description |
|---|---|
| LOA-Full | Complete algorithm |
| NoYakka | Yakka-lite disabled |
| NoGaro | Garo restart disabled |
| None | Both mechanisms disabled |

Removing Yakka-lite drives success rate to exactly **0%** on all Local-Trap sub-experiments, establishing it as a necessary condition for trap escape.

---

## Citation

> File, S.G. (2026). Escaping Spatially Displaced Global Optima: The Luwa Optimization Algorithm and Mechanistic Ablation Evidence. *Scientific Reports*. *(under review)*

---

## License

MIT License — see [LICENSE](LICENSE) for details.
