# TDA-Complex-Systems

**Topological Data Analysis of Phase Transitions in Complex Physical Systems**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18408156.svg)](https://doi.org/10.5281/zenodo.18408156)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-2021-orange.svg)](https://www.rust-lang.org/)

Extension of TDA-CUSUM methodology to 3D systems with H1/H2 homology, glass transitions, and molecular fluids.

## Overview

This repository extends the topological approach to phase transitions from 2D to more complex physical systems:

- **3D Lennard-Jones** - Crystallization with H1 (loops) and H2 (voids) homology
- **Kob-Andersen Glass** - Binary mixture vitrification with variable quench rates
- **TIP4P Water** - Realistic water model with SHAKE constraints and Wolf electrostatics

### Important Note: Euler Approximation

All persistence entropy calculations in this repository use the **Euler approximation**:

```
beta_1 = E - V + beta_0 - F
```

This approximation:
- Provides fast computation O(n^2)
- Counts features but loses birth/death information
- Detects transitions AFTER physical signatures (not precursors)
- Useful for phase classification, not early warning

For **precursor detection** (t_CUSUM < t_physical), use exact persistent homology as implemented in [TDA-Information-Dynamics](https://github.com/Yatrogenesis/TDA-Information-Dynamics).

## Results Summary

| System | N | Transition | Success | Q6 (Steinhardt) | Method |
|--------|---|------------|---------|-----------------|--------|
| LJ 3D (pure) | 256 | CRYSTAL | 100% | 0.56 | Euler H1+H2 |
| LJ 3D (pure) | 500 | CRYSTAL | 100% | 0.56 | Euler H1+H2 |
| LJ Glass (KA) | 256 | GLASS | 100% | 0.29 | Euler H1 |
| TIP4P Water | 64 | LIQUID* | - | Q4=0.19 | H-bond |

**Steinhardt Q6 Reference Values** (PRB 28, 784, 1983):
- FCC crystal: 0.574
- HCP crystal: 0.485
- Liquid/Glass: 0.28-0.35

\* TIP4P remains liquid (ice nucleation requires microseconds)

\* TIP4P remains liquid (ice nucleation requires microseconds)

## Systems

### 1. LJ 3D - Lennard-Jones Crystallization

3D monocomponent LJ system with FCC crystallization.

**Physics:**
- Velocity Verlet integration
- Berendsen thermostat
- Melt-recrystallize protocol

**Order Parameters:**
- Steinhardt Q6 (spherical harmonics Y_6m)
- H1 entropy (loop distribution)
- H2 entropy (void distribution)

**Result:** Q6 = 0.56 confirms FCC crystallization (literature: 0.574)

```bash
cd lj_3d && cargo run --release
```

### 2. LJ Glass - Kob-Andersen Binary Mixture

80% A + 20% B mixture with size disparity that frustrates crystallization.

**Physics:**
- Binary LJ with KA parameters (PRE 51, 4626, 1995)
- Variable quench rate study (TTT diagram)
- Dynamical arrest detection (MSD plateau)

**Quench Rate Study:**

| Quench Rate | Steps | Glass% | Q6 | MSD |
|-------------|-------|--------|-----|-----|
| ~10^14 K/s | 1 | 100% | 0.29 | 1.06 |
| ~10^12 K/s | 100 | 100% | 0.29 | 0.22 |
| ~10^11 K/s | 1000 | 100% | 0.29 | 0.08 |
| ~10^10 K/s | 5000 | 100% | 0.29 | 0.03 |
| ~10^9 K/s | 20000 | 100% | 0.30 | 0.01 |

**Result:** KA mixture vitrifies at ALL quench rates (excellent glass-former)

```bash
cd lj_glass && cargo run --release
```

### 3. TIP4P Water

Realistic TIP4P/2005 water model.

**Physics:**
- 4-site model (O, H1, H2, M)
- SHAKE constraints for rigid geometry
- Wolf summation for electrostatics
- Q4 tetrahedral order parameter

**Status:** Physically correct but ice nucleation requires longer timescales (microseconds)

```bash
cd tip4p && cargo run --release
```

## Technical Details

### Steinhardt Order Parameters

Correct implementation using real spherical harmonics:

```rust
// Q6 = sqrt(4*pi/13 * sum_m |q_6m|^2)
// where q_6m = (1/N_b) * sum_j Y_6m(theta_ij, phi_ij)
```

Normalized spherical harmonics Y_6m with:
- 13 components (m = -6 to +6)
- Correct normalization: integral |Y_lm|^2 dOmega = 1
- Reference: Steinhardt, Nelson, Ronchetti, PRB 28, 784 (1983)

### Euler Approximation for Persistence

H1 entropy approximation from edge distribution:
```rust
fn persistence_entropy_h1(dm: &Array2<f64>) -> f64 {
    // Bin edge distances, compute histogram differences
    // Shannon entropy over "lifetime" proxies
}
```

H2 entropy approximation from void distribution:
```rust
fn persistence_entropy_h2(dm: &Array2<f64>) -> f64 {
    // Analyze gaps in distance distribution
    // Entropy of empty space statistics
}
```

**Limitation:** These approximations detect phase changes but NOT precursors.

## Repository Structure

```
TDA-Complex-Systems/
+-- lj_3d/                  # 3D Lennard-Jones crystallization
|   +-- src/main.rs         # ~730 lines
|   +-- Cargo.toml
+-- lj_glass/               # Kob-Andersen glass-former
|   +-- src/main.rs         # ~740 lines
|   +-- Cargo.toml
+-- tip4p/                  # TIP4P/2005 water
|   +-- src/main.rs         # ~650 lines
|   +-- Cargo.toml
+-- results/                # JSON output files
|   +-- lj3d_N256.json
|   +-- lj3d_N500.json
|   +-- lj_glass_variable_quench.json
|   +-- tip4p_N64.json
+-- figures/                # (empty, for future plots)
+-- paper/                  # (empty, for future paper)
```

## Key Findings

1. **Crystal vs Glass Discrimination:**
   - LJ pure: Crystallizes to FCC (Q6 ~ 0.56)
   - KA mixture: Vitrifies to glass (Q6 ~ 0.29)
   - Steinhardt Q6 correctly classifies phases

2. **Quench Rate Independence (KA):**
   - Glass forms at ALL quench rates tested
   - Q6 remains constant (~0.29) regardless of cooling speed
   - MSD decreases with slower quench (better relaxation)

3. **TDA Method Comparison:**
   - Euler approximation: Phase classification (post-transition)
   - Exact persistence: Precursor detection (pre-transition)
   - For early warning, use [TDA-Information-Dynamics](https://github.com/Yatrogenesis/TDA-Information-Dynamics)

## Requirements

- Rust 1.70+ with cargo
- Dependencies: ndarray, rand, rayon, serde, num-complex

## Build and Run

```bash
# Build all projects
cd lj_3d && cargo build --release
cd ../lj_glass && cargo build --release
cd ../tip4p && cargo build --release

# Run validation
cd lj_3d && cargo run --release
cd ../lj_glass && cargo run --release
cd ../tip4p && cargo run --release
```

## Related Projects

- **[TDA-Phase-Transitions](https://github.com/Yatrogenesis/TDA-Phase-Transitions)** (Python)
  DOI: [10.5281/zenodo.18220298](https://doi.org/10.5281/zenodo.18220298)
  2D Lennard-Jones with exact persistent homology (Ripser)

- **[TDA-Information-Dynamics](https://github.com/Yatrogenesis/TDA-Information-Dynamics)** (Rust)
  General TDA-CUSUM framework with exact AND Euler persistence
  5 dynamical systems validated

## References

1. Steinhardt, P. J., Nelson, D. R., & Ronchetti, M. (1983). Bond-orientational order in liquids and glasses. *Physical Review B*, 28(2), 784.

2. Kob, W., & Andersen, H. C. (1995). Testing mode-coupling theory for a supercooled binary Lennard-Jones mixture. *Physical Review E*, 51(5), 4626.

3. Abascal, J. L. F., & Vega, C. (2005). A general purpose model for the condensed phases of water: TIP4P/2005. *Journal of Chemical Physics*, 123(23), 234505.

4. Wolf, D., et al. (1999). Exact method for the simulation of Coulombic systems by spherically truncated, pairwise r^-1 summation. *Journal of Chemical Physics*, 110(17), 8254.

## Author

**Francisco Molina-Burgos**
Avermex Research Division
Merida, Yucatan, Mexico
2026

## License

MIT License - See [LICENSE](LICENSE) for details.
