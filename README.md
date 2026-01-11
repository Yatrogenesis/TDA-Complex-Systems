# TDA-Complex-Systems

**Topological Data Analysis of Phase Transitions in Complex Systems**

Companion project to [TDA-Phase-Transitions](https://github.com/Yatrogenesis/TDA-Phase-Transitions), extending the TDA-CUSUM methodology to more complex physical systems.

## Results Summary

| System | N | Transition | Success | Q6 (Steinhardt) | Time/trial |
|--------|---|------------|---------|-----------------|------------|
| LJ 3D (pure) | 256 | CRYSTAL | 100% | 0.56 | 2.9s |
| LJ 3D (pure) | 500 | CRYSTAL | 100% | 0.56 | 9.7s |
| LJ Glass (KA) | 256 | GLASS | 100% | 0.29 | ~10s |
| TIP4P Water | 64 | LIQUID* | 0% | Q4=0.19 | 0.5s |

**Steinhardt Q6 Reference Values** (PRB 28, 784, 1983):
- FCC crystal: 0.574
- HCP crystal: 0.485
- Liquid/Glass: 0.28-0.35

\* TIP4P remains liquid (ice nucleation requires microseconds)

## Systems Studied

### 1. LJ 3D - Lennard-Jones Pure (Crystallization)
- **Physics**: Monocomponent LJ in 3D with gradual cooling
- **Transition**: Liquid → FCC crystal
- **Detection**: Steinhardt Q6 (spherical harmonics Y_6m) + H1/H2 homology
- **Result**: Q6 = 0.56 (FCC literature: 0.574)
- **Status**: COMPLETE - 100% crystallization

### 2. LJ Glass - Kob-Andersen Binary Mixture (Vitrification)
- **Physics**: 80% A + 20% B with size disparity (frustrates crystallization)
- **Transition**: Liquid → Amorphous glass
- **Detection**: Steinhardt Q6 + MSD plateau, D→0 (dynamical arrest)
- **Variable Quench Study**:

| Quench Rate | Steps | Glass% | Q6 | MSD |
|-------------|-------|--------|-----|-----|
| instantaneous (~10^14 K/s) | 1 | 100% | 0.29 | 1.06 |
| ultra_fast (~10^12 K/s) | 100 | 100% | 0.29 | 0.22 |
| fast (~10^11 K/s) | 1000 | 100% | 0.29 | 0.08 |
| moderate (~10^10 K/s) | 5000 | 100% | 0.29 | 0.03 |
| slow (~10^9 K/s) | 20000 | 100% | 0.30 | 0.01 |

- **Status**: COMPLETE - System vitrifies at all quench rates (Q6 confirms disorder)

### 3. TIP4P Water (Ice Nucleation)
- **Physics**: TIP4P/2005 model with SHAKE constraints + Wolf electrostatics
- **Transition**: Liquid → Ice Ih (not observed in ns timescale)
- **Detection**: Q4 tetrahedral order + H-bond network topology
- **Status**: Physically correct (needs longer simulations for nucleation)

## Methodology

Extension of CUSUM-based precursor detection from 2D to:

1. **3D Homology**: Both H1 (loops) and H2 (voids/cavities)
2. **Steinhardt Q6**: Real spherical harmonics Y_6m implementation
   - 13 components (m = -6 to +6)
   - Correctly normalized: ∫|Y_lm|² dΩ = 1
   - Reference: Steinhardt, Nelson, Ronchetti, PRB 28, 784 (1983)
3. **Molecular Systems**: Hydrogen bond network topology
4. **Amorphous States**: Topological signatures without long-range order
5. **Variable Quench Rates**: TTT (Time-Temperature-Transformation) study

## Repository Structure

```
TDA-Complex-Systems/
├── lj_3d/              # 3D Lennard-Jones (crystallization)
│   └── src/main.rs     # ~570 lines, H1+H2 homology
├── lj_glass/           # Kob-Andersen glass-former
│   └── src/main.rs     # ~737 lines, variable quench
├── tip4p/              # TIP4P/2005 water
│   └── src/main.rs     # SHAKE + Wolf summation
└── results/
    ├── lj3d_N256.json
    ├── lj3d_N500.json
    ├── lj_glass_variable_quench.json
    └── tip4p_N64.json
```

## Key Findings

1. **Crystallization vs Vitrification** (validated with real Steinhardt Q6):
   - LJ pure: Crystallizes to FCC (Q6 ~ 0.56, matches literature 0.574)
   - KA mixture: Vitrifies (Q6 ~ 0.29, matches liquid/glass 0.28-0.35)
   - The Q6 order parameter correctly discriminates crystal from glass

2. **Quench Rate Effects**:
   - Fast quench: High residual stress (MSD~1.0), frozen disorder
   - Slow quench: Well-relaxed glass (MSD~0.01), local relaxation
   - KA mixture vitrifies at ALL rates (excellent glass-former)
   - Q6 remains constant (~0.29) regardless of quench rate

3. **TDA Applicability**:
   - Entropy approximation detects AFTER physical transition
   - Real TDA (ripser) expected to detect precursors
   - Steinhardt Q6 provides ground truth for phase classification

## Requirements

- Rust 1.70+ (primary implementation)
- Python 3.10+ with ripser, numpy (analysis)

## Author

Francisco Molina Burgos
Independent Researcher

## Related Work

- Original 2D study: [TDA-Phase-Transitions](https://github.com/Yatrogenesis/TDA-Phase-Transitions)

## License

MIT License
