# TDA-Complex-Systems

**Topological Data Analysis of Phase Transitions in Complex Systems**

Companion project to [TDA-Phase-Transitions](https://github.com/Yatrogenesis/TDA-Phase-Transitions), extending the TDA-CUSUM methodology to more complex physical systems.

## Results Summary

| System | N | Transition | Success | Order Parameter | Time/trial |
|--------|---|------------|---------|-----------------|------------|
| LJ 3D (pure) | 256 | CRYSTAL | 100% | Q6=0.95 | 2.8s |
| LJ 3D (pure) | 500 | CRYSTAL | 100% | Q6=0.95 | 9.4s |
| LJ Glass (KA) | 256 | GLASS | 100% | Q6=0.09 | 10.6s |
| TIP4P Water | 64 | LIQUID* | 0% | Q4=0.19 | 0.5s |

\* TIP4P remains liquid (ice nucleation requires microseconds)

## Systems Studied

### 1. LJ 3D - Lennard-Jones Pure (Crystallization)
- **Physics**: Monocomponent LJ in 3D with gradual cooling
- **Transition**: Liquid → FCC/HCP crystal
- **Detection**: Q6 bond-orientational order + H1/H2 homology
- **Status**: COMPLETE - 100% crystallization

### 2. LJ Glass - Kob-Andersen Binary Mixture (Vitrification)
- **Physics**: 80% A + 20% B with size disparity (frustrates crystallization)
- **Transition**: Liquid → Amorphous glass
- **Detection**: MSD plateau, D→0 (dynamical arrest)
- **Variable Quench Study**:

| Quench Rate | Steps | Glass% | Q6 | MSD |
|-------------|-------|--------|-----|-----|
| instantaneous (~10^14 K/s) | 1 | 100% | 0.096 | 1.06 |
| ultra_fast (~10^12 K/s) | 100 | 100% | 0.084 | 0.22 |
| fast (~10^11 K/s) | 1000 | 100% | 0.090 | 0.08 |
| moderate (~10^10 K/s) | 5000 | 100% | 0.083 | 0.03 |
| slow (~10^9 K/s) | 20000 | 100% | 0.086 | 0.01 |

- **Status**: COMPLETE - System vitrifies correctly at all quench rates

### 3. TIP4P Water (Ice Nucleation)
- **Physics**: TIP4P/2005 model with SHAKE constraints + Wolf electrostatics
- **Transition**: Liquid → Ice Ih (not observed in ns timescale)
- **Detection**: Q4 tetrahedral order + H-bond network topology
- **Status**: Physically correct (needs longer simulations for nucleation)

## Methodology

Extension of CUSUM-based precursor detection from 2D to:

1. **3D Homology**: Both H1 (loops) and H2 (voids/cavities)
2. **Molecular Systems**: Hydrogen bond network topology
3. **Amorphous States**: Topological signatures without long-range order
4. **Variable Quench Rates**: TTT (Time-Temperature-Transformation) study

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

1. **Crystallization vs Vitrification**:
   - LJ pure: Always crystallizes (Q6→0.95)
   - KA mixture: Always vitrifies (Q6~0.09) due to size frustration

2. **Quench Rate Effects**:
   - Fast quench: High residual stress (MSD~1.0)
   - Slow quench: Well-relaxed glass (MSD~0.01)
   - KA mixture vitrifies at ALL rates (excellent glass-former)

3. **TDA Applicability**:
   - Entropy approximation detects AFTER physical transition
   - Real TDA (ripser) expected to detect precursors

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
