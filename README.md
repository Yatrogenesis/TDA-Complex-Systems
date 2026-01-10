# TDA-Complex-Systems

**Topological Data Analysis of Phase Transitions in Complex Systems**

Companion paper to [TDA-Phase-Transitions](https://github.com/Yatrogenesis/TDA-Phase-Transitions), extending the TDA-CUSUM methodology to more complex physical systems.

## Systems Studied

| System | Potential | Transition | TDA Features | Status |
|--------|-----------|------------|--------------|--------|
| LJ 3D | Lennard-Jones | Liquid → FCC/HCP | H1 + H2 | In progress |
| TIP4P Water | 4-site rigid | Liquid → Ice Ih | H1 (H-bonds) | Planned |
| LJ Glass | Lennard-Jones | Supercooled → Glass | H1 (no order) | Planned |

## Methodology

Extension of the CUSUM-based precursor detection from 2D to:

1. **3D Homology**: Both H1 (loops) and H2 (voids/cavities)
2. **Molecular Systems**: Hydrogen bond topology
3. **Amorphous States**: Topological signatures without long-range order

## Repository Structure

```
TDA-Complex-Systems/
├── lj_3d/              # 3D Lennard-Jones simulations
│   ├── src/            # Rust implementation
│   └── results/        # Validation data
├── tip4p/              # TIP4P water simulations
│   ├── src/
│   └── results/
├── lj_glass/           # LJ glass-forming system
│   ├── src/
│   └── results/
├── paper/              # LaTeX manuscript
└── figures/            # Publication figures
```

## Key Questions

1. Does H2 (voids) provide additional precursor information in 3D?
2. Do hydrogen bond networks show topological precursors before ice nucleation?
3. How does topological entropy behave during vitrification vs crystallization?

## Requirements

- Rust 1.70+ (primary implementation)
- Python 3.10+ with ripser, numpy (analysis)
- Optional: CUDA for GPU acceleration

## Author

Francisco Molina Burgos
Independent Researcher

## Related Work

- Original 2D study: [TDA-Phase-Transitions](https://github.com/Yatrogenesis/TDA-Phase-Transitions)

## License

MIT License
