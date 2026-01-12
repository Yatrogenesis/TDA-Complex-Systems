# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-01-10

### Added

**LJ 3D Crystallization** (`lj_3d/`)
- 3D Lennard-Jones simulation with Velocity Verlet integration
- Steinhardt Q6 order parameter with correct spherical harmonics (Y_6m)
- H1 (loops) and H2 (voids) entropy approximation
- Melt-recrystallize protocol for crystallization
- FCC lattice initialization
- Parallel force computation with rayon
- Results: 100% crystallization, Q6 = 0.56 (FCC literature: 0.574)

**LJ Glass** (`lj_glass/`)
- Kob-Andersen binary mixture (80% A, 20% B)
- Standard KA parameters (PRE 51, 4626, 1995)
- Variable quench rate study (TTT diagram)
- Dynamical arrest detection via MSD plateau
- Results: 100% vitrification at all quench rates, Q6 ~ 0.29

**TIP4P Water** (`tip4p/`)
- TIP4P/2005 model parameters
- SHAKE algorithm for rigid molecule constraints
- Wolf summation for electrostatics
- Q4 tetrahedral order parameter
- H-bond network topology analysis
- Status: Physically correct, ice nucleation requires longer timescales

**Infrastructure**
- Results directory with JSON output files
- Comprehensive README with results tables
- MIT License

### Technical Notes

- All persistence entropy uses **Euler approximation** (not exact)
- Euler detects transitions AFTER physical signatures
- For precursor detection, see TDA-Information-Dynamics (exact persistence)
- Steinhardt Q6 correctly discriminates crystal (0.57) from glass (0.29)

---

## Links

- [Repository](https://github.com/Yatrogenesis/TDA-Complex-Systems)
- [Related: TDA-Phase-Transitions (Python)](https://github.com/Yatrogenesis/TDA-Phase-Transitions)
- [Related: TDA-Information-Dynamics (Rust)](https://github.com/Yatrogenesis/TDA-Information-Dynamics)
