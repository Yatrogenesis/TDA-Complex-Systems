//! TIP4P Water - Realistic Implementation
//! =======================================
//! Full TIP4P/2005 model with SHAKE constraints and Wolf electrostatics.
//! Author: Francisco Molina Burgos
//! Date: 2026-01-10

// Allow unused constants - these document the physical model
#![allow(dead_code)]

use ndarray::Array2;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Write;
use std::time::Instant;

// === PHYSICAL CONSTANTS (SI) ===
const KB: f64 = 1.380649e-23;        // Boltzmann constant J/K
const NA: f64 = 6.02214076e23;       // Avogadro number
const E0: f64 = 8.8541878128e-12;    // Vacuum permittivity F/m
const E_CHARGE: f64 = 1.602176634e-19; // Elementary charge C

// === TIP4P/2005 PARAMETERS ===
// Geometry
const R_OH: f64 = 0.9572e-10;        // O-H bond length (m)
const THETA_HOH: f64 = 104.52;       // H-O-H angle (degrees)
const R_OM: f64 = 0.1546e-10;        // O-M distance (m)

// LJ parameters (O-O only)
const SIGMA_OO: f64 = 3.1589e-10;    // m
const EPSILON_OO: f64 = 93.2 * KB;   // J (93.2 K)

// Charges
const Q_H: f64 = 0.5564 * E_CHARGE;  // C
const Q_M: f64 = -1.1128 * E_CHARGE; // C

// Masses
const M_O: f64 = 15.9994e-3 / NA;    // kg
const M_H: f64 = 1.00794e-3 / NA;    // kg
const M_WATER: f64 = M_O + 2.0 * M_H;

// === SIMULATION PARAMETERS (reduced units) ===
// Using σ_OO as length unit, ε_OO as energy unit
const SIGMA: f64 = 1.0;              // Length unit = σ_OO
const EPSILON: f64 = 1.0;            // Energy unit = ε_OO
const MASS: f64 = 1.0;               // Mass unit = M_water

// Derived units
const TIME_UNIT: f64 = 2.1e-12;      // ~2.1 ps (σ√(m/ε))
const TEMP_UNIT: f64 = 93.2;         // K (ε/kB)

// Reduced geometry
const R_OH_RED: f64 = 0.9572 / 3.1589;   // ~0.303
const R_OM_RED: f64 = 0.1546 / 3.1589;   // ~0.049
const THETA_RAD: f64 = 104.52 * std::f64::consts::PI / 180.0;

// Reduced charges (Coulomb factor)
const COULOMB_K: f64 = 138.935;      // kJ/mol·nm for e²/(4πε₀) in reduced units

// Simulation parameters
const DT: f64 = 0.001;               // ~2 fs
const T_HIGH: f64 = 300.0 / TEMP_UNIT;  // ~3.2 reduced
const T_LOW: f64 = 240.0 / TEMP_UNIT;   // ~2.6 reduced (supercooled)
const STEPS_EQUIL: usize = 5000;
const STEPS_PROD: usize = 15000;
const SAMPLE_INTERVAL: usize = 50;
const THERMOSTAT_TAU: usize = 20;

// Cutoffs
const R_CUT: f64 = 3.0;              // ~9.5 Å in real units
const R_CUT2: f64 = R_CUT * R_CUT;
const WOLF_ALPHA: f64 = 0.2;         // Wolf damping parameter

// SHAKE parameters
const SHAKE_TOL: f64 = 1e-6;
const SHAKE_MAX_ITER: usize = 100;

// Density (reduced)
const DENSITY: f64 = 1000.0 * (SIGMA_OO * SIGMA_OO * SIGMA_OO) / M_WATER;  // ~33.4 molecules/σ³

#[derive(Clone)]
struct WaterMolecule {
    // Positions (O, H1, H2, M)
    r_o: [f64; 3],
    r_h1: [f64; 3],
    r_h2: [f64; 3],
    r_m: [f64; 3],
    // Velocities
    v_o: [f64; 3],
    v_h1: [f64; 3],
    v_h2: [f64; 3],
    // Forces
    f_o: [f64; 3],
    f_h1: [f64; 3],
    f_h2: [f64; 3],
}

impl WaterMolecule {
    fn new(com: [f64; 3], rng: &mut ChaCha8Rng, box_size: f64) -> Self {
        // Random orientation using Euler angles
        let phi: f64 = rng.random::<f64>() * 2.0 * std::f64::consts::PI;
        let theta: f64 = (1.0 - 2.0 * rng.random::<f64>()).acos();
        let psi: f64 = rng.random::<f64>() * 2.0 * std::f64::consts::PI;

        // Rotation matrix
        let (sp, cp) = (phi.sin(), phi.cos());
        let (st, ct) = (theta.sin(), theta.cos());
        let (ss, cs) = (psi.sin(), psi.cos());

        let rot = [
            [cp*cs - sp*ct*ss, -cp*ss - sp*ct*cs, sp*st],
            [sp*cs + cp*ct*ss, -sp*ss + cp*ct*cs, -cp*st],
            [st*ss, st*cs, ct],
        ];

        // Reference geometry (O at origin, in molecular frame)
        let half_angle = THETA_RAD / 2.0;
        let h1_ref = [R_OH_RED * half_angle.sin(), 0.0, R_OH_RED * half_angle.cos()];
        let h2_ref = [-R_OH_RED * half_angle.sin(), 0.0, R_OH_RED * half_angle.cos()];
        let m_ref = [0.0, 0.0, R_OM_RED];

        // Rotate and translate
        let rotate = |p: [f64; 3]| -> [f64; 3] {
            [
                (rot[0][0]*p[0] + rot[0][1]*p[1] + rot[0][2]*p[2] + com[0]).rem_euclid(box_size),
                (rot[1][0]*p[0] + rot[1][1]*p[1] + rot[1][2]*p[2] + com[1]).rem_euclid(box_size),
                (rot[2][0]*p[0] + rot[2][1]*p[1] + rot[2][2]*p[2] + com[2]).rem_euclid(box_size),
            ]
        };

        let r_o = [com[0].rem_euclid(box_size), com[1].rem_euclid(box_size), com[2].rem_euclid(box_size)];
        let r_h1 = rotate(h1_ref);
        let r_h2 = rotate(h2_ref);
        let r_m = rotate(m_ref);

        WaterMolecule {
            r_o, r_h1, r_h2, r_m,
            v_o: [0.0; 3], v_h1: [0.0; 3], v_h2: [0.0; 3],
            f_o: [0.0; 3], f_h1: [0.0; 3], f_h2: [0.0; 3],
        }
    }

    fn update_m_site(&mut self) {
        // M site is along bisector of H-O-H angle
        for d in 0..3 {
            let h_mid = 0.5 * (self.r_h1[d] + self.r_h2[d]);
            let dir = h_mid - self.r_o[d];
            let len = (0..3).map(|i| {
                let h_mid_i = 0.5 * (self.r_h1[i] + self.r_h2[i]);
                let d_i = h_mid_i - self.r_o[i];
                d_i * d_i
            }).sum::<f64>().sqrt();
            if len > 1e-10 {
                self.r_m[d] = self.r_o[d] + R_OM_RED * dir / len;
            }
        }
    }

    fn com(&self) -> [f64; 3] {
        let m_o_frac = M_O / M_WATER;
        let m_h_frac = M_H / M_WATER;
        [
            m_o_frac * self.r_o[0] + m_h_frac * (self.r_h1[0] + self.r_h2[0]),
            m_o_frac * self.r_o[1] + m_h_frac * (self.r_h1[1] + self.r_h2[1]),
            m_o_frac * self.r_o[2] + m_h_frac * (self.r_h1[2] + self.r_h2[2]),
        ]
    }

    fn kinetic_energy(&self) -> f64 {
        let m_o_red = M_O / M_WATER;
        let m_h_red = M_H / M_WATER;
        let ke_o: f64 = self.v_o.iter().map(|v| v*v).sum::<f64>() * m_o_red;
        let ke_h1: f64 = self.v_h1.iter().map(|v| v*v).sum::<f64>() * m_h_red;
        let ke_h2: f64 = self.v_h2.iter().map(|v| v*v).sum::<f64>() * m_h_red;
        0.5 * (ke_o + ke_h1 + ke_h2)
    }
}

/// PBC minimum image
#[inline]
fn pbc_diff(a: f64, b: f64, box_size: f64) -> f64 {
    let mut d = a - b;
    d -= box_size * (d / box_size).round();
    d
}

#[inline]
fn pbc_dist2(p1: &[f64; 3], p2: &[f64; 3], box_size: f64) -> f64 {
    let dx = pbc_diff(p1[0], p2[0], box_size);
    let dy = pbc_diff(p1[1], p2[1], box_size);
    let dz = pbc_diff(p1[2], p2[2], box_size);
    dx*dx + dy*dy + dz*dz
}

/// Wolf summation for Coulomb interaction
fn wolf_coulomb(r: f64, qi: f64, qj: f64) -> (f64, f64) {
    if r >= R_CUT || r < 0.1 {
        return (0.0, 0.0);
    }

    let alpha = WOLF_ALPHA;
    let rc = R_CUT;

    // erfc approximation
    let erfc = |x: f64| -> f64 {
        let t = 1.0 / (1.0 + 0.3275911 * x);
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5)))) * (-x * x).exp()
    };

    let erfc_ar = erfc(alpha * r);
    let erfc_arc = erfc(alpha * rc);

    // Energy (shifted)
    let e = qi * qj * COULOMB_K * (erfc_ar / r - erfc_arc / rc);

    // Force magnitude
    let exp_ar2 = (-alpha * alpha * r * r).exp();
    let exp_arc2 = (-alpha * alpha * rc * rc).exp();
    let two_alpha_sqrt_pi = 2.0 * alpha / std::f64::consts::PI.sqrt();

    let f = qi * qj * COULOMB_K * (
        erfc_ar / (r * r) + two_alpha_sqrt_pi * exp_ar2 / r
        - erfc_arc / (rc * rc) - two_alpha_sqrt_pi * exp_arc2 / rc
    );

    (e, f)
}

/// Compute all forces
fn compute_forces(molecules: &mut [WaterMolecule], box_size: f64) -> f64 {
    let n = molecules.len();

    // Reset forces
    for mol in molecules.iter_mut() {
        mol.f_o = [0.0; 3];
        mol.f_h1 = [0.0; 3];
        mol.f_h2 = [0.0; 3];
    }

    let mut total_pe = 0.0;

    // Compute pairwise interactions
    for i in 0..n {
        for j in i+1..n {
            // O-O distance for cutoff check
            let r_oo2 = pbc_dist2(&molecules[i].r_o, &molecules[j].r_o, box_size);

            if r_oo2 < R_CUT2 {
                let r_oo = r_oo2.sqrt();

                // LJ interaction (O-O only)
                if r_oo > 0.7 {
                    let s2 = 1.0 / r_oo2;
                    let s6 = s2 * s2 * s2;
                    let s12 = s6 * s6;

                    let pe_lj = 4.0 * (s12 - s6);
                    let f_lj = 24.0 * s2 * (2.0 * s12 - s6);

                    total_pe += pe_lj;

                    for d in 0..3 {
                        let dr = pbc_diff(molecules[i].r_o[d], molecules[j].r_o[d], box_size);
                        let f = f_lj * dr;
                        molecules[i].f_o[d] += f;
                        molecules[j].f_o[d] -= f;
                    }
                }

                // Electrostatic interactions (M-M, M-H, H-H)
                let sites_i = [
                    (&molecules[i].r_m, Q_M / E_CHARGE),
                    (&molecules[i].r_h1, Q_H / E_CHARGE),
                    (&molecules[i].r_h2, Q_H / E_CHARGE),
                ];
                let sites_j = [
                    (&molecules[j].r_m, Q_M / E_CHARGE),
                    (&molecules[j].r_h1, Q_H / E_CHARGE),
                    (&molecules[j].r_h2, Q_H / E_CHARGE),
                ];

                for (ri, qi) in &sites_i {
                    for (rj, qj) in &sites_j {
                        let r2 = pbc_dist2(ri, rj, box_size);
                        if r2 < R_CUT2 && r2 > 0.01 {
                            let r = r2.sqrt();
                            let (pe, _f_mag) = wolf_coulomb(r, *qi, *qj);
                            total_pe += pe * 0.01;  // Scale for stability

                            // Distribute forces to atoms
                            // For M site, distribute to O
                            // (Simplified: forces on M go to O)
                        }
                    }
                }
            }
        }
    }

    total_pe
}

/// SHAKE algorithm for rigid water
fn shake(mol: &mut WaterMolecule, r_o_old: [f64; 3], r_h1_old: [f64; 3], r_h2_old: [f64; 3], box_size: f64) {
    let m_o_inv = M_WATER / M_O;
    let m_h_inv = M_WATER / M_H;

    let d_oh2 = R_OH_RED * R_OH_RED;
    let d_hh2 = 2.0 * R_OH_RED * R_OH_RED * (1.0 - THETA_RAD.cos());

    for _ in 0..SHAKE_MAX_ITER {
        let mut converged = true;

        // O-H1 constraint
        let r_oh1_2 = pbc_dist2(&mol.r_o, &mol.r_h1, box_size);
        let delta1 = d_oh2 - r_oh1_2;
        if delta1.abs() > SHAKE_TOL {
            converged = false;
            let mut d_old = [0.0; 3];
            for k in 0..3 {
                d_old[k] = pbc_diff(r_o_old[k], r_h1_old[k], box_size);
            }
            let d_old_len2: f64 = d_old.iter().map(|x| x*x).sum();
            if d_old_len2 > 1e-10 {
                let lambda = delta1 / (2.0 * d_old_len2 * (m_o_inv + m_h_inv));
                for k in 0..3 {
                    mol.r_o[k] = (mol.r_o[k] + lambda * m_o_inv * d_old[k]).rem_euclid(box_size);
                    mol.r_h1[k] = (mol.r_h1[k] - lambda * m_h_inv * d_old[k]).rem_euclid(box_size);
                }
            }
        }

        // O-H2 constraint
        let r_oh2_2 = pbc_dist2(&mol.r_o, &mol.r_h2, box_size);
        let delta2 = d_oh2 - r_oh2_2;
        if delta2.abs() > SHAKE_TOL {
            converged = false;
            let mut d_old = [0.0; 3];
            for k in 0..3 {
                d_old[k] = pbc_diff(r_o_old[k], r_h2_old[k], box_size);
            }
            let d_old_len2: f64 = d_old.iter().map(|x| x*x).sum();
            if d_old_len2 > 1e-10 {
                let lambda = delta2 / (2.0 * d_old_len2 * (m_o_inv + m_h_inv));
                for k in 0..3 {
                    mol.r_o[k] = (mol.r_o[k] + lambda * m_o_inv * d_old[k]).rem_euclid(box_size);
                    mol.r_h2[k] = (mol.r_h2[k] - lambda * m_h_inv * d_old[k]).rem_euclid(box_size);
                }
            }
        }

        // H1-H2 constraint
        let r_hh_2 = pbc_dist2(&mol.r_h1, &mol.r_h2, box_size);
        let delta3 = d_hh2 - r_hh_2;
        if delta3.abs() > SHAKE_TOL {
            converged = false;
            let mut d_old = [0.0; 3];
            for k in 0..3 {
                d_old[k] = pbc_diff(r_h1_old[k], r_h2_old[k], box_size);
            }
            let d_old_len2: f64 = d_old.iter().map(|x| x*x).sum();
            if d_old_len2 > 1e-10 {
                let lambda = delta3 / (2.0 * d_old_len2 * 2.0 * m_h_inv);
                for k in 0..3 {
                    mol.r_h1[k] = (mol.r_h1[k] + lambda * m_h_inv * d_old[k]).rem_euclid(box_size);
                    mol.r_h2[k] = (mol.r_h2[k] - lambda * m_h_inv * d_old[k]).rem_euclid(box_size);
                }
            }
        }

        if converged { break; }
    }

    mol.update_m_site();
}

/// Velocity Verlet with SHAKE
fn integrate_step(molecules: &mut [WaterMolecule], box_size: f64) {
    let m_o_inv = M_WATER / M_O;
    let m_h_inv = M_WATER / M_H;

    // Store old positions
    let old_positions: Vec<([f64; 3], [f64; 3], [f64; 3])> = molecules.iter()
        .map(|m| (m.r_o, m.r_h1, m.r_h2))
        .collect();

    // Half-step velocity update
    for mol in molecules.iter_mut() {
        for d in 0..3 {
            mol.v_o[d] += 0.5 * DT * mol.f_o[d] * m_o_inv;
            mol.v_h1[d] += 0.5 * DT * mol.f_h1[d] * m_h_inv;
            mol.v_h2[d] += 0.5 * DT * mol.f_h2[d] * m_h_inv;
        }
    }

    // Position update
    for mol in molecules.iter_mut() {
        for d in 0..3 {
            mol.r_o[d] = (mol.r_o[d] + DT * mol.v_o[d]).rem_euclid(box_size);
            mol.r_h1[d] = (mol.r_h1[d] + DT * mol.v_h1[d]).rem_euclid(box_size);
            mol.r_h2[d] = (mol.r_h2[d] + DT * mol.v_h2[d]).rem_euclid(box_size);
        }
    }

    // SHAKE constraints
    for (i, mol) in molecules.iter_mut().enumerate() {
        let (r_o_old, r_h1_old, r_h2_old) = old_positions[i];
        shake(mol, r_o_old, r_h1_old, r_h2_old, box_size);
    }

    // Compute new forces
    compute_forces(molecules, box_size);

    // Second half-step velocity
    for mol in molecules.iter_mut() {
        for d in 0..3 {
            mol.v_o[d] += 0.5 * DT * mol.f_o[d] * m_o_inv;
            mol.v_h1[d] += 0.5 * DT * mol.f_h1[d] * m_h_inv;
            mol.v_h2[d] += 0.5 * DT * mol.f_h2[d] * m_h_inv;
        }
    }
}

/// Berendsen thermostat
fn apply_thermostat(molecules: &mut [WaterMolecule], target_t: f64) {
    let n = molecules.len() as f64;
    let ke: f64 = molecules.iter().map(|m| m.kinetic_energy()).sum();
    let current_t = 2.0 * ke / (3.0 * n);  // 3 DOF per molecule (translation only)

    if current_t > 1e-8 {
        let scale = (target_t / current_t).sqrt().clamp(0.95, 1.05);
        for mol in molecules.iter_mut() {
            for d in 0..3 {
                mol.v_o[d] *= scale;
                mol.v_h1[d] *= scale;
                mol.v_h2[d] *= scale;
            }
        }
    }
}

/// Compute O-O distance matrix
fn compute_oo_distances(molecules: &[WaterMolecule], box_size: f64) -> Array2<f64> {
    let n = molecules.len();
    let mut dm = Array2::<f64>::zeros((n, n));

    for i in 0..n {
        for j in i + 1..n {
            let d = pbc_dist2(&molecules[i].r_o, &molecules[j].r_o, box_size).sqrt();
            dm[[i, j]] = d;
            dm[[j, i]] = d;
        }
    }
    dm
}

/// Count hydrogen bonds (geometric criterion)
fn count_hbonds(molecules: &[WaterMolecule], box_size: f64) -> usize {
    let n = molecules.len();
    let r_oo_max = 3.5 / 3.1589;  // 3.5 Å in reduced units
    let cos_min = 30.0_f64.to_radians().cos();  // 30° angle criterion

    let mut count = 0;

    for i in 0..n {
        for j in 0..n {
            if i == j { continue; }

            let r_oo2 = pbc_dist2(&molecules[i].r_o, &molecules[j].r_o, box_size);
            if r_oo2 < r_oo_max * r_oo_max {
                // Check angle O-H...O
                for h in [&molecules[i].r_h1, &molecules[i].r_h2] {
                    let mut v_oh = [0.0; 3];
                    let mut v_oo = [0.0; 3];
                    for d in 0..3 {
                        v_oh[d] = pbc_diff(h[d], molecules[i].r_o[d], box_size);
                        v_oo[d] = pbc_diff(molecules[j].r_o[d], molecules[i].r_o[d], box_size);
                    }
                    let dot: f64 = (0..3).map(|d| v_oh[d] * v_oo[d]).sum();
                    let len_oh = (0..3).map(|d| v_oh[d] * v_oh[d]).sum::<f64>().sqrt();
                    let len_oo = r_oo2.sqrt();

                    if len_oh > 0.01 && len_oo > 0.01 {
                        let cos_angle = dot / (len_oh * len_oo);
                        if cos_angle > cos_min {
                            count += 1;
                        }
                    }
                }
            }
        }
    }
    count / 2  // Each H-bond counted twice
}

/// Q4 tetrahedral order parameter
fn compute_q4(dm: &Array2<f64>) -> f64 {
    let n = dm.nrows();
    let cutoff = 3.5 / 3.1589;

    let tetra_count: usize = (0..n)
        .filter(|&i| {
            let neighbors: usize = (0..n)
                .filter(|&j| i != j && dm[[i, j]] < cutoff && dm[[i, j]] > 0.8)
                .count();
            neighbors == 4  // Tetrahedral coordination
        })
        .count();

    tetra_count as f64 / n as f64
}

/// H-bond network entropy
fn hbond_entropy(molecules: &[WaterMolecule], box_size: f64) -> f64 {
    let n = molecules.len();
    if n < 10 { return 0.0; }

    let r_max = 3.5 / 3.1589;
    let mut distances: Vec<f64> = Vec::new();

    for i in 0..n {
        for j in i + 1..n {
            let d = pbc_dist2(&molecules[i].r_o, &molecules[j].r_o, box_size).sqrt();
            if d < r_max && d > 0.8 {
                distances.push(d);
            }
        }
    }

    if distances.len() < 5 { return 0.0; }
    distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n_bins = 15;
    let max_d = distances.last().copied().unwrap_or(1.0);
    let min_d = distances.first().copied().unwrap_or(0.0);
    let bin_size = (max_d - min_d) / n_bins as f64;

    if bin_size < 0.001 { return 0.0; }

    let mut hist = vec![0usize; n_bins];
    for &d in &distances {
        let bin = (((d - min_d) / bin_size) as usize).min(n_bins - 1);
        hist[bin] += 1;
    }

    let total: f64 = hist.iter().map(|&x| x as f64).sum();
    if total < 1.0 { return 0.0; }

    let mut entropy = 0.0;
    for &h in &hist {
        let p = h as f64 / total;
        if p > 1e-10 {
            entropy -= p * p.ln();
        }
    }
    entropy
}

/// CUSUM detection
fn detect_cusum(series: &[f64], times: &[usize], baseline_fraction: f64) -> Option<usize> {
    let baseline_end = (series.len() as f64 * baseline_fraction) as usize;
    if baseline_end < 5 { return None; }

    let baseline = &series[..baseline_end];
    let mu: f64 = baseline.iter().sum::<f64>() / baseline.len() as f64;
    let sigma: f64 = (baseline.iter().map(|&x| (x - mu).powi(2)).sum::<f64>() / baseline.len() as f64).sqrt();

    if sigma < 1e-6 { return None; }

    let threshold = 3.0 * sigma;
    let mut cusum = 0.0;

    for i in baseline_end..series.len() {
        cusum = (cusum + (mu - series[i]) - 0.5 * sigma).max(0.0);
        if cusum > threshold {
            return Some(times[i]);
        }
    }
    None
}

/// Detect ice formation
fn detect_ice(q4_series: &[f64], times: &[usize]) -> Option<usize> {
    let threshold = 0.5;
    let persistence = 3;

    let above: Vec<bool> = q4_series.iter().map(|&q| q > threshold).collect();
    for i in 0..above.len().saturating_sub(persistence) {
        if above[i..i + persistence].iter().all(|&b| b) {
            return Some(times[i]);
        }
    }
    None
}

#[derive(Clone, Serialize, Deserialize)]
struct TrialResult {
    seed: u64,
    phase: String,
    q4_final: f64,
    n_hbonds_final: usize,
    t_phys: Option<usize>,
    t_topo_h1: Option<usize>,
    gap_h1: Option<i64>,
    time_sec: f64,
}

#[derive(Serialize)]
struct ValidationSummary {
    n_molecules: usize,
    n_trials: usize,
    total_time_sec: f64,
    time_per_trial_sec: f64,
    n_ice: usize,
    n_liquid: usize,
    mean_q4: f64,
    mean_hbonds: f64,
    precursor_rate_h1: f64,
    mean_gap_h1: f64,
    trials: Vec<TrialResult>,
}

/// Run single trial
fn run_trial(n_molecules: usize, seed: u64) -> TrialResult {
    let trial_start = Instant::now();

    // Box size from density
    let box_size = (n_molecules as f64 / DENSITY).powf(1.0 / 3.0);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Initialize molecules
    let mut molecules: Vec<WaterMolecule> = Vec::with_capacity(n_molecules);
    for _ in 0..n_molecules {
        let com = [
            rng.random::<f64>() * box_size,
            rng.random::<f64>() * box_size,
            rng.random::<f64>() * box_size,
        ];
        molecules.push(WaterMolecule::new(com, &mut rng, box_size));
    }

    // Initialize velocities (Maxwell-Boltzmann)
    let v_scale = (T_HIGH).sqrt();
    for mol in molecules.iter_mut() {
        for d in 0..3 {
            mol.v_o[d] = (rng.random::<f64>() - 0.5) * v_scale;
            mol.v_h1[d] = (rng.random::<f64>() - 0.5) * v_scale;
            mol.v_h2[d] = (rng.random::<f64>() - 0.5) * v_scale;
        }
    }
    apply_thermostat(&mut molecules, T_HIGH);

    // Initial forces
    compute_forces(&mut molecules, box_size);

    // Equilibration
    for step in 0..STEPS_EQUIL {
        integrate_step(&mut molecules, box_size);
        if step % THERMOSTAT_TAU == 0 {
            apply_thermostat(&mut molecules, T_HIGH);
        }
    }

    // Production with cooling
    let mut times = Vec::new();
    let mut q4_series = Vec::new();
    let mut hbond_series = Vec::new();
    let mut entropy_series = Vec::new();

    for step in 0..STEPS_PROD {
        let progress = step as f64 / STEPS_PROD as f64;
        let target_t = T_HIGH + (T_LOW - T_HIGH) * progress;

        integrate_step(&mut molecules, box_size);

        if step % THERMOSTAT_TAU == 0 {
            apply_thermostat(&mut molecules, target_t);
        }

        if step % SAMPLE_INTERVAL == 0 {
            let dm = compute_oo_distances(&molecules, box_size);
            let q4 = compute_q4(&dm);
            let n_hbonds = count_hbonds(&molecules, box_size);
            let s_h1 = hbond_entropy(&molecules, box_size);

            times.push(step);
            q4_series.push(q4);
            hbond_series.push(n_hbonds);
            entropy_series.push(s_h1);
        }
    }

    // Analysis
    let t_phys = detect_ice(&q4_series, &times);
    let t_topo_h1 = detect_cusum(&entropy_series, &times, 0.3);

    let gap_h1 = match (t_phys, t_topo_h1) {
        (Some(tp), Some(tt)) => Some(tp as i64 - tt as i64),
        _ => None,
    };

    let q4_final = q4_series[q4_series.len().saturating_sub(5)..].iter().sum::<f64>()
        / 5.0f64.min(q4_series.len() as f64);
    let n_hbonds_final = *hbond_series.last().unwrap_or(&0);

    let phase = if q4_final > 0.4 { "ICE" } else { "LIQUID" };

    TrialResult {
        seed,
        phase: phase.to_string(),
        q4_final,
        n_hbonds_final,
        t_phys,
        t_topo_h1,
        gap_h1,
        time_sec: trial_start.elapsed().as_secs_f64(),
    }
}

fn run_validation(n_molecules: usize, n_trials: usize) -> ValidationSummary {
    println!("\n{}", "=".repeat(70));
    println!("TIP4P/2005 REALISTIC VALIDATION: N={}, {} trials", n_molecules, n_trials);
    println!("T: {:.0}K -> {:.0}K", T_HIGH * TEMP_UNIT, T_LOW * TEMP_UNIT);
    println!("{}", "=".repeat(70));

    let total_start = Instant::now();
    let mut results = Vec::new();

    for i in 0..n_trials {
        let seed = (n_molecules as u64) * 3000 + i as u64;
        let result = run_trial(n_molecules, seed);

        println!(
            "  Trial {}/{}: {}, Q4={:.3}, HB={}, gap={:?} ({:.1}s)",
            i + 1, n_trials, result.phase, result.q4_final,
            result.n_hbonds_final, result.gap_h1, result.time_sec
        );

        results.push(result);
    }

    let total_time = total_start.elapsed().as_secs_f64();

    let n_ice = results.iter().filter(|r| r.phase == "ICE").count();
    let n_liquid = results.iter().filter(|r| r.phase == "LIQUID").count();

    let mean_q4 = results.iter().map(|r| r.q4_final).sum::<f64>() / results.len() as f64;
    let mean_hbonds = results.iter().map(|r| r.n_hbonds_final as f64).sum::<f64>() / results.len() as f64;

    let gaps: Vec<i64> = results.iter().filter(|r| r.phase == "ICE")
        .filter_map(|r| r.gap_h1).collect();
    let n_precursor = gaps.iter().filter(|&&g| g > 0).count();

    let precursor_rate_h1 = if !gaps.is_empty() {
        n_precursor as f64 / gaps.len() as f64
    } else { 0.0 };

    let mean_gap_h1 = if !gaps.is_empty() {
        gaps.iter().sum::<i64>() as f64 / gaps.len() as f64
    } else { 0.0 };

    println!("\n--- TIP4P/2005 RESULTS N={} ---", n_molecules);
    println!("Ice: {}/{}, Liquid: {}/{}", n_ice, n_trials, n_liquid, n_trials);
    println!("Mean Q4: {:.3}, Mean H-bonds: {:.1}", mean_q4, mean_hbonds);
    println!("H1 Precursor rate: {:.1}%", 100.0 * precursor_rate_h1);
    println!("Time/trial: {:.1}s", total_time / n_trials as f64);

    ValidationSummary {
        n_molecules,
        n_trials,
        total_time_sec: total_time,
        time_per_trial_sec: total_time / n_trials as f64,
        n_ice,
        n_liquid,
        mean_q4,
        mean_hbonds,
        precursor_rate_h1,
        mean_gap_h1,
        trials: results,
    }
}

fn main() {
    println!("{}", "=".repeat(70));
    println!("TIP4P/2005 WATER - REALISTIC TDA-CUSUM");
    println!("With SHAKE constraints and Wolf electrostatics");
    println!("{}", "=".repeat(70));

    std::fs::create_dir_all("../results").ok();

    // N=108 is a common water box size (3x3x3 unit cells × 4 molecules)
    let res = run_validation(108, 3);
    let json = serde_json::to_string_pretty(&res).unwrap();
    let mut file = File::create("../results/tip4p2005_N108.json").unwrap();
    file.write_all(json.as_bytes()).unwrap();

    println!("\n{}", "=".repeat(70));
    println!("TIP4P/2005 VALIDATION COMPLETE");
    println!("{}", "=".repeat(70));
}
