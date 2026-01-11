//! LJ Glass-Forming System - Kob-Andersen Binary Mixture
//! ======================================================
//! Variable quench rate study: glass vs crystal formation
//! Author: Francisco Molina Burgos
//! Date: 2026-01-10

use ndarray::Array2;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;
use std::fs::File;
use std::io::Write;
use std::time::Instant;
use num_complex::Complex64;

// === KOB-ANDERSEN PARAMETERS ===
// Standard parameters from Kob & Andersen, PRE 51, 4626 (1995)
const SIGMA_AA: f64 = 1.0;
const SIGMA_AB: f64 = 0.80;
const SIGMA_BB: f64 = 0.88;
const EPSILON_AA: f64 = 1.0;
const EPSILON_AB: f64 = 1.5;
const EPSILON_BB: f64 = 0.5;
const FRACTION_A: f64 = 0.8;   // 80% A, 20% B
const DENSITY: f64 = 1.2;      // Standard KA density

// === SIMULATION PARAMETERS ===
const DT: f64 = 0.005;         // Larger timestep OK for LJ
const T_LIQUID: f64 = 1.0;     // Equilibrate as liquid (T > Tm ~ 0.75)
const T_GLASS: f64 = 0.1;      // Far below Tg ~ 0.435

// Base timing
const STEPS_EQUIL_HOT: usize = 10000;   // Long equilibration at T_LIQUID
const STEPS_ANNEAL: usize = 15000;      // Anneal at T_GLASS
const SAMPLE_INTERVAL: usize = 50;
const THERMOSTAT_TAU: usize = 20;

const R_CUT: f64 = 2.5;
const R_MIN: f64 = 0.7;

// === QUENCH RATE CONFIGURATIONS ===
// Quench rate = (T_LIQUID - T_GLASS) / (steps_quench * dt) in reduced units
// Higher steps = slower quench = more time for nucleation = crystal
// Lower steps = faster quench = freeze disorder = glass

#[derive(Clone, Copy, Debug)]
struct QuenchConfig {
    name: &'static str,
    steps_quench: usize,       // Steps during quench phase
    rate_description: &'static str,
}

const QUENCH_RATES: [QuenchConfig; 5] = [
    QuenchConfig {
        name: "instantaneous",
        steps_quench: 1,           // Single step - ~10^14 K/s equivalent
        rate_description: "~10^14 K/s"
    },
    QuenchConfig {
        name: "ultra_fast",
        steps_quench: 100,         // ~10^12 K/s equivalent
        rate_description: "~10^12 K/s"
    },
    QuenchConfig {
        name: "fast",
        steps_quench: 1000,        // ~10^11 K/s equivalent
        rate_description: "~10^11 K/s"
    },
    QuenchConfig {
        name: "moderate",
        steps_quench: 5000,        // ~10^10 K/s - near critical cooling rate
        rate_description: "~10^10 K/s (near Rc)"
    },
    QuenchConfig {
        name: "slow",
        steps_quench: 20000,       // ~10^9 K/s - allows crystallization
        rate_description: "~10^9 K/s (crystallizes)"
    },
];

#[derive(Clone, Serialize, Deserialize)]
struct TrialResult {
    seed: u64,
    quench_name: String,
    quench_steps: usize,
    quench_rate: String,
    phase: String,
    q6_final: f64,
    msd_final: f64,
    diffusion_coeff: f64,
    t_arrest: Option<usize>,
    t_topo_h1: Option<usize>,
    gap_h1: Option<i64>,
}

#[derive(Serialize)]
struct QuenchSummary {
    quench_name: String,
    quench_steps: usize,
    quench_rate: String,
    n_trials: usize,
    n_glass: usize,
    n_crystal: usize,
    n_liquid: usize,
    glass_fraction: f64,
    mean_q6: f64,
    mean_msd: f64,
    mean_diff_coeff: f64,
}

#[derive(Serialize)]
struct ValidationSummary {
    n_particles: usize,
    n_trials_per_rate: usize,
    total_quench_rates: usize,
    total_time_sec: f64,
    quench_results: Vec<QuenchSummary>,
    trials: Vec<TrialResult>,
}

/// Get LJ parameters for particle pair
#[inline]
fn get_lj_params(type_i: u8, type_j: u8) -> (f64, f64) {
    match (type_i, type_j) {
        (0, 0) => (SIGMA_AA, EPSILON_AA),  // A-A
        (1, 1) => (SIGMA_BB, EPSILON_BB),  // B-B
        _ => (SIGMA_AB, EPSILON_AB),        // A-B
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
fn pbc_dist2(p1: &[f64], p2: &[f64], box_size: f64) -> f64 {
    let dx = pbc_diff(p1[0], p2[0], box_size);
    let dy = pbc_diff(p1[1], p2[1], box_size);
    let dz = pbc_diff(p1[2], p2[2], box_size);
    dx*dx + dy*dy + dz*dz
}

/// Compute distance matrix
fn compute_distance_matrix(pos: &Array2<f64>, box_size: f64) -> Array2<f64> {
    let n = pos.nrows();
    let mut dm = Array2::<f64>::zeros((n, n));

    for i in 0..n {
        for j in i + 1..n {
            let d = pbc_dist2(
                pos.row(i).as_slice().unwrap(),
                pos.row(j).as_slice().unwrap(),
                box_size
            ).sqrt();
            dm[[i, j]] = d;
            dm[[j, i]] = d;
        }
    }
    dm
}

/// Compute forces for binary LJ mixture
fn compute_forces(pos: &Array2<f64>, types: &[u8], box_size: f64) -> Array2<f64> {
    let n = pos.nrows();
    let r_min2 = R_MIN * R_MIN;

    let results: Vec<(usize, f64, f64, f64)> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut fx = 0.0;
            let mut fy = 0.0;
            let mut fz = 0.0;
            let pi = pos.row(i);
            let ti = types[i];

            for j in 0..n {
                if i == j { continue; }
                let pj = pos.row(j);
                let tj = types[j];

                let dx = pbc_diff(pi[0], pj[0], box_size);
                let dy = pbc_diff(pi[1], pj[1], box_size);
                let dz = pbc_diff(pi[2], pj[2], box_size);

                let mut r2 = dx*dx + dy*dy + dz*dz;
                let (sigma, epsilon) = get_lj_params(ti, tj);
                let sigma2 = sigma * sigma;
                let r_cut2 = R_CUT * R_CUT * sigma2;

                if r2 < r_cut2 {
                    if r2 < r_min2 * sigma2 {
                        r2 = r_min2 * sigma2;
                    }
                    let s2_r2 = sigma2 / r2;
                    let s6_r6 = s2_r2 * s2_r2 * s2_r2;
                    let f_mag = 48.0 * epsilon * s6_r6 * (s6_r6 - 0.5) / r2;
                    let f_mag = f_mag.clamp(-100.0, 100.0);

                    fx += f_mag * dx;
                    fy += f_mag * dy;
                    fz += f_mag * dz;
                }
            }
            (i, fx, fy, fz)
        })
        .collect();

    let mut forces = Array2::<f64>::zeros((n, 3));
    for (i, fx, fy, fz) in results {
        forces[[i, 0]] = fx;
        forces[[i, 1]] = fy;
        forces[[i, 2]] = fz;
    }
    forces
}

/// Velocity Verlet step
fn velocity_verlet_step(
    pos: &mut Array2<f64>,
    vel: &mut Array2<f64>,
    forces: &mut Array2<f64>,
    types: &[u8],
    box_size: f64,
) {
    let n = pos.nrows();

    // Half-step velocity
    for i in 0..n {
        for d in 0..3 {
            vel[[i, d]] += 0.5 * DT * forces[[i, d]];
        }
    }

    // Position update
    for i in 0..n {
        for d in 0..3 {
            pos[[i, d]] = (pos[[i, d]] + DT * vel[[i, d]]).rem_euclid(box_size);
        }
    }

    // New forces
    *forces = compute_forces(pos, types, box_size);

    // Second half-step velocity
    for i in 0..n {
        for d in 0..3 {
            vel[[i, d]] += 0.5 * DT * forces[[i, d]];
        }
    }
}

/// Berendsen thermostat with strong coupling
fn apply_thermostat(vel: &mut Array2<f64>, target_t: f64, tau: f64) {
    let n = vel.nrows() as f64;
    let ke: f64 = vel.iter().map(|v| 0.5 * v * v).sum();
    let current_t = 2.0 * ke / (3.0 * n);

    if current_t > 1e-8 {
        // Berendsen with coupling constant
        let lambda = (1.0 + (DT / tau) * (target_t / current_t - 1.0)).sqrt();
        let lambda = lambda.clamp(0.9, 1.1);
        vel.mapv_inplace(|v| v * lambda);
    }
}

/// Instantaneous velocity rescaling (for quench)
fn rescale_velocities(vel: &mut Array2<f64>, target_t: f64) {
    let n = vel.nrows() as f64;
    let ke: f64 = vel.iter().map(|v| 0.5 * v * v).sum();
    let current_t = 2.0 * ke / (3.0 * n);

    if current_t > 1e-8 {
        let scale = (target_t / current_t).sqrt();
        vel.mapv_inplace(|v| v * scale);
    }
}

/// Spherical harmonics Y_lm for l=6
/// Using explicit formulas for computational efficiency
fn spherical_harmonic_y6m(m: i32, cos_theta: f64, phi: f64) -> Complex64 {
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
    let x = cos_theta;

    // Correctly normalized spherical harmonics for l=6
    // Using the standard convention: Y_lm = N_lm * P_l^|m|(cos θ) * e^(imφ)
    // N_lm = sqrt((2l+1)/(4π) * (l-|m|)!/(l+|m|)!)
    //
    // The associated Legendre polynomials P_l^m include the derivative factors.
    // We compute the FULLY NORMALIZED Y_lm directly.

    let y_lm_value = match m.abs() {
        0 => {
            // Y_60 = sqrt(13/π)/32 * (231*x^6 - 315*x^4 + 105*x^2 - 5)
            let coeff = (13.0 / PI).sqrt() / 32.0;
            coeff * (231.0*x.powi(6) - 315.0*x.powi(4) + 105.0*x.powi(2) - 5.0)
        }
        1 => {
            // Y_61 = -sqrt(273/(2π))/16 * sin(θ) * (33*x^5 - 30*x^3 + 5*x)
            let coeff = -(273.0 / (2.0 * PI)).sqrt() / 16.0;
            coeff * sin_theta * (33.0*x.powi(5) - 30.0*x.powi(3) + 5.0*x)
        }
        2 => {
            // Y_62 = sqrt(1365/(2π))/64 * sin^2(θ) * (33*x^4 - 18*x^2 + 1)
            let coeff = (1365.0 / (2.0 * PI)).sqrt() / 64.0;
            let sin2 = sin_theta * sin_theta;
            coeff * sin2 * (33.0*x.powi(4) - 18.0*x.powi(2) + 1.0)
        }
        3 => {
            // Y_63 = -sqrt(1365/π)/32 * sin^3(θ) * (11*x^3 - 3*x)
            let coeff = -(1365.0 / PI).sqrt() / 32.0;
            let sin3 = sin_theta.powi(3);
            coeff * sin3 * (11.0*x.powi(3) - 3.0*x)
        }
        4 => {
            // Y_64 = 3*sqrt(91/(2π))/32 * sin^4(θ) * (11*x^2 - 1)
            let coeff = 3.0 * (91.0 / (2.0 * PI)).sqrt() / 32.0;
            let sin4 = sin_theta.powi(4);
            coeff * sin4 * (11.0*x.powi(2) - 1.0)
        }
        5 => {
            // Y_65 = -3*sqrt(1001/π)/32 * sin^5(θ) * x
            let coeff = -3.0 * (1001.0 / PI).sqrt() / 32.0;
            let sin5 = sin_theta.powi(5);
            coeff * sin5 * x
        }
        6 => {
            // Y_66 = sqrt(3003/π)/64 * sin^6(θ)
            let coeff = (3003.0 / PI).sqrt() / 64.0;
            let sin6 = sin_theta.powi(6);
            coeff * sin6
        }
        _ => 0.0,
    };

    // Apply phase factor e^(imφ)
    let phase = Complex64::from_polar(1.0, m as f64 * phi);

    if m < 0 {
        // Y_l,-m = (-1)^|m| * conj(Y_l,|m|)
        let sign = if m.abs() % 2 == 0 { 1.0 } else { -1.0 };
        sign * Complex64::new(y_lm_value, 0.0) * phase.conj()
    } else {
        Complex64::new(y_lm_value, 0.0) * phase
    }
}

/// Steinhardt Q6 order parameter using spherical harmonics
/// Q6 ~ 0.57 for FCC, ~0.48 for HCP, ~0.28-0.35 for liquid/glass
fn compute_q6(pos: &Array2<f64>, dm: &Array2<f64>, box_size: f64, _types: &[u8]) -> f64 {
    let n = pos.nrows();
    let cutoff = 1.5;

    let q6_values: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut neighbors: Vec<usize> = Vec::new();
            for j in 0..n {
                if i != j && dm[[i, j]] < cutoff && dm[[i, j]] > 0.5 {
                    neighbors.push(j);
                }
            }

            if neighbors.is_empty() {
                return 0.0;
            }

            let mut q6m: [Complex64; 13] = [Complex64::new(0.0, 0.0); 13];

            for &j in &neighbors {
                let mut dx = pos[[j, 0]] - pos[[i, 0]];
                let mut dy = pos[[j, 1]] - pos[[i, 1]];
                let mut dz = pos[[j, 2]] - pos[[i, 2]];

                dx -= box_size * (dx / box_size).round();
                dy -= box_size * (dy / box_size).round();
                dz -= box_size * (dz / box_size).round();

                let r = (dx*dx + dy*dy + dz*dz).sqrt();
                if r < 1e-10 { continue; }

                let cos_theta = dz / r;
                let phi = dy.atan2(dx);

                for m_idx in 0..13 {
                    let m = m_idx as i32 - 6;
                    q6m[m_idx] += spherical_harmonic_y6m(m, cos_theta, phi);
                }
            }

            let n_b = neighbors.len() as f64;
            for m_idx in 0..13 {
                q6m[m_idx] /= n_b;
            }

            let sum_sq: f64 = q6m.iter().map(|c| c.norm_sqr()).sum();
            (4.0 * PI / 13.0 * sum_sq).sqrt()
        })
        .collect();

    if q6_values.is_empty() {
        0.0
    } else {
        q6_values.iter().sum::<f64>() / q6_values.len() as f64
    }
}

/// Compute MSD from reference positions (unwrapped)
fn compute_msd(pos: &Array2<f64>, pos_ref: &Array2<f64>, _box_size: f64) -> f64 {
    let n = pos.nrows();
    let mut total_msd = 0.0;

    for i in 0..n {
        for d in 0..3 {
            let dr = pos[[i, d]] - pos_ref[[i, d]];
            total_msd += dr * dr;
        }
    }

    total_msd / n as f64
}

/// Track unwrapped positions for MSD
fn update_unwrapped(
    pos: &Array2<f64>,
    pos_prev: &Array2<f64>,
    pos_unwrap: &mut Array2<f64>,
    box_size: f64,
) {
    let n = pos.nrows();
    for i in 0..n {
        for d in 0..3 {
            let mut dr = pos[[i, d]] - pos_prev[[i, d]];
            // Unwrap if crossed boundary
            if dr > box_size / 2.0 {
                dr -= box_size;
            } else if dr < -box_size / 2.0 {
                dr += box_size;
            }
            pos_unwrap[[i, d]] += dr;
        }
    }
}

/// H1 entropy approximation
fn persistence_entropy_h1(dm: &Array2<f64>) -> f64 {
    let n = dm.nrows();
    if n < 10 { return 0.0; }

    let mut edges: Vec<f64> = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in i + 1..n {
            let d = dm[[i, j]];
            if d > 0.0 && d < 3.0 {
                edges.push(d);
            }
        }
    }

    if edges.len() < 10 { return 0.0; }
    edges.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n_bins = 20;
    let max_d = edges.last().copied().unwrap_or(1.0);
    let bin_size = max_d / n_bins as f64;

    let mut hist = vec![0usize; n_bins];
    for &e in &edges {
        let bin = ((e / bin_size) as usize).min(n_bins - 1);
        hist[bin] += 1;
    }

    let mut lifetimes: Vec<f64> = Vec::new();
    for i in 1..n_bins {
        let diff = (hist[i] as i64 - hist[i - 1] as i64).abs() as f64;
        if diff > 0.0 {
            lifetimes.push(diff * bin_size);
        }
    }

    if lifetimes.is_empty() { return 0.0; }

    let total: f64 = lifetimes.iter().sum();
    if total < 1e-10 { return 0.0; }

    let mut entropy = 0.0;
    for l in &lifetimes {
        let p = l / total;
        if p > 1e-10 {
            entropy -= p * p.ln();
        }
    }
    entropy
}

/// Detect dynamical arrest (glass transition)
fn detect_arrest(msd_series: &[f64], times: &[usize]) -> Option<usize> {
    if msd_series.len() < 20 { return None; }

    // Glass: MSD plateau (subdiffusive behavior)
    // Look for where MSD growth rate drops significantly
    let window = 5;

    for i in window + 5..msd_series.len() - window {
        let early_rate = (msd_series[i] - msd_series[i - window]) / (window as f64);
        let late_rate = (msd_series[i + window] - msd_series[i]) / (window as f64);

        // Arrest when diffusion slows dramatically
        if early_rate > 0.001 && late_rate < early_rate * 0.1 {
            return Some(times[i]);
        }
    }

    // Alternative: check if MSD is very small at end
    if let Some(&last_msd) = msd_series.last() {
        if last_msd < 1.0 {  // Particles moved less than 1 σ on average
            return Some(times[msd_series.len() / 2]);
        }
    }

    None
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

/// Run single glass trial with variable quench rate
fn run_trial_glass(n_particles: usize, seed: u64, quench: &QuenchConfig) -> TrialResult {
    let box_size = (n_particles as f64 / DENSITY).powf(1.0 / 3.0);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Initialize particle types (80% A, 20% B)
    let n_a = (n_particles as f64 * FRACTION_A) as usize;
    let mut types: Vec<u8> = vec![0; n_a];
    types.extend(vec![1; n_particles - n_a]);
    // Shuffle types
    types.shuffle(&mut rng);

    // Initialize positions on lattice then randomize slightly
    let n_side = (n_particles as f64).powf(1.0 / 3.0).ceil() as usize;
    let spacing = box_size / n_side as f64;

    let mut pos = Array2::<f64>::zeros((n_particles, 3));
    let mut idx = 0;
    'outer: for ix in 0..n_side {
        for iy in 0..n_side {
            for iz in 0..n_side {
                if idx >= n_particles { break 'outer; }
                pos[[idx, 0]] = (ix as f64 + 0.5 + (rng.random::<f64>() - 0.5) * 0.1) * spacing;
                pos[[idx, 1]] = (iy as f64 + 0.5 + (rng.random::<f64>() - 0.5) * 0.1) * spacing;
                pos[[idx, 2]] = (iz as f64 + 0.5 + (rng.random::<f64>() - 0.5) * 0.1) * spacing;
                idx += 1;
            }
        }
    }

    // Apply PBC
    for i in 0..n_particles {
        for d in 0..3 {
            pos[[i, d]] = pos[[i, d]].rem_euclid(box_size);
        }
    }

    // Initialize velocities
    let mut vel = Array2::<f64>::zeros((n_particles, 3));
    for i in 0..n_particles {
        for d in 0..3 {
            vel[[i, d]] = (rng.random::<f64>() - 0.5) * 2.0;
        }
    }

    // Remove COM velocity
    for d in 0..3 {
        let com: f64 = vel.column(d).sum() / n_particles as f64;
        for i in 0..n_particles {
            vel[[i, d]] -= com;
        }
    }
    rescale_velocities(&mut vel, T_LIQUID);

    let mut forces = compute_forces(&pos, &types, box_size);

    // ========================================
    // PHASE 1: Equilibrate at high temperature
    // ========================================
    for step in 0..STEPS_EQUIL_HOT {
        velocity_verlet_step(&mut pos, &mut vel, &mut forces, &types, box_size);
        if step % THERMOSTAT_TAU == 0 {
            apply_thermostat(&mut vel, T_LIQUID, 0.5);  // Strong coupling
        }
    }

    // ========================================
    // PHASE 2: Variable rate quench
    // ========================================
    // Temperature decreases linearly from T_LIQUID to T_GLASS over steps_quench steps
    let steps_quench = quench.steps_quench;
    let dt_temp = (T_LIQUID - T_GLASS) / steps_quench as f64;

    for step in 0..steps_quench {
        velocity_verlet_step(&mut pos, &mut vel, &mut forces, &types, box_size);

        // Apply temperature ramp
        let target_t = T_LIQUID - dt_temp * step as f64;
        if step % 10 == 0 || steps_quench <= 10 {
            rescale_velocities(&mut vel, target_t.max(T_GLASS));
        }
    }

    // Final rescale to exact T_GLASS
    rescale_velocities(&mut vel, T_GLASS);

    // Save reference for MSD
    let pos_ref = pos.clone();
    let mut pos_unwrap = pos.clone();
    let mut pos_prev = pos.clone();

    // ========================================
    // PHASE 3: Anneal and sample at T_GLASS
    // ========================================
    let mut times = Vec::new();
    let mut q6_series = Vec::new();
    let mut msd_series = Vec::new();
    let mut s_h1_series = Vec::new();

    for step in 0..STEPS_ANNEAL {
        velocity_verlet_step(&mut pos, &mut vel, &mut forces, &types, box_size);

        if step % THERMOSTAT_TAU == 0 {
            apply_thermostat(&mut vel, T_GLASS, 1.0);
        }

        if step % SAMPLE_INTERVAL == 0 {
            // Update unwrapped positions
            update_unwrapped(&pos, &pos_prev, &mut pos_unwrap, box_size);
            pos_prev.assign(&pos);

            let dm = compute_distance_matrix(&pos, box_size);
            let q6 = compute_q6(&pos, &dm, box_size, &types);
            let msd = compute_msd(&pos_unwrap, &pos_ref, box_size);
            let s_h1 = persistence_entropy_h1(&dm);

            times.push(step);
            q6_series.push(q6);
            msd_series.push(msd);
            s_h1_series.push(s_h1);
        }
    }

    // Analysis
    let t_arrest = detect_arrest(&msd_series, &times);
    let t_topo_h1 = detect_cusum(&s_h1_series, &times, 0.2);

    let gap_h1 = match (t_arrest, t_topo_h1) {
        (Some(ta), Some(tt)) => Some(ta as i64 - tt as i64),
        _ => None,
    };

    let q6_final = q6_series[q6_series.len().saturating_sub(5)..].iter().sum::<f64>()
        / 5.0f64.min(q6_series.len() as f64);
    let msd_final = *msd_series.last().unwrap_or(&0.0);

    // Diffusion coefficient from late-time MSD
    let diffusion_coeff = if msd_series.len() > 10 {
        let n_late = msd_series.len() / 2;
        let late_msd = &msd_series[n_late..];
        let late_times = &times[n_late..];
        if late_times.len() > 1 {
            let dt = (late_times.last().unwrap() - late_times.first().unwrap()) as f64 * DT;
            let dmsd = late_msd.last().unwrap() - late_msd.first().unwrap();
            dmsd / (6.0 * dt)  // D = MSD / (6t) in 3D
        } else {
            0.0
        }
    } else {
        0.0
    };

    // Classification using real Steinhardt Q6:
    // - Crystal (FCC/HCP): Q6 > 0.45 (FCC~0.57, HCP~0.48)
    // - Liquid: Q6 ~ 0.28-0.35, high MSD, high D
    // - Glass: Q6 ~ 0.28-0.35, low MSD (< 1.0), low D (dynamical arrest)
    let phase = if q6_final > 0.45 {
        "CRYSTAL"
    } else if msd_final < 1.0 && diffusion_coeff < 0.01 {
        "GLASS"
    } else if msd_final > 5.0 || diffusion_coeff > 0.1 {
        "LIQUID"
    } else {
        "GLASS"  // Default to glass for intermediate cases at low T
    };

    TrialResult {
        seed,
        quench_name: quench.name.to_string(),
        quench_steps: quench.steps_quench,
        quench_rate: quench.rate_description.to_string(),
        phase: phase.to_string(),
        q6_final,
        msd_final,
        diffusion_coeff,
        t_arrest,
        t_topo_h1,
        gap_h1,
    }
}

fn run_validation_glass(n_particles: usize, n_trials_per_rate: usize) -> ValidationSummary {
    println!("\n{}", "=".repeat(70));
    println!("LJ GLASS - VARIABLE QUENCH RATE STUDY (Kob-Andersen)");
    println!("N={}, {} trials per rate, {} quench rates", n_particles, n_trials_per_rate, QUENCH_RATES.len());
    println!("{}", "=".repeat(70));

    let total_start = Instant::now();
    let mut all_results = Vec::new();
    let mut quench_summaries = Vec::new();

    for quench in QUENCH_RATES.iter() {
        println!("\n--- Quench: {} ({}) ---", quench.name, quench.rate_description);
        println!("    Steps: {} (dt={:.3})", quench.steps_quench, DT);

        let mut rate_results = Vec::new();

        for i in 0..n_trials_per_rate {
            let seed = (n_particles as u64) * 5000 + (quench.steps_quench as u64) * 100 + i as u64;
            let trial_start = Instant::now();
            let result = run_trial_glass(n_particles, seed, quench);
            let elapsed = trial_start.elapsed().as_secs_f64();

            println!(
                "  Trial {}/{}: {} | Q6={:.3}, MSD={:.2}, D={:.4} ({:.1}s)",
                i + 1, n_trials_per_rate, result.phase, result.q6_final,
                result.msd_final, result.diffusion_coeff, elapsed
            );

            rate_results.push(result.clone());
            all_results.push(result);
        }

        // Summarize this quench rate
        let n_glass = rate_results.iter().filter(|r| r.phase == "GLASS").count();
        let n_crystal = rate_results.iter().filter(|r| r.phase == "CRYSTAL").count();
        let n_liquid = rate_results.iter().filter(|r| r.phase == "LIQUID").count();
        let glass_fraction = n_glass as f64 / n_trials_per_rate as f64;
        let mean_q6 = rate_results.iter().map(|r| r.q6_final).sum::<f64>() / rate_results.len() as f64;
        let mean_msd = rate_results.iter().map(|r| r.msd_final).sum::<f64>() / rate_results.len() as f64;
        let mean_diff = rate_results.iter().map(|r| r.diffusion_coeff).sum::<f64>() / rate_results.len() as f64;

        println!("  => Glass: {}, Crystal: {}, Liquid: {} (Glass fraction: {:.0}%)",
                 n_glass, n_crystal, n_liquid, glass_fraction * 100.0);

        quench_summaries.push(QuenchSummary {
            quench_name: quench.name.to_string(),
            quench_steps: quench.steps_quench,
            quench_rate: quench.rate_description.to_string(),
            n_trials: n_trials_per_rate,
            n_glass,
            n_crystal,
            n_liquid,
            glass_fraction,
            mean_q6,
            mean_msd,
            mean_diff_coeff: mean_diff,
        });
    }

    let total_time = total_start.elapsed().as_secs_f64();

    // Print summary table
    println!("\n{}", "=".repeat(70));
    println!("QUENCH RATE SUMMARY (TTT Diagram)");
    println!("{}", "=".repeat(70));
    println!("{:<15} {:>10} {:>8} {:>8} {:>8} {:>10}",
             "Rate", "Steps", "Glass%", "Cryst%", "Q6", "MSD");
    println!("{}", "-".repeat(70));
    for qs in &quench_summaries {
        println!("{:<15} {:>10} {:>7.0}% {:>7.0}% {:>8.3} {:>10.2}",
                 qs.quench_name, qs.quench_steps,
                 qs.glass_fraction * 100.0,
                 (qs.n_crystal as f64 / qs.n_trials as f64) * 100.0,
                 qs.mean_q6, qs.mean_msd);
    }
    println!("{}", "-".repeat(70));
    println!("Total time: {:.1}s", total_time);

    ValidationSummary {
        n_particles,
        n_trials_per_rate,
        total_quench_rates: QUENCH_RATES.len(),
        total_time_sec: total_time,
        quench_results: quench_summaries,
        trials: all_results,
    }
}

fn main() {
    println!("{}", "=".repeat(70));
    println!("LJ GLASS - VARIABLE QUENCH RATE STUDY");
    println!("Testing: instantaneous -> ultra_fast -> fast -> moderate -> slow");
    println!("{}", "=".repeat(70));

    std::fs::create_dir_all("../results").ok();

    // Run 3 trials per quench rate for statistical significance
    let res = run_validation_glass(256, 3);
    let json = serde_json::to_string_pretty(&res).unwrap();
    let mut file = File::create("../results/lj_glass_variable_quench.json").unwrap();
    file.write_all(json.as_bytes()).unwrap();

    println!("\n{}", "=".repeat(70));
    println!("VARIABLE QUENCH STUDY COMPLETE");
    println!("Results saved to: results/lj_glass_variable_quench.json");
    println!("{}", "=".repeat(70));
}
