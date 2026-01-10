//! LJ Glass-Forming System - TDA During Vitrification
//! ====================================================
//! Kob-Andersen binary mixture with rapid quench.
//! Author: Francisco Molina Burgos
//! Date: 2026-01-10

use ndarray::Array2;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Write;
use std::time::Instant;

// === PARAMETERS ===
const DT: f64 = 0.002;
const T_HIGH: f64 = 1.5;       // Start in liquid
const T_LOW: f64 = 0.3;        // Below Tg ~ 0.435
const STEPS_EQUIL: usize = 5000;
const STEPS_PROD: usize = 10000;  // Longer for glass
const SAMPLE_INTERVAL: usize = 50;
const THERMOSTAT_TAU: usize = 50;
const R_CUT: f64 = 2.5;
const R_MIN: f64 = 0.7;
const DENSITY: f64 = 1.2;      // Kob-Andersen density
const FRACTION_A: f64 = 0.8;   // 80% A particles

// Kob-Andersen parameters
const SIGMA_AA: f64 = 1.0;
const SIGMA_AB: f64 = 0.8;
const SIGMA_BB: f64 = 0.88;
const EPSILON_AA: f64 = 1.0;
const EPSILON_AB: f64 = 1.5;
const EPSILON_BB: f64 = 0.5;

#[derive(Clone, Serialize, Deserialize)]
struct TrialResult {
    seed: u64,
    phase: String,
    q6_final: f64,
    msd_final: f64,
    t_glass: Option<usize>,
    t_topo_h1: Option<usize>,
    gap_h1: Option<i64>,
}

#[derive(Serialize)]
struct ValidationSummary {
    n_particles: usize,
    n_trials: usize,
    total_time_sec: f64,
    time_per_trial_sec: f64,
    n_glass: usize,
    n_crystal: usize,
    precursor_rate_h1: f64,
    mean_gap_h1: f64,
    trials: Vec<TrialResult>,
}

/// Get LJ parameters for particle pair
#[inline]
fn get_lj_params(type_i: u8, type_j: u8) -> (f64, f64) {
    match (type_i, type_j) {
        (0, 0) => (SIGMA_AA, EPSILON_AA),
        (1, 1) => (SIGMA_BB, EPSILON_BB),
        _ => (SIGMA_AB, EPSILON_AB),
    }
}

/// Compute PBC distance in 3D
#[inline]
fn pbc_distance_3d(p1: &[f64], p2: &[f64], box_size: f64) -> f64 {
    let mut dx = p1[0] - p2[0];
    let mut dy = p1[1] - p2[1];
    let mut dz = p1[2] - p2[2];
    dx -= box_size * (dx / box_size).round();
    dy -= box_size * (dy / box_size).round();
    dz -= box_size * (dz / box_size).round();
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Compute distance matrix
fn compute_distance_matrix_3d(pos: &Array2<f64>, box_size: f64) -> Array2<f64> {
    let n = pos.nrows();
    let mut dm = Array2::<f64>::zeros((n, n));

    let results: Vec<(usize, usize, f64)> = (0..n)
        .into_par_iter()
        .flat_map(|i| {
            (i + 1..n)
                .map(|j| {
                    let pi = pos.row(i);
                    let pj = pos.row(j);
                    let d = pbc_distance_3d(pi.as_slice().unwrap(), pj.as_slice().unwrap(), box_size);
                    (i, j, d)
                })
                .collect::<Vec<_>>()
        })
        .collect();

    for (i, j, d) in results {
        dm[[i, j]] = d;
        dm[[j, i]] = d;
    }
    dm
}

/// Compute forces for binary LJ mixture
fn compute_forces_binary(pos: &Array2<f64>, types: &[u8], box_size: f64) -> Array2<f64> {
    let n = pos.nrows();
    let r_cut2 = R_CUT * R_CUT;
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

                let mut dx = pi[0] - pj[0];
                let mut dy = pi[1] - pj[1];
                let mut dz = pi[2] - pj[2];
                dx -= box_size * (dx / box_size).round();
                dy -= box_size * (dy / box_size).round();
                dz -= box_size * (dz / box_size).round();

                let mut r2 = dx * dx + dy * dy + dz * dz;
                let (sigma, epsilon) = get_lj_params(ti, tj);
                let sigma2 = sigma * sigma;
                let r_cut2_scaled = r_cut2 * sigma2;

                if r2 < r_cut2_scaled {
                    if r2 < r_min2 * sigma2 {
                        r2 = r_min2 * sigma2;
                    }
                    let s2_r2 = sigma2 / r2;
                    let s6_r6 = s2_r2 * s2_r2 * s2_r2;
                    let f_mag = 48.0 * epsilon * s6_r6 * (s6_r6 - 0.5) / r2;
                    let f_mag = f_mag.clamp(-50.0, 50.0);

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

    for i in 0..n {
        vel[[i, 0]] += 0.5 * DT * forces[[i, 0]];
        vel[[i, 1]] += 0.5 * DT * forces[[i, 1]];
        vel[[i, 2]] += 0.5 * DT * forces[[i, 2]];
    }

    for i in 0..n {
        pos[[i, 0]] = (pos[[i, 0]] + DT * vel[[i, 0]]).rem_euclid(box_size);
        pos[[i, 1]] = (pos[[i, 1]] + DT * vel[[i, 1]]).rem_euclid(box_size);
        pos[[i, 2]] = (pos[[i, 2]] + DT * vel[[i, 2]]).rem_euclid(box_size);
    }

    let new_forces = compute_forces_binary(pos, types, box_size);
    *forces = new_forces;

    for i in 0..n {
        vel[[i, 0]] += 0.5 * DT * forces[[i, 0]];
        vel[[i, 1]] += 0.5 * DT * forces[[i, 1]];
        vel[[i, 2]] += 0.5 * DT * forces[[i, 2]];
    }
}

/// Berendsen thermostat
fn apply_thermostat(vel: &mut Array2<f64>, target_t: f64) {
    let n = vel.nrows() as f64;
    let ke: f64 = vel.iter().map(|v| 0.5 * v * v).sum();
    let current_t = 2.0 * ke / (3.0 * n);

    if current_t > 1e-6 {
        let scale = (target_t / current_t).sqrt().clamp(0.9, 1.1);
        vel.mapv_inplace(|v| v * scale);
    }
}

/// Compute Q6 order parameter (crystallinity indicator)
fn compute_q6(dm: &Array2<f64>) -> f64 {
    let n = dm.nrows();
    let cutoff = 1.5;

    let ordered_count: usize = (0..n)
        .into_par_iter()
        .filter(|&i| {
            let mut n_neigh = 0;
            for j in 0..n {
                if i != j && dm[[i, j]] < cutoff && dm[[i, j]] > 0.5 {
                    n_neigh += 1;
                }
            }
            n_neigh >= 11
        })
        .count();

    ordered_count as f64 / n as f64
}

/// Compute MSD from initial positions
fn compute_msd(pos: &Array2<f64>, pos0: &Array2<f64>, box_size: f64) -> f64 {
    let n = pos.nrows();
    let mut total_msd = 0.0;

    for i in 0..n {
        let mut dx = pos[[i, 0]] - pos0[[i, 0]];
        let mut dy = pos[[i, 1]] - pos0[[i, 1]];
        let mut dz = pos[[i, 2]] - pos0[[i, 2]];
        // Note: For MSD we don't apply PBC to capture diffusion
        dx -= box_size * (dx / box_size).round();
        dy -= box_size * (dy / box_size).round();
        dz -= box_size * (dz / box_size).round();
        total_msd += dx * dx + dy * dy + dz * dz;
    }

    total_msd / n as f64
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

/// Detect glass transition (MSD plateau)
fn detect_glass(msd_series: &[f64], times: &[usize]) -> Option<usize> {
    if msd_series.len() < 10 { return None; }

    // Glass transition: MSD stops growing (plateau)
    let window = 5;
    for i in window..msd_series.len() - window {
        let before = msd_series[i - window..i].iter().sum::<f64>() / window as f64;
        let after = msd_series[i..i + window].iter().sum::<f64>() / window as f64;

        // If MSD growth slows significantly (ratio < 1.1)
        if after / before < 1.1 && before > 0.1 {
            return Some(times[i]);
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

/// Run single glass trial
fn run_trial_glass(n_particles: usize, seed: u64) -> TrialResult {
    let box_size = (n_particles as f64 / DENSITY).powf(1.0 / 3.0);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Initialize particle types (80% A, 20% B)
    let n_a = (n_particles as f64 * FRACTION_A) as usize;
    let mut types: Vec<u8> = vec![0; n_a];
    types.extend(vec![1; n_particles - n_a]);

    // Initialize positions
    let mut pos = Array2::<f64>::zeros((n_particles, 3));
    for i in 0..n_particles {
        pos[[i, 0]] = rng.random::<f64>() * box_size;
        pos[[i, 1]] = rng.random::<f64>() * box_size;
        pos[[i, 2]] = rng.random::<f64>() * box_size;
    }

    // Initialize velocities
    let mut vel = Array2::<f64>::zeros((n_particles, 3));
    for i in 0..n_particles {
        vel[[i, 0]] = rng.random::<f64>() * 2.0 - 1.0;
        vel[[i, 1]] = rng.random::<f64>() * 2.0 - 1.0;
        vel[[i, 2]] = rng.random::<f64>() * 2.0 - 1.0;
    }

    // Remove COM velocity
    let com: Vec<f64> = (0..3).map(|d| vel.column(d).sum() / n_particles as f64).collect();
    for i in 0..n_particles {
        vel[[i, 0]] -= com[0];
        vel[[i, 1]] -= com[1];
        vel[[i, 2]] -= com[2];
    }
    apply_thermostat(&mut vel, T_HIGH);

    let mut forces = compute_forces_binary(&pos, &types, box_size);

    // Equilibration at high T
    for step in 0..STEPS_EQUIL {
        velocity_verlet_step(&mut pos, &mut vel, &mut forces, &types, box_size);
        if step % THERMOSTAT_TAU == 0 {
            apply_thermostat(&mut vel, T_HIGH);
        }
    }

    // Save initial positions for MSD
    let pos0 = pos.clone();

    // Production with rapid quench
    let mut times = Vec::new();
    let mut q6_series = Vec::new();
    let mut msd_series = Vec::new();
    let mut s_h1_series = Vec::new();

    for step in 0..STEPS_PROD {
        let progress = step as f64 / STEPS_PROD as f64;
        let target_t = T_HIGH + (T_LOW - T_HIGH) * progress;

        velocity_verlet_step(&mut pos, &mut vel, &mut forces, &types, box_size);

        if step % THERMOSTAT_TAU == 0 {
            apply_thermostat(&mut vel, target_t);
        }

        if step % SAMPLE_INTERVAL == 0 {
            let dm = compute_distance_matrix_3d(&pos, box_size);
            let q6 = compute_q6(&dm);
            let msd = compute_msd(&pos, &pos0, box_size);
            let s_h1 = persistence_entropy_h1(&dm);

            times.push(step);
            q6_series.push(q6);
            msd_series.push(msd);
            s_h1_series.push(s_h1);
        }
    }

    // Detection
    let t_glass = detect_glass(&msd_series, &times);
    let t_topo_h1 = detect_cusum(&s_h1_series, &times, 0.3);

    let gap_h1 = match (t_glass, t_topo_h1) {
        (Some(tg), Some(tt)) => Some(tg as i64 - tt as i64),
        _ => None,
    };

    let q6_final = q6_series[q6_series.len().saturating_sub(5)..].iter().sum::<f64>()
        / 5.0f64.min(q6_series.len() as f64);
    let msd_final = msd_series.last().copied().unwrap_or(0.0);

    // Glass if low Q6 (no crystallization) and low MSD (frozen)
    let phase = if q6_final < 0.3 && msd_final < 2.0 {
        "GLASS"
    } else if q6_final > 0.5 {
        "CRYSTAL"
    } else {
        "LIQUID"
    };

    TrialResult {
        seed,
        phase: phase.to_string(),
        q6_final,
        msd_final,
        t_glass,
        t_topo_h1,
        gap_h1,
    }
}

/// Run validation
fn run_validation_glass(n_particles: usize, n_trials: usize) -> ValidationSummary {
    println!("\n{}", "=".repeat(70));
    println!("LJ GLASS VALIDATION: N={}, {} trials", n_particles, n_trials);
    println!("{}", "=".repeat(70));

    let total_start = Instant::now();
    let mut results = Vec::new();

    for i in 0..n_trials {
        let seed = (n_particles as u64) * 1000 + i as u64;
        let trial_start = Instant::now();
        let result = run_trial_glass(n_particles, seed);
        let elapsed = trial_start.elapsed().as_secs_f64();

        println!(
            "  Trial {}/{}: {}, Q6={:.3}, MSD={:.2}, gap_H1={:?} ({:.1}s)",
            i + 1, n_trials, result.phase, result.q6_final, result.msd_final,
            result.gap_h1, elapsed
        );

        results.push(result);
    }

    let total_time = total_start.elapsed().as_secs_f64();

    let n_glass = results.iter().filter(|r| r.phase == "GLASS").count();
    let n_crystal = results.iter().filter(|r| r.phase == "CRYSTAL").count();

    let gaps: Vec<i64> = results.iter().filter(|r| r.phase == "GLASS")
        .filter_map(|r| r.gap_h1).collect();
    let n_precursor = gaps.iter().filter(|&&g| g > 0).count();

    let precursor_rate_h1 = if !gaps.is_empty() {
        n_precursor as f64 / gaps.len() as f64
    } else { 0.0 };

    let mean_gap_h1 = if !gaps.is_empty() {
        gaps.iter().sum::<i64>() as f64 / gaps.len() as f64
    } else { 0.0 };

    println!("\n--- GLASS RESULTS N={} ---", n_particles);
    println!("Glass: {}/{}", n_glass, n_trials);
    println!("Crystal: {}/{}", n_crystal, n_trials);
    println!("H1 Precursor rate: {:.1}%", 100.0 * precursor_rate_h1);
    println!("Mean gap H1: {:.1}", mean_gap_h1);
    println!("Time/trial: {:.1}s", total_time / n_trials as f64);

    ValidationSummary {
        n_particles,
        n_trials,
        total_time_sec: total_time,
        time_per_trial_sec: total_time / n_trials as f64,
        n_glass,
        n_crystal,
        precursor_rate_h1,
        mean_gap_h1,
        trials: results,
    }
}

fn main() {
    println!("{}", "=".repeat(70));
    println!("LJ GLASS TDA-CUSUM (Kob-Andersen Binary Mixture)");
    println!("{}", "=".repeat(70));

    std::fs::create_dir_all("../results").ok();

    let res_256 = run_validation_glass(256, 5);
    let json = serde_json::to_string_pretty(&res_256).unwrap();
    let mut file = File::create("../results/lj_glass_N256.json").unwrap();
    file.write_all(json.as_bytes()).unwrap();

    println!("\n{}", "=".repeat(70));
    println!("GLASS VALIDATION COMPLETE");
    println!("{}", "=".repeat(70));
}
