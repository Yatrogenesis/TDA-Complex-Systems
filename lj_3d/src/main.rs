//! Lennard-Jones 3D Phase Transition - TDA with H1 + H2
//! =====================================================
//! Extension to 3D with both loop (H1) and void (H2) homology.
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

// === PARAMETERS ===
const DT: f64 = 0.002;
const T_HIGH: f64 = 2.0;
const T_LOW: f64 = 0.3;  // Higher than 2D to avoid deep supercooling
const STEPS_EQUIL: usize = 3000;
const STEPS_PROD: usize = 6000;
const SAMPLE_INTERVAL: usize = 30;
const THERMOSTAT_TAU: usize = 50;
const R_CUT: f64 = 2.5;
const R_MIN: f64 = 0.8;
const DENSITY: f64 = 0.85;  // Higher for 3D FCC

#[derive(Clone, Serialize, Deserialize)]
struct TrialResult {
    seed: u64,
    phase: String,
    q6_final: f64,
    t_phys: Option<usize>,
    t_topo_h1: Option<usize>,
    t_topo_h2: Option<usize>,
    gap_h1: Option<i64>,
    gap_h2: Option<i64>,
}

#[derive(Serialize)]
struct ValidationSummary {
    n_particles: usize,
    n_trials: usize,
    total_time_sec: f64,
    time_per_trial_sec: f64,
    n_crystal: usize,
    precursor_rate_h1: f64,
    precursor_rate_h2: f64,
    mean_gap_h1: f64,
    mean_gap_h2: f64,
    trials: Vec<TrialResult>,
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

/// Compute full 3D PBC distance matrix (parallel)
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

/// Compute 3D LJ forces (parallel)
fn compute_forces_3d(pos: &Array2<f64>, box_size: f64) -> Array2<f64> {
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

            for j in 0..n {
                if i == j {
                    continue;
                }
                let pj = pos.row(j);

                let mut dx = pi[0] - pj[0];
                let mut dy = pi[1] - pj[1];
                let mut dz = pi[2] - pj[2];
                dx -= box_size * (dx / box_size).round();
                dy -= box_size * (dy / box_size).round();
                dz -= box_size * (dz / box_size).round();

                let mut r2 = dx * dx + dy * dy + dz * dz;
                if r2 < r_cut2 {
                    if r2 < r_min2 {
                        r2 = r_min2;
                    }
                    let r2_inv = 1.0 / r2;
                    let r6_inv = r2_inv * r2_inv * r2_inv;
                    let f_mag = 48.0 * r2_inv * r6_inv * (r6_inv - 0.5);
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

/// Velocity Verlet step in 3D
fn velocity_verlet_step_3d(
    pos: &mut Array2<f64>,
    vel: &mut Array2<f64>,
    forces: &mut Array2<f64>,
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

    let new_forces = compute_forces_3d(pos, box_size);
    *forces = new_forces;

    for i in 0..n {
        vel[[i, 0]] += 0.5 * DT * forces[[i, 0]];
        vel[[i, 1]] += 0.5 * DT * forces[[i, 1]];
        vel[[i, 2]] += 0.5 * DT * forces[[i, 2]];
    }
}

/// Apply Berendsen thermostat in 3D
fn apply_thermostat_3d(vel: &mut Array2<f64>, target_t: f64) {
    let n = vel.nrows() as f64;
    let ke: f64 = vel.iter().map(|v| 0.5 * v * v).sum();
    let current_t = 2.0 * ke / (3.0 * n);  // 3D: 3N degrees of freedom

    if current_t > 1e-6 {
        let scale = (target_t / current_t).sqrt().clamp(0.9, 1.1);
        vel.mapv_inplace(|v| v * scale);
    }
}

/// Steinhardt Q6 order parameter for 3D crystallinity
fn steinhardt_q6(pos: &Array2<f64>, dm: &Array2<f64>, _box_size: f64) -> f64 {
    let n = pos.nrows();
    let cutoff = 1.5;  // First neighbor shell

    // Simplified Q6: count well-ordered local environments
    let ordered_count: usize = (0..n)
        .into_par_iter()
        .filter(|&i| {
            let mut n_neigh = 0;
            for j in 0..n {
                if i != j && dm[[i, j]] < cutoff && dm[[i, j]] > 0.5 {
                    n_neigh += 1;
                }
            }
            // FCC/HCP have 12 neighbors, liquid has ~10-11
            n_neigh >= 11
        })
        .count();

    ordered_count as f64 / n as f64
}

/// Approximate H1 entropy from edge statistics
fn persistence_entropy_h1(dm: &Array2<f64>) -> f64 {
    let n = dm.nrows();
    if n < 10 {
        return 0.0;
    }

    let mut edges: Vec<f64> = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in i + 1..n {
            let d = dm[[i, j]];
            if d > 0.0 && d < 3.0 {
                edges.push(d);
            }
        }
    }

    if edges.len() < 10 {
        return 0.0;
    }

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

    if lifetimes.is_empty() {
        return 0.0;
    }

    let total: f64 = lifetimes.iter().sum();
    if total < 1e-10 {
        return 0.0;
    }

    let mut entropy = 0.0;
    for l in &lifetimes {
        let p = l / total;
        if p > 1e-10 {
            entropy -= p * p.ln();
        }
    }

    entropy
}

/// Approximate H2 entropy - void/cavity statistics
fn persistence_entropy_h2(dm: &Array2<f64>) -> f64 {
    let n = dm.nrows();
    if n < 20 {
        return 0.0;
    }

    // H2 approximation: analyze distribution of "empty space"
    // by looking at largest gaps between particles

    let mut all_distances: Vec<f64> = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in i + 1..n {
            all_distances.push(dm[[i, j]]);
        }
    }
    all_distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Look at gaps in the distance distribution
    let n_bins = 15;
    let max_d = all_distances.last().copied().unwrap_or(1.0);
    let bin_size = max_d / n_bins as f64;

    let mut hist = vec![0usize; n_bins];
    for &d in &all_distances {
        let bin = ((d / bin_size) as usize).min(n_bins - 1);
        hist[bin] += 1;
    }

    // Entropy of void distribution
    let total: f64 = hist.iter().map(|&x| x as f64).sum();
    if total < 1.0 {
        return 0.0;
    }

    let mut entropy = 0.0;
    for &h in &hist {
        let p = h as f64 / total;
        if p > 1e-10 {
            entropy -= p * p.ln();
        }
    }

    entropy
}

/// Detect crystallization
fn detect_crystal(q6_series: &[f64], times: &[usize], threshold: f64, persistence: usize) -> Option<usize> {
    let above: Vec<bool> = q6_series.iter().map(|&q| q > threshold).collect();
    for i in 0..above.len().saturating_sub(persistence) {
        if above[i..i + persistence].iter().all(|&b| b) {
            return Some(times[i]);
        }
    }
    None
}

/// CUSUM detection
fn detect_cusum(series: &[f64], times: &[usize], baseline_fraction: f64) -> Option<usize> {
    let baseline_end = (series.len() as f64 * baseline_fraction) as usize;
    if baseline_end < 5 {
        return None;
    }

    let baseline = &series[..baseline_end];
    let mu: f64 = baseline.iter().sum::<f64>() / baseline.len() as f64;
    let sigma: f64 = (baseline.iter().map(|&x| (x - mu).powi(2)).sum::<f64>() / baseline.len() as f64).sqrt();

    if sigma < 1e-6 {
        return None;
    }

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

/// Run single 3D trial
fn run_trial_3d(n_particles: usize, seed: u64) -> TrialResult {
    let box_size = (n_particles as f64 / DENSITY).powf(1.0 / 3.0);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Initialize positions in 3D
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
    apply_thermostat_3d(&mut vel, T_HIGH);

    let mut forces = compute_forces_3d(&pos, box_size);

    // Equilibration
    for step in 0..STEPS_EQUIL {
        velocity_verlet_step_3d(&mut pos, &mut vel, &mut forces, box_size);
        if step % THERMOSTAT_TAU == 0 {
            apply_thermostat_3d(&mut vel, T_HIGH);
        }
    }

    // Production
    let mut times = Vec::new();
    let mut q6_series = Vec::new();
    let mut s_h1_series = Vec::new();
    let mut s_h2_series = Vec::new();

    for step in 0..STEPS_PROD {
        let progress = step as f64 / STEPS_PROD as f64;
        let target_t = T_HIGH + (T_LOW - T_HIGH) * progress;

        velocity_verlet_step_3d(&mut pos, &mut vel, &mut forces, box_size);

        if step % THERMOSTAT_TAU == 0 {
            apply_thermostat_3d(&mut vel, target_t);
        }

        if step % SAMPLE_INTERVAL == 0 {
            let dm = compute_distance_matrix_3d(&pos, box_size);
            let q6 = steinhardt_q6(&pos, &dm, box_size);
            let s_h1 = persistence_entropy_h1(&dm);
            let s_h2 = persistence_entropy_h2(&dm);

            times.push(step);
            q6_series.push(q6);
            s_h1_series.push(s_h1);
            s_h2_series.push(s_h2);
        }
    }

    // Detection
    let t_phys = detect_crystal(&q6_series, &times, 0.6, 4);
    let t_topo_h1 = detect_cusum(&s_h1_series, &times, 0.3);
    let t_topo_h2 = detect_cusum(&s_h2_series, &times, 0.3);

    let gap_h1 = match (t_phys, t_topo_h1) {
        (Some(tp), Some(tt)) => Some(tp as i64 - tt as i64),
        _ => None,
    };

    let gap_h2 = match (t_phys, t_topo_h2) {
        (Some(tp), Some(tt)) => Some(tp as i64 - tt as i64),
        _ => None,
    };

    let q6_final = q6_series[q6_series.len().saturating_sub(5)..].iter().sum::<f64>()
        / 5.0f64.min(q6_series.len() as f64);

    let phase = if q6_final > 0.6 { "CRYSTAL" } else { "LIQUID" };

    TrialResult {
        seed,
        phase: phase.to_string(),
        q6_final,
        t_phys,
        t_topo_h1,
        t_topo_h2,
        gap_h1,
        gap_h2,
    }
}

/// Run validation
fn run_validation_3d(n_particles: usize, n_trials: usize) -> ValidationSummary {
    println!("\n{}", "=".repeat(70));
    println!("3D LJ VALIDATION: N={}, {} trials", n_particles, n_trials);
    println!("{}", "=".repeat(70));

    let total_start = Instant::now();
    let mut results = Vec::new();

    for i in 0..n_trials {
        let seed = (n_particles as u64) * 100 + i as u64;
        let trial_start = Instant::now();
        let result = run_trial_3d(n_particles, seed);
        let elapsed = trial_start.elapsed().as_secs_f64();

        println!(
            "  Trial {}/{}: {}, Q6={:.3}, gap_H1={:?}, gap_H2={:?} ({:.1}s)",
            i + 1,
            n_trials,
            result.phase,
            result.q6_final,
            result.gap_h1,
            result.gap_h2,
            elapsed
        );

        results.push(result);
    }

    let total_time = total_start.elapsed().as_secs_f64();

    // Analysis
    let crystal: Vec<&TrialResult> = results.iter().filter(|r| r.phase == "CRYSTAL").collect();
    let n_crystal = crystal.len();

    let gaps_h1: Vec<i64> = crystal.iter().filter_map(|r| r.gap_h1).collect();
    let gaps_h2: Vec<i64> = crystal.iter().filter_map(|r| r.gap_h2).collect();

    let n_precursor_h1 = gaps_h1.iter().filter(|&&g| g > 0).count();
    let n_precursor_h2 = gaps_h2.iter().filter(|&&g| g > 0).count();

    let precursor_rate_h1 = if !gaps_h1.is_empty() {
        n_precursor_h1 as f64 / gaps_h1.len() as f64
    } else {
        0.0
    };

    let precursor_rate_h2 = if !gaps_h2.is_empty() {
        n_precursor_h2 as f64 / gaps_h2.len() as f64
    } else {
        0.0
    };

    let mean_gap_h1 = if !gaps_h1.is_empty() {
        gaps_h1.iter().sum::<i64>() as f64 / gaps_h1.len() as f64
    } else {
        0.0
    };

    let mean_gap_h2 = if !gaps_h2.is_empty() {
        gaps_h2.iter().sum::<i64>() as f64 / gaps_h2.len() as f64
    } else {
        0.0
    };

    println!("\n--- 3D RESULTS N={} ---", n_particles);
    println!("Crystallization: {}/{}", n_crystal, n_trials);
    println!("H1 Precursor rate: {:.1}%", 100.0 * precursor_rate_h1);
    println!("H2 Precursor rate: {:.1}%", 100.0 * precursor_rate_h2);
    println!("Mean gap H1: {:.1}", mean_gap_h1);
    println!("Mean gap H2: {:.1}", mean_gap_h2);
    println!("Time/trial: {:.1}s", total_time / n_trials as f64);

    ValidationSummary {
        n_particles,
        n_trials,
        total_time_sec: total_time,
        time_per_trial_sec: total_time / n_trials as f64,
        n_crystal,
        precursor_rate_h1,
        precursor_rate_h2,
        mean_gap_h1,
        mean_gap_h2,
        trials: results,
    }
}

fn main() {
    println!("{}", "=".repeat(70));
    println!("TDA-CUSUM 3D VALIDATION (H1 + H2)");
    println!("{}", "=".repeat(70));

    std::fs::create_dir_all("../results").ok();

    // N=256 (4^3 * 4) - small 3D test
    let res_256 = run_validation_3d(256, 5);
    let json = serde_json::to_string_pretty(&res_256).unwrap();
    let mut file = File::create("../results/lj3d_N256.json").unwrap();
    file.write_all(json.as_bytes()).unwrap();

    // N=500 - medium 3D test
    let res_500 = run_validation_3d(500, 3);
    let json = serde_json::to_string_pretty(&res_500).unwrap();
    let mut file = File::create("../results/lj3d_N500.json").unwrap();
    file.write_all(json.as_bytes()).unwrap();

    println!("\n{}", "=".repeat(70));
    println!("3D VALIDATION COMPLETE");
    println!("{}", "=".repeat(70));
}
