//! TIP4P Water Ice Nucleation - TDA with H-bond Topology
//! ======================================================
//! Simplified TIP4P model for ice nucleation detection.
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

// === TIP4P PARAMETERS (in reduced units) ===
const DT: f64 = 0.001;          // Smaller for rigid molecules
const T_HIGH: f64 = 1.0;        // ~300K equivalent
const T_LOW: f64 = 0.75;        // ~225K, below Tm
const STEPS_EQUIL: usize = 3000;
const STEPS_PROD: usize = 8000;
const SAMPLE_INTERVAL: usize = 40;
const THERMOSTAT_TAU: usize = 40;
const R_CUT: f64 = 3.0;
const DENSITY: f64 = 0.9;       // Water density

// TIP4P geometry (in reduced units)
const R_OH: f64 = 0.9572;       // O-H bond length (scaled)
const THETA_HOH: f64 = 104.52;  // H-O-H angle in degrees
const R_OM: f64 = 0.15;         // O-M distance

// LJ parameters for O-O
const SIGMA_OO: f64 = 3.154;
const EPSILON_OO: f64 = 0.648;

// Charges (scaled)
const Q_H: f64 = 0.52;
const Q_M: f64 = -1.04;

// H-bond criteria
const HBOND_DIST: f64 = 3.5;    // O-O distance
const HBOND_ANGLE: f64 = 30.0;  // Max angle deviation

#[derive(Clone, Serialize, Deserialize)]
struct TrialResult {
    seed: u64,
    phase: String,
    q4_final: f64,
    n_hbonds_final: usize,
    t_phys: Option<usize>,
    t_topo_h1: Option<usize>,
    gap_h1: Option<i64>,
}

#[derive(Serialize)]
struct ValidationSummary {
    n_molecules: usize,
    n_trials: usize,
    total_time_sec: f64,
    time_per_trial_sec: f64,
    n_ice: usize,
    precursor_rate_h1: f64,
    mean_gap_h1: f64,
    trials: Vec<TrialResult>,
}

/// Simple water molecule representation (center of mass + orientation)
#[derive(Clone)]
struct WaterMolecule {
    com: [f64; 3],      // Center of mass
    vel: [f64; 3],      // COM velocity
    // Simplified: store O, H1, H2, M positions directly
    o_pos: [f64; 3],
    h1_pos: [f64; 3],
    h2_pos: [f64; 3],
    m_pos: [f64; 3],
}

impl WaterMolecule {
    fn new(com: [f64; 3], rng: &mut ChaCha8Rng) -> Self {
        // Random orientation
        let theta: f64 = rng.random::<f64>() * std::f64::consts::PI;
        let phi: f64 = rng.random::<f64>() * 2.0 * std::f64::consts::PI;

        let half_angle = (THETA_HOH / 2.0) * std::f64::consts::PI / 180.0;

        // O at COM (simplified)
        let o_pos = com;

        // H atoms
        let h1_pos = [
            com[0] + R_OH * half_angle.sin() * phi.cos(),
            com[1] + R_OH * half_angle.sin() * phi.sin(),
            com[2] + R_OH * half_angle.cos(),
        ];
        let h2_pos = [
            com[0] - R_OH * half_angle.sin() * phi.cos(),
            com[1] - R_OH * half_angle.sin() * phi.sin(),
            com[2] + R_OH * half_angle.cos(),
        ];

        // M site
        let m_pos = [
            com[0],
            com[1],
            com[2] + R_OM,
        ];

        WaterMolecule {
            com,
            vel: [0.0, 0.0, 0.0],
            o_pos,
            h1_pos,
            h2_pos,
            m_pos,
        }
    }

    fn update_positions(&mut self, box_size: f64) {
        // Update all sites based on COM (simplified rigid body)
        let half_angle = (THETA_HOH / 2.0) * std::f64::consts::PI / 180.0;

        self.o_pos = self.com;

        // Keep relative positions (simplified - no rotation)
        self.h1_pos = [
            (self.com[0] + R_OH * half_angle.sin()).rem_euclid(box_size),
            self.com[1].rem_euclid(box_size),
            (self.com[2] + R_OH * half_angle.cos()).rem_euclid(box_size),
        ];
        self.h2_pos = [
            (self.com[0] - R_OH * half_angle.sin()).rem_euclid(box_size),
            self.com[1].rem_euclid(box_size),
            (self.com[2] + R_OH * half_angle.cos()).rem_euclid(box_size),
        ];
        self.m_pos = [
            self.com[0].rem_euclid(box_size),
            self.com[1].rem_euclid(box_size),
            (self.com[2] + R_OM).rem_euclid(box_size),
        ];
    }
}

/// PBC distance
#[inline]
fn pbc_distance(p1: &[f64; 3], p2: &[f64; 3], box_size: f64) -> f64 {
    let mut dx = p1[0] - p2[0];
    let mut dy = p1[1] - p2[1];
    let mut dz = p1[2] - p2[2];
    dx -= box_size * (dx / box_size).round();
    dy -= box_size * (dy / box_size).round();
    dz -= box_size * (dz / box_size).round();
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Compute O-O distance matrix
fn compute_oo_distances(molecules: &[WaterMolecule], box_size: f64) -> Array2<f64> {
    let n = molecules.len();
    let mut dm = Array2::<f64>::zeros((n, n));

    for i in 0..n {
        for j in i + 1..n {
            let d = pbc_distance(&molecules[i].o_pos, &molecules[j].o_pos, box_size);
            dm[[i, j]] = d;
            dm[[j, i]] = d;
        }
    }
    dm
}

/// Count hydrogen bonds
fn count_hbonds(molecules: &[WaterMolecule], dm: &Array2<f64>) -> usize {
    let n = molecules.len();
    let mut count = 0;

    for i in 0..n {
        for j in i + 1..n {
            if dm[[i, j]] < HBOND_DIST && dm[[i, j]] > 2.0 {
                // Simple H-bond criterion: O-O distance in range
                count += 1;
            }
        }
    }
    count
}

/// Compute forces on molecules (simplified LJ + electrostatic)
fn compute_forces(molecules: &[WaterMolecule], box_size: f64) -> Vec<[f64; 3]> {
    let n = molecules.len();
    let r_cut2 = R_CUT * R_CUT;

    let forces: Vec<[f64; 3]> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut fx = 0.0;
            let mut fy = 0.0;
            let mut fz = 0.0;

            for j in 0..n {
                if i == j { continue; }

                // O-O LJ
                let oi = &molecules[i].o_pos;
                let oj = &molecules[j].o_pos;

                let mut dx = oi[0] - oj[0];
                let mut dy = oi[1] - oj[1];
                let mut dz = oi[2] - oj[2];
                dx -= box_size * (dx / box_size).round();
                dy -= box_size * (dy / box_size).round();
                dz -= box_size * (dz / box_size).round();

                let r2 = dx * dx + dy * dy + dz * dz;

                if r2 < r_cut2 && r2 > 0.5 {
                    // LJ force
                    let sigma2 = SIGMA_OO * SIGMA_OO / 100.0;  // Scale down
                    let s2_r2 = sigma2 / r2;
                    let s6_r6 = s2_r2 * s2_r2 * s2_r2;
                    let f_lj = 48.0 * EPSILON_OO * s6_r6 * (s6_r6 - 0.5) / r2;
                    let f_lj = f_lj.clamp(-20.0, 20.0);

                    // Simplified electrostatic (reaction field)
                    let r = r2.sqrt();
                    let f_elec = Q_M * Q_M / (r2 * r) * 0.1;  // Scaled
                    let f_elec = f_elec.clamp(-10.0, 10.0);

                    let f_total = f_lj + f_elec;
                    fx += f_total * dx;
                    fy += f_total * dy;
                    fz += f_total * dz;
                }
            }
            [fx, fy, fz]
        })
        .collect();

    forces
}

/// Velocity Verlet for molecules
fn velocity_verlet_molecules(
    molecules: &mut [WaterMolecule],
    forces: &mut Vec<[f64; 3]>,
    box_size: f64,
) {
    let n = molecules.len();

    // Half-step velocity
    for i in 0..n {
        molecules[i].vel[0] += 0.5 * DT * forces[i][0];
        molecules[i].vel[1] += 0.5 * DT * forces[i][1];
        molecules[i].vel[2] += 0.5 * DT * forces[i][2];
    }

    // Position update
    for i in 0..n {
        molecules[i].com[0] = (molecules[i].com[0] + DT * molecules[i].vel[0]).rem_euclid(box_size);
        molecules[i].com[1] = (molecules[i].com[1] + DT * molecules[i].vel[1]).rem_euclid(box_size);
        molecules[i].com[2] = (molecules[i].com[2] + DT * molecules[i].vel[2]).rem_euclid(box_size);
        molecules[i].update_positions(box_size);
    }

    // New forces
    *forces = compute_forces(molecules, box_size);

    // Second half-step velocity
    for i in 0..n {
        molecules[i].vel[0] += 0.5 * DT * forces[i][0];
        molecules[i].vel[1] += 0.5 * DT * forces[i][1];
        molecules[i].vel[2] += 0.5 * DT * forces[i][2];
    }
}

/// Thermostat for molecules
fn apply_thermostat_mol(molecules: &mut [WaterMolecule], target_t: f64) {
    let n = molecules.len() as f64;
    let ke: f64 = molecules.iter()
        .map(|m| 0.5 * (m.vel[0].powi(2) + m.vel[1].powi(2) + m.vel[2].powi(2)))
        .sum();
    let current_t = 2.0 * ke / (3.0 * n);

    if current_t > 1e-6 {
        let scale = (target_t / current_t).sqrt().clamp(0.9, 1.1);
        for m in molecules.iter_mut() {
            m.vel[0] *= scale;
            m.vel[1] *= scale;
            m.vel[2] *= scale;
        }
    }
}

/// Compute Q4 order parameter (tetrahedral order for ice)
fn compute_q4(dm: &Array2<f64>) -> f64 {
    let n = dm.nrows();
    let cutoff = 3.5;

    // Count molecules with 4 neighbors (ice-like)
    let ice_like: usize = (0..n)
        .filter(|&i| {
            let neighbors: usize = (0..n)
                .filter(|&j| i != j && dm[[i, j]] < cutoff && dm[[i, j]] > 2.0)
                .count();
            neighbors == 4
        })
        .count();

    ice_like as f64 / n as f64
}

/// H1 entropy from H-bond network
fn hbond_entropy(dm: &Array2<f64>) -> f64 {
    let n = dm.nrows();
    if n < 10 { return 0.0; }

    // H-bond distances only
    let mut hbond_dists: Vec<f64> = Vec::new();
    for i in 0..n {
        for j in i + 1..n {
            let d = dm[[i, j]];
            if d > 2.0 && d < HBOND_DIST {
                hbond_dists.push(d);
            }
        }
    }

    if hbond_dists.len() < 5 { return 0.0; }
    hbond_dists.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n_bins = 15;
    let max_d = hbond_dists.last().copied().unwrap_or(1.0);
    let min_d = hbond_dists.first().copied().unwrap_or(0.0);
    let bin_size = (max_d - min_d) / n_bins as f64;

    if bin_size < 0.01 { return 0.0; }

    let mut hist = vec![0usize; n_bins];
    for &d in &hbond_dists {
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

/// Detect ice formation
fn detect_ice(q4_series: &[f64], times: &[usize]) -> Option<usize> {
    let threshold = 0.4;  // Ice has high tetrahedral order
    let persistence = 3;

    let above: Vec<bool> = q4_series.iter().map(|&q| q > threshold).collect();
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

/// Run single TIP4P trial
fn run_trial_tip4p(n_molecules: usize, seed: u64) -> TrialResult {
    let box_size = (n_molecules as f64 / DENSITY).powf(1.0 / 3.0) * 3.0;  // Scale for molecular size
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Initialize molecules
    let mut molecules: Vec<WaterMolecule> = Vec::with_capacity(n_molecules);
    for _ in 0..n_molecules {
        let com = [
            rng.random::<f64>() * box_size,
            rng.random::<f64>() * box_size,
            rng.random::<f64>() * box_size,
        ];
        molecules.push(WaterMolecule::new(com, &mut rng));
    }

    // Initialize velocities
    for m in molecules.iter_mut() {
        m.vel = [
            rng.random::<f64>() * 2.0 - 1.0,
            rng.random::<f64>() * 2.0 - 1.0,
            rng.random::<f64>() * 2.0 - 1.0,
        ];
    }

    // Remove COM velocity
    let n = molecules.len() as f64;
    let com_vel: [f64; 3] = [
        molecules.iter().map(|m| m.vel[0]).sum::<f64>() / n,
        molecules.iter().map(|m| m.vel[1]).sum::<f64>() / n,
        molecules.iter().map(|m| m.vel[2]).sum::<f64>() / n,
    ];
    for m in molecules.iter_mut() {
        m.vel[0] -= com_vel[0];
        m.vel[1] -= com_vel[1];
        m.vel[2] -= com_vel[2];
    }
    apply_thermostat_mol(&mut molecules, T_HIGH);

    let mut forces = compute_forces(&molecules, box_size);

    // Equilibration
    for step in 0..STEPS_EQUIL {
        velocity_verlet_molecules(&mut molecules, &mut forces, box_size);
        if step % THERMOSTAT_TAU == 0 {
            apply_thermostat_mol(&mut molecules, T_HIGH);
        }
    }

    // Production
    let mut times = Vec::new();
    let mut q4_series = Vec::new();
    let mut hbond_series = Vec::new();
    let mut s_h1_series = Vec::new();

    for step in 0..STEPS_PROD {
        let progress = step as f64 / STEPS_PROD as f64;
        let target_t = T_HIGH + (T_LOW - T_HIGH) * progress;

        velocity_verlet_molecules(&mut molecules, &mut forces, box_size);

        if step % THERMOSTAT_TAU == 0 {
            apply_thermostat_mol(&mut molecules, target_t);
        }

        if step % SAMPLE_INTERVAL == 0 {
            let dm = compute_oo_distances(&molecules, box_size);
            let q4 = compute_q4(&dm);
            let n_hbonds = count_hbonds(&molecules, &dm);
            let s_h1 = hbond_entropy(&dm);

            times.push(step);
            q4_series.push(q4);
            hbond_series.push(n_hbonds);
            s_h1_series.push(s_h1);
        }
    }

    // Detection
    let t_phys = detect_ice(&q4_series, &times);
    let t_topo_h1 = detect_cusum(&s_h1_series, &times, 0.3);

    let gap_h1 = match (t_phys, t_topo_h1) {
        (Some(tp), Some(tt)) => Some(tp as i64 - tt as i64),
        _ => None,
    };

    let q4_final = q4_series[q4_series.len().saturating_sub(5)..].iter().sum::<f64>()
        / 5.0f64.min(q4_series.len() as f64);
    let n_hbonds_final = *hbond_series.last().unwrap_or(&0);

    let phase = if q4_final > 0.35 { "ICE" } else { "LIQUID" };

    TrialResult {
        seed,
        phase: phase.to_string(),
        q4_final,
        n_hbonds_final,
        t_phys,
        t_topo_h1,
        gap_h1,
    }
}

/// Run validation
fn run_validation_tip4p(n_molecules: usize, n_trials: usize) -> ValidationSummary {
    println!("\n{}", "=".repeat(70));
    println!("TIP4P WATER VALIDATION: N={}, {} trials", n_molecules, n_trials);
    println!("{}", "=".repeat(70));

    let total_start = Instant::now();
    let mut results = Vec::new();

    for i in 0..n_trials {
        let seed = (n_molecules as u64) * 2000 + i as u64;
        let trial_start = Instant::now();
        let result = run_trial_tip4p(n_molecules, seed);
        let elapsed = trial_start.elapsed().as_secs_f64();

        println!(
            "  Trial {}/{}: {}, Q4={:.3}, HB={}, gap_H1={:?} ({:.1}s)",
            i + 1, n_trials, result.phase, result.q4_final,
            result.n_hbonds_final, result.gap_h1, elapsed
        );

        results.push(result);
    }

    let total_time = total_start.elapsed().as_secs_f64();

    let n_ice = results.iter().filter(|r| r.phase == "ICE").count();

    let gaps: Vec<i64> = results.iter().filter(|r| r.phase == "ICE")
        .filter_map(|r| r.gap_h1).collect();
    let n_precursor = gaps.iter().filter(|&&g| g > 0).count();

    let precursor_rate_h1 = if !gaps.is_empty() {
        n_precursor as f64 / gaps.len() as f64
    } else { 0.0 };

    let mean_gap_h1 = if !gaps.is_empty() {
        gaps.iter().sum::<i64>() as f64 / gaps.len() as f64
    } else { 0.0 };

    println!("\n--- TIP4P RESULTS N={} ---", n_molecules);
    println!("Ice: {}/{}", n_ice, n_trials);
    println!("H1 Precursor rate: {:.1}%", 100.0 * precursor_rate_h1);
    println!("Mean gap H1: {:.1}", mean_gap_h1);
    println!("Time/trial: {:.1}s", total_time / n_trials as f64);

    ValidationSummary {
        n_molecules,
        n_trials,
        total_time_sec: total_time,
        time_per_trial_sec: total_time / n_trials as f64,
        n_ice,
        precursor_rate_h1,
        mean_gap_h1,
        trials: results,
    }
}

fn main() {
    println!("{}", "=".repeat(70));
    println!("TIP4P WATER TDA-CUSUM (H-bond Network Topology)");
    println!("{}", "=".repeat(70));

    std::fs::create_dir_all("../results").ok();

    // Smaller system for water (more expensive)
    let res_64 = run_validation_tip4p(64, 5);
    let json = serde_json::to_string_pretty(&res_64).unwrap();
    let mut file = File::create("../results/tip4p_N64.json").unwrap();
    file.write_all(json.as_bytes()).unwrap();

    println!("\n{}", "=".repeat(70));
    println!("TIP4P VALIDATION COMPLETE");
    println!("{}", "=".repeat(70));
}
