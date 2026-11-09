//! Aides communes des tests d'intégration du CLI : appel du binaire et boule en mm.

#![allow(dead_code)]

use std::path::Path;
use std::process::Command;

pub fn gyroxide(args: &[&str]) {
    let out = gyroxide_output(args);
    assert!(
        out.status.success(),
        "gyroxide {args:?}\n{}\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}

pub fn gyroxide_output(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_gyroxide"))
        .args(args)
        .output()
        .unwrap()
}

/// Netgen mesh of a voxel ball scaled to a "brain" in mm (radius 30 mm, off-centre).
pub fn ball_in_mm(path: &Path) {
    gyroxide(&["ball", "--n", "5", "--out", path.to_str().unwrap()]);
    let text = std::fs::read_to_string(path).unwrap();
    let mut lines: Vec<String> = text.lines().map(str::to_owned).collect();
    let n: usize = lines[0].trim().parse().unwrap();
    for l in &mut lines[1..=n] {
        let p: Vec<f64> = l.split_whitespace().map(|v| v.parse().unwrap()).collect();
        *l = format!("{} {} {}", 30.0 * p[0] - 20.0, 30.0 * p[1] + 5.0, 30.0 * p[2] + 12.0);
    }
    std::fs::write(path, lines.join("\n") + "\n").unwrap();
}
