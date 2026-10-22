//! `gyroxide run --bids` on a small ball (CPU, f64): run directory, sequence.tsv, status.json,
//! config.toml, dataset_description.json and GIfTI surfaces in image space (gyroxide#2, #3).

use std::path::Path;
use std::process::Command;

fn gyroxide(args: &[&str]) {
    let out = Command::new(env!("CARGO_BIN_EXE_gyroxide"))
        .args(args)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "gyroxide {args:?}\n{}\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}

/// Netgen mesh of a voxel ball scaled to a "brain" in mm (radius 30 mm, off-centre).
fn ball_in_mm(path: &Path) {
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

#[test]
fn bids_run_directory() {
    let root = std::env::temp_dir().join(format!("gyroxide-bids-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(root.join("sourcedata")).unwrap();
    let mesh = root.join("ball.mesh");
    ball_in_mm(&mesh);
    let config = root.join("params.toml");
    let preset = Command::new(env!("CARGO_BIN_EXE_gyroxide"))
        .args(["preset", "braingrowth"])
        .output()
        .unwrap();
    let toml = String::from_utf8(preset.stdout)
        .unwrap()
        .replace("normalization = \"none\"", "normalization = \"brain_growth_whole\"");
    assert!(toml.contains("brain_growth_whole"), "{toml}");
    std::fs::write(&config, toml).unwrap();
    gyroxide(&[
        "run",
        "--config",
        config.to_str().unwrap(),
        "--mesh",
        mesh.to_str().unwrap(),
        "--backend",
        "ndarray",
        "--uniform-growth",
        "--no-midplane",
        "--mesh-spacing",
        "0.4",
        "--steps",
        "50",
        "--surface-every",
        "20",
        "--bids",
        root.join("sourcedata").to_str().unwrap(),
        "--participant",
        "01",
        "--session",
        "2",
        "--label",
        "essai",
        "--hemi",
        "R",
    ]);
    // sourcedata/ -> derivatives/ next to it
    let pipeline = root.join("derivatives/gyroxide");
    let run = pipeline.join("sub-01/ses-2/anat/run-essai");
    assert_eq!(gyroxide_io::lister_runs(run.parent().unwrap()).len(), 1);
    let status = gyroxide_io::lire_status(&run.join("status.json")).unwrap();
    assert_eq!(status.state, gyroxide_io::State::Done);
    assert_eq!((status.step, status.total_steps), (Some(50), Some(50)));
    let seq = gyroxide_io::lire_sequence(&run.join("sequence.tsv")).unwrap();
    assert_eq!(seq.iter().map(|r| r.step).collect::<Vec<_>>(), [0, 20, 40, 50]);
    for r in &seq {
        assert!(run.join(&r.file).is_file(), "{}", r.file);
        assert!(r.gi.unwrap() >= 1.0 && r.energy.is_some() && r.n_contacts.is_some());
        assert_eq!(r.ga_weeks, None, "braingrowth: no ga_weeks");
    }
    assert_eq!(seq[1].file, "sub-01_ses-2_hemi-R_desc-sim_step-0000020_pial.surf.gii");
    // step 0: the initial surface, back in the frame of the input (mm)
    let (s0, meta) = surfoxide_io::gifti::read_surface(run.join(&seq[0].file)).unwrap();
    assert_eq!(meta.primary.as_deref(), Some("CortexRight"));
    let (lo, hi) = s0
        .vertices()
        .iter()
        .fold(([f64::MAX; 3], [f64::MIN; 3]), |(lo, hi), p| {
            (
                std::array::from_fn(|k| lo[k].min(p[k])),
                std::array::from_fn(|k| hi[k].max(p[k])),
            )
        });
    assert!(
        (lo[0] + 50.0).abs() < 1e-3 && (hi[0] - 10.0).abs() < 1e-3,
        "x in mm: {lo:?} {hi:?}"
    );
    assert!(
        (lo[2] + 18.0).abs() < 1e-3 && (hi[2] - 42.0).abs() < 1e-3,
        "z in mm: {lo:?} {hi:?}"
    );
    let config = std::fs::read_to_string(run.join("config.toml")).unwrap();
    assert!(
        config.contains("initial_state = \"../../../../../../ball.mesh\""),
        "{config}"
    );
    assert!(config.contains("[material]"));
    let desc: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(pipeline.join("dataset_description.json")).unwrap()).unwrap();
    assert_eq!(desc["GeneratedBy"][0]["Name"], "gyroxide run");
    std::fs::remove_dir_all(&root).unwrap();
}
