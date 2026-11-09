//! `gyroxide run --bids` on a small ball (CPU, f64): run directory, sequence.tsv, status.json,
//! config.toml, dataset_description.json and GIfTI surfaces in image space (gyroxide#2, #3).

mod common;

use common::{ball_in_mm, gyroxide, gyroxide_output};
use std::process::Command;

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

/// Growth mask from the `cortex` field of a session .vtu (`--growth-field cortex`): the name is kept in
/// config.toml, a missing field is an error.
#[test]
fn growth_mask_from_vtu_field() {
    let root = std::env::temp_dir().join(format!("gyroxide-growth-field-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(root.join("sourcedata")).unwrap();
    let netgen = root.join("ball.mesh");
    ball_in_mm(&netgen);
    let mesh = gyroxide_mesh::netgen::read(&netgen).unwrap();
    // "medial wall": no growth for x < -20 mm
    let cortex: Vec<f64> = mesh
        .nodes
        .iter()
        .map(|p| if p[0] < -20.0 { 0.0 } else { 1.0 })
        .collect();
    assert!(cortex.contains(&0.0) && cortex.contains(&1.0));
    let vtu = root.join("ball_tet.vtu");
    let field = |name: &str, values: Vec<f64>| gyroxide_mesh::vtu::PointField {
        name: name.into(),
        values,
    };
    gyroxide_mesh::vtu::write(&vtu, &mesh, &[field("cortex", cortex)]).unwrap();
    let run = |growth_field: &str, label: &str| {
        gyroxide_output(&[
            "run",
            "--preset",
            "braingrowth",
            "--mesh",
            vtu.to_str().unwrap(),
            "--backend",
            "ndarray",
            "--growth-field",
            growth_field,
            "--no-midplane",
            "--mesh-spacing",
            "0.4",
            "--steps",
            "20",
            "--bids",
            root.join("sourcedata").to_str().unwrap(),
            "--participant",
            "01",
            "--label",
            label,
            "--hemi",
            "L",
        ])
    };
    let out = run("cortex", "masque");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let config = std::fs::read_to_string(root.join("derivatives/gyroxide/sub-01/anat/run-masque/config.toml")).unwrap();
    assert!(
        config.contains("kind = \"mesh_field\"") && config.contains("name = \"cortex\""),
        "{config}"
    );
    let out = run("sulc", "absent");
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("no node field 'sulc'"));
    std::fs::remove_dir_all(&root).unwrap();
}
