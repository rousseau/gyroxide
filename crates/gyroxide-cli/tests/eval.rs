//! `gyroxide eval` sur une petite boule (CPU, f64) : un vrai run `run --bids`, puis son tableau de bord contre un
//! « atlas » synthétique (la surface initiale de la boule), la persistance et la comparaison de deux exécutions.
//! Le calcul des métriques est en Python (tools/scorecard.py) : le test est sauté sans python3 + numpy, scipy,
//! nibabel (la CI les installe).

mod common;

use common::{ball_in_mm, gyroxide};
use serde_json::Value;
use std::path::Path;
use std::process::Command;

fn python_ready() -> bool {
    Command::new("python3")
        .args(["-c", "import numpy, scipy, nibabel"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn json(path: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

#[test]
fn eval_on_a_ball_run() {
    if !python_ready() {
        eprintln!("skipped: python3 with numpy, scipy and nibabel not available");
        return;
    }
    let root = std::env::temp_dir().join(format!("gyroxide-eval-{}", std::process::id()));
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
        "25",
        "--bids",
        root.join("sourcedata").to_str().unwrap(),
        "--participant",
        "01",
        "--label",
        "essai",
        "--hemi",
        "L",
    ]);
    let run = root.join("derivatives/gyroxide/sub-01/anat/run-essai");
    // atlas synthétique : la boule initiale, aux semaines 21 (triangulation) et 22 (cible)
    let atlas = root.join("atlas");
    std::fs::create_dir(&atlas).unwrap();
    let step0 = run.join("sub-01_hemi-L_desc-sim_step-0000000_pial.surf.gii");
    for w in [21, 22] {
        std::fs::copy(&step0, atlas.join(format!("fetal.week{w}.left.pial.surf.gii"))).unwrap();
    }
    let meshes = root.join("meshes"); // vide : pas de maillage de l'atlas, masque = tous les sommets
    std::fs::create_dir(&meshes).unwrap();
    let (atlas_s, meshes_s) = (atlas.to_str().unwrap(), meshes.to_str().unwrap());

    let mut cards = vec![];
    for k in 1..=2 {
        let out = root.join(format!("scorecard{k}.json"));
        gyroxide(&[
            "eval",
            run.to_str().unwrap(),
            "--week",
            "22",
            "--atlas-dir",
            atlas_s,
            "--meshes-dir",
            meshes_s,
            "--out",
            out.to_str().unwrap(),
        ]);
        cards.push(out);
    }
    // deux exécutions : mêmes valeurs à 1e-6 (échoue avec un code non nul sinon)
    gyroxide(&[
        "eval",
        "--compare",
        cards[0].to_str().unwrap(),
        cards[1].to_str().unwrap(),
    ]);
    let card = json(&cards[0]);
    assert_eq!(card["week"], 22);
    assert_eq!(card["label"], "run-essai");
    assert_eq!(card["inputs"]["mask"], "all");
    assert_eq!(card["inputs"]["vertex_correspondence"], true);
    let m = &card["metrics"];
    let dev = m["dev_rigid_mm"]["vertex"]["rms"].as_f64().unwrap();
    assert!(dev > 0.0 && dev.is_finite(), "la boule a grandi : écart non nul, {dev}");
    for k in ["fold_depth_mm", "area_mm2", "gi", "folding_index"] {
        let r = m[k]["ratio"].as_f64().unwrap();
        assert!(r.is_finite() && r > 0.0, "{k}: {r}");
    }
    assert!(
        m["area_mm2"]["ratio"].as_f64().unwrap() > 1.0,
        "la croissance augmente l'aire"
    );
    assert!(m["curvature_corr"].as_f64().unwrap().is_finite());
    assert_eq!(m["curvature_bands"]["scales_mm"].as_array().unwrap().len(), 4);
    assert!(
        m["thickness_mm"].is_null(),
        "pas de tétraèdres de l'atlas : pas d'épaisseur"
    );
    assert_eq!(m["diagnostics"]["residual"]["source"], "sequence.tsv");

    // persistance : atlas 21 contre atlas 22, ici identiques → écart nul, rapports 1
    let pers = root.join("persistence");
    gyroxide(&[
        "eval",
        "--persistence",
        "--weeks",
        "22-22",
        "--atlas-dir",
        atlas_s,
        "--meshes-dir",
        meshes_s,
        "--out",
        pers.to_str().unwrap(),
    ]);
    let p = json(&pers.join("persistence_sg22.json"));
    assert_eq!(p["label"], "persistence");
    assert!(p["metrics"]["dev_rigid_mm"]["vertex"]["rms"].as_f64().unwrap() < 1e-9);
    assert!((p["metrics"]["area_mm2"]["ratio"].as_f64().unwrap() - 1.0).abs() < 1e-12);
    std::fs::remove_dir_all(&root).unwrap();
}
