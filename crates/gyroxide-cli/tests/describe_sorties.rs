//! `gyroxide --describe` against reality: every `required` output declared for `run` is written by
//! `gyroxide run --bids` (small ball, CPU, f64, as in `bids.rs`), and `--describe` reads and writes nothing.

mod common;

use common::{ball_in_mm, gyroxide, gyroxide_output};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::Path;

/// Resolves a path pattern: `{x}` replaced by its value; an optional group `[…]` is kept (without brackets) when every
/// mark in it has a value, dropped otherwise.
fn resolve(pattern: &str, vars: &BTreeMap<&str, String>) -> String {
    let mut out = String::new();
    let mut rest = pattern;
    while let Some(i) = rest.find('[') {
        out.push_str(&substitute(&rest[..i], vars).expect("mark without value outside an optional group"));
        let j = rest[i..].find(']').expect("unclosed bracket") + i;
        if let Some(inner) = substitute(&rest[i + 1..j], vars) {
            out.push_str(&inner);
        }
        rest = &rest[j + 1..];
    }
    out.push_str(&substitute(rest, vars).expect("mark without value"));
    out
}

fn substitute(text: &str, vars: &BTreeMap<&str, String>) -> Option<String> {
    let mut out = String::new();
    let mut rest = text;
    while let Some(i) = rest.find('{') {
        out.push_str(&rest[..i]);
        let j = rest[i..].find('}')? + i;
        out.push_str(vars.get(&rest[i + 1..j])?);
        rest = &rest[j + 1..];
    }
    out.push_str(rest);
    Some(out)
}

#[test]
fn the_pattern_resolver_handles_optional_groups() {
    let mut v = BTreeMap::new();
    v.insert("sub", "01".to_string());
    v.insert("ses", "2".to_string());
    let p = "sub-{sub}[/ses-{ses}]/anat/sub-{sub}[_ses-{ses}]_x.nii.gz";
    assert_eq!(resolve(p, &v), "sub-01/ses-2/anat/sub-01_ses-2_x.nii.gz");
    v.remove("ses");
    assert_eq!(resolve(p, &v), "sub-01/anat/sub-01_x.nii.gz");
}

#[test]
fn describe_prints_json_and_nothing_else() {
    let out = gyroxide_output(&["--describe"]);
    assert!(out.status.success());
    assert!(out.stderr.is_empty());
    let d: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(d["schema"], "oxide-describe/1");
    let ids: Vec<&str> = d["commands"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c["id"].as_str().unwrap())
        .collect();
    assert_eq!(ids, ["mesh", "run"]);
}

#[test]
fn every_required_output_of_run_is_written() {
    let root = std::env::temp_dir().join(format!("gyroxide-describe-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(root.join("sourcedata")).unwrap();
    let mesh = root.join("ball.mesh");
    ball_in_mm(&mesh);
    let config = root.join("params.toml");
    let preset = gyroxide_output(&["preset", "braingrowth"]);
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

    let doc: Value = serde_json::from_slice(&gyroxide_output(&["--describe"]).stdout).unwrap();
    let run = doc["commands"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["id"] == "run")
        .unwrap();
    let pipeline = root.join("derivatives").join(run["pipeline"].as_str().unwrap());
    let mut vars = BTreeMap::new();
    vars.insert("sub", "01".to_string());
    vars.insert("ses", "2".to_string());
    vars.insert("label", "essai".to_string());
    vars.insert("hemi", "R".to_string());

    let mut missing = Vec::new();
    let mut check = |path: &Path| {
        if !path.exists() {
            missing.push(path.display().to_string());
        }
    };
    for o in run["outputs"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|o| o["required"] == true)
    {
        let dir = pipeline.join(resolve(o["pattern"].as_str().unwrap(), &vars));
        check(&dir);
        for c in o["contents"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|c| c["required"] == true)
        {
            let pattern = c["pattern"].as_str().unwrap();
            if let Some(series) = c.get("series") {
                // The files of a series are listed by the table named in `index_output`.
                let index = o["contents"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|x| x["id"] == series["index_output"])
                    .unwrap();
                let rows = gyroxide_io::lire_sequence(&dir.join(index["pattern"].as_str().unwrap())).unwrap();
                assert!(!rows.is_empty());
                for row in rows {
                    let mut v = vars.clone();
                    v.insert("step", format!("{:07}", row.step));
                    let expected = resolve(pattern, &v);
                    assert_eq!(
                        row.file, expected,
                        "the sequence.tsv file name differs from the declared pattern"
                    );
                    check(&dir.join(expected));
                }
            } else {
                check(&dir.join(pattern));
            }
        }
    }
    let _ = std::fs::remove_dir_all(&root);
    assert!(missing.is_empty(), "declared but not written: {missing:#?}");
}
