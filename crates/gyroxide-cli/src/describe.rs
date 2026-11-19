//! `gyroxide --describe`: JSON description of the commands guioxide can launch (schema `oxide-describe/1`,
//! `docs/donnees.md` of guioxide, section 5).
//!
//! The document is generated, not written by hand: parameters come from the clap definitions
//! (`Cli::command()`), file names from `gyroxide_io::contrat` (the constants the writing code uses), the rest from
//! this module. Nothing is read or written, and no GPU is touched. Only the commands with a BIDS mode are described
//! (`mesh`, `run`). The model parameters (material, growth schedule…) live in `config.toml`, not in clap: this
//! schema describes the command line only (`--preset`, `--config`).

use std::any::TypeId;

use clap::{Arg, ArgAction, Command, CommandFactory};
use gyroxide_io::contrat::{self, ANAT_PREFIX};
use serde_json::{json, Map, Value};

use crate::Cli;

/// Schema version.
const SCHEMA: &str = "oxide-describe/1";
/// Version of the data contract of guioxide the tool follows.
const CONTRACT: &str = "1.0";
/// Source repository.
const CODE_URL: &str = "https://github.com/rousseau/gyroxide";
/// Output folder of `surfx run`, input of `gyroxide mesh`.
const PIPELINE_SURFACES: &str = "surfx-surfaces";

/// BIDS arguments of clap: they make guioxide's BIDS command line, they are not parameters.
const BIDS_ARGS: [&str; 4] = ["bids", "participant", "session", "output_dir"];
/// Free-mode paths and inputs (outside BIDS): listed apart, not parameters.
const FREE_IO: [&str; 3] = ["mesh", "ref_static", "out"];
/// Parameters shown at once in a form; all others are `advanced`.
const BASIC: [&str; 6] = ["label", "preset", "hemi", "steps", "desc", "tetgen"];
/// Known lower bounds: (parameter, minimum). Nothing is guessed.
const MINIMA: [(&str, i64); 2] = [("steps", 1), ("surface_every", 0)];

/// Full document: `{"schema", "contract", "tool", "commands"}`.
pub fn document() -> Value {
    let cli = Cli::command();
    let sub = |name: &str| {
        cli.get_subcommands()
            .find(|c| c.get_name() == name)
            .expect("known subcommand")
            .clone()
    };
    json!({
        "schema": SCHEMA,
        "contract": CONTRACT,
        "tool": {
            "name": "gyroxide",
            "version": env!("CARGO_PKG_VERSION"),
            "commit": option_env!("GYROXIDE_COMMIT"),
            "code_url": CODE_URL,
        },
        "commands": [command_mesh(&sub("mesh")), command_run(&sub("run"))],
    })
}

/// First sentence of clap's `about`.
fn summary(c: &Command) -> String {
    let about = c.get_about().map(|a| a.to_string()).unwrap_or_default();
    let flat = about.split_whitespace().collect::<Vec<_>>().join(" ");
    match flat.find(". ") {
        Some(i) => flat[..=i].to_string(),
        None => flat,
    }
}

/// JSON value of a clap default, by parameter type.
fn default_value(kind: &str, arg: &Arg) -> Value {
    let Some(text) = arg.get_default_values().first().and_then(|v| v.to_str()) else {
        return if kind == "boolean" { json!(false) } else { Value::Null };
    };
    match kind {
        "number" => text.parse::<f64>().map_or(Value::Null, |v| json!(v)),
        "integer" => text.parse::<i64>().map_or(Value::Null, |v| json!(v)),
        "boolean" => json!(text == "true"),
        _ => json!(text),
    }
}

/// Schema type from clap's value parser.
fn kind_of(arg: &Arg, has_choices: bool) -> &'static str {
    if matches!(arg.get_action(), ArgAction::SetTrue | ArgAction::SetFalse) {
        return "boolean";
    }
    if has_choices {
        return "enum";
    }
    let id = arg.get_value_parser().type_id();
    if id == TypeId::of::<f64>() || id == TypeId::of::<f32>() {
        "number"
    } else if [
        TypeId::of::<usize>(),
        TypeId::of::<u64>(),
        TypeId::of::<u32>(),
        TypeId::of::<i64>(),
        TypeId::of::<i32>(),
    ]
    .iter()
    .any(|t| id == *t)
    {
        "integer"
    } else if id == TypeId::of::<std::path::PathBuf>() {
        "path"
    } else {
        "string"
    }
}

/// Schema parameter for a clap argument.
fn parameter(arg: &Arg) -> Value {
    let id = arg.get_id().as_str();
    let choices: Vec<String> = arg
        .get_possible_values()
        .iter()
        .map(|v| v.get_name().to_string())
        .collect();
    let kind = kind_of(arg, !choices.is_empty());
    let mut m = Map::new();
    m.insert("id".into(), id.into());
    m.insert("flag".into(), format!("--{}", arg.get_long().unwrap_or(id)).into());
    m.insert("type".into(), kind.into());
    if !choices.is_empty() {
        m.insert("choices".into(), json!(choices));
    }
    m.insert("default".into(), default_value(kind, arg));
    if let Some((_, min)) = MINIMA.iter().find(|(name, _)| *name == id) {
        m.insert("min".into(), json!(min));
    }
    m.insert("optional".into(), true.into());
    m.insert(
        "level".into(),
        (if BASIC.contains(&id) { "basic" } else { "advanced" }).into(),
    );
    m.insert(
        "help".into(),
        arg.get_help().map(|h| h.to_string()).unwrap_or_default().into(),
    );
    Value::Object(m)
}

/// Parameters of a command: every long option except the BIDS arguments and the free-mode paths.
fn parameters(c: &Command) -> Vec<Value> {
    c.get_arguments()
        .filter(|a| {
            let id = a.get_id().as_str();
            a.get_long().is_some()
                && !BIDS_ARGS.contains(&id)
                && !FREE_IO.contains(&id)
                && id != "help"
                && id != "version"
        })
        .map(parameter)
        .collect()
}

/// `{"hemi": {"param": "hemi", "all": [...]}}` or without `all`.
fn expand_hemi(all: bool) -> Value {
    if all {
        json!({ "hemi": { "param": "hemi", "all": ["L", "R"] } })
    } else {
        json!({ "hemi": { "param": "hemi" } })
    }
}

/// `gyroxide mesh`: tetrahedral mesh of a session hemisphere, from the surfaces of `surfx run`.
fn command_mesh(clap: &Command) -> Value {
    let from_surfx = |id: &str, nature: &str, suffix: &str| {
        json!({
            "id": id,
            "nature": nature,
            "pipeline": PIPELINE_SURFACES,
            "pattern": format!("{ANAT_PREFIX}{suffix}"),
            "expand": expand_hemi(false),
            "cardinality": "session",
            "count": "one",
            "required": true,
        })
    };
    json!({
        "id": "mesh",
        "argv": ["mesh"],
        "summary": summary(clap),
        "stage": "simulation",
        "pipeline": gyroxide_io::PIPELINE,
        "generated_by": "gyroxide mesh",
        "bids": {
            "mode": "required",
            "root": "--bids",
            "participant": "--participant",
            "session": "--session",
            "output_dir": "--output-dir",
        },
        "inputs": [
            from_surfx("white", "surface", contrat::ENTREE_BLANCHE),
            from_surfx("pial", "surface", contrat::ENTREE_PIALE),
            from_surfx("cortex", "field", contrat::ENTREE_CORTEX),
        ],
        "outputs": [{
            "id": "mesh",
            "nature": "mesh",
            "required": true,
            "cardinality": "session",
            "expand": expand_hemi(false),
            "pattern": format!("{ANAT_PREFIX}{}", contrat::SUFFIXE_MAILLAGE),
        }],
        "parameters": parameters(clap),
        "presets": [],
        "default_preset": null,
        "resources": { "gpu": "none", "device_param": null, "memory_gb": null },
        "progress": null,
    })
}

/// `gyroxide run`: a simulation of the session hemisphere, in `run-<label>/`.
fn command_run(clap: &Command) -> Value {
    let in_run = |id: &str, nature: &str, pattern: &str| json!({ "id": id, "nature": nature, "required": true, "pattern": pattern });
    let mut surfaces = in_run(
        "surfaces",
        "surface",
        &format!("sub-{{sub}}[_ses-{{ses}}]{}", contrat::SURFACE_SIMULEE),
    );
    surfaces["expand"] = expand_hemi(false);
    surfaces["series"] = json!({ "placeholder": "step", "index_output": "sequence" });
    json!({
        "id": "run",
        "argv": ["run"],
        "summary": summary(clap),
        "stage": "simulation",
        "pipeline": gyroxide_io::PIPELINE,
        "generated_by": "gyroxide run",
        "bids": {
            "mode": "supported",
            "root": "--bids",
            "participant": "--participant",
            "session": "--session",
            "output_dir": "--output-dir",
            "extra_required": ["label"],
        },
        "inputs": [{
            "id": "mesh",
            "nature": "mesh",
            "pipeline": gyroxide_io::PIPELINE,
            "pattern": format!("{ANAT_PREFIX}{}", contrat::SUFFIXE_MAILLAGE.replace("{desc}", "{mesh_desc}")),
            "expand": expand_hemi(false),
            "cardinality": "session",
            "count": "one",
            "required": true,
        }],
        "outputs": [{
            "id": "run",
            "nature": "run",
            "required": true,
            "cardinality": "session",
            "pattern": format!("sub-{{sub}}[/ses-{{ses}}]/anat/{}{{label}}/", contrat::PREFIXE_DOSSIER_RUN),
            "contents": [
                in_run("config", "metadata", contrat::FICHIER_CONFIG),
                in_run("status", "metadata", contrat::FICHIER_STATUS),
                in_run("sequence", "table", contrat::FICHIER_SEQUENCE),
                surfaces,
            ],
        }],
        "parameters": parameters(clap),
        "presets": [{
            "id": "session-hemisphere",
            "label": "Hémisphère d'une session",
            "params": { "growth_field": "cortex", "no_midplane": true },
        }],
        "default_preset": "session-hemisphere",
        "resources": { "gpu": "optional", "device_param": "backend", "memory_gb": null },
        "progress": {
            "pattern": format!("sub-{{sub}}[/ses-{{ses}}]/anat/{}{{label}}/{}", contrat::PREFIXE_DOSSIER_RUN, contrat::FICHIER_STATUS),
            "format": "status-json",
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    fn cmd<'a>(d: &'a Value, id: &str) -> &'a Value {
        d["commands"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["id"] == id)
            .unwrap()
    }

    /// Form of the document: schema, two commands, patterns built from the writer constants.
    #[test]
    fn the_document_has_mesh_and_run() {
        let d = document();
        assert_eq!(d["schema"], "oxide-describe/1");
        assert_eq!(d["tool"]["name"], "gyroxide");
        let mesh = cmd(&d, "mesh");
        assert_eq!(mesh["bids"]["mode"], "required");
        assert_eq!(
            mesh["outputs"][0]["pattern"],
            "sub-{sub}[/ses-{ses}]/anat/sub-{sub}[_ses-{ses}]_hemi-{hemi}_desc-{desc}_tet.vtu"
        );
        let run = cmd(&d, "run");
        assert_eq!(run["bids"]["extra_required"], json!(["label"]));
        assert_eq!(run["outputs"][0]["pattern"], "sub-{sub}[/ses-{ses}]/anat/run-{label}/");
        assert_eq!(
            run["inputs"][0]["pattern"],
            "sub-{sub}[/ses-{ses}]/anat/sub-{sub}[_ses-{ses}]_hemi-{hemi}_desc-{mesh_desc}_tet.vtu"
        );
        assert_eq!(run["progress"]["format"], "status-json");
    }

    /// Every long option of each described command is in the document, except BIDS and free-mode paths; clap defaults.
    #[test]
    fn every_clap_option_is_described_with_its_default() {
        let cli = Cli::command();
        let d = document();
        for id in ["mesh", "run"] {
            let clap = cli.get_subcommands().find(|c| c.get_name() == id).unwrap();
            let params = cmd(&d, id)["parameters"].as_array().unwrap();
            for a in clap.get_arguments().filter(|a| a.get_long().is_some()) {
                let name = a.get_id().as_str();
                let excluded =
                    BIDS_ARGS.contains(&name) || FREE_IO.contains(&name) || name == "help" || name == "version";
                assert_eq!(params.iter().any(|p| p["id"] == name), !excluded, "{id}: {name}");
            }
            for free in FREE_IO {
                assert!(
                    clap.get_arguments().any(|a| a.get_id() == free) || id == "mesh",
                    "{id}: {free} unknown to clap"
                );
            }
        }
        let run = cmd(&d, "run")["parameters"].as_array().unwrap();
        let par = |name: &str| run.iter().find(|p| p["id"] == name).unwrap();
        assert_eq!(par("hemi")["type"], "enum");
        assert_eq!(par("hemi")["choices"], json!(["L", "R"]));
        assert_eq!(par("hemi")["default"], "L");
        assert_eq!(par("preset")["default"], "tallinen2016");
        assert_eq!(par("steps")["type"], "integer");
        assert_eq!(par("steps")["default"], Value::Null);
        assert_eq!(par("surface_every")["default"], 0);
        assert_eq!(par("no_midplane")["type"], "boolean");
        assert_eq!(par("no_midplane")["default"], false);
        assert_eq!(par("backend")["type"], "enum");
        assert_eq!(par("label")["level"], "basic");
        assert_eq!(par("print_every")["level"], "advanced");
        for basic in BASIC {
            let in_mesh = cmd(&d, "mesh")["parameters"]
                .as_array()
                .unwrap()
                .iter()
                .any(|p| p["id"] == basic);
            assert!(
                in_mesh || run.iter().any(|p| p["id"] == basic),
                "basic parameter unknown: {basic}"
            );
        }
    }

    /// Every preset is accepted as is by clap, and its parameter ids exist.
    #[test]
    fn every_preset_is_accepted_by_clap() {
        let d = document();
        for c in d["commands"].as_array().unwrap() {
            for p in c["presets"].as_array().unwrap() {
                let mut argv: Vec<String> = vec!["gyroxide".into()];
                argv.extend(
                    c["argv"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_str().unwrap().to_string()),
                );
                if c["id"] == "run" {
                    argv.extend(["--bids", "r", "--participant", "p", "--label", "x"].map(String::from));
                }
                for (id, value) in p["params"].as_object().unwrap() {
                    let param = c["parameters"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .find(|q| q["id"] == id.as_str())
                        .unwrap();
                    argv.push(param["flag"].as_str().unwrap().to_string());
                    if param["type"] != "boolean" {
                        argv.push(value.as_str().unwrap().to_string());
                    }
                }
                Cli::try_parse_from(&argv).unwrap_or_else(|e| panic!("preset {}: {e}", p["id"]));
            }
        }
        assert_eq!(cmd(&d, "run")["default_preset"], "session-hemisphere");
    }

    /// The patterns are the names the writing code builds (with and without session).
    #[test]
    fn the_patterns_equal_the_names_the_writers_build() {
        assert_eq!(
            contrat::suffixe_maillage('L', "{desc}"),
            contrat::SUFFIXE_MAILLAGE.replace("{hemi}", "L")
        );
        assert_eq!(
            gyroxide_io::surface_name("sub-01_ses-2", 'R', 20),
            "sub-01_ses-2_hemi-R_desc-sim_step-0000020_pial.surf.gii"
        );
        assert_eq!(
            contrat::suffixe_surface_simulee('R', 20),
            contrat::SURFACE_SIMULEE
                .replace("{hemi}", "R")
                .replace("{step}", "0000020")
        );
    }
}
