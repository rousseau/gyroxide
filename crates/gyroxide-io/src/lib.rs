//! Runs de gyroxide, selon le contrat de données de guioxide (`docs/donnees.md`, section 4), sans Burn :
//! écriture et lecture de `status.json` et de `sequence.tsv`, liste des dossiers `run-<label>`, indice de
//! gyrification `gi`.
//!
//! ```text
//! <étude>/derivatives/gyroxide/sub-X[/ses-Y]/anat/run-<label>/
//!   config.toml        paramètres complets, chemin relatif de l'état initial
//!   status.json        {"state", "step", "total_steps", "started", "updated", "message"}
//!   sequence.tsv       step  time  ga_weeks  file  gi  energy  n_contacts
//!   sub-X[_ses-Y]_hemi-L_desc-sim_step-0000000_pial.surf.gii …
//! ```

use std::fmt::Write as _;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Nom du pipeline, dossier de `derivatives/`.
pub const PIPELINE: &str = "gyroxide";
/// Colonnes de `sequence.tsv`, dans l'ordre.
pub const SEQUENCE_COLUMNS: [&str; 7] = ["step", "time", "ga_weeks", "file", "gi", "energy", "n_contacts"];

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{path} : {source}")]
    Io { path: PathBuf, source: std::io::Error },
    #[error("{path} : {message}")]
    Format { path: PathBuf, message: String },
}

impl Error {
    fn io(path: &Path, source: std::io::Error) -> Self {
        Self::Io {
            path: path.to_path_buf(),
            source,
        }
    }
    fn format(path: &Path, message: impl Into<String>) -> Self {
        Self::Format {
            path: path.to_path_buf(),
            message: message.into(),
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;

// ------------------------------------------------------------------------------------------------
// status.json

/// État d'un run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum State {
    Running,
    Done,
    Failed,
}

impl State {
    pub fn as_str(self) -> &'static str {
        match self {
            State::Running => "running",
            State::Done => "done",
            State::Failed => "failed",
        }
    }
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "running" => Some(State::Running),
            "done" => Some(State::Done),
            "failed" => Some(State::Failed),
            _ => None,
        }
    }
}

/// Contenu de `status.json`.
#[derive(Debug, Clone, PartialEq)]
pub struct Status {
    pub state: State,
    pub step: Option<u64>,
    pub total_steps: Option<u64>,
    /// Dates ISO 8601 (UTC).
    pub started: Option<String>,
    pub updated: Option<String>,
    pub message: Option<String>,
}

impl Status {
    /// Avancement dans [0, 1], si le nombre total de pas est connu.
    pub fn progress(&self) -> Option<f64> {
        let (s, t) = (self.step?, self.total_steps?);
        (t > 0).then(|| (s as f64 / t as f64).clamp(0.0, 1.0))
    }

    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "state": self.state.as_str(),
            "step": self.step,
            "total_steps": self.total_steps,
            "started": self.started,
            "updated": self.updated,
            "message": self.message,
        })
    }
}

/// Écrit `status.json` de façon atomique : fichier temporaire dans le même dossier, puis renommage (un
/// lecteur ne voit jamais un fichier à moitié écrit).
pub fn ecrire_status(path: &Path, status: &Status) -> Result<()> {
    let text = serde_json::to_string_pretty(&status.to_json()).expect("JSON") + "\n";
    write_atomic(path, text.as_bytes())
}

/// Lit `status.json` ; seul `state` est obligatoire.
pub fn lire_status(path: &Path) -> Result<Status> {
    let text = std::fs::read_to_string(path).map_err(|e| Error::io(path, e))?;
    let v: serde_json::Value = serde_json::from_str(&text).map_err(|e| Error::format(path, e.to_string()))?;
    let s = |k: &str| v[k].as_str().map(str::to_owned);
    let state = s("state").ok_or_else(|| Error::format(path, "champ « state » absent"))?;
    Ok(Status {
        state: State::parse(&state)
            .ok_or_else(|| Error::format(path, format!("état « {state} » inconnu (running, done, failed)")))?,
        step: v["step"].as_u64(),
        total_steps: v["total_steps"].as_u64(),
        started: s("started"),
        updated: s("updated"),
        message: s("message"),
    })
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let tmp = path.with_file_name(format!(".{name}.tmp"));
    std::fs::write(&tmp, bytes).map_err(|e| Error::io(&tmp, e))?;
    std::fs::rename(&tmp, path).map_err(|e| Error::io(path, e))
}

/// Date et heure UTC courantes, ISO 8601 à la seconde (`2026-10-21T14:03:52Z`).
pub fn now_iso8601() -> String {
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs());
    iso8601(secs)
}

/// Secondes depuis l'époque Unix -> ISO 8601 UTC (calendrier grégorien proleptique).
pub fn iso8601(secs: u64) -> String {
    let (days, rem) = ((secs / 86_400) as i64, secs % 86_400);
    // Jours -> date civile (H. Hinnant, « chrono-compatible low-level date algorithms »).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        rem / 60 % 60,
        rem % 60
    )
}

// ------------------------------------------------------------------------------------------------
// sequence.tsv

/// Une ligne de `sequence.tsv` : une surface écrite au pas `step`.
#[derive(Debug, Clone, PartialEq)]
pub struct SequenceRow {
    pub step: u64,
    /// Temps du modèle.
    pub time: Option<f64>,
    /// Âge gestationnel (semaines), interpolé depuis `schedule.ga_weeks`.
    pub ga_weeks: Option<f64>,
    /// Nom du fichier de surface, relatif au dossier du run.
    pub file: String,
    /// Aire de la surface / aire de son enveloppe convexe ([`gi`]).
    pub gi: Option<f64>,
    /// Énergie élastique Ue.
    pub energy: Option<f64>,
    pub n_contacts: Option<u64>,
}

fn cell(v: Option<f64>) -> String {
    v.filter(|x| x.is_finite()).map_or("n/a".into(), |x| format!("{x}"))
}

impl SequenceRow {
    fn tsv(&self) -> String {
        let mut s = String::new();
        let _ = write!(
            s,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}",
            self.step,
            cell(self.time),
            cell(self.ga_weeks),
            self.file,
            cell(self.gi),
            cell(self.energy),
            self.n_contacts.map_or("n/a".into(), |n| n.to_string())
        );
        s
    }
}

/// `sequence.tsv` en cours d'écriture : en-tête à la création, puis une ligne par surface, vidée sur
/// disque aussitôt (guioxide relit le fichier pendant le run).
pub struct SequenceWriter {
    path: PathBuf,
    file: std::fs::File,
}

impl SequenceWriter {
    /// Crée (ou remplace) `sequence.tsv` avec son en-tête.
    pub fn create(path: &Path) -> Result<Self> {
        let mut file = std::fs::File::create(path).map_err(|e| Error::io(path, e))?;
        writeln!(file, "{}", SEQUENCE_COLUMNS.join("\t")).map_err(|e| Error::io(path, e))?;
        file.sync_data().map_err(|e| Error::io(path, e))?;
        Ok(Self {
            path: path.to_path_buf(),
            file,
        })
    }

    pub fn append(&mut self, row: &SequenceRow) -> Result<()> {
        writeln!(self.file, "{}", row.tsv()).map_err(|e| Error::io(&self.path, e))?;
        self.file.sync_data().map_err(|e| Error::io(&self.path, e))
    }
}

/// Lit `sequence.tsv` (colonnes repérées par l'en-tête ; `step` et `file` obligatoires, `n/a` ou cellule
/// vide pour une valeur inconnue). Les lignes sont triées par pas.
pub fn lire_sequence(path: &Path) -> Result<Vec<SequenceRow>> {
    let text = std::fs::read_to_string(path).map_err(|e| Error::io(path, e))?;
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let header: Vec<&str> = lines
        .next()
        .ok_or_else(|| Error::format(path, "fichier vide"))?
        .split('\t')
        .collect();
    let col = |name: &str| header.iter().position(|h| h.trim() == name);
    let (Some(i_step), Some(i_file)) = (col("step"), col("file")) else {
        return Err(Error::format(path, "colonnes « step » et « file » obligatoires"));
    };
    let mut rows = Vec::new();
    for (k, line) in lines.enumerate() {
        let cells: Vec<&str> = line.split('\t').collect();
        let get = |i: Option<usize>| {
            i.and_then(|i| cells.get(i))
                .map(|c| c.trim())
                .filter(|c| !c.is_empty() && *c != "n/a")
        };
        let bad = |c: &str, v: &str| Error::format(path, format!("ligne {} : {c} = « {v} »", k + 2));
        let num =
            |c: &str| -> Result<Option<f64>> { get(col(c)).map(|v| v.parse().map_err(|_| bad(c, v))).transpose() };
        let step = get(Some(i_step)).ok_or_else(|| Error::format(path, format!("ligne {} : step vide", k + 2)))?;
        rows.push(SequenceRow {
            step: step.parse().map_err(|_| bad("step", step))?,
            time: num("time")?,
            ga_weeks: num("ga_weeks")?,
            file: get(Some(i_file))
                .ok_or_else(|| Error::format(path, format!("ligne {} : file vide", k + 2)))?
                .to_owned(),
            gi: num("gi")?,
            energy: num("energy")?,
            n_contacts: get(col("n_contacts"))
                .map(|v| v.parse().map_err(|_| bad("n_contacts", v)))
                .transpose()?,
        });
    }
    rows.sort_by_key(|r| r.step);
    Ok(rows)
}

// ------------------------------------------------------------------------------------------------
// Runs et noms de fichiers

/// Dossiers `run-<label>` d'un dossier `anat/` de gyroxide, triés : `(label, chemin)`.
pub fn lister_runs(dir: &Path) -> Vec<(String, PathBuf)> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut runs: Vec<(String, PathBuf)> = entries
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir())
        .filter_map(|p| Some((p.file_name()?.to_str()?.strip_prefix("run-")?.to_owned(), p.clone())))
        .collect();
    runs.sort();
    runs
}

/// Nom de la surface simulée au pas `step` : `<prefix>_hemi-<H>_desc-sim_step-<7 chiffres>_pial.surf.gii`,
/// `prefix` étant `sub-X[_ses-Y]`.
pub fn surface_name(prefix: &str, hemi: char, step: u64) -> String {
    format!("{prefix}_hemi-{hemi}_desc-sim_step-{step:07}_pial.surf.gii")
}

// ------------------------------------------------------------------------------------------------
// Indice de gyrification

fn area(points: &[[f64; 3]], faces: &[[u32; 3]]) -> f64 {
    faces
        .iter()
        .map(|f| {
            let [a, b, c] = f.map(|i| points[i as usize]);
            let (u, v) = (
                [b[0] - a[0], b[1] - a[1], b[2] - a[2]],
                [c[0] - a[0], c[1] - a[1], c[2] - a[2]],
            );
            let n = [
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            ];
            0.5 * (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt()
        })
        .sum()
}

/// Indice de gyrification : aire de la surface / aire de son enveloppe convexe (1 pour une surface
/// convexe, plus grand pour une surface plissée). `None` si les sommets sont coplanaires.
pub fn gi(vertices: &[[f64; 3]], faces: &[[u32; 3]]) -> Option<f64> {
    let hull = surfoxide_metrics::hull::convex_hull(vertices)?;
    let h = area(vertices, &hull);
    (h > 0.0).then(|| area(vertices, faces) / h)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("gyroxide-io-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn status_ecrit_atomiquement_puis_relu() {
        let d = tmp("status");
        let p = d.join("status.json");
        let s = Status {
            state: State::Running,
            step: Some(250),
            total_steps: Some(1000),
            started: Some("2026-10-21T10:00:00Z".into()),
            updated: Some(now_iso8601()),
            message: None,
        };
        ecrire_status(&p, &s).unwrap();
        assert!(
            !d.join(".status.json.tmp").exists(),
            "pas de fichier temporaire restant"
        );
        let r = lire_status(&p).unwrap();
        assert_eq!(r, s);
        assert_eq!(r.progress(), Some(0.25));
        std::fs::write(&p, r#"{"step": 3}"#).unwrap();
        assert!(lire_status(&p).unwrap_err().to_string().contains("state"));
        std::fs::write(&p, r#"{"state": "paused"}"#).unwrap();
        assert!(lire_status(&p).unwrap_err().to_string().contains("paused"));
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn sequence_ecrite_ligne_par_ligne_puis_relue() {
        let d = tmp("sequence");
        let p = d.join("sequence.tsv");
        let mut w = SequenceWriter::create(&p).unwrap();
        let row = |step, file: &str, ga| SequenceRow {
            step,
            time: Some(step as f64 * 1e-3),
            ga_weeks: ga,
            file: file.into(),
            gi: Some(1.25),
            energy: None,
            n_contacts: Some(3),
        };
        w.append(&row(500, "b.surf.gii", Some(21.5))).unwrap();
        w.append(&row(0, "a.surf.gii", None)).unwrap();
        // relu pendant l'écriture : les lignes sont déjà sur disque
        let s = lire_sequence(&p).unwrap();
        assert_eq!(s.iter().map(|r| r.step).collect::<Vec<_>>(), [0, 500], "triée par pas");
        assert_eq!(s[0], row(0, "a.surf.gii", None));
        assert_eq!(s[1].ga_weeks, Some(21.5));
        let text = std::fs::read_to_string(&p).unwrap();
        assert!(
            text.starts_with("step\ttime\tga_weeks\tfile\tgi\tenergy\tn_contacts\n"),
            "{text}"
        );
        assert!(text.contains("\tn/a\t"), "valeur inconnue écrite n/a");
        // fichier d'un autre outil : colonnes dans un autre ordre, colonnes manquantes
        std::fs::write(&p, "file\tstep\n x.gii \t7\n").unwrap();
        let s = lire_sequence(&p).unwrap();
        assert_eq!((s[0].step, s[0].file.as_str(), s[0].gi), (7, "x.gii", None));
        std::fs::write(&p, "step\tfile\nx\ta.gii\n").unwrap();
        assert!(lire_sequence(&p).unwrap_err().to_string().contains("ligne 2"));
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn runs_listes_et_noms_de_surface() {
        let d = tmp("runs");
        for n in ["run-b", "run-a", "autre"] {
            std::fs::create_dir_all(d.join(n)).unwrap();
        }
        std::fs::write(d.join("run-fichier"), b"").unwrap();
        let labels: Vec<String> = lister_runs(&d).into_iter().map(|(l, _)| l).collect();
        assert_eq!(labels, ["a", "b"]);
        assert!(lister_runs(&d.join("absent")).is_empty());
        assert_eq!(
            surface_name("sub-0avbc_ses-02", 'L', 8769),
            "sub-0avbc_ses-02_hemi-L_desc-sim_step-0008769_pial.surf.gii"
        );
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn gi_d_une_surface_convexe_et_d_une_surface_creusee() {
        // cube unité : convexe, gi = 1
        let v = vec![
            [0., 0., 0.],
            [1., 0., 0.],
            [1., 1., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
            [1., 0., 1.],
            [1., 1., 1.],
            [0., 1., 1.],
        ];
        let f: Vec<[u32; 3]> = vec![
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ];
        assert!((gi(&v, &f).unwrap() - 1.0).abs() < 1e-12);
        // face du dessus creusée en pyramide inversée de profondeur 0.5 : aire plus grande, même enveloppe
        let mut v2 = v.clone();
        v2.push([0.5, 0.5, 0.5]);
        let mut f2: Vec<[u32; 3]> = f[..2].to_vec();
        f2.extend_from_slice(&f[4..]);
        f2.extend_from_slice(&[[4, 5, 8], [5, 6, 8], [6, 7, 8], [7, 4, 8]]);
        let expected = (5.0 + 4.0 * 0.5 * (0.5f64 * 0.5 + 0.5 * 0.5).sqrt()) / 6.0;
        assert!((gi(&v2, &f2).unwrap() - expected).abs() < 1e-12);
        assert_eq!(gi(&v[..4], &f[..2]), None, "points coplanaires");
    }

    #[test]
    fn dates_iso_8601() {
        assert_eq!(iso8601(0), "1970-01-01T00:00:00Z");
        assert_eq!(iso8601(951_782_400), "2000-02-29T00:00:00Z");
        assert_eq!(iso8601(1_792_675_220), "2026-10-22T13:20:20Z");
    }
}
