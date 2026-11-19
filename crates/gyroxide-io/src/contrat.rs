//! Noms de fichiers BIDS de gyroxide : source unique du code d'écriture (`gyroxide mesh`, `gyroxide run --bids`) et
//! de `gyroxide --describe` (schéma `oxide-describe/1`, `docs/donnees.md` de guioxide, section 5).
//!
//! Un nom de fichier est `sub-X[_ses-Y]` suivi d'un suffixe ; `{hemi}` vaut `L` ou `R`, `{desc}` l'entité `desc-`
//! du maillage, `{step}` le pas (7 chiffres). Les motifs de chemin complets se composent avec [`ANAT_PREFIX`].

/// Début des motifs de chemin d'une sortie par session, relatif à la racine du pipeline.
pub const ANAT_PREFIX: &str = "sub-{sub}[/ses-{ses}]/anat/sub-{sub}[_ses-{ses}]";

/// Suffixe de la surface blanche de `surfx run`, entrée de `gyroxide mesh`.
pub const ENTREE_BLANCHE: &str = "_hemi-{hemi}_white.surf.gii";
/// Suffixe de la surface piale de `surfx run`, entrée de `gyroxide mesh`.
pub const ENTREE_PIALE: &str = "_hemi-{hemi}_pial.surf.gii";
/// Suffixe du masque du cortex de `surfx run`, entrée de `gyroxide mesh`.
pub const ENTREE_CORTEX: &str = "_hemi-{hemi}_cortex.shape.gii";
/// Suffixe du maillage tétraédrique d'un hémisphère de session (`gyroxide mesh`, lu par `gyroxide run --bids`).
pub const SUFFIXE_MAILLAGE: &str = "_hemi-{hemi}_desc-{desc}_tet.vtu";
/// Début du nom du dossier d'un run : `run-<label>`.
pub const PREFIXE_DOSSIER_RUN: &str = "run-";
/// Paramètres complets du run, dans le dossier du run.
pub const FICHIER_CONFIG: &str = "config.toml";
/// État d'avancement du run (`status.json`, section 4 du contrat).
pub const FICHIER_STATUS: &str = "status.json";
/// Surfaces écrites par le run (`sequence.tsv`, section 4 du contrat).
pub const FICHIER_SEQUENCE: &str = "sequence.tsv";
/// Suffixe d'une surface simulée, dans le dossier du run ; `{step}` : 7 chiffres.
pub const SURFACE_SIMULEE: &str = "_hemi-{hemi}_desc-sim_step-{step}_pial.surf.gii";

/// `suffixe` pour l'hémisphère `hemi` (`L` ou `R`).
pub fn avec_hemi(suffixe: &str, hemi: char) -> String {
    suffixe.replace("{hemi}", &hemi.to_string())
}

/// Suffixe du maillage de session d'un hémisphère et d'une méthode (`desc`).
pub fn suffixe_maillage(hemi: char, desc: &str) -> String {
    avec_hemi(SUFFIXE_MAILLAGE, hemi).replace("{desc}", desc)
}

/// Suffixe de la surface simulée au pas `step`.
pub fn suffixe_surface_simulee(hemi: char, step: u64) -> String {
    avec_hemi(SURFACE_SIMULEE, hemi).replace("{step}", &format!("{step:07}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Les noms d'avant la table n'ont pas changé.
    #[test]
    fn noms_inchanges() {
        assert_eq!(suffixe_maillage('L', "tetgen"), "_hemi-L_desc-tetgen_tet.vtu");
        assert_eq!(
            suffixe_surface_simulee('R', 20),
            "_hemi-R_desc-sim_step-0000020_pial.surf.gii"
        );
        assert_eq!(avec_hemi(ENTREE_PIALE, 'L'), "_hemi-L_pial.surf.gii");
        assert_eq!(avec_hemi(ENTREE_BLANCHE, 'L'), "_hemi-L_white.surf.gii");
        assert_eq!(avec_hemi(ENTREE_CORTEX, 'R'), "_hemi-R_cortex.shape.gii");
        assert_eq!(
            (FICHIER_CONFIG, FICHIER_STATUS, FICHIER_SEQUENCE),
            ("config.toml", "status.json", "sequence.tsv")
        );
        assert_eq!(PREFIXE_DOSSIER_RUN, "run-");
    }
}
