# gyroxide

Simulation du plissement cortical par croissance tangentielle (modèle de
Tallinen et al., *Nature Physics* 2016), en Rust sur [Burn](https://burn.dev),
avec tout l'état de la simulation résident sur le GPU.

Version 1 : modèle de Tallinen à l'identique (néo-Hookéen avec J̄ nodal,
branche des éléments inversés, croissance tangentielle totale, dynamique
explicite amortie, auto-contact point–triangle, plan médian), implémenté en
opérations de tenseurs Burn. Deux préréglages : `tallinen2016` (code C de
l'article) et `braingrowth`.

```sh
cargo test                                   # tests en f64 sur CPU
cargo run --release --bin gyroxide -- run --ball 8          # petite boule sur GPU (wgpu)
cargo run --release --bin gyroxide -- run --preset braingrowth \
    --mesh ~/Sync/Code/BrainGrowth/data/sphere5.mesh --surface-every 5000 --out out/s5
cargo run --release --bin gyroxide -- preset tallinen2016   # paramètres en TOML
```

Options utiles de `run` : `--backend ndarray` (CPU f64), `--steps N`,
`--surface-every N` (surfaces GIfTI), `--hemi L|R`, `--trace-every N` (traces NPY),
`--vtk-every N` (surfaces VTK de débogage, ParaView), `--no-contact`, `--no-midplane`,
`--uniform-growth`, `--config params.toml`.

## Surfaces écrites

Le calcul se fait sur le maillage normalisé (demi-taille du cerveau ~ 1). Les surfaces sont
écrites en GIfTI (`*.surf.gii`, via surfoxide-io) après la transformée inverse de la
normalisation : coordonnées RAS en mm, dans le repère de l'image d'où vient l'état initial.
Seuls les nœuds de surface sont écrits, faces orientées vers l'extérieur,
`AnatomicalStructurePrimary` = `CortexLeft` ou `CortexRight` selon `--hemi`.

Avec `normalization = "none"` (maillages déjà normalisés : maillages de Tallinen, maillage
de l'atlas `w21_left_610k`), il n'y a pas de transformée à inverser : les coordonnées
écrites sont celles du maillage, en unités normalisées, **pas en mm**. Les surfaces VTK de
`--vtk-every` restent dans le repère normalisé du calcul.

Documentation : [modèle](docs/model.md), [architecture](docs/architecture.md),
[validation](docs/validation.md).
