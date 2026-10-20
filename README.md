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
    --mesh ~/Sync/Code/BrainGrowth/data/sphere5.mesh --vtk-every 5000 --out out/s5
cargo run --release --bin gyroxide -- preset tallinen2016   # paramètres en TOML
```

Options utiles de `run` : `--backend ndarray` (CPU f64), `--steps N`,
`--trace-every N` (traces NPY), `--vtk-every N` (surfaces ParaView),
`--no-contact`, `--no-midplane`, `--uniform-growth`, `--config params.toml`.

Documentation : [modèle](docs/model.md), [architecture](docs/architecture.md),
[validation](docs/validation.md).

## Périmètre et lien avec surfoxide

gyroxide génère lui-même les maillages tétraédriques sur lesquels il simule. Les surfaces
corticales réelles viennent de [surfoxide](https://github.com/rousseau/surfoxide), qui ne traite
que des surfaces : surfaces blanche et piale en GIfTI (coordonnées RAS en mm), fermées, 2-variétés,
orientées vers l'extérieur, de genre 0, sans auto-intersection, la blanche strictement dans la
piale. La tétraédrisation de ces surfaces est faite ici, dans gyroxide. En retour, gyroxide
renvoie sa surface simulée (surface triangulaire, pas de maillage volumique), que surfoxide mesure
avec le même code que les surfaces réelles.
