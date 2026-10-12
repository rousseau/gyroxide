# Architecture

## Crates

| Crate | Rôle |
|---|---|
| `gyroxide-mesh` | `TetMesh`, lecture/écriture Netgen `.mesh`, normalisation (et sa transformée inverse), surface, normales de référence, incidence ELL, plus proche nœud de surface (grille), boule voxelisée de test, sorties VTK |
| `gyroxide-model` | `Params` (TOML via serde), préréglages, lois de croissance, masque gr |
| `gyroxide-sim` | `Simulation<B: Backend>` : état sur le device, pas de temps, élasticité, contact ; trait `TraceSink` |
| `gyroxide-validate` | E/S NPY, écrivain de traces, comparaison à un oracle |
| `gyroxide-cli` | binaire `gyroxide` : `run`, `preset`, `ball`, `compare` |

## État sur le device

Structure de tableaux : chaque champ vectoriel est trois tenseurs 1-D (`V3`),
chaque champ matriciel neuf (`M3`), dans `algebra.rs`. C'est la disposition
qu'aura un noyau CubeCL (un thread par tétraèdre, tout en registres).

- Par nœud : x, v [N].
- Par tétraèdre : indices [T]×4, arêtes de référence A_r0 (M3), normale de
  croissance n, d moyen, gr moyen, `growth_prev` (= g_m a_t du pas précédent).
- Incidence nœud → (sommet local k, tétraèdre t), slot `k·T + t`, au format ELL
  [N, largeur max], complétée par un slot nul.
- Contact : paires (point, triangle) et leur propre table ELL nœud → (rôle, paire).

## Un pas de temps

1. (tous les `check_every` pas) déplacement max de la surface → 1 scalaire lu ;
   si besoin, positions relues, paires reconstruites sur CPU et téléversées.
2. Forces de contact par paire, gather vers les nœuds.
3. Élasticité : g_m, G, F, volumes nodaux (gather), J̄, deux branches calculées
   et sélectionnées par masque, forces locales [4T], gather vers les nœuds.
4. Plan médian.
5. Amortissement et Euler semi-implicite.

Aucune lecture CPU entre deux sorties hormis le scalaire de l'étape 1.
L'assemblage par gather évite les additions atomiques flottantes (absentes de
WGSL) et rend le résultat déterministe.

## Niveau « référence tenseurs » (v1) et suite

La v1 est écrite en opérations Burn élément par élément ; le backend Fusion de
wgpu en regroupe une partie, mais le pas reste à quelques centaines
d'opérations. Mesures sur un Mac (Metal), sphere5 (6 761 nœuds, 35 497
tétraèdres) : 5,2 ms/pas sur GPU f32, 51 ms/pas sur CPU f64. La branche des
éléments inversés est calculée pour tous les tétraèdres puis masquée : coût
doublé, acceptable pour un oracle interne.

Étape suivante (phase 4 de la synthèse) : quatre noyaux CubeCL (volumes
nodaux, forces tétraédriques, contact, assemblage + intégration), validés contre
cette référence, puis la reconstruction des listes de contact sur GPU.

## Backends

- `--backend wgpu` (défaut) : Metal / Vulkan / DX12, f32, Fusion.
- `--backend ndarray` : CPU, f64, sert d'oracle interne et de base de comparaison
  avec le code C.
- Feature `cuda` prévue pour la station Linux (non testée).
