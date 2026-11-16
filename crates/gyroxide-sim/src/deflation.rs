//! Déflation des équilibres déjà connus pour le Newton de l'équilibre (Farrell, Birkisson et Funke 2015).
//!
//! Le résidu F(x) est multiplié par M(x) = sum_i d(x, x_i)^-p + sigma, où x_i sont les équilibres connus et
//! d la distance moyenne quadratique des nœuds, en unités de longueur (`scale` : unité de position en mm). Une
//! racine de F déjà connue n'est plus une racine de M F (M y tend vers l'infini) : le Newton repart ailleurs
//! ou échoue, sans la retrouver. Les autres racines de F restent des racines de M F.
//!
//! Le pas de Newton sur M F se déduit de celui sur F (sans refaire de résolution) par Sherman-Morrison :
//! `delta = tau delta0`, `tau = 1 / (1 - grad M . delta0 / M)`.

/// Équilibres à écarter et réglages de l'opérateur de déflation.
#[derive(Debug, Clone)]
pub struct Deflation {
    /// Positions des nœuds de chaque équilibre connu.
    pub known: Vec<Vec<[f64; 3]>>,
    /// Exposant p (2 par défaut : le pas de Newton est renversé au voisinage de la racine).
    pub p: f64,
    /// Décalage sigma (valeur de M loin des équilibres connus).
    pub sigma: f64,
    /// Unité de position en mm.
    pub scale: f64,
}

impl Deflation {
    pub fn is_empty(&self) -> bool {
        self.known.is_empty()
    }

    /// Résidu de x par rapport à l'équilibre i, après le meilleur déplacement rigide (translation et rotation) de
    /// celui-ci sur x : le modèle n'a aucune condition aux limites, deux états qui diffèrent d'un mouvement rigide
    /// sont le même équilibre. Le résidu est de somme nulle.
    pub fn residual(&self, x: &[[f64; 3]], i: usize) -> Vec<[f64; 3]> {
        let k = &self.known[i];
        assert_eq!(
            x.len(),
            k.len(),
            "équilibre déflaté et état n'ont pas le même nombre de nœuds"
        );
        let n = x.len() as f64;
        let cen = |v: &[[f64; 3]]| [0, 1, 2].map(|d| v.iter().map(|a| a[d]).sum::<f64>() / n);
        let (cx, ck) = (cen(x), cen(k));
        // H = sum (k - ck)(x - cx)^T ; la rotation R qui envoie k sur x est le facteur orthogonal de H^T... on
        // cherche R maximisant sum (x - cx) . R (k - ck) : R = polaire de sum (x - cx)(k - ck)^T.
        let mut h = [[0.0; 3]; 3];
        for (a, b) in x.iter().zip(k) {
            for r in 0..3 {
                for c in 0..3 {
                    h[r][c] += (a[r] - cx[r]) * (b[c] - ck[c]);
                }
            }
        }
        let r = polar(h);
        x.iter()
            .zip(k)
            .map(|(a, b)| {
                let kb = [0, 1, 2].map(|d| b[d] - ck[d]);
                [0, 1, 2].map(|d| (a[d] - cx[d]) - (0..3).map(|e| r[d][e] * kb[e]).sum::<f64>())
            })
            .collect()
    }

    /// Distance de x à l'équilibre i : racine de la moyenne des carrés des déplacements des nœuds, en mm, au
    /// mouvement rigide près.
    pub fn dist(&self, x: &[[f64; 3]], i: usize) -> f64 {
        let r = self.residual(x, i);
        let s: f64 = r.iter().map(|v| v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sum();
        self.scale * (s / x.len() as f64).sqrt()
    }

    /// M(x) ; 1 sans équilibre connu (la déflation vide ne change rien).
    pub fn factor(&self, x: &[[f64; 3]]) -> f64 {
        if self.is_empty() {
            return 1.0;
        }
        self.sigma
            + (0..self.known.len())
                .map(|i| self.dist(x, i).max(1e-300).powf(-self.p))
                .sum::<f64>()
    }

    /// Facteur tau du pas déflaté pour le pas de Newton `delta0` sur F (convention du simulateur :
    /// `delta0 = K^-1 F`, K = -dF/dx, le pas est `x + alpha tau delta0`). Borné à |tau| <= `max`.
    pub fn tau(&self, x: &[[f64; 3]], delta0: &[[f64; 3]], max: f64) -> f64 {
        if self.is_empty() {
            return 1.0;
        }
        let n = x.len() as f64;
        let m = self.factor(x);
        let mut dm = 0.0;
        for i in 0..self.known.len() {
            // Le gradient de la distance au mouvement rigide près est le résidu aligné (théorème de l'enveloppe) :
            // d(dist) . delta0 = scale^2 sum r . delta0 / (n dist).
            let r = self.residual(x, i);
            let d = (self.scale * (r.iter().map(|v| v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sum::<f64>() / n).sqrt())
                .max(1e-300);
            let dot: f64 = r
                .iter()
                .zip(delta0)
                .map(|(a, c)| a[0] * c[0] + a[1] * c[1] + a[2] * c[2])
                .sum();
            dm += -self.p * d.powf(-self.p - 1.0) * self.scale * self.scale * dot / (n * d);
        }
        let den = 1.0 - dm / m;
        if den.abs() < 1.0 / max {
            max.copysign(den)
        } else {
            (1.0 / den).clamp(-max, max)
        }
    }
}

/// Facteur orthogonal (rotation) de la décomposition polaire d'une matrice 3x3 inversible, par l'itération de
/// Newton X <- (X + X^-T) / 2. Une matrice singulière (états dégénérés) rend l'identité.
fn polar(h: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let inv_t = |m: &[[f64; 3]; 3]| -> Option<[[f64; 3]; 3]> {
        let c = |r: usize, s: usize| {
            let (r1, r2, c1, c2) = ((r + 1) % 3, (r + 2) % 3, (s + 1) % 3, (s + 2) % 3);
            m[r1][c1] * m[r2][c2] - m[r1][c2] * m[r2][c1]
        };
        let det = m[0][0] * c(0, 0) + m[0][1] * c(0, 1) + m[0][2] * c(0, 2);
        if det.abs() < 1e-300 {
            return None;
        }
        // (M^-T)[r][s] = cofacteur(r, s) / det
        Some([0, 1, 2].map(|r| [0, 1, 2].map(|s| c(r, s) / det)))
    };
    let mut x = h;
    for _ in 0..50 {
        let Some(t) = inv_t(&x) else {
            return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        };
        let next = [0, 1, 2].map(|r| [0, 1, 2].map(|s| 0.5 * (x[r][s] + t[r][s])));
        let delta: f64 = (0..3)
            .flat_map(|r| (0..3).map(move |s| (r, s)))
            .map(|(r, s)| (next[r][s] - x[r][s]).abs())
            .sum();
        x = next;
        if delta < 1e-14 {
            break;
        }
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Cas jouet : un repère de quatre nœuds non coplanaires dont le premier a pour abscisse x ; force
    /// F(x) = x - x^3 sur cette abscisse (équilibres -1, 0 et 1 ; 0 est instable) ; K = -dF/dx = 3 x^2 - 1 ;
    /// delta0 = F / K. La distance est au mouvement rigide près (les trois autres nœuds en absorbent une partie).
    fn state(x: f64) -> Vec<[f64; 3]> {
        vec![[x, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0], [1.0, 1.0, 3.0]]
    }

    fn force(x: f64) -> f64 {
        x - x * x * x
    }

    /// Newton déflaté pur (pas `tau delta0`, sans recherche linéaire : c'est la forme de l'article).
    fn newton(x0: f64, defl: &Deflation, steps: usize) -> (f64, bool) {
        let mut x = x0;
        for _ in 0..steps {
            if force(x).abs() < 1e-12 {
                return (x, true);
            }
            let step = force(x) / (3.0 * x * x - 1.0);
            let d0 = [[step, 0.0, 0.0], [0.0; 3], [0.0; 3], [0.0; 3]];
            let tau = defl.tau(&state(x), &d0, 10.0);
            x += tau * step;
        }
        (x, force(x).abs() < 1e-12)
    }

    fn deflation(known: &[f64]) -> Deflation {
        Deflation {
            known: known.iter().map(|&k| state(k)).collect(),
            p: 2.0,
            sigma: 1.0,
            scale: 1.0,
        }
    }

    #[test]
    fn sans_deflation_newton_retrouve_l_equilibre_le_plus_proche() {
        let (x, ok) = newton(0.9, &deflation(&[]), 60);
        assert!(ok && (x - 1.0).abs() < 1e-9, "x = {x}");
    }

    #[test]
    fn un_equilibre_connu_n_est_plus_retrouve() {
        for x0 in [0.9, 1.2, 2.0] {
            let (x, ok) = newton(x0, &deflation(&[1.0]), 200);
            assert!(ok, "départ {x0} : pas de convergence");
            assert!(
                (x - 1.0).abs() > 0.5,
                "départ {x0} : retombé sur l'équilibre connu ({x})"
            );
            assert!(force(x).abs() < 1e-10);
        }
    }

    #[test]
    fn deflation_successive_jusqu_a_epuisement() {
        // Deux équilibres écartés : il reste -1 ou 0 ; les trois : plus aucune racine, échec propre (pas de panique).
        let (x, ok) = newton(0.9, &deflation(&[1.0, 0.0]), 300);
        assert!(ok && (x + 1.0).abs() < 1e-9, "x = {x}");
        let (_, ok) = newton(0.9, &deflation(&[1.0, 0.0, -1.0]), 300);
        assert!(!ok, "les trois racines sont écartées : le Newton ne peut pas converger");
    }

    #[test]
    fn distance_en_mm_et_facteur() {
        let mut d = deflation(&[0.0]);
        assert!(d.dist(&state(0.0), 0) < 1e-12);
        let d1 = d.dist(&state(0.5), 0);
        assert!(d1 > 0.0);
        d.scale = 2.0;
        d.sigma = 0.5;
        assert!((d.dist(&state(0.5), 0) - 2.0 * d1).abs() < 1e-12);
        assert!((d.factor(&state(0.5)) - (0.5 + (2.0 * d1).powi(-2))).abs() < 1e-9);
        assert_eq!(deflation(&[]).factor(&state(0.5)), 1.0);
    }

    #[test]
    fn un_mouvement_rigide_n_est_pas_une_distance() {
        // Quatre nœuds non coplanaires, tournés de 0,3 rad autour de z et translatés : distance nulle ; déformés : non.
        let k = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]];
        let (c, s) = (0.3f64.cos(), 0.3f64.sin());
        let moved: Vec<[f64; 3]> = k
            .iter()
            .map(|v| [c * v[0] - s * v[1] + 5.0, s * v[0] + c * v[1] - 2.0, v[2] + 0.7])
            .collect();
        let d = Deflation {
            known: vec![k.clone()],
            p: 2.0,
            sigma: 1.0,
            scale: 1.0,
        };
        assert!(d.dist(&moved, 0) < 1e-12, "dist = {}", d.dist(&moved, 0));
        let mut bent = moved.clone();
        bent[3][2] += 0.4;
        assert!(d.dist(&bent, 0) > 0.05);
        // le résidu est de somme nulle
        let r = d.residual(&bent, 0);
        for e in 0..3 {
            assert!(r.iter().map(|v| v[e]).sum::<f64>().abs() < 1e-12);
        }
    }

    #[test]
    fn tau_vaut_moins_un_pres_de_la_racine_pour_p_2() {
        // Près de x* = 1, delta0 ~ -(x - 1) : le pas déflaté (p = 2) est renversé, tau → -1.
        let d = deflation(&[1.0]);
        let e = 1e-4;
        let d0 = [[-e, 0.0, 0.0], [0.0; 3], [0.0; 3], [0.0; 3]];
        let tau = d.tau(&state(1.0 + e), &d0, 10.0);
        assert!((tau + 1.0).abs() < 1e-3, "tau = {tau}");
    }

    #[test]
    fn tau_ne_depend_pas_de_l_unite_de_longueur() {
        // Sans décalage (sigma = 0) M est homogène en la distance : tau est sans dimension.
        let (e, d0) = (0.05, [[-0.02, 0.0, 0.0], [0.0; 3], [0.0; 3], [0.0; 3]]);
        let mut d = deflation(&[1.0]);
        d.sigma = 0.0;
        let t1 = d.tau(&state(1.0 + e), &d0, 100.0);
        d.scale = 3.0;
        let t3 = d.tau(&state(1.0 + e), &d0, 100.0);
        assert!((t1 - t3).abs() < 1e-9 * t1.abs().max(1.0), "{t1} contre {t3}");
    }

    #[test]
    fn tau_est_borne_des_deux_cotes() {
        // Pas de Newton qui annule 1 - grad M . delta0 / M : tau diverge, il est borné à +-max, du signe de 1 - ...
        let d = deflation(&[0.0]);
        // À distance e de la racine 0, delta0 = -e/2 donne grad M . delta0 / M = p/2 = 1 : dénominateur nul.
        let e = 0.3;
        let d0 = [[-e / 2.0, 0.0, 0.0], [0.0; 3], [0.0; 3], [0.0; 3]];
        let tau = d.tau(&state(e), &d0, 7.0);
        assert!(tau.abs() <= 7.0 + 1e-12);
        // Pas loin d'un dénominateur nul : borné sans changer de signe
        let tau_pos = d.tau(
            &state(e),
            &[[-e / 2.0 * 0.5, 0.0, 0.0], [0.0; 3], [0.0; 3], [0.0; 3]],
            1.5,
        );
        assert!(tau_pos > 0.0 && tau_pos <= 1.5 + 1e-12, "{tau_pos}");
        let tau_neg = d.tau(&state(e), &[[-e * 2.0, 0.0, 0.0], [0.0; 3], [0.0; 3], [0.0; 3]], 0.5);
        assert!((-0.5 - 1e-12..0.0).contains(&tau_neg), "{tau_neg}");
    }

    #[test]
    fn sans_equilibre_connu_le_facteur_et_tau_valent_un() {
        let mut d = deflation(&[]);
        d.sigma = 0.5;
        assert_eq!(d.factor(&state(0.3)), 1.0);
        assert_eq!(
            d.tau(&state(0.3), &[[1.0, 0.0, 0.0], [0.0; 3], [0.0; 3], [0.0; 3]], 10.0),
            1.0
        );
    }

    #[test]
    fn tau_sature_a_plus_ou_moins_max_de_part_et_d_autre_du_denominateur_nul() {
        let d = deflation(&[0.0]);
        let e = 0.3;
        let taus: Vec<f64> = (0..400)
            .map(|k| {
                let c = 0.5 + 1.5 * k as f64 / 400.0;
                d.tau(&state(e), &[[-e * c, 0.0, 0.0], [0.0; 3], [0.0; 3], [0.0; 3]], 7.0)
            })
            .collect();
        assert!(taus.contains(&7.0), "pas de saturation à +max");
        assert!(taus.iter().any(|&t| t == -7.0), "pas de saturation à -max");
        assert!(taus.iter().all(|&t| t.abs() <= 7.0));
    }
}
