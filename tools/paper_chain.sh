#!/bin/sh
# Fitted-growth chain for the ISBI paper, from SG 21: the inverse problem solved every week from SG 22
# to 36 with one protocol (law + lobar core, multiplicative thickening, --keep-best; equilibrium
# settings of the lbm chain up to SG 31, of the lbmf chain after), chain material mu_g/mu_w = 2, K = 5 (the realistic material
# does not chain past SG 29; the fitted growth does not depend on it, tools/material_chain.sh
# re-equilibrates the fits afterwards).
#  1. inputs from the measured-growth chain (out/remodel_a0): tools/make_full.py, data/full_paper;
#     SG 21 = rest state; law maps of SG 21-25 added to data/full/law (atlas depth, as make_full)
#  2. SG 22 fit from the measured-growth state, law maps of SG 21 (predictive), out/paper/fit22_lbm
#  3. chain SG 22 -> 36 (tools/full_chain.sh, LAW=1), out/paper/fit<SG>_lbm
#   [FROM=<SG>] tools/paper_chain.sh [first step, default 1]
set -e
cd "$(dirname "$0")/.."
BIN=${BIN:-target/mu/adjoint_week_v8}
D=data/full_paper
L=data/full/law
LB=$(ls $L/lb40/basis_lb*_s100.npy | paste -sd,)
SOLVER="--newton-cg 600 --force-tol 5e-5 --ls-tol 3e-3"
S=${1:-1}
mkdir -p out/paper
if [ "$S" -le 1 ]; then
    python3 tools/make_full.py --out $D --weeks $(seq 22 36) --full "out/remodel_a0/w{w}/eq/x_released.npy"
    python3 - <<'EOF'
import json, shutil, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "tools")
from make_full import depth_bases
D, L = Path("data/full_paper"), Path("data/full/law")
meta = json.loads(Path("data/meshes/w21_left_610k_s.json").read_text())
ns, mm = meta["n_surface"], meta["maxd_mm"]
ck = D / "ckpt_w21"                                 # rest state (start of the measured-growth chain)
shutil.rmtree(ck, ignore_errors=True)
shutil.copytree(D / "ckpt_w22", ck)
x = np.load("out/remodel_a0/w22/eq/x_0.npy")
np.save(ck / "x.npy", x)
(ck / "state.json").write_text(json.dumps({"step": 0, "t": 0.0}))
p = Path("data/adjoint_prep_s/w22")
faces, cortex = np.load(p / "faces.npy").astype(int), np.load(p / "cortex.npy")[:ns] > 0.5
for w in range(21, 26):
    y = x[:ns] if w == 21 else np.load(Path("data/adjoint_prep_s") / f"w{w}" / "target.npy")[:ns]
    d30, fine = depth_bases(y, faces, cortex, mm)
    for name, v in (("d30", d30), ("d3o", fine)):
        f = L / f"basis_{name}_w{w}_s100.npy"
        if not f.exists():
            np.save(f, v / 100)
print("SG 21 rest state and law maps of SG 21-25 written")
EOF
    cp configs/mu_test/mu2_full.toml configs/mu_test/mu2_full_paper.toml
fi
if [ "$S" -le 2 ]; then
    # shellcheck disable=SC2086
    $BIN configs/mu_test/mu2_full_paper.toml data/meshes/w21_left_610k_s.mesh $D/ckpt_w22 $D/prep/w22 \
        out/paper/fit22_lbm --absolute --multiscale 10:30,0:30 --cumulative --theta0-table --relax 300 \
        --relax0 3000 --newton 40 --min-growth 0 --max-step 0.05 --cg 600 --w-vertex 1 --w-normal 1 --w-lap 1 \
        --w-area 0.5 --growth-band 0.5 --fit-fields c,k --field-band 50 \
        --field-law $L/basis_d30_w21_s100.npy,$L/basis_d3o_w21_s100.npy,$L/basis_one_s100.npy \
        --field-law-k $LB --thick-mult --keep-best $SOLVER > out/paper/fit22_lbm.log 2>&1
    grep -E "^start mm k=  0|^final mm k=  0|panicked|inverted" out/paper/fit22_lbm.log
fi
# Equilibrium settings of the earlier lobar chain: default up to SG 31, the late-week ones after.
# FROM=<SG>: resume the chain from the finished fit of that week.
W=${FROM:-22}
P=""
[ "$W" -gt 22 ] && P=lbm$W
if [ "$W" -lt 31 ]; then
    # shellcheck disable=SC2086
    LAW=1 DATA=$D OUTD=out/paper PREV0=$P BIN=$BIN tools/full_chain.sh lbm out/paper/fit${W}_lbm $W 31 \
        --field-law-k $LB --thick-mult --keep-best
    W=31; P=lbm31
fi
# shellcheck disable=SC2086
LAW=1 DATA=$D OUTD=out/paper PREV0=$P RELAX=300 BIN=$BIN tools/full_chain.sh lbm out/paper/fit${W}_lbm $W 36 \
    --field-law-k $LB --thick-mult --keep-best $SOLVER
