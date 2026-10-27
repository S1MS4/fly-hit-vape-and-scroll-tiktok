"""Turn raw MaleCNS v1.0 tables into simulator-ready files in data/build/.

Outputs
  graph.npz    CSR connectome (presynaptic rows), signed synapse counts, body ids
  groups.json  named neuron groups (indices) used as sensory inputs / motor readouts
  retina.npz   photoreceptor index, eye (0=L, 1=R), u, v in [0, 1]
  somata.bin   float32 [x, y, z, neuron_index, category] per neuron with a soma
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "build"

ANNOT = RAW / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
NT = RAW / "body-neurotransmitters-male-cns-v1.0.feather"
WEIGHTS = RAW / "connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather"

# Transmitter -> synapse sign. ACh excites; GABA and glutamate inhibit (as in
# Shiu et al.). Photoreceptors are histaminergic and hyperpolarise lamina cells,
# but a spiking model can't carry signal through inhibition of silent cells, so
# photoreceptor output is treated as an excitatory relay; other histamine
# synapses inhibit. Dopamine, serotonin and octopamine act mostly through slow
# metabotropic receptors, so they get no fast synaptic weight. Unknown -> weak +.
SIGN = {"acetylcholine": 1.0, "gaba": -1.0, "glutamate": -1.0, "histamine": -1.0,
        "dopamine": 0.0, "serotonin": 0.0, "octopamine": 0.0}
UNKNOWN_SCALE = 0.5

# Fruity-ester-responsive olfactory glomeruli (Or42b/DM1, Or22a/DM2, Or59b/DM4,
# Or43b/VM2, Or92a/VA2): stand-in for a sweet e-liquid smell.
ELIQUID_ORNS = ["ORN_DM1", "ORN_DM2", "ORN_DM4", "ORN_VM2", "ORN_VA2"]

# Coarse categories for colouring the 3D brain.
CATEGORIES = ["other", "optic", "central", "sensory", "descending", "motor",
              "mushroom_body", "dopamine", "vnc"]


def category_of(row):
    t, sc, cl = str(row.type), str(row.superclass), str(row["class"])
    if cl == "DAN":
        return "dopamine"
    if cl in ("Kenyon_Cell", "MBON"):
        return "mushroom_body"
    if sc.startswith("ol_") or sc.startswith("visual"):
        return "optic"
    if "sensory" in sc:
        return "sensory"
    if sc.startswith("descending"):
        return "descending"
    if "motor" in sc:
        return "motor"
    if sc.startswith("cb_"):
        return "central"
    if sc.startswith("vnc") or sc.startswith("ascending"):
        return "vnc"
    return "other"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("loading tables ...")
    ann = pd.read_feather(ANNOT).drop_duplicates("bodyId").set_index("bodyId")
    nt = pd.read_feather(NT).drop_duplicates("body").set_index("body")
    w = pd.read_feather(WEIGHTS, columns=["body_pre", "body_post", "weight"])
    w = w[w.body_pre != w.body_post]

    ids = np.union1d(w.body_pre.values, w.body_post.values)
    n = len(ids)
    pre = np.searchsorted(ids, w.body_pre.values)
    post = np.searchsorted(ids, w.body_post.values)
    print(f"{n} neurons, {len(w)} connections")

    # Per-neuron transmitter: consensus, then cell-type prediction, then own prediction.
    t = nt.reindex(ids)
    tx = t.consensus_nt.where(t.consensus_nt.notna() & (t.consensus_nt != "unclear"),
                              t.celltype_predicted_nt)
    tx = tx.where(tx.notna() & (tx != "unclear"), t.predicted_nt).fillna("unclear")
    sign = tx.map(SIGN).fillna(UNKNOWN_SCALE).to_numpy(np.float32)
    is_photoreceptor = ann.type.reindex(ids).fillna("").str.match(r"^R[1-8]").to_numpy()
    sign[is_photoreceptor] = 1.0
    is_ach = (tx == "acetylcholine").to_numpy()

    counts = np.clip(w.weight.to_numpy(), 0, 32767).astype(np.float32) * sign[pre]
    keep = counts != 0
    pre, post, counts = pre[keep], post[keep], counts[keep]
    w = w[keep]
    order = np.argsort(pre, kind="stable")
    indptr = np.concatenate([[0], np.cumsum(np.bincount(pre, minlength=n))]).astype(np.int64)
    np.savez_compressed(
        OUT / "graph.npz", indptr=indptr, indices=post[order].astype(np.int32),
        counts=np.rint(counts[order]).astype(np.int16), is_ach=is_ach, body_ids=ids)
    print("wrote graph.npz")

    a = ann.reindex(ids)
    typ = a.type.fillna("")
    side = a.instance.fillna("").str.extract(r"_([LR])$")[0].fillna("")
    idx_where = lambda mask: np.flatnonzero(np.asarray(mask)).tolist()

    # Descending neurons with the most direct synapses onto front-leg (T1) motor neurons.
    t1_mn = a.superclass.eq("vnc_motor") & a.somaNeuromere.eq("T1")
    dn = a.superclass.eq("descending_neuron")
    edge_mask = dn.to_numpy()[pre] & t1_mn.to_numpy()[post]
    leg_dn_types = (pd.Series(w.weight.to_numpy()[edge_mask], index=typ.to_numpy()[pre[edge_mask]])
                    .groupby(level=0).sum().sort_values(ascending=False).head(8).index.tolist())

    groups = {
        "eliquid_orns": idx_where(typ.isin(ELIQUID_ORNS)),
        "photoreceptors": idx_where(typ.str.match(r"^R[1-8]")),
        "pam": idx_where(typ.str.startswith("PAM")),
        "ppl1": idx_where(typ.str.startswith("PPL1")),
        "mbon": idx_where(typ.str.startswith("MBON")),
        "kc": idx_where(a["class"].eq("Kenyon_Cell")),
        "steer_left": idx_where(typ.isin(["DNa01", "DNa02"]) & side.eq("L")),
        "steer_right": idx_where(typ.isin(["DNa01", "DNa02"]) & side.eq("R")),
        "forward": idx_where(typ.eq("DNp09")),
        "backward": idx_where(typ.eq("MDN")),
        "escape": idx_where(typ.eq("DNp01")),
        "proboscis": idx_where(typ.eq("MN9")),
        "front_leg": idx_where(typ.isin(leg_dn_types)),
    }
    meta = {"n": int(n), "edges": int(len(w)), "front_leg_dn_types": leg_dn_types,
            "eliquid_orn_types": ELIQUID_ORNS, "categories": CATEGORIES,
            "groups": groups}
    (OUT / "groups.json").write_text(json.dumps(meta))
    print("groups:", {k: len(v) for k, v in groups.items()})
    print("front-leg DN types:", leg_dn_types)

    build_retina(a, typ, side, w, pre, post, ids)
    build_somata(a)


def build_retina(a, typ, side, w, pre, post, ids):
    """Place each photoreceptor at the optic-lobe column of its strongest columnar partner."""
    hex1, hex2 = a.assignedOlHex1.to_numpy(), a.assignedOlHex2.to_numpy()
    is_pr = typ.str.match(r"^R[1-8]").to_numpy()
    mask = is_pr[pre] & ~np.isnan(hex1[post])
    e = pd.DataFrame({"pre": pre[mask], "post": post[mask], "w": w.weight.to_numpy()[mask]})
    best = e.sort_values("w", ascending=False).drop_duplicates("pre")

    h1, h2 = hex1[best.post.to_numpy()], hex2[best.post.to_numpy()]
    x, y = h1 - h2, h1 + h2  # axial hex coordinates -> planar
    eye = (side.to_numpy()[best.pre.to_numpy()] == "R").astype(np.int8)
    u, v = np.zeros_like(x), np.zeros_like(y)
    for e_id in (0, 1):
        m = eye == e_id
        u[m] = (x[m] - x[m].min()) / max(np.ptp(x[m]), 1)
        v[m] = (y[m] - y[m].min()) / max(np.ptp(y[m]), 1)
    np.savez_compressed(OUT / "retina.npz", idx=best.pre.to_numpy().astype(np.int64),
                        eye=eye, u=u.astype(np.float32), v=v.astype(np.float32))
    print(f"wrote retina.npz ({len(best)} photoreceptors mapped)")


def build_somata(a):
    has = a.somaLocation.notna().to_numpy()
    idx = np.flatnonzero(has)
    xyz = np.stack(a.somaLocation.to_numpy()[has]).astype(np.float64)
    xyz -= (xyz.min(0) + xyz.max(0)) / 2
    xyz /= np.abs(xyz).max()
    cat = a.iloc[idx].apply(category_of, axis=1).map({c: i for i, c in enumerate(CATEGORIES)})
    rec = np.column_stack([xyz, idx, cat.to_numpy()]).astype(np.float32)
    rec.tofile(OUT / "somata.bin")
    print(f"wrote somata.bin ({len(idx)} somata)")


if __name__ == "__main__":
    main()
