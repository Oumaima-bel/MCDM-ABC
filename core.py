import numpy as np, pandas as pd

# ---- Table 1 : conversion qualitatif -> score normalisé
T1 = {
 "Risk": {"High": 0.47, "Normal": 0.35, "Low": 0.18},
 "Demand fluctuation": {"Increasing": 0.36, "Stable": 0.28, "Unknown": 0.20, "Decreasing": 0.16, "Ending": 0.0},
 "Consignment stock": {"No": 0.80, "Yes": 0.20},
 "Unit size": {"Large": 0.53, "Medium": 0.31, "Small": 0.13},
}
# ---- Table 2 : poids additifs des critères agrégés
T2 = {
 "Criticality": {"Risk": 0.78, "Demand fluctuation": 0.22},
 "Demand": {"Daily usage": 0.71, "Average stock": 0.29},
 "Supply": {"Lead time": 0.75, "Consignment stock": 0.25},
}
QUANT = ["Average stock", "Daily usage", "Unit cost", "Lead time"]
QUAL = list(T1)
# ---- Poids AHP (Kartal et al., 2016, Table 6)
AHP_ORDER = ["Criticality", "Unit cost", "Supply", "Demand", "Unit size"]
AHP_MATRIX = np.array([
 [1.00, 2.47, 2.76, 2.90, 0.89],
 [0.41, 1.00, 0.69, 0.69, 0.58],
 [0.36, 1.44, 1.00, 1.44, 1.00],
 [0.34, 1.44, 0.69, 1.00, 1.00],
 [1.13, 1.71, 1.00, 1.00, 1.00]])
PAPER_W = dict(zip(AHP_ORDER, [0.33, 0.12, 0.18, 0.15, 0.22]))

def ahp_weights(M=AHP_MATRIX):
    vals, vecs = np.linalg.eig(M)
    k = np.argmax(vals.real); w = np.abs(vecs[:, k].real); w /= w.sum()
    n = len(M); ci = (vals[k].real - n) / (n - 1); cr = ci / 1.12  # RI(5)=1.12
    return w, cr

def encode(df):
    """Étape 2 : scores numériques normalisés. Quantitatifs : x / max (comme SAW du papier)."""
    e = pd.DataFrame(index=df.index)
    for c, m in T1.items(): e[c] = df[c].map(m)
    for c in QUANT: e[c] = df[c] / df[c].max()
    return e[["Risk","Demand fluctuation","Average stock","Daily usage","Unit cost","Lead time","Consignment stock","Unit size"]]

def criteria(e):
    """Étape 3 : critères agrégés (somme pondérée avec poids additifs)."""
    c = pd.DataFrame(index=e.index)
    for k, parts in T2.items(): c[k] = sum(e[a] * w for a, w in parts.items())
    c["Unit cost"] = e["Unit cost"]; c["Unit size"] = e["Unit size"]
    return c[AHP_ORDER]

def topsis(C, w):
    """Étape 4-5 : matrice normalisée (vectorielle), pondérée, solutions idéales, proximité."""
    w = np.array([w[k] for k in C.columns]); w = w / w.sum()
    R = C / np.sqrt((C ** 2).sum()); V = R * w            # tous les critères = bénéfice
    ap, an = V.max(), V.min()
    dp = np.sqrt(((V - ap) ** 2).sum(axis=1)); dn = np.sqrt(((V - an) ** 2).sum(axis=1))
    return R, V, dp, dn, dn / (dp + dn)

def abc(score, a=0.20, b=0.30):
    rk = score.rank(ascending=False, method="first"); n = len(score)
    return pd.Series(np.where(rk <= a * n, "A", np.where(rk <= (a + b) * n, "B", "C")), index=score.index), rk.astype(int)
