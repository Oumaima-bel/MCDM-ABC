import streamlit as st, pandas as pd, numpy as np, plotly.express as px, plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from core import *

st.set_page_config(page_title="ABC Multi-Attributs", page_icon="🌸", layout="wide")
PINK = ["#ff2e93", "#ff8fc7", "#c2185b", "#ffc2e0", "#7b1048"]
CLS = {"A": "#d6197b", "B": "#ff7ab8", "C": "#ffc9e3"}
st.markdown("""<style>
.stApp{background:linear-gradient(135deg,#fff0f7 0%,#ffe0f0 55%,#ffd1e8 100%)}
[data-testid=stSidebar]{background:linear-gradient(180deg,#ff5fa8,#c2185b)}
[data-testid=stSidebar] *{color:#fff !important}
.hero{background:linear-gradient(120deg,#ff2e93,#ff8fc7);padding:2rem 2.5rem;border-radius:24px;color:#fff;
box-shadow:0 12px 35px rgba(255,46,147,.35);margin-bottom:1.2rem}
.hero h1{margin:0;font-size:2.4rem;color:#fff}.hero p{margin:.4rem 0 0;opacity:.95}
.kpi{background:#fff;border-radius:20px;padding:1.1rem;text-align:center;border:2px solid #ffc2e0;
box-shadow:0 6px 18px rgba(255,46,147,.15)}.kpi b{font-size:2rem;color:#d6197b}.kpi span{display:block;color:#7b1048}
.stTabs [data-baseweb=tab]{background:#fff;border-radius:14px 14px 0 0;padding:.6rem 1rem;color:#c2185b;font-weight:600}
.stTabs [aria-selected=true]{background:#ff2e93 !important;color:#fff !important}
h2,h3{color:#c2185b}
.box{background:#fff;border-left:6px solid #ff2e93;padding:1rem 1.3rem;border-radius:14px;margin:.6rem 0}
</style>""", unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>🌸 Classification ABC Multi-Attributs</h1><p>Réalisé par oumaima belhaddad et kaka fatima zahra</p></div>', unsafe_allow_html=True)

def kpi(col, v, l): col.markdown(f'<div class="kpi"><b>{v}</b><span>{l}</span></div>', unsafe_allow_html=True)
def style(fig): fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.7)", font_color="#7b1048"); return fig

# ---------------- Sidebar
st.sidebar.header("⚙️ Paramètres")
up = st.sidebar.file_uploader("Base CSV", type="csv")
df = pd.read_csv(up) if up else pd.read_csv("inventory_data.csv")
st.sidebar.subheader("Pondération des critères")
mode = st.sidebar.radio("Méthode", ["AHP (valeurs de l'article)", "AHP recalculé (vecteur propre)", "Manuelle"])
w_ahp, cr = ahp_weights()
if mode.startswith("AHP ("): W = dict(PAPER_W)
elif "recalculé" in mode: W = dict(zip(AHP_ORDER, w_ahp.round(4)))
else: W = {k: st.sidebar.slider(k, 0.0, 1.0, PAPER_W[k], 0.01) for k in AHP_ORDER}
sa = st.sidebar.slider("Seuil classe A (% articles)", 5, 40, 20); sb = st.sidebar.slider("Seuil classe B (% articles)", 10, 50, 30)
st.sidebar.caption(f"Classe C = {100-sa-sb} % restants")

E = encode(df); C = criteria(E); R, V, dp, dn, cc = topsis(C, W)
cl, rk = abc(cc, sa / 100, sb / 100)
out = df.copy(); out["TOPSIS_score"] = cc.round(5); out["Rang"] = rk; out["Classe"] = cl

tabs = st.tabs(["📋 1 Données", "🔢 2 Transformation", "🧩 3 Critères", "🎯 4-5 TOPSIS", "🅰️ 6-7 ABC", "🤖 8-9 Machine Learning", "🌫️ 10 Flou", "🆕 Nouvel article", "📘 Méthodologie"])

with tabs[0]:
    c = st.columns(4); kpi(c[0], len(df), "Articles"); kpi(c[1], df.shape[1], "Variables"); kpi(c[2], 4, "Quantitatives"); kpi(c[3], 4, "Qualitatives")
    nat = pd.DataFrame({"Variable": df.columns, "Type pandas": df.dtypes.astype(str).values,
        "Nature": ["Qualitative ordinale" if v in ("Risk","Demand fluctuation","Unit size") else "Qualitative nominale (binaire)" if v=="Consignment stock" else "Quantitative continue" if v in ("Average stock","Daily usage","Unit cost") else "Quantitative discrète" for v in df.columns],
        "Manquants": df.isna().sum().values})
    st.dataframe(nat, width='stretch', hide_index=True); st.dataframe(df.head(15), width='stretch')
    st.dataframe(df.describe().T.round(2), width='stretch')

with tabs[1]:
    st.markdown('<div class="box"><b>Pourquoi transformer ?</b> Les méthodes MCDM calculent des sommes, distances et moyennes : elles exigent des échelles numériques comparables. La conversion (Table 1) encode l\'ordre et l\'importance relative des modalités ; la normalisation évite qu\'un critère à grande échelle (stock) écrase un critère à petite échelle (coût).</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame([(a, k, v) for a, m in T1.items() for k, v in m.items()], columns=["Attribut", "Modalité", "Score normalisé"]), hide_index=True)
    st.caption("Variables quantitatives : normalisées par x / max(x) ∈ [0,1].")
    st.dataframe(E.head(15).round(3), width='stretch')

with tabs[2]:
    st.markdown('<div class="box"><b>Rôle des critères agrégés :</b> ils réduisent 8 attributs à 5 critères décisionnels (Criticité, Coût, Approvisionnement, Demande, Taille), ce qui simplifie les comparaisons par paires AHP et reflète la logique métier.<br><code>Critère = Σ poids_additif × score_attribut</code></div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame([(k, a, w) for k, p in T2.items() for a, w in p.items()], columns=["Critère global", "Attribut", "Poids additif"]), hide_index=True)
    st.dataframe(C.head(15).round(4), width='stretch')
    st.plotly_chart(style(px.box(C.melt(), x="variable", y="value", color="variable", color_discrete_sequence=PINK)), width='stretch')

with tabs[3]:
    st.subheader("Poids des critères")
    wd = pd.Series(W) / pd.Series(W).sum()
    c1, c2 = st.columns([1, 1])
    c1.plotly_chart(style(px.pie(values=wd.values, names=wd.index, hole=.5, color_discrete_sequence=PINK)), width='stretch')
    c2.markdown(f"**Matrice AHP de comparaison par paires (échelle de Saaty 1–9)** — ratio de cohérence **CR = {cr*100:.2f} %** (< 10 % ⇒ valide)")
    c2.dataframe(pd.DataFrame(AHP_MATRIX, index=AHP_ORDER, columns=AHP_ORDER).round(2))
    st.markdown("**Étapes TOPSIS** : (1) normalisation vectorielle rᵢⱼ = xᵢⱼ/√Σxᵢⱼ² → (2) pondération vᵢⱼ = wⱼ·rᵢⱼ → (3) solutions idéale A⁺ (max) et anti-idéale A⁻ (min) → (4) distances euclidiennes D⁺, D⁻ → (5) proximité **Cᵢ = D⁻/(D⁺+D⁻)**.")
    st.write("Matrice normalisée (extrait)"); st.dataframe(R.head(8).round(4), width='stretch')
    st.write("Matrice pondérée (extrait)"); st.dataframe(V.head(8).round(4), width='stretch')
    res = out.assign(D_plus=dp.round(4), D_moins=dn.round(4)).sort_values("Rang")
    st.write("Classement décroissant du coefficient de proximité"); st.dataframe(res, width='stretch')
    st.plotly_chart(style(px.histogram(out, x="TOPSIS_score", nbins=40, color_discrete_sequence=["#ff2e93"])), width='stretch')

with tabs[4]:
    cnt = out.Classe.value_counts().reindex(list("ABC")); c = st.columns(3)
    for i, k in enumerate("ABC"): kpi(c[i], f"{cnt[k]} ({cnt[k]/len(out):.0%})", f"Classe {k}")
    st.markdown(f'<div class="box"><b>Justification des seuils :</b> règle de Pareto adaptée à la littérature multi-attributs (Kartal & Cebi) : <b>A = top {sa} %</b> (articles vitaux, suivi strict), <b>B = {sb} %</b> suivants (contrôle modéré), <b>C = {100-sa-sb} %</b> restants (gestion simplifiée). Les seuils portent sur le <i>rang TOPSIS</i> (et non sur une valeur cumulée) car le score est un indice d\'importance multi-critères, pas une valeur monétaire.</div>', unsafe_allow_html=True)
    s = out.sort_values("Rang"); s["Cumul % score"] = s.TOPSIS_score.cumsum() / s.TOPSIS_score.sum() * 100; s["Cumul % articles"] = np.arange(1, len(s)+1) / len(s) * 100
    f = go.Figure(go.Scatter(x=s["Cumul % articles"], y=s["Cumul % score"], mode="lines", line=dict(color="#ff2e93", width=4), fill="tozeroy", fillcolor="rgba(255,46,147,.2)"))
    f.update_layout(xaxis_title="% articles", yaxis_title="% cumulé du score"); st.plotly_chart(style(f), width='stretch')
    st.plotly_chart(style(px.scatter(out, x="Rang", y="TOPSIS_score", color="Classe", color_discrete_map=CLS)), width='stretch')
    st.dataframe(out.sort_values("Rang"), width='stretch')
    st.download_button("💾 Télécharger la base avec la variable Classe", out.to_csv(index=False).encode(), "inventory_classified.csv", "text/csv")

MODELS = {"Naïve Bayes": GaussianNB(), "Random Forest": RandomForestClassifier(200, random_state=0),
  "SVM (RBF)": make_pipeline(StandardScaler(), SVC(C=10, probability=True, random_state=0)),
  "Réseau de neurones (MLP)": make_pipeline(StandardScaler(), MLPClassifier((32, 16), max_iter=2000, random_state=0)),
  "k-NN": make_pipeline(StandardScaler(), KNeighborsClassifier(7)),
  "Régression logistique": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))}

@st.cache_data(show_spinner=False)
def run_ml(X, y, seed):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=1/3, stratify=y, random_state=seed)
    cv = StratifiedKFold(10, shuffle=True, random_state=seed); rows, cms = [], {}
    for n, m in MODELS.items():
        m.fit(Xtr, ytr); p = m.predict(Xte); pr, rc, f1, _ = precision_recall_fscore_support(yte, p, average="weighted", zero_division=0)
        _, rA, fA, _ = precision_recall_fscore_support(yte, p, labels=["A"], zero_division=0)
        rows.append([n, accuracy_score(cross_val_predict(m, X, y, cv=cv), y), accuracy_score(yte, p), pr, rc, f1, rA[0], fA[0]])
        cms[n] = confusion_matrix(yte, p, labels=list("ABC"))
    return pd.DataFrame(rows, columns=["Modèle", "Accuracy CV-10", "Accuracy test (33 %)", "Précision", "Rappel", "F1", "Rappel classe A", "F1 classe A"]), cms

def pick_best(t):
    return t.sort_values(["F1", "Accuracy CV-10", "Rappel classe A"], ascending=False).Modèle.iloc[0]

with tabs[5]:
    st.markdown('<div class="box"><b>Protocole :</b> entrées = 8 attributs (scores normalisés), cible = Classe ABC issue de TOPSIS. Évaluation : validation croisée 10 plis + découpage stratifié 66/33. Le rappel de la classe A est suivi séparément : c\'est la classe la plus critique (Sun et al., 2007).<br><i>Remarque :</i> les classes étant une fonction déterministe des attributs, une accuracy élevée est attendue ; c\'est une mesure de la capacité des modèles à reproduire la logique MCDM.</div>', unsafe_allow_html=True)
    seed = st.number_input("Graine aléatoire", 0, 999, 42)
    if st.button("🚀 Entraîner les modèles"):
        with st.spinner("Entraînement…"): st.session_state.ml = run_ml(E, out.Classe, seed)
    if "ml" in st.session_state:
        t, cms = st.session_state.ml
        st.dataframe(t.style.format({c: "{:.3f}" for c in t.columns[1:]}).background_gradient(cmap="RdPu", subset=t.columns[1:]), width='stretch', hide_index=True)
        st.plotly_chart(style(px.bar(t.melt("Modèle", ["Accuracy CV-10", "Accuracy test (33 %)", "F1"]), x="Modèle", y="value", color="variable", barmode="group", color_discrete_sequence=PINK)), width='stretch')
        best = pick_best(t); st.success(f"🏆 Meilleur modèle (F1) : {best}")
        pick = st.selectbox("Matrice de confusion", list(cms), index=list(cms).index(best))
        st.plotly_chart(style(px.imshow(cms[pick], x=list("ABC"), y=list("ABC"), text_auto=True, color_continuous_scale="RdPu", labels=dict(x="Prédit", y="Réel"))), width='stretch')


with tabs[6]:
    st.markdown('<div class="box"><b>Classer un nouvel article.</b> Deux méthodes en parallèle : (1) <b>TOPSIS</b> avec les mêmes normes, poids et solutions idéale/anti-idéale que la base de référence, puis rang du score parmi les 700 articles ; (2) le <b>modèle ML retenu</b> (meilleur F1, départage par accuracy CV puis rappel de A), ré-entraîné sur toute la base.</div>', unsafe_allow_html=True)
    a, b, c3, d4 = st.columns(4)
    nr = {"Risk": a.selectbox("Risque", list(T1["Risk"]), 1), "Demand fluctuation": b.selectbox("Fluctuation demande", list(T1["Demand fluctuation"]), 1),
          "Consignment stock": c3.selectbox("Stock en consignation", list(T1["Consignment stock"]), 0), "Unit size": d4.selectbox("Taille unité", list(T1["Unit size"]), 1)}
    a, b, c3, d4 = st.columns(4)
    nr["Average stock"] = a.number_input("Stock moyen", 0.0, 10000.0, float(df["Average stock"].median()))
    nr["Daily usage"] = b.number_input("Utilisation quotidienne", 0.0, 1000.0, float(df["Daily usage"].median()))
    nr["Unit cost"] = c3.number_input("Coût unitaire", 0.0, 100000.0, float(df["Unit cost"].median()))
    nr["Lead time"] = d4.number_input("Délai de livraison (jours)", 0, 1000, int(df["Lead time"].median()))
    row = pd.DataFrame([nr])[list(df.columns)]
    # --- encodage avec les maxima de la base de référence
    en = pd.DataFrame([{**{k: T1[k][nr[k]] for k in T1}, **{q: nr[q] / df[q].max() for q in QUANT}}])[list(E.columns)]
    Cn = criteria(en); wv = pd.Series(W)[C.columns]; wv = wv / wv.sum()
    nrm = np.sqrt((C ** 2).sum()); Vn = Cn / nrm * wv
    ap, an = V.max(), V.min()
    dpn = float(np.sqrt(((Vn - ap) ** 2).sum(axis=1)).iloc[0]); dnn = float(np.sqrt(((Vn - an) ** 2).sum(axis=1)).iloc[0])
    ccn = dnn / (dpn + dnn); rkn = int((cc > ccn).sum()) + 1; frac = rkn / (len(cc) + 1)
    cls_t = "A" if frac <= sa / 100 else "B" if frac <= (sa + sb) / 100 else "C"
    k = st.columns(3); kpi(k[0], f"{ccn:.4f}", "Score TOPSIS"); kpi(k[1], f"{rkn} / {len(cc)+1}", "Rang"); kpi(k[2], cls_t, "Classe (TOPSIS)")
    if "ml" not in st.session_state:
        with st.spinner("Évaluation des modèles…"): st.session_state.ml = run_ml(E, out.Classe, 42)
    t, _ = st.session_state.ml; best = pick_best(t)
    mdl = MODELS[best].fit(E, out.Classe); pred = mdl.predict(en)[0]; pr = pd.Series(mdl.predict_proba(en)[0], index=mdl.classes_)
    st.success(f"🏆 Modèle retenu : **{best}** (F1 = {t.set_index('Modèle').loc[best,'F1']:.3f}, accuracy CV-10 = {t.set_index('Modèle').loc[best,'Accuracy CV-10']:.3f}) → classe prédite : **{pred}**")
    c1, c2 = st.columns(2)
    c1.plotly_chart(style(px.bar(x=pr.index, y=pr.values, color=pr.index, color_discrete_map=CLS, labels=dict(x="Classe", y="Probabilité"), title="Probabilités du modèle retenu")), width='stretch')
    allp = {n: MODELS[n].fit(E, out.Classe).predict(en)[0] for n in MODELS}
    c2.write("Prédiction de tous les modèles"); c2.dataframe(pd.DataFrame({"Modèle": allp.keys(), "Classe prédite": allp.values()}), hide_index=True, width='stretch')
    if pred != cls_t: st.warning(f"Désaccord TOPSIS ({cls_t}) / ML ({pred}) : article proche d'une frontière de classe. La classe TOPSIS fait référence ; le ML l'approxime.")
    else: st.info(f"TOPSIS et ML concordent : classe **{pred}**.")
    st.caption("Remarque : TOPSIS est relatif à la base. Ajouter un article ne recalcule pas les normes ni les idéaux ; pour l'intégrer, ajoutez-le au CSV.")

with tabs[8]:
    st.markdown(f"""### 📘 Méthode de pondération
**1. Poids additifs (Table 2)** : agrégation des attributs en critères — Criticité = 0.78·Risque + 0.22·Fluctuation ; Demande = 0.71·Usage + 0.29·Stock ; Approvisionnement = 0.75·Délai + 0.25·Consignation.

**2. Poids des critères par AHP** (Saaty, 1980) : matrice de comparaison par paires 5×5 (échelle 1–9) ; les poids sont le **vecteur propre principal normalisé** ; cohérence vérifiée par CR = CI/RI avec CI = (λmax−n)/(n−1) et RI(5)=1.12.

| Critère | Poids article | Poids recalculé |
|---|---|---|
""" + "\n".join(f"| {k} | {PAPER_W[k]:.2f} | {w_ahp[i]:.3f} |" for i, k in enumerate(AHP_ORDER)) + f"""

CR recalculé = **{cr*100:.2f} %** (< 10 %, jugements cohérents).

**3. Pipeline** : CSV → scores normalisés → 5 critères → pondération AHP → TOPSIS → classement → ABC (20/30/50) → ML

**Référence** : Kartal, Oztekin, Gunasekaran & Cebi (2016), *Computers & Industrial Engineering*.""")