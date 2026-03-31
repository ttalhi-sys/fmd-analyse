import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.colors import LinearSegmentedColormap
from scipy.special import gamma as gamma_func
from scipy.optimize import minimize_scalar
import io
import base64
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Analyse FMD — Ligne Rim Glue #22",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cacher les icônes toolbar (GitHub, crayon, Deploy, etc.)
st.markdown("""
<style>
    [data-testid="stToolbar"] {display: none !important;}
    [data-testid="stDecoration"] {display: none !important;}
    #MainMenu {display: none !important;}
    header {visibility: hidden !important;}
</style>
""", unsafe_allow_html=True)

# Cacher les icônes toolbar (GitHub, crayon, Deploy, etc.)
st.markdown("""
<style>
    [data-testid="stToolbar"] {display: none !important;}
    [data-testid="stDecoration"] {display: none !important;}
    #MainMenu {display: none !important;}
    footer {display: none !important;}
    header {visibility: hidden !important;}
</style>
""", unsafe_allow_html=True)


MODES = {
    'tampon': {
        'label': 'MODE Tampon',
        'description': 'Ligne avec tampon (C5 actif)',
        'ss_actifs': ['A1','A2','A3','A4','B1','B2','B4','C1','C3','C4','C5','C6','C7'],
        'ss_exclus': ['A5','B3','C2'],
    },
    'rimglue': {
        'label': 'MODE RimGlue',
        'description': 'Ligne avec RimGlue (C2 actif, A5 ajouté)',
        'ss_actifs': ['A1','A2','A3','A4','A5','B1','B2','B3','B4','C1','C2','C6','C7'],
        'ss_exclus': ['C3','C4','C5'],
    },
    'mixte': {
        'label': 'MODE Tampon + RimGlue',
        'description': 'Ligne complète (C2 et C5 actifs)',
        'ss_actifs': ['A1','A2','A3','A4','A5','B1','B2','B3','B4','C1','C2','C3','C4','C5','C6','C7'],
        'ss_exclus': [],
    },
}

SS_MAP = {
    ('module a', '«denester»'): 'A3', ('module a', 'denester'): 'A3',
    ('module a', 'poussoir'): 'A3',
    ('module a', 'chaîne'): 'A4', ('module a', 'chaine'): 'A4',
    ('module a', 'guide chaîne'): 'A4', ('module a', 'guide chaine'): 'A4',
    ('module a', 'convoyeur blanc'): 'A1',
    ("module a", "convoyeur d'entree"): 'A1', ('module a', 'convoyeur d entree'): 'A1',
    ('module a', 'convoyeur bleu'): 'A2',
    ('module a', 'laser'): 'A5',
    ('module b', 'carroussel'): 'B2', ('module b', 'carrousel'): 'B2',
    ('module b', 'pick & place - prélèvement'): 'B1', ('module b', 'pick & place - prelevement'): 'B1',
    ('module b', 'pick & place - dépôt'): 'B4', ('module b', 'pick & place - depot'): 'B4',
    ('module b', 'système de colle- général'): 'B3', ('module b', 'systeme de colle- general'): 'B3',
    ('module b', 'système de colle'): 'B3', ('module b', 'systeme de colle'): 'B3',
    ('module b', 'colle rim'): 'B3',
    ('module c', 'caméra uv'): 'C2', ('module c', 'camera uv'): 'C2',
    ('module c', 'chaîne'): 'C1', ('module c', 'chaine'): 'C1',
    ('module c', 'guide chaîne'): 'C1', ('module c', 'guide chaine'): 'C1',
    ('module c', 'convoyeur d entree'): 'C1',
    ('module c', 'convoyeur de sortie'): 'C7',
    ('module c', 'couteau'): 'C4',
    ('module c', 'tamponneur'): 'C4', ('module c', 'tamponneuse'): 'C4',
    ('module c', 'empileuse'): 'C6',
    ('module c', 'poussoir'): 'C6',
    ('module c', 'système table de sortie'): 'C6', ('module c', 'systeme table de sortie'): 'C6',
    ('module c', 'colle tampon'): 'C3',
    ('système de colle', 'colle rim'): 'B3', ('systeme de colle', 'colle rim'): 'B3',
    ('système de colle', 'colle tampon'): 'C3', ('systeme de colle', 'colle tampon'): 'C3',
}

SOUS_SYSTEMES = [
    ('A1', 'A', 'Convoyeur entrée blanc'), ('A2', 'A', 'Entrée magasins'),
    ('A3', 'A', 'Magasins de stockage'), ('A4', 'A', 'Transport barquettes Mod A'),
    ('A5', 'A', 'Marquage laser'), ('B1', 'B', 'Bras prélèvement (Pick & Place)'),
    ('B2', 'B', 'Table rotative (Carrousel)'), ('B3', 'B', 'Station collage (Rim Glue)'),
    ('B4', 'B', 'Bras dépôt (Pick & Place)'), ('C1', 'C', 'Transport barquettes Mod C'),
    ('C2', 'C', 'Unité vérification colle'), ('C3', 'C', 'Fusil à colle tampon'),
    ('C4', 'C', 'Station tampon'), ('C5', 'C', 'Système caméra'),
    ('C6', 'C', 'Magasin de dépôt'), ('C7', 'C', 'Sortie magasin'),
]

SS_NOMS = {ss: nom for ss, _, nom in SOUS_SYSTEMES}
MOD_COLOR = {'A': '#1565C0', 'B': '#2E7D32', 'C': '#E65100'}
MOD_LABEL = {
    'A': 'MODULE A — Préparation & Marquage',
    'B': 'MODULE B — Collage',
    'C': 'MODULE C — Vérification & Sortie',
}

def _fc(d):
    return '#43A047' if d >= 95 else '#FFA000' if d >= 85 else '#E53935' if d > 0 else '#9E9E9E'

def identifier_ss(r2, r3):
    v2 = str(r2).strip().lower() if pd.notna(r2) else ''
    v3 = str(r3).strip().lower() if pd.notna(r3) else ''
    if (v2, v3) in SS_MAP:
        return SS_MAP[(v2, v3)]
    for (k2, k3), ss_id in SS_MAP.items():
        if k2 in v2 and k3 in v3:
            return ss_id
    return None
  # ============================================================================
# DÉTECTION FORMAT + EXTRACTION MES + NETTOYAGE
# ============================================================================

def detecter_format(df):
    for i in range(min(5, len(df))):
        row_vals = [str(v).strip() for v in df.iloc[i]]
        if "Ordre*" in row_vals and df.shape[1] <= 10:
            return "B", i
    return "A", None

def extraire_format_A(raw_bytes):
    all_sheets = pd.read_excel(io.BytesIO(raw_bytes), sheet_name=None, header=None)
    sheet_name = next(
        (s for s in all_sheets if "explorateur" not in s.lower()),
        list(all_sheets.keys())[0]
    )
    df = all_sheets[sheet_name]
    temps_fonctionnement = 0.0
    for i, row in df.iterrows():
        if "sommaire" in str(row.iloc[0]).lower():
            for j in range(i+1, min(i+8, len(df))):
                row_j = [str(v).strip() for v in df.iloc[j]]
                if any("en fonction" in v.lower() for v in row_j):
                    idx_f = next(k for k, v in enumerate(row_j) if "en fonction" in v.lower())
                    for jj in range(j+1, min(j+4, len(df))):
                        row_jj = [str(v).strip() for v in df.iloc[jj]]
                        if "(min)" in row_jj:
                            col_f = next((k for k in range(idx_f, len(row_jj)) if row_jj[k] == "(min)"), None)
                            for jjj in range(jj+1, min(jj+4, len(df))):
                                r = df.iloc[jjj]
                                v0 = str(r.iloc[0]).strip()
                                if v0 and v0.lower() not in ("nan", "", "total :", "département", "departement"):
                                    if col_f is not None:
                                        val = pd.to_numeric(r.iloc[col_f], errors="coerce")
                                        if pd.notna(val) and val > 0:
                                            temps_fonctionnement = float(val)
                                    break
                            break
                    break
            break
    header_row = None
    COL_ORDRE = COL_DATE = COL_MIN_ARRET = None
    COL_RAISON1 = COL_RAISON2 = COL_RAISON3 = COL_COMMENTS = None
    for i, row in df.iterrows():
        rv = [str(v).strip() for v in row]
        if "Ordre*" in rv:
            header_row = i
            above = [str(v).strip() for v in df.iloc[i-1]] if i > 0 else []
            idx_a = next((k for k, v in enumerate(above) if "en arrêt" in v.lower() or "en arret" in v.lower()), None)
            for k, v in enumerate(rv):
                if v == "(min)" and idx_a is not None and k >= idx_a:
                    COL_MIN_ARRET = k; break
            for k, v in enumerate(rv):
                vl = v.lower()
                if v == "Ordre*": COL_ORDRE = k
                elif "date début" in vl or "date debut" in vl: COL_DATE = k
                elif v == "Raison 1": COL_RAISON1 = k
                elif v == "Raison 2": COL_RAISON2 = k
                elif v == "Raison 3": COL_RAISON3 = k
                elif "commentaire" in vl: COL_COMMENTS = k
            break
    if header_row is None:
        raise ValueError("En-tête 'Ordre*' introuvable dans le fichier MES brut.")
    bris_rows = []
    total_mec = total_elec = total_oper = 0.0
    for i in range(header_row + 1, len(df)):
        row = df.iloc[i]
        r1 = str(row.iloc[COL_RAISON1]).strip() if COL_RAISON1 is not None and pd.notna(row.iloc[COL_RAISON1]) else ""
        mn = pd.to_numeric(row.iloc[COL_MIN_ARRET], errors="coerce") if COL_MIN_ARRET is not None else 0.0
        mn = float(mn) if pd.notna(mn) else 0.0
        if r1 == "Bris - Mécanique": total_mec += mn
        elif r1 == "Bris - Électrique": total_elec += mn
        elif r1 == "Opérationnel": total_oper += mn
        if r1 in ("Bris - Mécanique", "Bris - Électrique"):
            dv = row.iloc[COL_DATE] if COL_DATE is not None else None
            ds = dv.strftime("%Y-%m-%d %H:%M:%S") if hasattr(dv, "strftime") else (str(dv).strip() if dv is not None and pd.notna(dv) else "")
            bris_rows.append({
                "Ordre*": str(row.iloc[COL_ORDRE]).strip() if COL_ORDRE is not None and pd.notna(row.iloc[COL_ORDRE]) else "",
                "Date Début": ds, "(min) En Arrêt": round(mn, 2), "Raison 1": r1,
                "Raison 2": str(row.iloc[COL_RAISON2]).strip() if COL_RAISON2 is not None and pd.notna(row.iloc[COL_RAISON2]) else "",
                "Raison 3": str(row.iloc[COL_RAISON3]).strip() if COL_RAISON3 is not None and pd.notna(row.iloc[COL_RAISON3]) else "",
                "Commentaires": str(row.iloc[COL_COMMENTS]).strip() if COL_COMMENTS is not None and pd.notna(row.iloc[COL_COMMENTS]) else "",
            })
    temps_prog_min = temps_fonctionnement + total_mec + total_elec + total_oper
    df_clean = pd.DataFrame(bris_rows)
    if len(df_clean) > 0:
        df_clean["Date Début"] = pd.to_datetime(df_clean["Date Début"], errors="coerce")
    info = {
        'format': 'A', 'temps_fonc_min': temps_fonctionnement,
        'total_mec_min': total_mec, 'total_elec_min': total_elec,
        'total_oper_min': total_oper, 'temps_prog_h': temps_prog_min / 60.0,
        'nb_bris': len(bris_rows), 'sheet_name': sheet_name,
    }
    return df_clean, info

def nettoyer_fichier(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    col_map = {}
    for col in df.columns:
        cl = col.lower()
        if 'ordre' in cl: col_map['ordre'] = col
        elif 'début' in cl or 'debut' in cl: col_map['debut'] = col
        elif 'fin' in cl: col_map['fin'] = col
        elif 'arrêt' in cl or 'arret' in cl or ('min' in cl and 'comment' not in cl): col_map['duree'] = col
        elif 'raison 1' in cl or 'raison1' in cl: col_map['r1'] = col
        elif 'raison 2' in cl or 'raison2' in cl: col_map['r2'] = col
        elif 'raison 3' in cl or 'raison3' in cl: col_map['r3'] = col
        elif 'comment' in cl: col_map['comm'] = col
    manq = [c for c in ['debut', 'duree', 'r2', 'r3'] if c not in col_map]
    if manq:
        raise ValueError(f"Colonnes non détectées : {manq}\nColonnes : {list(df.columns)}")
    df.dropna(how='all', inplace=True)
    df[col_map['debut']] = pd.to_datetime(df[col_map['debut']], errors='coerce')
    if 'fin' in col_map:
        df[col_map['fin']] = pd.to_datetime(df[col_map['fin']], errors='coerce')
    df[col_map['duree']] = pd.to_numeric(df[col_map['duree']], errors='coerce')
    if 'fin' in col_map:
        mask = df[col_map['duree']].isna()
        df.loc[mask, col_map['duree']] = ((df.loc[mask, col_map['fin']] - df.loc[mask, col_map['debut']]).dt.total_seconds() / 60)
    df = df[df[col_map['duree']] > 0].copy()
    df = df.dropna(subset=[col_map['debut']]).copy()
    df.reset_index(drop=True, inplace=True)
    df['ss_id'] = df.apply(lambda row: identifier_ss(row[col_map['r2']], row[col_map['r3']]), axis=1)
    return df, col_map

def nettoyer_format_A(df_extracted):
    df = df_extracted.copy()
    col_map = {'ordre': 'Ordre*', 'debut': 'Date Début', 'duree': '(min) En Arrêt',
               'r1': 'Raison 1', 'r2': 'Raison 2', 'r3': 'Raison 3', 'comm': 'Commentaires'}
    df[col_map['debut']] = pd.to_datetime(df[col_map['debut']], errors='coerce')
    df[col_map['duree']] = pd.to_numeric(df[col_map['duree']], errors='coerce')
    df = df[df[col_map['duree']] > 0].copy()
    df = df.dropna(subset=[col_map['debut']]).copy()
    df.reset_index(drop=True, inplace=True)
    df['ss_id'] = df.apply(lambda row: identifier_ss(row[col_map['r2']], row[col_map['r3']]), axis=1)
    return df, col_map

# ============================================================================
# WEIBULL + CALCULS FMD
# ============================================================================

def estimer_weibull(durees_h):
    data = np.array([d for d in durees_h if d > 0])
    n = len(data)
    if n < 2:
        return 1.0, (np.mean(data) if n == 1 else 1.0)
    def neg_ll(beta):
        if beta <= 0.01: return 1e12
        sum_xb = np.sum(data ** beta)
        eta_mle = (sum_xb / n) ** (1.0 / beta)
        if eta_mle <= 0: return 1e12
        return -(n * np.log(beta) - n * beta * np.log(eta_mle) + (beta - 1) * np.sum(np.log(data)) - sum_xb / (eta_mle ** beta))
    try:
        result = minimize_scalar(neg_ll, bounds=(0.1, 10.0), method='bounded')
        beta = result.x; eta = (np.sum(data ** beta) / n) ** (1.0 / beta)
    except: beta, eta = 1.0, np.mean(data)
    return round(beta, 4), round(eta, 4)

def fiabilite_weibull(t, beta, eta):
    return np.exp(-(t / eta) ** beta) * 100 if eta > 0 else 0.0

def maintenabilite_weibull(t, beta, eta):
    return (1 - np.exp(-(t / eta) ** beta)) * 100 if eta > 0 else 0.0

def mtbf_weibull(beta, eta):
    return eta * gamma_func(1 + 1.0 / beta) if beta > 0 else eta

def calculer_fmd_ss(durees_min, periode_h, t_fiab=100, t_maint=10,
                    distribution='exponentielle', tbf_list_h=None, ttr_list_h=None):
    durees = [d for d in durees_min if d > 0]
    wb_f = {'beta': 1.0, 'eta': 0.0}; wb_m = {'beta': 1.0, 'eta': 0.0}
    if not durees:
        mtbf_val = round(periode_h, 3); wb_f = {'beta': 1.0, 'eta': mtbf_val}
        return dict(nb=0, t_arret_h=0.0, t_fonct_h=round(periode_h, 2), MTBF=mtbf_val, MTTR=0.0, D=100.0,
                    F=round(np.exp(-t_fiab / max(mtbf_val, 0.01)) * 100, 2), M=0.0,
                    distribution=distribution, weibull_fiab=wb_f, weibull_maint=wb_m)
    nb = len(durees); t_arret_h = sum(durees) / 60; t_fonct_h = max(periode_h - t_arret_h, 0.0)
    MTTR_exp = t_arret_h / nb; MTBF_exp = t_fonct_h / nb if nb > 0 else periode_h
    if distribution == 'weibull' and tbf_list_h is not None and ttr_list_h is not None:
        tbf_data = [t for t in tbf_list_h if t > 0]
        if len(tbf_data) >= 2:
            beta_f, eta_f = estimer_weibull(tbf_data); MTBF = mtbf_weibull(beta_f, eta_f); F = fiabilite_weibull(t_fiab, beta_f, eta_f)
        else: beta_f, eta_f = 1.0, MTBF_exp; MTBF = MTBF_exp; F = np.exp(-t_fiab / max(MTBF, 0.01)) * 100
        ttr_data = [t for t in ttr_list_h if t > 0]
        if len(ttr_data) >= 2:
            beta_m, eta_m = estimer_weibull(ttr_data); MTTR = mtbf_weibull(beta_m, eta_m); M = maintenabilite_weibull(t_maint, beta_m, eta_m)
        else: beta_m, eta_m = 1.0, MTTR_exp; MTTR = MTTR_exp; M = (1 - np.exp(-t_maint / max(MTTR, 0.01))) * 100
        wb_f = {'beta': beta_f, 'eta': eta_f}; wb_m = {'beta': beta_m, 'eta': eta_m}
    else:
        MTBF, MTTR = MTBF_exp, MTTR_exp
        F = np.exp(-t_fiab / max(MTBF, 0.01)) * 100; M = (1 - np.exp(-t_maint / max(MTTR, 0.01))) * 100
        wb_f = {'beta': 1.0, 'eta': round(MTBF, 4)}; wb_m = {'beta': 1.0, 'eta': round(MTTR, 4)}
    D = (MTBF / (MTBF + MTTR)) * 100 if (MTBF + MTTR) > 0 else 100.0
    return dict(nb=nb, t_arret_h=round(t_arret_h, 3), t_fonct_h=round(t_fonct_h, 2),
                MTBF=round(MTBF, 3), MTTR=round(MTTR, 3), D=round(D, 2), F=round(F, 2), M=round(M, 2),
                distribution=distribution, weibull_fiab=wb_f, weibull_maint=wb_m)

def calculer_fmd_module_topologie(resultats_ss, mode, module, t_fiab=100, t_maint=10):
    BYPASS_SS = {'C2', 'C5'}
    rs = [r for r in resultats_ss if r['nb'] > 0]
    if not rs: return dict(nb=0, t_arret_h=0.0, MTBF=0.0, MTTR=0.0, D=100.0, F=0.0, M=0.0)
    nb_tot = sum(r['nb'] for r in rs); t_arret = sum(r['t_arret_h'] for r in rs)
    rs_crit = [r for r in rs if r.get('id') not in BYPASS_SS]
    if not rs_crit: return dict(nb=nb_tot, t_arret_h=round(t_arret, 3), MTBF=0.0, MTTR=0.0, D=100.0, F=100.0, M=0.0)
    sl = sum(1.0/r['MTBF'] for r in rs_crit if r['MTBF']>0); MTBF = 1.0/sl if sl>0 else 0.0
    slm = sum((1.0/r['MTBF'])*r['MTTR'] for r in rs_crit if r['MTBF']>0); MTTR = slm/sl if sl>0 else 0.0
    D = 1.0
    for r in resultats_ss:
        if r.get('id') not in BYPASS_SS: D *= (r['D']/100.0)
    D *= 100.0
    F = 1.0
    for r in resultats_ss:
        if r.get('id') not in BYPASS_SS: F *= (r['F']/100.0)
    F *= 100.0
    M = (1-np.exp(-t_maint/max(MTTR,0.001)))*100 if MTTR>0 else 0.0
    return dict(nb=nb_tot, t_arret_h=round(t_arret,3), MTBF=round(MTBF,3), MTTR=round(MTTR,3), D=round(D,2), F=round(F,2), M=round(M,2))

def calculer_fmd_machine_topologie(resultats_modules, t_fiab=100, t_maint=10):
    rs = [r for r in resultats_modules if r['nb']>0]
    if not rs: return dict(nb=0, t_arret_h=0.0, MTBF=0.0, MTTR=0.0, D=100.0, F=0.0, M=0.0)
    nb_tot = sum(r['nb'] for r in resultats_modules); t_arret = sum(r['t_arret_h'] for r in resultats_modules)
    sl = sum(1.0/r['MTBF'] for r in rs if r['MTBF']>0); MTBF = 1.0/sl if sl>0 else 0.0
    slm = sum((1.0/r['MTBF'])*r['MTTR'] for r in rs if r['MTBF']>0); MTTR = slm/sl if sl>0 else 0.0
    dd = [r['D']/100 for r in resultats_modules if r['D']>0]; D = np.prod(dd)*100 if dd else 100.0
    ff = [r['F']/100 for r in resultats_modules]; F = np.prod(ff)*100 if ff else 100.0
    M = (1-np.exp(-t_maint/max(MTTR,0.001)))*100 if MTTR>0 else 0.0
    return dict(nb=nb_tot, t_arret_h=round(t_arret,3), MTBF=round(MTBF,3), MTTR=round(MTTR,3), D=round(D,2), F=round(F,2), M=round(M,2))

def simulation_monte_carlo(mtbf, mttr, t_fiab=100, t_maint=10, n_sim=10000):
    if mtbf==0 or mttr==0:
        return {k:(100.0 if 'D' in k else 0.0) for k in ['D_moy','D_std','D_p5','D_p95','F_moy','F_std','F_p5','F_p95','M_moy','M_std','M_p5','M_p95']}
    mtbf_s = np.maximum(np.random.normal(mtbf, mtbf*0.10, n_sim), 0.01)
    mttr_s = np.maximum(np.random.normal(mttr, mttr*0.10, n_sim), 0.01)
    D_s=(mtbf_s/(mtbf_s+mttr_s))*100; F_s=np.exp(-t_fiab/mtbf_s)*100; M_s=(1-np.exp(-t_maint/mttr_s))*100
    def _s(a): return round(np.mean(a),2),round(np.std(a),2),round(np.percentile(a,5),2),round(np.percentile(a,95),2)
    dm,ds,d5,d95=_s(D_s); fm,fs,f5,f95=_s(F_s); mm,ms,m5,m95=_s(M_s)
    return dict(D_moy=dm,D_std=ds,D_p5=d5,D_p95=d95,F_moy=fm,F_std=fs,F_p5=f5,F_p95=f95,M_moy=mm,M_std=ms,M_p5=m5,M_p95=m95)
  # ============================================================================
# PIPELINE COMPLET
# ============================================================================

def run_pipeline(df_brut, d1, d2, mode, t_fiab, t_maint, temps_prog_h, distribution, format_type='B'):
    periode_h = (d2 - d1).total_seconds() / 3600
    if format_type == 'A':
        df, col_map = nettoyer_format_A(df_brut)
    else:
        df, col_map = nettoyer_fichier(df_brut)
    df = df[(df[col_map['debut']] >= d1) & (df[col_map['debut']] <= d2)].copy()
    df.reset_index(drop=True, inplace=True)
    p_h = temps_prog_h or periode_h
    mode_cfg = MODES[mode]; ss_actifs = set(mode_cfg['ss_actifs'])
    def _get_tbf_ttr(ss_id):
        mask = df['ss_id'] == ss_id
        sub = df[mask].sort_values(col_map['debut']).copy()
        ttr_list = (sub[col_map['duree']] / 60).tolist()
        dates = sub[col_map['debut']].tolist()
        tbf_list = []
        for i in range(1, len(dates)):
            delta_h = (dates[i] - dates[i-1]).total_seconds() / 3600
            if delta_h > 0: tbf_list.append(delta_h)
        return tbf_list, ttr_list
    fmd_ss = {}
    for ss_id, module, nom in SOUS_SYSTEMES:
        durees = df.loc[df['ss_id'] == ss_id, col_map['duree']].tolist()
        tbf_list, ttr_list = _get_tbf_ttr(ss_id)
        fmd_ss[ss_id] = {'id': ss_id, 'module': module, 'nom': nom, 'actif': ss_id in ss_actifs,
            **calculer_fmd_ss(durees, p_h, t_fiab, t_maint, distribution=distribution, tbf_list_h=tbf_list, ttr_list_h=ttr_list)}
    fmd_mod = {}
    for mod in ['A','B','C']:
        ids = [ss for ss,m,_ in SOUS_SYSTEMES if m==mod and ss in ss_actifs]
        fmd_mod[mod] = calculer_fmd_module_topologie([fmd_ss[ss] for ss in ids], mode, mod, t_fiab, t_maint)
    fmd_mach = calculer_fmd_machine_topologie(list(fmd_mod.values()), t_fiab, t_maint)
    return {'df': df, 'col_map': col_map, 'fmd_ss': fmd_ss, 'fmd_mod': fmd_mod, 'fmd_mach': fmd_mach,
            'periode_h': periode_h, 'temps_prog_h': temps_prog_h, 'mode': mode, 'mode_cfg': mode_cfg,
            'd1': d1, 'd2': d2, 't_fiab': t_fiab, 't_maint': t_maint, 'distribution': distribution}

# ============================================================================
# HEADER + SIDEBAR
# ============================================================================

st.markdown("""
<div style="background:linear-gradient(135deg,#1565C0,#0D47A1);color:white;padding:24px;border-radius:12px;text-align:center;margin-bottom:20px;">
    <h1 style="margin:0;font-size:28px;">⚙️ ANALYSE FMD — LIGNE TAMPONNEUSE RIM GLUE #22</h1>
    <p style="margin:8px 0 0;font-size:14px;opacity:0.85;">
        Import Excel (brut MES ou nettoyé) · Modes · MTBF/MTTR · Disponibilité · Weibull · Monte Carlo · Criticité · Pareto</p>
    <p style="margin:4px 0 0;font-size:11px;opacity:0.6;">Développé par <b>Nadir</b> — Stagiaire #138422 — Cascades Inopak, Drummondville</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Paramètres")
    uploaded_file = st.file_uploader("📂 Fichier Excel ou CSV", type=['xlsx', 'xls', 'csv'])
    mode = st.selectbox("🔧 Mode de fonctionnement", ['mixte', 'tampon', 'rimglue'],
                         format_func=lambda x: MODES[x]['label'])
    st.markdown(f"*{MODES[mode]['description']}*")
    col1, col2 = st.columns(2)
    with col1: d1 = st.date_input("📅 Début")
    with col2: d2 = st.date_input("📅 Fin")
    distribution = st.selectbox("📐 Distribution", ['exponentielle', 'weibull'],
                                 format_func=lambda x: 'Exponentielle (λ=1/MTBF)' if x=='exponentielle' else 'Weibull (β, η — MLE)')
    col3, col4 = st.columns(2)
    with col3: t_fiab = st.number_input("t Fiabilité (h)", value=100, min_value=1)
    with col4: t_maint = st.number_input("t Maintenabilité (h)", value=10, min_value=1)
    t_prog_manual = st.number_input("⏱️ Temps programmé (h)", value=0.0, min_value=0.0,
                                     help="Format B uniquement. Si 0 → période calendaire. Format A → extrait auto du MES.")
    n_sim = st.number_input("🎲 Nb simulations Monte Carlo", value=10000, min_value=1000, step=1000)
    btn_analyser = st.button("▶️ Analyser", type="primary", use_container_width=True)
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;font-size:11px;color:#888;padding:10px 0;">
        <b>Développé par Nadir</b><br>Stagiaire #138422<br>
        M. Ing. Gestion de projets d'ingénierie<br>ÉTS — Cascades Inopak<br>Drummondville, QC
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# ANALYSE PRINCIPALE
# ============================================================================

if btn_analyser and uploaded_file is not None:
    try:
        from datetime import datetime, time as dt_time
        d1_dt = datetime.combine(d1, dt_time.min)
        d2_dt = datetime.combine(d2, dt_time.max)
        raw_bytes = uploaded_file.read()
        uploaded_file.seek(0)

        if uploaded_file.name.endswith('.csv'):
            df_input = pd.read_csv(io.BytesIO(raw_bytes))
            format_type = 'B'; format_info = None
            temps_prog_h = t_prog_manual if t_prog_manual > 0 else None
        else:
            df_test = pd.read_excel(io.BytesIO(raw_bytes), sheet_name=None, header=None)
            sn = next((s for s in df_test if "explorateur" not in s.lower()), list(df_test.keys())[0])
            fmt_type, _ = detecter_format(df_test[sn])
            if fmt_type == 'A':
                format_type = 'A'
                df_input, format_info = extraire_format_A(raw_bytes)
                temps_prog_h = format_info['temps_prog_h']
            else:
                format_type = 'B'; format_info = None
                df_input = pd.read_excel(io.BytesIO(raw_bytes))
                temps_prog_h = t_prog_manual if t_prog_manual > 0 else None

        if format_type == 'A' and format_info:
            st.success(f"📄 **Fichier brut MES détecté** — {format_info['nb_bris']} bris extraits")
            with st.expander("📊 Détails extraction MES", expanded=True):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Fonctionnement", f"{format_info['temps_fonc_min']:.0f} min")
                c2.metric("Bris Méc.", f"{format_info['total_mec_min']:.0f} min")
                c3.metric("Bris Élec.", f"{format_info['total_elec_min']:.0f} min")
                c4.metric("Temps programmé", f"{format_info['temps_prog_h']:.2f} h")
        else:
            tp_label = f"{temps_prog_h:.2f}h" if temps_prog_h else "calendaire"
            st.success(f"✅ **Fichier nettoyé** — {len(df_input)} lignes — Temps réf : {tp_label}")

        res = run_pipeline(df_input, d1_dt, d2_dt, mode, t_fiab, t_maint, temps_prog_h, distribution, format_type)

        tabs = st.tabs(["🔍 Nettoyage", "📊 Résultats FMD", "📐 Méthode", "🎲 Monte Carlo",
                        "⚠️ Criticité", "📐 FBS", "📈 Graphiques & Pareto", "📋 Export"])

        df = res['df']; col_map = res['col_map']; fmd_ss = res['fmd_ss']
        fmd_mod = res['fmd_mod']; fmd_mach = res['fmd_mach']; mode_cfg = res['mode_cfg']
        ss_actifs = set(mode_cfg['ss_actifs'])
        p_h = temps_prog_h or res['periode_h']
      # ── ONGLET 1 : NETTOYAGE ──
        with tabs[0]:
            st.subheader("🔍 Nettoyage & Affectation")
            nb_id = int(df['ss_id'].notna().sum()); nb_non = int(df['ss_id'].isna().sum())
            c1, c2, c3 = st.columns(3)
            c1.metric("Total arrêts", len(df)); c2.metric("✅ Identifiés", nb_id); c3.metric("⚠️ Non identifiés", nb_non)
            if nb_id > 0:
                rep = (df[df['ss_id'].notna()].groupby('ss_id')
                       .agg(nb=(col_map['duree'],'count'), duree=(col_map['duree'],'sum'))
                       .reset_index().sort_values('ss_id'))
                rep['Sous-système'] = rep['ss_id'].map(SS_NOMS)
                rep['duree'] = rep['duree'].round(1)
                rep.columns = ['ID','Nb arrêts','Durée tot (min)','Sous-système']
                st.dataframe(rep[['ID','Sous-système','Nb arrêts','Durée tot (min)']], use_container_width=True)
            show = df[[col_map['debut'],col_map['duree'],col_map['r2'],col_map['r3'],'ss_id']].copy()
            show.columns = ['Date','Durée (min)','Raison 2','Raison 3','SS']
            show['SS'] = show['SS'].fillna('❌ NON ID'); show['Durée (min)'] = show['Durée (min)'].round(2)
            st.dataframe(show.head(50), use_container_width=True)

        # ── ONGLET 2 : RÉSULTATS FMD ──
        with tabs[1]:
            st.subheader("📊 Résultats FMD")
            tp_txt = f"{temps_prog_h:.2f}h" if temps_prog_h else f"{res['periode_h']:.1f}h (cal.)"
            st.info(f"**{mode_cfg['label']}** | {distribution.upper()} | Période : {d1} → {d2} | Temps réf : {tp_txt}")
            rows_ss = []
            for ss_id, mod, nom in SOUS_SYSTEMES:
                r = fmd_ss[ss_id]
                rows_ss.append({'ID':ss_id,'Sous-système':nom,'Module':mod,'Actif':'✅' if r['actif'] else '❌',
                    'n':r['nb'],'T.arrêt(h)':r['t_arret_h'],'T.fonct(h)':r['t_fonct_h'],
                    'MTBF(h)':r['MTBF'],'MTTR(h)':r['MTTR'],'D(%)':r['D'],'F(%)':r['F'],'M(%)':r['M']})
            st.markdown("**Par sous-système :**")
            st.dataframe(pd.DataFrame(rows_ss), use_container_width=True)
            st.markdown("**Par module :**")
            rows_mod = []
            for mod in ['A','B','C']:
                r = fmd_mod[mod]
                rows_mod.append({'Module':MOD_LABEL[mod],'n':r['nb'],'T.arrêt(h)':r['t_arret_h'],
                    'MTBF(h)':r['MTBF'],'MTTR(h)':r['MTTR'],'D(%)':r['D'],'F(%)':r['F'],'M(%)':r['M']})
            st.dataframe(pd.DataFrame(rows_mod), use_container_width=True)
            st.markdown("### 🏭 Machine complète")
            rm = fmd_mach
            c1,c2,c3,c4,c5 = st.columns(5)
            c1.metric("Nb pannes", rm['nb']); c2.metric("MTBF", f"{rm['MTBF']:.3f} h"); c3.metric("MTTR", f"{rm['MTTR']:.3f} h")
            c4.metric("Disponibilité", f"{rm['D']:.2f}%"); c5.metric("Fiabilité", f"{rm['F']:.2f}%")

        # ── ONGLET 3 : MÉTHODE ──
        with tabs[2]:
            st.subheader("📐 Méthode de calcul")
            st.markdown(f"""
**Distribution :** {distribution.upper()} | **Mode :** {mode_cfg['label']}

**1. Par sous-système :**
- n = nombre de pannes affectées
- T_arrêt = Σ durées (en heures)
- T_fonct = T_programmé − T_arrêt
- MTBF = T_fonct / n | MTTR = T_arrêt / n
- D = MTBF / (MTBF + MTTR) × 100

**2. Agrégation (série pure, C2/C5 bypass) :**
- λ_sys = Σ(1/MTBFᵢ) — hors bypass
- MTBF_sys = 1/λ_sys
- MTTR_sys = Σ(λᵢ·MTTRᵢ)/λ_sys
- D_sys = Π Dᵢ | F_sys = Π Fᵢ — hors bypass

**Chemin critique :**
- **Tampon** : A1→A2→A3→A4 → B1→B2→B4 → C1→C3→C4→C6→C7 *(C5 bypass)*
- **RimGlue** : A1→A2→A3→A4→A5 → B1→B2→B3→B4 → C1→C6→C7 *(C2 bypass)*
- **Mixte** : A1→A2→A3→A4→A5 → B1→B2→B3→B4 → C1→C3→C4→C6→C7 *(C2, C5 bypass)*
            """)
            if distribution == 'weibull':
                st.markdown("**Paramètres Weibull par sous-système :**")
                wb_rows = []
                for ss_id, mod, nom in SOUS_SYSTEMES:
                    if ss_id not in ss_actifs: continue
                    r = fmd_ss[ss_id]; wf = r.get('weibull_fiab',{}); wm = r.get('weibull_maint',{})
                    bf = wf.get('beta',1)
                    interp = '⬇️ Infantile' if bf<0.95 else ('➡️ Utile' if bf<=1.05 else '⬆️ Usure')
                    wb_rows.append({'ID':ss_id,'Nom':nom,'β_f':wf.get('beta',''),'η_f':wf.get('eta',''),
                        'MTBF':r['MTBF'],'β_m':wm.get('beta',''),'η_m':wm.get('eta',''),'MTTR':r['MTTR'],'Interp.':interp})
                st.dataframe(pd.DataFrame(wb_rows), use_container_width=True)

        # ── ONGLET 4 : MONTE CARLO ──
        with tabs[3]:
            st.subheader("🎲 Simulation Monte Carlo")
            rm = fmd_mach
            mc = simulation_monte_carlo(rm['MTBF'], rm['MTTR'], t_fiab, t_maint, n_sim)
            st.markdown(f"**Machine complète** — {n_sim:,} simulations")
            mc_df = pd.DataFrame([
                {'KPI':'D (%)','Déterministe':rm['D'],'Moyenne MC':mc['D_moy'],'Std':mc['D_std'],'P5':mc['D_p5'],'P95':mc['D_p95']},
                {'KPI':'F (%)','Déterministe':rm['F'],'Moyenne MC':mc['F_moy'],'Std':mc['F_std'],'P5':mc['F_p5'],'P95':mc['F_p95']},
                {'KPI':'M (%)','Déterministe':rm['M'],'Moyenne MC':mc['M_moy'],'Std':mc['M_std'],'P5':mc['M_p5'],'P95':mc['M_p95']},
            ])
            st.dataframe(mc_df, use_container_width=True)
            for mod in ['A','B','C']:
                rm_m = fmd_mod[mod]; mc_m = simulation_monte_carlo(rm_m['MTBF'],rm_m['MTTR'],t_fiab,t_maint,n_sim)
                with st.expander(f"{MOD_LABEL[mod]}"):
                    st.dataframe(pd.DataFrame([
                        {'KPI':'D','Dét.':rm_m['D'],'Moy':mc_m['D_moy'],'P5':mc_m['D_p5'],'P95':mc_m['D_p95']},
                        {'KPI':'F','Dét.':rm_m['F'],'Moy':mc_m['F_moy'],'P5':mc_m['F_p5'],'P95':mc_m['F_p95']},
                        {'KPI':'M','Dét.':rm_m['M'],'Moy':mc_m['M_moy'],'P5':mc_m['M_p5'],'P95':mc_m['M_p95']},
                    ]), use_container_width=True)
            fig_mc, axes = plt.subplots(1, 3, figsize=(15, 4))
            fig_mc.suptitle(f'Distributions Monte Carlo ({n_sim:,} sim)', fontweight='bold')
            mtbf_s = np.maximum(np.random.normal(rm['MTBF'],rm['MTBF']*0.1,n_sim),0.01)
            mttr_s = np.maximum(np.random.normal(rm['MTTR'],rm['MTTR']*0.1,n_sim),0.01)
            for ax,data,color,label in [
                (axes[0],(mtbf_s/(mtbf_s+mttr_s))*100,'#43A047','Disponibilité (%)'),
                (axes[1],np.exp(-t_fiab/mtbf_s)*100,'#1565C0',f'Fiabilité (%)'),
                (axes[2],(1-np.exp(-t_maint/mttr_s))*100,'#E65100',f'Maintenabilité (%)')]:
                ax.hist(data,bins=50,color=color,alpha=0.7,edgecolor='black')
                ax.set_xlabel(label); ax.set_title(label.split('(')[0]); ax.grid(alpha=0.3)
            plt.tight_layout(); st.pyplot(fig_mc)

        # ── ONGLET 5 : CRITICITÉ ──
        with tabs[4]:
            st.subheader("⚠️ Analyse de criticité")
            nb_mois = max(p_h / (30*24), 1/30)
            st.markdown("### 🚨 Arrêts graves (durée > 60 min)")
            graves = df[df[col_map['duree']] > 60].sort_values(col_map['duree'], ascending=False)
            if len(graves) == 0: st.success("✅ Aucun arrêt > 60 min")
            else:
                show_g = graves[[col_map['debut'],col_map['duree'],col_map['r2'],col_map['r3'],'ss_id']].copy()
                show_g.columns = ['Date','Durée (min)','Raison 2','Raison 3','SS']
                show_g['SS'] = show_g['SS'].apply(lambda x: f'{x} — {SS_NOMS.get(x,"")}' if pd.notna(x) else '❌')
                st.dataframe(show_g, use_container_width=True)
            st.markdown("### 📊 Matrice Gravité × Fréquence")
            rows_c = []
            for ss_id, mod, nom in SOUS_SYSTEMES:
                if ss_id not in ss_actifs: continue
                sub = df[df['ss_id']==ss_id]; nb=len(sub)
                if nb==0: continue
                dt = sub[col_map['duree']].sum(); npm = nb/nb_mois
                fs = 1 if npm<=2 else (2 if npm<=10 else (3 if npm<=30 else 4))
                gs = 1 if dt<=15 else (2 if dt<=60 else (3 if dt<=180 else 4))
                rows_c.append({'ID':ss_id,'Sous-système':nom,'Module':mod,'Nb':nb,'Nb/mois':round(npm,1),
                    'Durée tot':round(dt,1),'F':fs,'G':gs,'Score':fs*gs,
                    'Priorité':'🔴 Critique' if fs*gs>=12 else ('🟠 Élevée' if fs*gs>=8 else ('🟡 Moyenne' if fs*gs>=4 else '🟢 Faible'))})
            if rows_c:
                df_crit = pd.DataFrame(rows_c).sort_values('Score', ascending=False)
                st.dataframe(df_crit, use_container_width=True)
                fig_hm, ax = plt.subplots(figsize=(8,6))
                labels_mat = [['' for _ in range(4)] for _ in range(4)]
                for _, r in df_crit.iterrows():
                    fi,gi=int(r['F'])-1,int(r['G'])-1
                    if labels_mat[gi][fi]: labels_mat[gi][fi]+='\n'
                    labels_mat[gi][fi]+=r['ID']
                cm = np.array([[1,2,3,4],[2,4,6,8],[3,6,9,12],[4,8,12,16]])
                cmap = LinearSegmentedColormap.from_list('gf',['#4CAF50','#FFC107','#FF9800','#E53935'])
                ax.imshow(cm,cmap=cmap,aspect='auto',vmin=1,vmax=16,origin='lower')
                for gi in range(4):
                    for fi in range(4):
                        sc=(fi+1)*(gi+1); txt=labels_mat[gi][fi]; c='white' if sc>=8 else 'black'
                        ax.text(fi,gi,f'{sc}\n{txt}',ha='center',va='center',fontsize=8,fontweight='bold',color=c)
                ax.set_xticks(range(4)); ax.set_xticklabels(['F=1\n≤2/m','F=2\n3-10/m','F=3\n11-30/m','F=4\n>30/m'])
                ax.set_yticks(range(4)); ax.set_yticklabels(['G=1\n≤15min','G=2\n16-60min','G=3\n61-180min','G=4\n>180min'])
                ax.set_xlabel('Fréquence'); ax.set_ylabel('Gravité'); ax.set_title('Matrice de Criticité G × F')
                plt.tight_layout(); st.pyplot(fig_hm)
              # ── ONGLET 6 : FBS ──
        with tabs[5]:
            st.subheader("📐 Diagramme FBS")
            blocs = [{'id':ss,'module':mod,'nom':nom,'MTBF':fmd_ss[ss]['MTBF'],'MTTR':fmd_ss[ss]['MTTR'],'D':fmd_ss[ss]['D']}
                     for ss,mod,nom in SOUS_SYSTEMES if ss in ss_actifs]
            nb_b = len(blocs); figh = nb_b*1.15+6
            fig_fbs, ax = plt.subplots(figsize=(10, figh))
            fig_fbs.patch.set_facecolor('white'); ax.set_xlim(0,10); ax.set_ylim(0,figh); ax.axis('off')
            ax.text(5,figh-0.45,'FBS — LIGNE TAMPONNEUSE RIM GLUE #22',ha='center',fontsize=13,fontweight='bold')
            ax.text(5,figh-0.90,f"{mode_cfg['label']}  —  {d1} → {d2}",ha='center',fontsize=9,style='italic',color='gray')
            CX,BW,BH,STEP=5,5.4,0.85,1.14; y0=figh-1.90
            ax.add_patch(plt.Circle((CX,y0),0.30,color='#1565C0',zorder=5))
            ax.text(CX,y0,'⊗',ha='center',va='center',fontsize=16,color='white',fontweight='bold',zorder=6)
            ax.text(CX+0.56,y0,'ENTRÉE',va='center',fontsize=10,fontweight='bold')
            cur_mod=None; y=y0
            for i,b in enumerate(blocs):
                y=y0-(i+1)*STEP
                if b['module']!=cur_mod:
                    cur_mod=b['module']; sy=y+BH/2+0.36; mc=MOD_COLOR[b['module']]
                    ax.plot([0.3,9.7],[sy,sy],color=mc,lw=2,linestyle='--')
                    ax.text(0.2,sy+0.07,f"MODULE {b['module']}",fontsize=9,fontweight='bold',color=mc,va='bottom')
                ax.annotate('',xy=(CX,y+BH/2+0.03),xytext=(CX,y+BH/2+0.46),arrowprops=dict(arrowstyle='->',color='black',lw=2.5))
                ax.add_patch(FancyBboxPatch((CX-BW/2,y-BH/2),BW,BH,boxstyle='round,pad=0.05',facecolor=_fc(b['D']),edgecolor='black',lw=2,zorder=4))
                ax.text(CX-BW/2+0.24,y+0.21,f"{b['id']} — {b['nom']}",va='center',fontsize=9,fontweight='bold',color='white',zorder=5)
                ax.text(CX-BW/2+0.24,y-0.19,f"D={b['D']:.1f}%  MTBF={b['MTBF']:.2f}h  MTTR={b['MTTR']:.3f}h",va='center',fontsize=8,color='white',zorder=5)
            ax.annotate('',xy=(CX,y-BH/2-0.28),xytext=(CX,y-BH/2-0.02),arrowprops=dict(arrowstyle='->',color='black',lw=2.5))
            ax.add_patch(plt.Circle((CX,y-BH/2-0.55),0.30,color='#1565C0',zorder=5))
            ax.text(CX,y-BH/2-0.55,'⊗',ha='center',va='center',fontsize=16,color='white',fontweight='bold',zorder=6)
            ax.text(CX+0.56,y-BH/2-0.55,'SORTIE',va='center',fontsize=10,fontweight='bold')
            d_sys=fmd_mach['D']
            ax.text(CX,y-BH/2-1.50,f"Disponibilité système = {d_sys:.2f}%",ha='center',fontsize=12,fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.6',facecolor=_fc(d_sys),edgecolor='black',lw=2.5),color='white')
            plt.tight_layout(); st.pyplot(fig_fbs)

        # ── ONGLET 7 : GRAPHIQUES & PARETO ──
        with tabs[6]:
            st.subheader("📈 Graphiques FMD & Pareto")
            ids = [ss for ss,_,_ in SOUS_SYSTEMES if ss in ss_actifs]
            disps=[fmd_ss[s]['D'] for s in ids]; mtbfs=[fmd_ss[s]['MTBF'] for s in ids]
            mttrs=[fmd_ss[s]['MTTR'] for s in ids]; narrs=[fmd_ss[s]['nb'] for s in ids]
            tarrs=[fmd_ss[s]['t_arret_h'] for s in ids]
            fig_d, ax = plt.subplots(figsize=(14,5))
            ax.bar(ids, disps, color=[_fc(d) for d in disps], edgecolor='black', width=0.65)
            ax.axhline(95,color='green',ls='--',lw=1.8,label='95%'); ax.axhline(85,color='orange',ls='--',lw=1.8,label='85%')
            ax.set_ylim(0,110); ax.set_title('Disponibilité (%)',fontweight='bold'); ax.legend(); ax.grid(axis='y',alpha=0.3)
            for i,v in enumerate(disps):
                if v>0: ax.text(i,v+1,f'{v:.1f}',ha='center',fontsize=8,fontweight='bold')
            plt.tight_layout(); st.pyplot(fig_d)
            col1, col2 = st.columns(2)
            with col1:
                fig_b, ax = plt.subplots(figsize=(7,4))
                ax.bar(ids, mtbfs, color=[MOD_COLOR[fmd_ss[s]['module']] for s in ids], edgecolor='black')
                ax.set_title('MTBF (h)',fontweight='bold'); ax.grid(axis='y',alpha=0.3)
                ax.set_xticks(range(len(ids))); ax.set_xticklabels(ids,rotation=45,ha='right')
                plt.tight_layout(); st.pyplot(fig_b)
            with col2:
                fig_t, ax = plt.subplots(figsize=(7,4))
                ax.bar(ids, mttrs, color=[MOD_COLOR[fmd_ss[s]['module']] for s in ids], edgecolor='black')
                ax.set_title('MTTR (h)',fontweight='bold'); ax.grid(axis='y',alpha=0.3)
                ax.set_xticks(range(len(ids))); ax.set_xticklabels(ids,rotation=45,ha='right')
                plt.tight_layout(); st.pyplot(fig_t)
            st.markdown("### 📊 Diagramme de Pareto")
            data_p = [{'id':ss,'nom':fmd_ss[ss]['nom'],'module':fmd_ss[ss]['module'],'nb':fmd_ss[ss]['nb'],'ta':fmd_ss[ss]['t_arret_h']}
                      for ss in ids if fmd_ss[ss]['nb']>0]
            if data_p:
                df_p = pd.DataFrame(data_p)
                fig_par, axes2 = plt.subplots(1,2,figsize=(16,6))
                for ax,col,ylabel in [(axes2[0],'nb','Nombre de pannes'),(axes2[1],'ta',"Temps d'arrêt (h)")]:
                    df_s = df_p.sort_values(col,ascending=False).reset_index(drop=True)
                    total = df_s[col].sum(); df_s['cumul'] = df_s[col].cumsum()/total*100
                    colors = [MOD_COLOR[m] for m in df_s['module']]
                    ax.bar(range(len(df_s)),df_s[col],color=colors,edgecolor='black',width=0.7,zorder=3)
                    ax2 = ax.twinx()
                    ax2.plot(range(len(df_s)),df_s['cumul'],color='#D32F2F',marker='o',lw=2.5,markersize=6,zorder=5)
                    ax2.axhline(80,color='#D32F2F',ls='--',lw=1.5,alpha=0.6); ax2.set_ylim(0,105)
                    ax.set_title(f'Pareto — {ylabel}',fontweight='bold'); ax.set_ylabel(ylabel)
                    ax.set_xticks(range(len(df_s))); ax.set_xticklabels(df_s['id'],rotation=45,ha='right')
                    ax.grid(axis='y',alpha=0.3,zorder=0)
                    idx_80 = df_s[df_s['cumul']<=80].index
                    if len(idx_80)>0: ax.axvspan(-0.5,idx_80[-1]+0.5,alpha=0.08,color='red',zorder=0)
                legend_elements = [Patch(facecolor=MOD_COLOR[m],edgecolor='black',label=f'Module {m}') for m in ['A','B','C']]
                fig_par.legend(handles=legend_elements,loc='lower center',ncol=3,fontsize=10,frameon=True)
                plt.tight_layout(rect=[0,0.06,1,0.95]); st.pyplot(fig_par)
                df_ta = df_p.sort_values('ta',ascending=False).reset_index(drop=True)
                df_ta['cumul'] = (df_ta['ta'].cumsum()/df_ta['ta'].sum()*100).round(1)
                df_ta['Zone'] = df_ta['cumul'].apply(lambda x:'🔴 A' if x<=80 else ('🟡 B' if x<=95 else '🟢 C'))
                st.dataframe(df_ta[['id','nom','module','nb','ta','cumul','Zone']], use_container_width=True)

        # ── ONGLET 8 : EXPORT ──
        with tabs[7]:
            st.subheader("📋 Export des résultats")
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine='openpyxl') as writer:
                pd.DataFrame([
                    {'Paramètre':'Mode','Valeur':mode_cfg['label']},
                    {'Paramètre':'Distribution','Valeur':distribution},
                    {'Paramètre':'Début','Valeur':str(d1)},{'Paramètre':'Fin','Valeur':str(d2)},
                    {'Paramètre':'Temps programmé (h)','Valeur':temps_prog_h if temps_prog_h else 'Calendaire'},
                    {'Paramètre':'Topologie','Valeur':'Série pure — C2, C5 = bypass'},
                ]).to_excel(writer, sheet_name='Paramètres', index=False)
                show_net = df[[col_map['debut'],col_map['duree'],col_map['r2'],col_map['r3'],'ss_id']].copy()
                show_net.columns = ['Date','Durée (min)','Raison 2','Raison 3','SS']
                show_net.to_excel(writer, sheet_name='Nettoyage', index=False)
                rows_ss_exp = []
                for ss_id,mod,nom in SOUS_SYSTEMES:
                    r = fmd_ss[ss_id]
                    rows_ss_exp.append({'ID':ss_id,'Module':mod,'Nom':nom,'Actif':'Oui' if r['actif'] else 'Non',
                        'n':r['nb'],'T.arrêt(h)':r['t_arret_h'],'MTBF(h)':r['MTBF'],'MTTR(h)':r['MTTR'],
                        'D(%)':r['D'],'F(%)':r['F'],'M(%)':r['M']})
                pd.DataFrame(rows_ss_exp).to_excel(writer, sheet_name='FMD_SS', index=False)
                rows_mod_exp = []
                for mod in ['A','B','C']:
                    r = fmd_mod[mod]
                    rows_mod_exp.append({'Module':MOD_LABEL[mod],'n':r['nb'],'T.arrêt(h)':r['t_arret_h'],
                        'MTBF(h)':r['MTBF'],'MTTR(h)':r['MTTR'],'D(%)':r['D'],'F(%)':r['F'],'M(%)':r['M']})
                rm = fmd_mach
                rows_mod_exp.append({'Module':'MACHINE','n':rm['nb'],'T.arrêt(h)':rm['t_arret_h'],
                    'MTBF(h)':rm['MTBF'],'MTTR(h)':rm['MTTR'],'D(%)':rm['D'],'F(%)':rm['F'],'M(%)':rm['M']})
                pd.DataFrame(rows_mod_exp).to_excel(writer, sheet_name='FMD_Modules', index=False)
                mc_rows = []
                for label,r_data in [('MACHINE',rm)]+[(f'MODULE {m}',fmd_mod[m]) for m in ['A','B','C']]:
                    mc_exp = simulation_monte_carlo(r_data['MTBF'],r_data['MTTR'],t_fiab,t_maint,10000)
                    mc_rows.append({'Niveau':label,'D_dét':r_data['D'],'D_moy':mc_exp['D_moy'],'D_P5':mc_exp['D_p5'],'D_P95':mc_exp['D_p95'],
                        'F_dét':r_data['F'],'F_moy':mc_exp['F_moy'],'F_P5':mc_exp['F_p5'],'F_P95':mc_exp['F_p95'],
                        'M_dét':r_data['M'],'M_moy':mc_exp['M_moy'],'M_P5':mc_exp['M_p5'],'M_P95':mc_exp['M_p95']})
                pd.DataFrame(mc_rows).to_excel(writer, sheet_name='Monte_Carlo', index=False)
                graves_exp = df[df[col_map['duree']]>60].sort_values(col_map['duree'],ascending=False)
                if len(graves_exp)>0:
                    g_show = graves_exp[[col_map['debut'],col_map['duree'],col_map['r2'],col_map['r3'],'ss_id']].copy()
                    g_show.columns = ['Date','Durée','R2','R3','SS']
                    g_show.to_excel(writer, sheet_name='Arrêts_graves', index=False)
                if rows_c:
                    pd.DataFrame(rows_c).to_excel(writer, sheet_name='Matrice_GxF', index=False)
            buf.seek(0)
            st.download_button("📥 Télécharger le rapport complet (Excel)", data=buf.getvalue(),
                              file_name=f"rapport_COMPLET_FMD_{mode}.xlsx",
                              mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                              use_container_width=True)

    except Exception as e:
        import traceback
        st.error(f"❌ Erreur : {e}")
        st.code(traceback.format_exc())

elif not uploaded_file:
    st.info("👈 Commencez par charger un fichier Excel dans la barre latérale, puis cliquez **Analyser**.")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align:center;font-size:12px;color:#888;padding:20px 0;">
    <b>⚙️ Analyse FMD — Ligne Tamponneuse Rim Glue #22</b><br>
    Développé par <b>Nadir</b> — Stagiaire #138422<br>
    M. Ing. Gestion de projets d'ingénierie — ÉTS<br>
    Cascades Inopak, Drummondville, QC<br>
    <em>Série pure | C2, C5 = bypass | MTBF = 1/Σ(1/MTBFᵢ) | D = ΠDᵢ</em>
</div>
""", unsafe_allow_html=True)
