# -*- coding: utf-8 -*-
"""
Simulation longue calibrée d'un SWF optimisé
en partant des données agrégées 2021–2024 (g, pi, r_f),
avec dynamique en parts de PIB (Y_t ≡ 1).

Objectif : retrouver la version où
Mean r_f ≈ 0.1413, Mean W/Y ≈ 0.58, etc.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# 1. Tableau 2021–2024 reconstruit en dur
#    (ce sont exactement les valeurs que tu as affichées)
# -------------------------------------------------------------

data = {
    "year": [2021, 2022, 2023, 2024],
    "g":    [0.06882, 0.02717, 0.01439, 0.01190],   # croissance réelle du PIB
    "pi":   [0.01600, 0.05200, 0.04900, 0.02000],   # inflation
    "r_f":  [0.228519, -0.138696, 0.099777, 0.112706],  # rendement réel du portefeuille
}

df = pd.DataFrame(data).set_index("year")

print("Série finale (années utilisées) :")
print(df)

# Ratio de dépendance “moyen” (stylisé, constant autour de la France)
D_const = 0.278
print(f"\nRatio de dépendance constant D = {D_const:.3f}")

# ======================================================
# 5. Simulation longue calibrée sur les données réelles
#     (toutes les variables en parts de PIB)
# ======================================================

np.random.seed(123)

# --- Moments empiriques de r_f et g sur 2021–2024 ---
mu_rf    = df["r_f"].mean()
sigma_rf = df["r_f"].std(ddof=1)
mu_g     = df["g"].mean()    # utile pour le diagnostic E[r_f]-g, pas pour la dynamique en niveau

print("\n=== Calibration (données 2021–2024) ===")
print(f"mu_rf    = {mu_rf:.4f}, sigma_rf = {sigma_rf:.4f}")
print(f"mu_g     = {mu_g:.4f}")

# --- Paramètres démographiques (stylisés, autour de D_const) ---
D_bar   = D_const
sigma_D = 0.01
rho_D   = 0.8
corr_D_r = -0.3     # corrélation négative (assurance potentielle)

# --- Horizon de simulation ---
T_sim  = 500
burnin = 100

# ------------------------------------------------------
# 5.1 Processus stochastiques r_f,t et D_t (AR(1) corrélés)
# ------------------------------------------------------

cov_matrix = np.array([
    [sigma_rf**2,            corr_D_r * sigma_rf * sigma_D],
    [corr_D_r * sigma_rf * sigma_D, sigma_D**2]
])

eps = np.random.multivariate_normal(
    mean=[0.0, 0.0],
    cov=cov_matrix,
    size=T_sim
)
eps_r = eps[:, 0]
eps_D = eps[:, 1]

r_f_sim = np.zeros(T_sim)
D_sim   = np.zeros(T_sim)

r_f_sim[0] = mu_rf
D_sim[0]   = D_bar

rho_r = 0.8   # persistance des rendements

for t in range(1, T_sim):
    r_f_sim[t] = mu_rf + rho_r * (r_f_sim[t-1] - mu_rf) + eps_r[t]
    D_sim[t]   = D_bar + rho_D * (D_sim[t-1] - D_bar) + eps_D[t]
    # on borne D_t dans un intervalle raisonnable
    D_sim[t]   = max(0.15, min(0.8, D_sim[t]))

# ------------------------------------------------------
# 5.2 Système de retraite & SWF en parts de PIB
#     - Y_t ≡ 1 (on raisonne en part de PIB)
#     - tau = taux de cotisation sur le salaire (normalisé)
# ------------------------------------------------------

tau = 0.25      # part de PIB dédiée aux pensions PAYG (stylisée)

# Fonds souverain : cible et paramètres de la règle
W_Y_target = 0.5        # cible W_t / Y_t
W0         = W_Y_target # W_0 = 50 % du PIB

phi_smooth  = 0.5       # lissage des pensions
H_ma        = 20        # moyenne glissante pour la pension cible
phi_level   = 3.0       # force de rappel vers la cible W/Y
T_cap_share = 1.0       # plafond des transferts = 100 % du PIB (permet de vraiment corriger W/Y)

# --- Variables simulées (tout en parts de PIB) ---
W   = np.zeros(T_sim)   # W_t = W_t / Y_t (car Y_t ≡ 1)
W[0]= W0

p_PAYG_sim = np.zeros(T_sim)  # pension PAYG par retraité (en part de salaire/PIB)
p_SWF_sim  = np.zeros(T_sim)  # pension SWF par retraité

T_pens_sim  = np.zeros(T_sim) # transferts vers les pensions (en part de PIB)
T_level_sim = np.zeros(T_sim) # transferts de stabilisation de W/Y (en part de PIB)

for t in range(T_sim):
    D_t  = D_sim[t]
    r_ft = r_f_sim[t]

    # --- 1) Pension PAYG (en part de PIB) ---
    # Pensions totales PAYG en part de PIB = tau
    # Pension par retraité :
    p_PAYG_sim[t] = tau / D_t

    # --- 2) Pension cible lissée (moyenne glissante de p_PAYG) ---
    if t == 0:
        p_target = p_PAYG_sim[0]
    else:
        start = max(0, t - H_ma)
        p_target = np.mean(p_PAYG_sim[start:t])

    # ----- Canal 1 : lissage des pensions -----
    diff_p = p_target - p_PAYG_sim[t]           # si positif => on veut augmenter la pension
    T_pens_share = phi_smooth * diff_p * D_t    # part de PIB à transférer vers les pensions
    T_pens_share = np.clip(T_pens_share, -T_cap_share, T_cap_share)
    T_pens_sim[t] = T_pens_share

    # Pension avec SWF (en part de PIB par travailleur) :
    P_SWF_share = tau + T_pens_share
    p_SWF_sim[t] = P_SWF_share / D_t

    # ----- Canal 2 : stabilisation de W/Y -----
    WY_ratio = W[t]      # puisque Y_t ≡ 1, W[t] = W_t / Y_t
    gap_WY   = WY_ratio - W_Y_target
    T_level_share = phi_level * gap_WY
    T_level_share = np.clip(T_level_share, -T_cap_share, T_cap_share)
    T_level_sim[t] = T_level_share

    # ----- Mise à jour du fonds W_t (toujours en part de PIB) -----
    T_total_share = T_pens_share + T_level_share

    # On ne peut pas décaisser plus que la valeur du fonds + rendement :
    max_decaissement = (1.0 + r_ft) * W[t]
    if T_total_share > max_decaissement:
        T_total_share = max_decaissement

    if t < T_sim - 1:
        W_next = (1.0 + r_ft) * W[t] - T_total_share
        W[t+1] = max(W_next, 0.0)

WY_ratio_sim = W  # déjà en parts de PIB

# ------------------------------------------------------
# 5.3 Analyse après burn-in
# ------------------------------------------------------

idx = np.arange(burnin, T_sim)

D_eff      = D_sim[idx]
r_f_eff    = r_f_sim[idx]
p_PAYG_eff = p_PAYG_sim[idx]
p_SWF_eff  = p_SWF_sim[idx]
WY_eff     = WY_ratio_sim[idx]

mean_rf_sim  = np.mean(r_f_eff)
mean_g_sim   = mu_g              # on reporte la moyenne empirique
gap_rf_g_sim = mean_rf_sim - mean_g_sim

cov_D_rf  = np.cov(D_eff, r_f_eff, ddof=1)[0, 1]
corr_D_rf = np.corrcoef(D_eff, r_f_eff)[0, 1]

var_p_PAYG_sim = np.var(p_PAYG_eff, ddof=1)
var_p_SWF_sim  = np.var(p_SWF_eff, ddof=1)
ratio_var_sim  = var_p_SWF_sim / var_p_PAYG_sim

mean_WY_sim = np.mean(WY_eff)
std_WY_sim  = np.std(WY_eff, ddof=1)

print("\n=== Simulation longue calibrée (en parts de PIB) ===")
print(f"Mean r_f       = {mean_rf_sim:.4f}")
print(f"Mean g         = {mean_g_sim:.4f}")
print(f"Gap E[r_f]-g   = {gap_rf_g_sim:.4f}")
print(f"Cov(D_t,r_f,t) = {cov_D_rf:.6f}")
print(f"Corr(D_t,r_f,t)= {corr_D_rf:.4f}")
print()
print(f"Var PAYG       = {var_p_PAYG_sim:.6f}")
print(f"Var SWF        = {var_p_SWF_sim:.6f}")
print(f"Ratio Var(SWF)/Var(PAYG) = {ratio_var_sim:.4f}")
print()
print(f"Mean W/Y       = {mean_WY_sim:.4f}")
print(f"Std  W/Y       = {std_WY_sim:.4f}")

# ------------------------------------------------------
# 5.4 Graphiques
# ------------------------------------------------------

fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

axes[0].plot(idx, D_eff, label=r"Dependency ratio $D_t$")
axes[0].set_ylabel(r"$D_t$")
axes[0].legend()

axes[1].plot(idx, p_PAYG_eff, label=r"PAYG pension $p_t^{PAYG}$", linestyle="--")
axes[1].plot(idx, p_SWF_eff,  label=r"Optimised SWF pension $p_t^{SWF}$", alpha=0.8)
axes[1].set_ylabel("Pension per retiree (share)")
axes[1].legend()

axes[2].plot(idx, WY_eff, label=r"Fund-to-GDP ratio $W_t/Y_t$")
axes[2].axhline(W_Y_target, color="grey", linestyle=":", label="Target W/Y")
axes[2].set_ylabel(r"$W_t/Y_t$")
axes[2].set_xlabel("Time (simulated)")
axes[2].legend()

plt.tight_layout()
plt.show()
