# -*- coding: utf-8 -*-
"""
Calibration + simulation académique d'un fonds souverain social.

PARTIE A : Calibration empirique d'un rendement r_f pour un portefeuille 60/40
           à partir d'ETF globaux via yfinance (ACWI / BNDX).

PARTIE B : Simulation Monte Carlo d'un fonds souverain :
           - Contribution annuelle = s_share % du PIB
           - Trois scénarios de rendement moyen (3 %, 5 %, 7 %)
           - Horizon = 40 ans
           - N trajectoires
"""

import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt

# ======================================================================
# Helper : récupération robuste d'une série de prix pour un ticker
# ======================================================================

def get_price_series(ticker, start, end=None):
    """
    Télécharge les prix via yfinance, compatible avec les formats :
    - colonnes simples : 'Adj Close', 'Close'
    - MultiIndex Price/Ticker : ('Adj Close','ACWI'), ('Close','ACWI'), ...
    Retourne une Series pandas contenant le prix ajusté.
    """
    data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)

    if data.empty:
        raise ValueError(f"Aucune donnée téléchargée pour le ticker {ticker}")

    cols = data.columns

    # --- CAS 1 : Index simple (pas MultiIndex) ---
    if not isinstance(cols, pd.MultiIndex):
        for c in ["Adj Close", "Close"]:
            if c in cols:
                return data[c]
        raise KeyError(f"Impossible de trouver 'Adj Close' ou 'Close' dans {list(cols)}")

    # --- CAS 2 : MultiIndex Price/Ticker (ton cas) ---
    price_level  = cols.names[0]  # "Price"
    # ticker_level = cols.names[1]  # pas nécessaire ici

    # On récupère les prix ajustés
    if "Adj Close" in cols.get_level_values(price_level):
        sub = data.xs("Adj Close", level=price_level, axis=1)
    elif "Close" in cols.get_level_values(price_level):
        sub = data.xs("Close", level=price_level, axis=1)
    else:
        raise KeyError("Ni 'Adj Close' ni 'Close' dans le niveau Price.")

    # sub est un DataFrame : colonnes = tickers
    if isinstance(sub, pd.DataFrame):
        if ticker in sub.columns:
            return sub[ticker]
        else:
            return sub.iloc[:, 0]
    else:
        return sub

# ======================================================================
# PARTIE A : CALIBRATION EMPIRIQUE DU PORTEFEUILLE 60/40
# ======================================================================

start = "2000-01-01"
end   = None  # jusqu'à aujourd'hui

equity_ticker = "ACWI"   # iShares MSCI ACWI (global equity)
bond_ticker   = "BNDX"   # Vanguard Total International Bond (global bonds, hedged)

print("Téléchargement des prix mensuels via yfinance...")

px_eq = get_price_series(equity_ticker, start=start, end=end)
px_bd = get_price_series(bond_ticker,   start=start, end=end)

# Passage en données mensuelles (dernier cours du mois)
px_eq_m = px_eq.resample("ME").last()   # 'ME' = month end, évite le FutureWarning
px_bd_m = px_bd.resample("ME").last()

# Alignement des dates communes
prices = pd.concat([px_eq_m, px_bd_m], axis=1, join="inner")
prices.columns = ["PX_EQ", "PX_BOND"]

# Rendements mensuels et portefeuille 60/40
ret_eq_m = prices["PX_EQ"].pct_change()
ret_bd_m = prices["PX_BOND"].pct_change()

returns_m = pd.concat([ret_eq_m, ret_bd_m], axis=1).dropna()
returns_m.columns = ["r_eq_m", "r_bd_m"]

w_eq   = 0.60
w_bond = 0.40
returns_m["r_port_m"] = w_eq * returns_m["r_eq_m"] + w_bond * returns_m["r_bd_m"]

# Agrégation en annuel (produit des (1+r_m) - 1)
returns_m["year"] = returns_m.index.year

def cumulate_to_annual(group):
    gross = (1.0 + group).prod()
    return gross - 1.0

annual = returns_m.groupby("year")["r_port_m"].apply(cumulate_to_annual).to_frame()
annual.columns = ["r_f"]

# On garde les années pleines (optionnel)
annual = annual.loc[annual.index >= 2003]

print("\n=== Moments empiriques annuels du portefeuille 60/40 ===")
print(annual["r_f"].describe())

# ----------------------------------------------------------------------
# Estimation AR(1) : r_t - mu = rho (r_{t-1} - mu) + eps_t
# ----------------------------------------------------------------------

r = annual["r_f"].values
r_lag   = r[:-1]
r_curr  = r[1:]

X = sm.add_constant(r_lag)   # ndarray -> params indexés par 0,1
model = sm.OLS(r_curr, X).fit()

rho_hat   = model.params[1]       # coefficient sur r_{t-1}
const_hat = model.params[0]       # constante
mu_hat    = const_hat / (1.0 - rho_hat)   # moyenne inconditionnelle
eps_hat   = model.resid
sigma_hat = np.std(eps_hat, ddof=1)

print("\n=== Estimation AR(1) pour r_f (annuel) ===")
print(model.summary())
print("\nParamètres calibrés (pour la simulation) :")
print(f"mu_rf    = {mu_hat:.4f}")
print(f"rho_rf   = {rho_hat:.4f}")
print(f"sigma_rf = {sigma_hat:.4f}")

# ======================================================================
# PARTIE B : SIMULATION DU FONDS SOUVERAIN
# ======================================================================

def simulate_r_f_ar1(mu_rf, rho_rf, sigma_rf, T, N):
    """
    Trajectoires AR(1) pour r_f :
    r_t - mu = rho (r_{t-1} - mu) + eps_t, eps_t ~ N(0, sigma^2)
    Sortie : matrice N x T de rendements annuels.
    """
    r = np.zeros((N, T))
    r[:, 0] = mu_rf
    for t in range(1, T):
        eps_t = np.random.normal(loc=0.0, scale=sigma_rf, size=N)
        r[:, t] = mu_rf + rho_rf * (r[:, t-1] - mu_rf) + eps_t
    return r

def simulate_fund_paths(mu_rf, rho_rf, sigma_rf,
                        T=40, N=10000,
                        g_mean=0.015, g_sigma=0.0,
                        s_share=0.015,
                        gdp0_eur=3000e9):
    """
    Simule N trajectoires sur T ans pour un fonds souverain :
    - Y_t : PIB réel (niveau, euros constants)
    - W_t : valeur du fonds (euros constants)
    - Contribution annuelle = s_share * Y_t
    - r_f simulé via AR(1)
    """
    r_f_paths = simulate_r_f_ar1(mu_rf, rho_rf, sigma_rf, T, N)

    Y = np.zeros((N, T+1))
    W = np.zeros((N, T+1))

    Y[:, 0] = gdp0_eur
    W[:, 0] = 0.0

    # croissance du PIB (ici déterministe, mais on peut mettre g_sigma > 0)
    g_t = np.random.normal(loc=g_mean, scale=g_sigma, size=(N, T))

    for t in range(T):
        Y[:, t+1] = Y[:, t] * (1.0 + g_t[:, t])
        contrib_t = s_share * Y[:, t]
        W[:, t+1] = (W[:, t] + contrib_t) * (1.0 + r_f_paths[:, t])

    WY = W / Y
    return {"Y": Y, "W": W, "WY": WY, "r_f": r_f_paths}

# Paramètres de simulation
T        = 40
N        = 10000
g_mean   = 0.015
g_sigma  = 0.0
s_share  = 0.015
gdp0_eur = 3000e9

# Trois scénarios de moyenne de r_f : 3 %, 5 %, 7 %
mu_list = [0.03, 0.05, 0.07]
results = {}

for mu_rf_scn in mu_list:
    print(f"\n--- Simulation fonds : scénario r_f moyen = {int(mu_rf_scn*100)} % ---")
    sim = simulate_fund_paths(mu_rf=mu_rf_scn,
                              rho_rf=rho_hat,
                              sigma_rf=sigma_hat,
                              T=T, N=N,
                              g_mean=g_mean, g_sigma=g_sigma,
                              s_share=s_share,
                              gdp0_eur=gdp0_eur)
    Y  = sim["Y"]
    W  = sim["W"]
    WY = sim["WY"]

    W_T  = W[:, -1]
    WY_T = WY[:, -1]

    q_W  = np.percentile(W_T,  [5, 50, 95])
    q_WY = np.percentile(WY_T, [5, 50, 95])
    q_WY_path = np.percentile(WY, [5, 50, 95], axis=0)

    results[mu_rf_scn] = {
        "Y": Y,
        "W": W,
        "WY": WY,
        "q_W": q_W,
        "q_WY": q_WY,
        "q_WY_path": q_WY_path
    }

# Résumé à l'horizon T
rows = []
for mu_rf_scn in mu_list:
    res = results[mu_rf_scn]
    q_W  = res["q_W"]
    q_WY = res["q_WY"]
    rows.append({
        "r_f moyen": f"{int(mu_rf_scn*100)} %",
        "W_T p5 (Mds €)" : q_W[0]  / 1e9,
        "W_T p50 (Mds €)": q_W[1]  / 1e9,
        "W_T p95 (Mds €)": q_W[2]  / 1e9,
        "W_T/Y_T p5"      : q_WY[0],
        "W_T/Y_T p50"     : q_WY[1],
        "W_T/Y_T p95"     : q_WY[2],
    })

summary = pd.DataFrame(rows)
print("\n=== Distribution du fonds à l'horizon T = {} ans ===".format(T))
print(summary.to_string(index=False,
                        float_format=lambda x: f"{x:,.2f}"))

# Fan chart comparatif
plt.figure(figsize=(10,6))
years = np.arange(T+1)

for mu_rf_scn in mu_list:
    res = results[mu_rf_scn]
    q_WY_path = res["q_WY_path"]

    plt.fill_between(years,
                     q_WY_path[0, :],
                     q_WY_path[2, :],
                     alpha=0.10,
                     label=f"{int(mu_rf_scn*100)} % – éventail 5–95 %")
    plt.plot(years,
             q_WY_path[1, :],
             linewidth=2,
             label=f"Médiane – {int(mu_rf_scn*100)} %")

plt.axhline(0.5, color="grey", linestyle="--", label="Référence : 50 % du PIB")
plt.xlabel("Années")
plt.ylabel("Ratio W_t / Y_t")
plt.title("Évolution simulée du ratio W_t/Y_t pour trois rendements moyens du fonds")
plt.legend(loc="upper left", ncol=2)
plt.tight_layout()
plt.show()
