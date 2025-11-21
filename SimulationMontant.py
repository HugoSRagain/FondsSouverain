# -*- coding: utf-8 -*-
"""
Academic calibration + Monte Carlo simulation of a social sovereign wealth fund (SWF).

PART A: Empirical calibration of an annual return r_f for a 60/40 global portfolio
        using ETFs from yfinance (ACWI / BNDX).

PART B: Monte Carlo simulation of a sovereign fund:
        - Annual contribution = s_share % of GDP
        - Three scenarios for the average real return (3%, 5%, 7%)
        - Horizon = 40 years
        - N paths
"""

import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt

# ======================================================================
# Helper: robust retrieval of a price series for a given ticker
# ======================================================================

def get_price_series(ticker, start, end=None):
    """
    Download prices via yfinance, compatible with:
    - single-level columns: 'Adj Close', 'Close'
    - MultiIndex Price/Ticker: ('Adj Close','ACWI'), ('Close','ACWI'), ...
    Returns a pandas Series with adjusted (or close) prices.
    """
    data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)

    if data.empty:
        raise ValueError(f"No data downloaded for ticker {ticker}")

    cols = data.columns

    # --- CASE 1: Simple Index (no MultiIndex) ---
    if not isinstance(cols, pd.MultiIndex):
        for c in ["Adj Close", "Close"]:
            if c in cols:
                return data[c]
        raise KeyError(f"Could not find 'Adj Close' or 'Close' in {list(cols)}")

    # --- CASE 2: MultiIndex Price/Ticker (as in your environment) ---
    price_level = cols.names[0]  # typically "Price"

    # We extract adjusted prices (or close prices as a fallback)
    if "Adj Close" in cols.get_level_values(price_level):
        sub = data.xs("Adj Close", level=price_level, axis=1)
    elif "Close" in cols.get_level_values(price_level):
        sub = data.xs("Close", level=price_level, axis=1)
    else:
        raise KeyError("Neither 'Adj Close' nor 'Close' found in the first level of the MultiIndex.")

    # 'sub' is a DataFrame: columns = tickers
    if isinstance(sub, pd.DataFrame):
        if ticker in sub.columns:
            return sub[ticker]
        else:
            # if the ticker column is not present explicitly, take the first one
            return sub.iloc[:, 0]
    else:
        return sub

# ======================================================================
# PART A: EMPIRICAL CALIBRATION OF THE 60/40 PORTFOLIO RETURN
# ======================================================================

start = "2000-01-01"
end   = None  # up to today

equity_ticker = "ACWI"   # iShares MSCI ACWI (global equity)
bond_ticker   = "BNDX"   # Vanguard Total International Bond (global bonds, hedged)

print("Downloading monthly prices via yfinance...")

px_eq = get_price_series(equity_ticker, start=start, end=end)
px_bd = get_price_series(bond_ticker,   start=start, end=end)

# Resample to monthly data (last price of each month)
px_eq_m = px_eq.resample("ME").last()   # 'ME' = month end (avoids FutureWarning)
px_bd_m = px_bd.resample("ME").last()

# Align on common dates
prices = pd.concat([px_eq_m, px_bd_m], axis=1, join="inner")
prices.columns = ["PX_EQ", "PX_BOND"]

# Monthly returns and 60/40 portfolio
ret_eq_m = prices["PX_EQ"].pct_change()
ret_bd_m = prices["PX_BOND"].pct_change()

returns_m = pd.concat([ret_eq_m, ret_bd_m], axis=1).dropna()
returns_m.columns = ["r_eq_m", "r_bd_m"]

w_eq   = 0.60
w_bond = 0.40
returns_m["r_port_m"] = w_eq * returns_m["r_eq_m"] + w_bond * returns_m["r_bd_m"]

# Aggregate to annual returns (product of (1 + r_m) - 1)
returns_m["year"] = returns_m.index.year

def cumulate_to_annual(group):
    gross = (1.0 + group).prod()
    return gross - 1.0

annual = returns_m.groupby("year")["r_port_m"].apply(cumulate_to_annual).to_frame()
annual.columns = ["r_f"]

# Optionally keep only full years (e.g., post-2003)
annual = annual.loc[annual.index >= 2003]

print("\n=== Empirical annual moments of the 60/40 portfolio ===")
print(annual["r_f"].describe())

# ----------------------------------------------------------------------
# AR(1) estimation: r_t - mu = rho (r_{t-1} - mu) + eps_t
# ----------------------------------------------------------------------

r      = annual["r_f"].values
r_lag  = r[:-1]
r_curr = r[1:]

X = sm.add_constant(r_lag)   # ndarray -> params indexed by [0] (const), [1] (r_{t-1})
model = sm.OLS(r_curr, X).fit()

rho_hat   = model.params[1]       # coefficient on r_{t-1}
const_hat = model.params[0]       # constant term
mu_hat    = const_hat / (1.0 - rho_hat)   # implied unconditional mean
eps_hat   = model.resid
sigma_hat = np.std(eps_hat, ddof=1)

print("\n=== AR(1) estimation for annual r_f ===")
print(model.summary())
print("\nCalibrated parameters for simulation:")
print(f"mu_rf    = {mu_hat:.4f}")
print(f"rho_rf   = {rho_hat:.4f}")
print(f"sigma_rf = {sigma_hat:.4f}")

# ======================================================================
# PART B: SOVEREIGN FUND MONTE CARLO SIMULATION
# ======================================================================

def simulate_r_f_ar1(mu_rf, rho_rf, sigma_rf, T, N):
    """
    Simulate N AR(1) paths for the annual real return r_f:
    r_t - mu = rho (r_{t-1} - mu) + eps_t,   eps_t ~ N(0, sigma^2)
    Returns an N x T matrix of annual returns.
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
    Simulate N paths over T years for a sovereign wealth fund:
    - Y_t: real GDP level (euros, constant prices)
    - W_t: fund value (euros, constant prices)
    - Annual contribution = s_share * Y_t
    - Annual real return r_f simulated via AR(1)
    """
    # Simulate returns
    r_f_paths = simulate_r_f_ar1(mu_rf, rho_rf, sigma_rf, T, N)

    # Allocate arrays for GDP and fund
    Y = np.zeros((N, T+1))
    W = np.zeros((N, T+1))

    Y[:, 0] = gdp0_eur
    W[:, 0] = 0.0

    # Deterministic (or stochastic) real GDP growth
    g_t = np.random.normal(loc=g_mean, scale=g_sigma, size=(N, T))

    for t in range(T):
        # GDP dynamics
        Y[:, t+1] = Y[:, t] * (1.0 + g_t[:, t])

        # Annual contribution in real euros
        contrib_t = s_share * Y[:, t]

        # Fund dynamics
        W[:, t+1] = (W[:, t] + contrib_t) * (1.0 + r_f_paths[:, t])

    # Fund-to-GDP ratio
    WY = W / Y
    return {"Y": Y, "W": W, "WY": WY, "r_f": r_f_paths}

# Simulation parameters
T        = 40
N        = 10000
g_mean   = 0.015   # 1.5% real GDP growth
g_sigma  = 0.0     # can be > 0 if you want stochastic GDP growth
s_share  = 0.015   # 1.5% of GDP contributed each year
gdp0_eur = 3000e9  # initial real GDP (3 000 bn euros)

# Three scenarios for the mean real return: 3%, 5%, 7%
mu_list = [0.03, 0.05, 0.07]
results = {}

for mu_rf_scn in mu_list:
    print(f"\n--- Fund simulation: scenario mean r_f = {int(mu_rf_scn*100)} % ---")
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

    # Distributions at horizon T
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

# Summary at horizon T
rows = []
for mu_rf_scn in mu_list:
    res = results[mu_rf_scn]
    q_W  = res["q_W"]
    q_WY = res["q_WY"]
    rows.append({
        "Mean r_f": f"{int(mu_rf_scn*100)} %",
        "W_T p5 (bn €)"  : q_W[0]  / 1e9,
        "W_T p50 (bn €)" : q_W[1]  / 1e9,
        "W_T p95 (bn €)" : q_W[2]  / 1e9,
        "W_T/Y_T p5"     : q_WY[0],
        "W_T/Y_T p50"    : q_WY[1],
        "W_T/Y_T p95"    : q_WY[2],
    })

summary = pd.DataFrame(rows)
print("\n=== Fund distribution at horizon T = {} years ===".format(T))
print(summary.to_string(index=False,
                        float_format=lambda x: f"{x:,.2f}"))

# Fan chart comparing the three return scenarios
plt.figure(figsize=(10, 6))
years = np.arange(T+1)

for mu_rf_scn in mu_list:
    res = results[mu_rf_scn]
    q_WY_path = res["q_WY_path"]

    # 5–95% band
    plt.fill_between(years,
                     q_WY_path[0, :],
                     q_WY_path[2, :],
                     alpha=0.10,
                     label=f"{int(mu_rf_scn*100)}% – 5–95% band")
    # median
    plt.plot(years,
             q_WY_path[1, :],
             linewidth=2,
             label=f"Median – {int(mu_rf_scn*100)}%")

plt.axhline(0.5, color="grey", linestyle="--", label="Reference: 50% of GDP")
plt.xlabel("Years")
plt.ylabel("Fund-to-GDP ratio $W_t / Y_t$")
plt.title("Simulated evolution of the fund-to-GDP ratio $W_t/Y_t$\nfor three mean real-return scenarios")
plt.legend(loc="upper left", ncol=2)
plt.tight_layout()
plt.show()
