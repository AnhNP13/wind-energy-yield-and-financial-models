import numpy as np
import pandas as pd
from pathlib import Path
from math import gamma

# ===================== Config =====================
MEASURE_FILE     = Path(r"C:\Users\anhnp\OneDrive\2024 EUREC_Hanze\S3. Thesis\Data playing\Python\ws_measure.xlsx")
HOR_FACTOR_FILE  = Path(r"C:\Users\anhnp\OneDrive\2024 EUREC_Hanze\S3. Thesis\Data playing\Python\ws_hor.extrapolation.xlsx")
POWER_CURVE_FILE = Path(r"C:\Users\anhnp\OneDrive\2024 EUREC_Hanze\S3. Thesis\Data playing\Python\ws_powercurve.xlsx")
OUTPUT_FILE      = Path(r"C:\Users\anhnp\OneDrive\2024 EUREC_Hanze\S3. Thesis\Data playing\Python\AEP_by_run_4UAV.xlsx")

MEASURE_HAS_HEADER = False
MEASURE_COLS       = ["periode", "wind_speed", "wind_direction"]

NUM_TURBINES   = 20
TURBINE_PREFIX = "T"
NUM_SECTORS    = 12
DEG_PER_SECTOR = 360 // NUM_SECTORS
HOURS_PER_YEAR = 8760.0
BIN_WIDTH      = 0.5

RUNS     = 5000
RNG_SEED = None

# Measurement uncertainty (per-run, additive)
WS_SIGMA_ABS = 0.5   # m/s
WD_SIGMA_DEG = 10.0    # deg

# Losses (%)
WAKE_LOSS = 0.08
WAKE_AFFECTED_TURBS = {
    "T3","T4","T5","T6","T7","T8","T9","T10","T11","T12","T13","T15","T16","T17"
}
AVAIL_LOSS = 0.03
ELEC_LOSS  = 0.02
CURT_LOSS  = 0.01
ICING_LOSS = 0.01

# ===================== Utils =====================
try:
    from numpy import trapezoid as _trapint
except ImportError:
    from numpy import trapz as _trapint

def choose_excel_engine():
    for eng in ("xlsxwriter", "openpyxl"):
        try: __import__(eng); return eng
        except Exception: pass
    raise RuntimeError("Install 'xlsxwriter' or 'openpyxl'.")

def to_number(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.replace("°","",regex=False).str.replace(",",".",regex=False)
    s = s.str.extract(r"(-?\d+(?:\.\d+)?)", expand=False)
    return pd.to_numeric(s, errors="coerce")

def sectorize_series(wd_deg: pd.Series, deg_per_sector=30, start_edge_deg=15.0) -> pd.Series:
    wd = pd.to_numeric(wd_deg, errors="coerce") % 360
    n = 360 // deg_per_sector
    sec = np.ceil(((wd - start_edge_deg) % 360) / float(deg_per_sector))
    sec = (sec.astype(int) % n) + 1
    out = pd.Series(sec, index=wd_deg.index)
    out[pd.isna(wd_deg)] = pd.NA
    return out.astype("Int64")

def sectorize_numpy(wd_deg_np: np.ndarray, deg_per_sector=30, start_edge_deg=15.0) -> np.ndarray:
    wd = np.asarray(wd_deg_np, float) % 360
    n = 360 // deg_per_sector
    sec = np.ceil(((wd - start_edge_deg) % 360) / float(deg_per_sector))
    sec = (sec.astype(int) % n) + 1
    return sec.astype(np.int64)

def weibull_fit_moments(v: np.ndarray):
    v = np.asarray(v, float)
    v = v[np.isfinite(v) & (v >= 0)]
    if v.size == 0: return 1.0, 0.0
    mu, sigma = v.mean(), v.std(ddof=0)
    if mu <= 0: return 1.0, 0.0
    if sigma <= 0: return 10.0, mu
    k = float(np.clip((sigma/mu)**-1.086, 0.1, 25.0))
    c = mu / gamma(1.0 + 1.0/k)
    return k, float(c)

def weibull_pdf(v: np.ndarray, k: float, c: float):
    v = np.asarray(v, float)
    if not (k>0 and c>0 and np.isfinite(k) and np.isfinite(c)): return np.zeros_like(v)
    pdf = (k/c) * (v/c)**(k-1) * np.exp(-(v/c)**k)
    pdf[v < 0] = 0.0
    area = _trapint(pdf, v)
    return pdf/area if (np.isfinite(area) and area > 0) else np.zeros_like(v)

def expected_power_from_pc(k: float, c: float, pc_df: pd.DataFrame, step: float):
    if not (np.isfinite(k) and np.isfinite(c) and k>0 and c>0) or pc_df is None or pc_df.empty: return 0.0
    vmax = max(40.0, float(pc_df["wind_speed"].max()) + 5.0)
    v = np.arange(0.0, vmax + step, step)
    p = np.interp(v, pc_df["wind_speed"].values, pc_df["power_kW"].values, left=0.0, right=0.0)
    pdf = weibull_pdf(v, k, c)
    return float(max(_trapint(p*pdf, v), 0.0))

def expected_power_sector_from_speeds(v_sector: np.ndarray, pc_df: pd.DataFrame, step: float):
    v = np.asarray(v_sector, float)
    v = v[np.isfinite(v) & (v >= 0)]
    if v.size == 0: return 0.0
    if v.size < 12:
        return float(np.interp(np.clip(v, 0, None), pc_df["wind_speed"].values, pc_df["power_kW"].values, left=0, right=0).mean())
    k, c = weibull_fit_moments(v)
    return expected_power_from_pc(k, c, pc_df, step)

def canon_tid(x: str) -> str:
    s = str(x).strip()
    if not s: return s
    return (s if s[0].upper() == TURBINE_PREFIX else f"{TURBINE_PREFIX}{s}").upper()

# ===================== IO =====================
def read_measurements(path: Path, has_header: bool, cols: list) -> pd.DataFrame:
    df = (pd.read_excel(path) if has_header else pd.read_excel(path, header=None).iloc[:, :3].rename(columns=dict(zip(range(3), cols))))
    if has_header:
        rename_map = {}
        for c in df.columns:
            cl = str(c).strip().lower()
            if "periode" in cl or "period" in cl or "time" in cl: rename_map[c] = "periode"
            elif ("wind" in cl and "speed" in cl) or cl == "ws": rename_map[c] = "wind_speed"
            elif ("wind" in cl and "dir" in cl) or "richtung" in cl: rename_map[c] = "wind_direction"
        df = df.rename(columns=rename_map)
    need = {"periode","wind_speed","wind_direction"}
    if not need.issubset(df.columns): raise ValueError("Measurement file must contain: periode, wind_speed, wind_direction.")
    df["wind_speed"]     = to_number(df["wind_speed"])
    df["wind_direction"] = to_number(df["wind_direction"])
    df = df.dropna(subset=["wind_speed","wind_direction"]).reset_index(drop=True)
    df["sector"] = sectorize_series(df["wind_direction"], DEG_PER_SECTOR, 15.0)
    return df[["periode","wind_speed","wind_direction","sector"]]

def read_hor_factors(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    cols = {c: str(c).strip().lower() for c in df.columns}
    sector_col = next((c for c in df.columns if cols[c] in {"sector","sektor"}), df.columns[0])
    df = df.rename(columns={sector_col:"Sector"})
    want = ["Sector"] + [f"{TURBINE_PREFIX}{i}" for i in range(1, NUM_TURBINES+1)]
    rename = {}
    for i in range(1, NUM_TURBINES+1):
        want_lower = f"{TURBINE_PREFIX}{i}".lower()
        cand = [c for c in df.columns if str(c).strip().lower() == want_lower]
        if cand: rename[cand[0]] = f"{TURBINE_PREFIX}{i}"
    df = df.rename(columns=rename)
    miss = [c for c in want if c not in df.columns]
    if miss: raise ValueError(f"Horizontal factor file missing columns: {miss}")
    df["Sector"] = pd.to_numeric(df["Sector"], errors="coerce").astype(int)
    df = df[want].set_index("Sector").sort_index().astype(float).fillna(1.0)
    missing_sec = [s for s in range(1, NUM_SECTORS+1) if s not in df.index]
    if missing_sec: raise ValueError(f"Missing sectors: {missing_sec}. Expected 1..{NUM_SECTORS}.")
    return df

def read_power_curve(path: Path) -> pd.DataFrame:
    pc = pd.read_excel(path, header=None).iloc[:, :2]
    pc.columns = ["wind_speed","power_kW"]
    pc = pc.apply(pd.to_numeric, errors="coerce").dropna()
    pc = pc[pc["wind_speed"] >= 0].sort_values("wind_speed").drop_duplicates("wind_speed").reset_index(drop=True)
    if pc.empty: raise ValueError("Power curve is empty/invalid.")
    return pc

# ===================== Core =====================
def aep_by_run(df_meas: pd.DataFrame, df_hor: pd.DataFrame, pc_df: pd.DataFrame):
    ws_base = df_meas["wind_speed"].to_numpy(float)
    wd_base = (df_meas["wind_direction"].to_numpy(float) % 360.0)
    n = ws_base.size
    rng = np.random.default_rng(RNG_SEED)
    all_rows, wb_rows = [], []

    farm_afterwake_by_run = {r: 0.0 for r in range(1, RUNS+1)}

    for i in range(1, NUM_TURBINES+1):
        t = f"{TURBINE_PREFIX}{i}"

        for r in range(RUNS):
            run_id = r + 1
            # 1) Add measurement uncertainty 
            wd_noise = rng.normal(0.0, WD_SIGMA_DEG, size=n)
            wd_new   = (wd_base + wd_noise) % 360.0

            sector_run = sectorize_numpy(wd_new, DEG_PER_SECTOR, 15.0)

            hfac_run   = df_hor.loc[sector_run, t].to_numpy()
            v_nominal  = ws_base * hfac_run

            ws_noise   = rng.normal(0.0, WS_SIGMA_ABS, size=n)  # add WS noise after factor 
            v          = np.clip(v_nominal + ws_noise, 0.0, None)

            eP_total = 0.0
            for s in range(1, NUM_SECTORS+1):
                idx = np.where(sector_run == s)[0]
                if idx.size == 0:
                    wb_rows.append({"run": run_id, "turbine": t, "sector": s, "n": 0,
                                    "k(weibull)": np.nan, "c(scale_Weibull)": np.nan})
                    continue
                v_s = v[idx]; w_s = idx.size / n
                if v_s.size >= 12:
                    k_s, c_s = weibull_fit_moments(v_s)
                    eP_s = expected_power_from_pc(k_s, c_s, pc_df, BIN_WIDTH)
                else:
                    k_s, c_s = (np.nan, np.nan)
                    eP_s = expected_power_sector_from_speeds(v_s, pc_df, BIN_WIDTH)
                eP_total += w_s * eP_s
                wb_rows.append({"run": run_id, "turbine": t, "sector": s, "n": int(v_s.size),
                                "k(weibull)": k_s, "c(scale_Weibull)": c_s})

            gross_mwh = eP_total * HOURS_PER_YEAR / 1_000.0
            if t in WAKE_AFFECTED_TURBS:
                after_wake_mwh = gross_mwh * (1.0 - WAKE_LOSS)
                wake_applied = WAKE_LOSS
            else:
                after_wake_mwh = gross_mwh
                wake_applied = 0.0
            
            farm_afterwake_by_run[run_id] += after_wake_mwh

            all_rows.append({
                "run": run_id,
                "turbine": t,
                "Gross_AEP (MWh)": gross_mwh,
                "Wake_Loss_Applied (%)": wake_applied,
                "After_Wake (MWh)": after_wake_mwh,
            })

    df_all = pd.DataFrame(all_rows)
    df_pivot = df_all.pivot(index="run", columns="turbine", values="After_Wake (MWh)").reset_index()

    net_factor_add = 1.0 - (AVAIL_LOSS + ELEC_LOSS + CURT_LOSS + ICING_LOSS)
    df_farm = (
        pd.DataFrame({
            "run": list(farm_afterwake_by_run.keys()),
            "Farm_Gross_AEP_afterwake (MWh)": list(farm_afterwake_by_run.values())
        })
        .sort_values("run")
        .assign(**{"Farm_Net_AEP (MWh)": lambda d: d["Farm_Gross_AEP_afterwake (MWh)"] * net_factor_add})
    )

    df_weib = pd.DataFrame(wb_rows).sort_values(["run","turbine","sector"]).reset_index(drop=True)

    return df_pivot, df_weib, df_farm

# ===================== Run =====================
def main():
    df_meas = read_measurements(MEASURE_FILE, MEASURE_HAS_HEADER, MEASURE_COLS)
    df_hor  = read_hor_factors(HOR_FACTOR_FILE)
    pc_df   = read_power_curve(POWER_CURVE_FILE)

    df_pivot, df_weib, df_farm = aep_by_run(df_meas, df_hor, pc_df)

    engine = choose_excel_engine()
    with pd.ExcelWriter(OUTPUT_FILE, engine=engine) as w:
        df_pivot.to_excel(w, sheet_name="AfterWake AEP_ALL", index=False)
        df_farm.to_excel(w,  sheet_name="Farm AEP", index=False)
        #df_weib.to_excel(w,  sheet_name="Weibull_By_Sector", index=False)

    print(f"✓ Done. Wrote {OUTPUT_FILE}")
    print("Sheets:")
    print(" - AfterWake AEP_ALL (run × turbine)")
    print(" - Farm AEP")
    #print(" - Weibull_By_Sector")

if __name__ == "__main__":
    main()