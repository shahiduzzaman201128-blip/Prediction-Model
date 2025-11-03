
import pandas as pd, numpy as np
from sklearn.linear_model import Ridge
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
TZ = "Asia/Dhaka"

DEMAND_DF = pd.read_csv(DATA_DIR / "demand.csv", parse_dates=["timestamp_utc"]).set_index("timestamp_utc")
WEATHER_DF = pd.read_csv(DATA_DIR / "weather.csv", parse_dates=["timestamp_utc"]).set_index("timestamp_utc")
HOLIDAYS = set(pd.to_datetime(pd.read_csv(DATA_DIR / "holidays.csv")["holiday_date"]).dt.date)

_BASELINE = None

def _design(idx, is_holiday=None, temp_c=None, rh=None):
    hod = (idx.tz_convert(TZ).tz_localize(None).to_series(index=idx).dt.floor('h').dt.hour.astype('float32'))
    t = (hod / 24.0).astype('float32')
    four = {f"sin{k}": np.sin(k*2*np.pi*t) for k in range(1, 6)}
    four.update({f"cos{k}": np.cos(k*2*np.pi*t) for k in range(1, 6)})
    X = pd.DataFrame(four, index=idx)

    def _aligned(series_like, name):
        if series_like is None:
            return None
        try:
            s = pd.Series(series_like)
            if hasattr(series_like, "index"):
                s.index = series_like.index
        except Exception:
            s = pd.Series(series_like, index=idx)
        return s.reindex(X.index).astype("float32").to_frame(name=name)

    cols = []
    a_h = _aligned(is_holiday, "is_holiday")
    if a_h is not None: cols.append(a_h)
    a_t = _aligned(temp_c, "temp_c")
    if a_t is not None: cols.append(a_t)
    a_r = _aligned(rh, "rel_humidity")
    if a_r is not None: cols.append(a_r)
    if cols:
        X = X.join(pd.concat(cols, axis=1))
    return X.fillna(0.0).astype("float32")

def _fit_baseline():
    wx = WEATHER_DF.copy()
    idx = DEMAND_DF.index.intersection(wx.index)
    y = DEMAND_DF.loc[idx, "demand_mw"].astype("float32")
    local_days = idx.tz_convert(TZ).tz_localize(None).normalize()
    is_h = local_days.isin(pd.to_datetime(list(HOLIDAYS)))
    X = _design(idx, temp_c=wx.loc[idx, "temp_c"], rh=wx.loc[idx, "rel_humidity"], is_holiday=is_h)
    ridge = Ridge(alpha=5.0, fit_intercept=True, random_state=42)
    ridge.fit(X.values, y.values)
    return ridge

def ensure_ready():
    global _BASELINE
    if _BASELINE is None:
        _BASELINE = _fit_baseline()

def predict_range(start_utc, end_utc):
    ensure_ready()
    idx = pd.date_range(start_utc, end_utc, freq='h', tz='UTC')
    wx = WEATHER_DF.reindex(idx).interpolate(limit_direction='both')
    local_days = idx.tz_convert(TZ).tz_localize(None).normalize()
    is_h = local_days.isin(pd.to_datetime(list(HOLIDAYS)))
    X = _design(idx, temp_c=wx['temp_c'], rh=wx['rel_humidity'], is_holiday=is_h)
    yhat = _BASELINE.predict(X.values).astype('float32')
    return pd.DataFrame({'pred_demand_mw': yhat}, index=idx)
