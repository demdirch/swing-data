"""
4h-Erweiterung der Swing-Data-Pipeline
Lädt 1h-Kerzen (Yahoo Finance, regulaere Session), baut daraus 4h-Kerzen im
TradingView-Schema fuer US-Aktien (09:30-13:30 und 13:30-16:00 ET) und schreibt:
  data/ohlcv_4h.csv    -> letzte 4h-Kerzen je Ticker (Long-Format)
  data/summary_4h.csv  -> eine Zeile pro Ticker mit 4h-Marktstruktur
Laeuft als eigener Workflow-Schritt: Fehler hier stoppen die Tages-Pipeline nicht.
"""
import json
import time
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from fetch_data import load_watchlist, atr, MARKET, SECTORS

ROOT = Path(__file__).parent
DATA = ROOT / "data"
PERIOD = "6mo"        # yfinance erlaubt 1h bis 730 Tage, 6 Monate reichen fuer Swing-Struktur
BARS_OUT = 200        # ca. 100 Handelstage an 4h-Kerzen je Ticker
PIVOT_N = 3           # Fraktal: Pivot braucht je 3 Kerzen links und rechts
TZ = "America/New_York"


def fetch_1h(tickers: list[str]) -> tuple[dict, list]:
    data, failed = {}, []
    for i in range(0, len(tickers), 15):
        chunk = tickers[i:i + 15]
        df = None
        for attempt in range(3):
            try:
                df = yf.download(chunk, period=PERIOD, interval="1h", auto_adjust=True,
                                 prepost=False, group_by="ticker", progress=False, threads=True)
                break
            except Exception as e:  # noqa: BLE001
                print(f"1h-Download-Fehler (Versuch {attempt + 1}): {e}")
                time.sleep(10 * (attempt + 1))
        for t in chunk:
            try:
                sub = df[t] if isinstance(df.columns, pd.MultiIndex) else df
                sub = sub[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
                if len(sub) < 50:
                    raise ValueError("zu wenig Daten")
                idx = pd.to_datetime(sub.index)
                idx = idx.tz_localize("UTC") if idx.tz is None else idx
                sub.index = idx.tz_convert(TZ)
                data[t] = sub
            except Exception:  # noqa: BLE001
                failed.append(t)
        time.sleep(2)
    return data, failed


def to_4h(h1: pd.DataFrame) -> pd.DataFrame:
    """1h -> 4h im Session-Schema: Block A 09:30-13:30, Block B 13:30-16:00 (ET)."""
    minutes = h1.index.hour * 60 + h1.index.minute
    block = np.where(minutes < 13 * 60 + 30, "09:30", "13:30")
    key = pd.to_datetime(h1.index.date.astype(str) + " " + block)
    g = h1.groupby(key)
    out = pd.DataFrame({
        "Open": g.Open.first(), "High": g.High.max(), "Low": g.Low.min(),
        "Close": g.Close.last(), "Volume": g.Volume.sum(),
    })
    out.index.name = "Bar_ET"
    return out


def pivots(df: pd.DataFrame, n: int = PIVOT_N) -> tuple[pd.Series, pd.Series]:
    """Bestaetigte Swing Highs/Lows (Fraktal). Die letzten n Kerzen koennen noch keinen Pivot bilden."""
    h, l = df.High, df.Low
    win = 2 * n + 1
    ph = h[(h == h.rolling(win, center=True).max())].dropna()
    pl = l[(l == l.rolling(win, center=True).min())].dropna()
    return ph, pl


def structure_4h(t: str, df: pd.DataFrame) -> dict:
    c, h, l, v = df.Close, df.High, df.Low, df.Volume.replace(0, np.nan)
    e21, e50 = c.ewm(span=21, adjust=False).mean(), c.ewm(span=50, adjust=False).mean()
    a = atr(df)
    ph, pl = pivots(df)
    last = c.iloc[-1]

    row = {
        "ticker": t,
        "last_bar_et": df.index[-1].strftime("%Y-%m-%d %H:%M"),
        "close": last,
        "ema21_4h": e21.iloc[-1],
        "ema50_4h": e50.iloc[-1],
        "above_ema21": bool(last > e21.iloc[-1]),
        "ema21_slope_5": (e21.iloc[-1] / e21.iloc[-6] - 1) * 100 if len(e21) > 6 else np.nan,
        "atr14_4h": a.iloc[-1],
        "ext_ema21_atr": (last - e21.iloc[-1]) / a.iloc[-1],
        "vol_ratio_20": v.iloc[-1] / v.rolling(20).mean().shift(1).iloc[-1],
        "close_pos_in_bar": (last - l.iloc[-1]) / (h.iloc[-1] - l.iloc[-1]) if h.iloc[-1] > l.iloc[-1] else np.nan,
    }

    ph_last = ph.iloc[-1] if len(ph) else np.nan
    ph_prev = ph.iloc[-2] if len(ph) > 1 else np.nan
    pl_last = pl.iloc[-1] if len(pl) else np.nan
    pl_prev = pl.iloc[-2] if len(pl) > 1 else np.nan
    row.update({
        "pivot_high": ph_last, "pivot_high_prev": ph_prev,
        "pivot_high_time": ph.index[-1].strftime("%Y-%m-%d %H:%M") if len(ph) else None,
        "pivot_low": pl_last, "pivot_low_prev": pl_prev,
        "pivot_low_time": pl.index[-1].strftime("%Y-%m-%d %H:%M") if len(pl) else None,
    })

    hh, hl = ph_last > ph_prev, pl_last > pl_prev
    if hh and hl:
        struct = "bull HH/HL"
    elif not hh and not hl:
        struct = "bear LH/LL"
    elif hh and not hl:
        struct = "expanding HH/LL"
    else:
        struct = "contracting LH/HL"
    row["structure"] = struct

    # Break of Structure: Schluss jenseits des letzten bestaetigten Pivots
    row["bos_up"] = bool(last > ph_last) if pd.notna(ph_last) else False
    row["bos_down"] = bool(last < pl_last) if pd.notna(pl_last) else False
    row["dist_pivot_high_atr"] = (ph_last - last) / a.iloc[-1] if pd.notna(ph_last) else np.nan
    row["dist_pivot_low_atr"] = (last - pl_last) / a.iloc[-1] if pd.notna(pl_last) else np.nan
    return row


def main() -> None:
    DATA.mkdir(exist_ok=True)
    extra = ["SMH", "XLK", "XLC", "XLE", "XLI", "XLB"]
    tickers = list(dict.fromkeys(MARKET + [s for s in SECTORS if s in extra] + load_watchlist()))

    h1, failed = fetch_1h(tickers)
    rows, bars = [], []
    for t, df in h1.items():
        try:
            b4 = to_4h(df)
            rows.append(structure_4h(t, b4))
            tail = b4.tail(BARS_OUT).copy()
            tail.insert(0, "ticker", t)
            bars.append(tail)
        except Exception as e:  # noqa: BLE001
            print(f"4h-Fehler {t}: {e}")
            failed.append(t)

    if not rows:
        print("Keine 4h-Daten erzeugt.")
        return
    pd.DataFrame(rows).round(3).to_csv(DATA / "summary_4h.csv", index=False)
    pd.concat(bars).reset_index().round(4).to_csv(DATA / "ohlcv_4h.csv", index=False)

    info = {"generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes"),
            "loaded": len(rows), "failed": sorted(set(failed))}
    (DATA / "meta_4h.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
