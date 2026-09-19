"""
Swing-Data-Pipeline
Lädt Tagesdaten (Yahoo Finance) für Watchlist, Indizes, Sektoren und Makro,
berechnet Swing-Kennzahlen und schreibt:
  data/summary.csv  -> eine Zeile pro Ticker mit allen Kennzahlen
  data/ohlcv.csv    -> Kursdaten der letzten 300 Handelstage (Long-Format)
  data/meta.json    -> Zeitstempel, letzter Handelstag, Fehlschläge
"""
import json
import sys
import time
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).parent
DATA = ROOT / "data"
BENCH = "SPY"
OHLCV_BARS = 300

# Öffentliche, frei zugängliche S&P-500-Konstituentenliste (täglich aktuell genug für ein Screening)
SP500_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
CANDIDATE_LIQUIDITY_MIN_M = 20   # Mindest-Dollarvolumen (20T-Schnitt) in Mio. USD
NEWS_MOVE_THRESHOLD = 4.0        # ab dieser 1-Tages-Bewegung (%) wird News gezogen
NEWS_MAX_PER_TICKER = 3

# Fest eingebaut: Markt, Breite-Proxy, Sektoren, Makro, Credit, Commodities
MARKET = ["SPY", "QQQ", "IWM", "DIA", "RSP"]
SECTORS = ["XLK", "XLE", "XLB", "XLI", "XLF", "XLV", "XLY", "XLP",
           "XLU", "XLRE", "XLC", "SMH", "XME", "COPX", "GDX"]
MACRO = ["^VIX", "^VIX3M", "^TNX", "^IRX", "DX-Y.NYB",
         "TLT", "HYG", "LQD", "CL=F", "GC=F", "HG=F"]


def load_sp500() -> list[str]:
    """S&P-500-Ticker für das breite Screening. Bei Fehler: leere Liste (Pipeline läuft trotzdem weiter)."""
    try:
        df = pd.read_csv(SP500_URL)
        tickers = df["Symbol"].str.replace(".", "-", regex=False).str.strip().str.upper().tolist()
        return tickers
    except Exception as e:  # noqa: BLE001
        print(f"S&P-500-Liste konnte nicht geladen werden: {e}")
        return []


def load_watchlist() -> list[str]:
    tickers = []
    for line in (ROOT / "watchlist.txt").read_text(encoding="utf-8").splitlines():
        t = line.split("#")[0].strip().upper()
        if t:
            tickers.append(t)
    return tickers


def fetch(tickers: list[str]) -> tuple[dict, list]:
    data, failed = {}, []
    for i in range(0, len(tickers), 20):
        chunk = tickers[i:i + 20]
        df = None
        for attempt in range(3):
            try:
                df = yf.download(chunk, period="2y", interval="1d", auto_adjust=True,
                                 group_by="ticker", progress=False, threads=True)
                break
            except Exception as e:  # noqa: BLE001
                print(f"Download-Fehler (Versuch {attempt + 1}): {e}")
                time.sleep(10 * (attempt + 1))
        for t in chunk:
            try:
                sub = df[t] if isinstance(df.columns, pd.MultiIndex) else df
                sub = sub[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
                if len(sub) < 30:
                    raise ValueError("zu wenig Daten")
                sub.index = pd.to_datetime(sub.index).tz_localize(None)
                data[t] = sub
            except Exception:  # noqa: BLE001
                failed.append(t)
        time.sleep(2)
    return data, failed


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df.Close.shift()
    tr = pd.concat([df.High - df.Low, (df.High - pc).abs(), (df.Low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()  # Wilder


def pct(s: pd.Series, n: int) -> float:
    return (s.iloc[-1] / s.iloc[-1 - n] - 1) * 100 if len(s) > n else np.nan


def distribution_days(df: pd.DataFrame, window: int = 25) -> int:
    c, v = df.Close, df.Volume.replace(0, np.nan)
    dd = (c.pct_change() < -0.002) & (v > v.shift())
    return int(dd.tail(window).sum())


def score_candidate(r: pd.Series) -> float:
    """Grobes, deterministisches Scoring fuers breite Screening (0 bis 100).
    Bewusst einfacher als das manuelle Scoring aus der Chat-Analyse, aber gleiche Grundidee:
    Trend, relative Staerke, Nicht-Extended, saubere Struktur, gesundes Volumen, Liquiditaet."""
    if pd.isna(r.get("rs_1m_vs_spy")) or r.get("dollar_vol_20d_m", 0) < CANDIDATE_LIQUIDITY_MIN_M:
        return -1.0  # zu illiquide oder keine RS-Basis (z. B. Benchmark selbst)

    s = 0.0
    s += 14 if r.ma_stack_bull else (7 if r.close > r.sma200 else 0)          # Trend, 15
    s += max(0, min(15, 7 + 0.35 * r.rs_1m_vs_spy))                            # RS 1M, 15
    s += max(0, min(10, 5 + 0.25 * r.rs_3m_vs_spy))                            # RS 3M, 10
    ext = abs(r.ext_21ema_atr)                                                  # Extension, 15
    s += 13 if ext < 0.5 else (15 if ext < 1.5 else (7 if ext < 3 else 1))
    struct = 6
    if r.inside_day or r.nr7:
        struct += 5
    dist_to_high = (r.close / r.prior_20d_high - 1) * 100 if r.prior_20d_high else -99
    if -3 <= dist_to_high <= 1:
        struct += 4
    elif dist_to_high > 3:
        struct -= 3
    s += max(0, min(15, struct))                                               # Struktur, 15
    vr = r.vol_ratio_50d
    s += 8 if 1.1 <= vr <= 2.2 else (5 if vr < 1.1 else (3 if r.chg_1d_pct > 0 else 1))  # Volumen, 10
    s += max(0, min(10, 4 + 0.06 * -r.dist_52w_high_pct * -1 + 6))  # Naehe 52W-Hoch bevorzugt, 10 (grob)
    s += max(0, min(10, 10 - r.dist_days_25 * 0.7))                             # wenig Distribution, 10
    return round(min(100, max(0, s)), 1)


def fetch_news(tickers: list[str]) -> dict:
    """Holt aktuelle Schlagzeilen (yfinance, kein API-Key noetig) fuer eine kleine Liste von Tickern.
    Nur fuer Titel mit auffaelliger Tagesbewegung aufgerufen, nicht fuer das ganze Universum."""
    news = {}
    for t in tickers:
        try:
            items = yf.Ticker(t).news or []
        except Exception as e:  # noqa: BLE001
            print(f"News-Fehler {t}: {e}")
            continue
        cleaned = []
        for it in items[:NEWS_MAX_PER_TICKER]:
            c = it.get("content", it)  # yfinance-Schema variiert je nach Version
            title = c.get("title") or it.get("title")
            publisher = (c.get("provider") or {}).get("displayName") if isinstance(c.get("provider"), dict) else it.get("publisher")
            link = (c.get("canonicalUrl") or {}).get("url") if isinstance(c.get("canonicalUrl"), dict) else it.get("link")
            published = c.get("pubDate") or it.get("providerPublishTime")
            if title:
                cleaned.append({"title": title, "publisher": publisher, "link": link, "published": published})
        if cleaned:
            news[t] = cleaned
        time.sleep(0.5)
    return news


def metrics(t: str, df: pd.DataFrame, bench: pd.Series | None, group: str) -> dict:
    c, h, l, v = df.Close, df.High, df.Low, df.Volume.replace(0, np.nan)
    e10, e21 = c.ewm(span=10, adjust=False).mean(), c.ewm(span=21, adjust=False).mean()
    s50, s200 = c.rolling(50).mean(), c.rolling(200).mean()
    a = atr(df)
    last = -1
    rng = h - l

    row = {
        "ticker": t,
        "group": group,
        "date": c.index[last].date().isoformat(),
        "close": c.iloc[last],
        "chg_1d_pct": pct(c, 1),
        "ema10": e10.iloc[last],
        "ema21": e21.iloc[last],
        "sma50": s50.iloc[last],
        "sma200": s200.iloc[last],
        "ma_stack_bull": bool(c.iloc[last] > e10.iloc[last] > e21.iloc[last] > s50.iloc[last] > s200.iloc[last]),
        "ma_stack_bear": bool(c.iloc[last] < e10.iloc[last] < e21.iloc[last] < s50.iloc[last] < s200.iloc[last]),
        "atr14": a.iloc[last],
        "atr_pct": a.iloc[last] / c.iloc[last] * 100,
        "atr_vs_50d": a.iloc[last] / a.rolling(50).mean().iloc[last],
        "ext_21ema_atr": (c.iloc[last] - e21.iloc[last]) / a.iloc[last],
        "ext_50sma_atr": (c.iloc[last] - s50.iloc[last]) / a.iloc[last],
        "volume": df.Volume.iloc[last],
        "vol_ratio_50d": v.iloc[last] / v.rolling(50).mean().shift(1).iloc[last],
        "dollar_vol_20d_m": (c * v).rolling(20).mean().iloc[last] / 1e6,
        "close_pos_in_bar": (c.iloc[last] - l.iloc[last]) / rng.iloc[last] if rng.iloc[last] else np.nan,
        "prior_20d_high": h.iloc[-21:-1].max(),
        "prior_20d_low": l.iloc[-21:-1].min(),
        "high_52w": h.tail(252).max(),
        "low_52w": l.tail(252).min(),
        "dist_52w_high_pct": (c.iloc[last] / h.tail(252).max() - 1) * 100,
        "inside_day": bool(h.iloc[last] < h.iloc[-2] and l.iloc[last] > l.iloc[-2]),
        "nr7": bool(rng.iloc[last] <= rng.tail(7).min()),
        "perf_1m": pct(c, 21),
        "perf_3m": pct(c, 63),
        "perf_6m": pct(c, 126),
        "dist_days_25": distribution_days(df),
    }

    if bench is not None and t != BENCH:
        b = bench.reindex(c.index).ffill()
        rs = c / b
        row["rs_1m_vs_spy"] = row["perf_1m"] - pct(b, 21)
        row["rs_3m_vs_spy"] = row["perf_3m"] - pct(b, 63)
        row["rs_6m_vs_spy"] = row["perf_6m"] - pct(b, 126)
        rs_hi = rs.tail(252).max()
        row["rs_line_52w_high"] = bool(rs.iloc[last] >= rs_hi * 0.999)
        # RS-Linie auf 52W-Hoch, Kurs aber noch nicht: klassisches Stärkesignal
        row["rs_leads_price"] = bool(row["rs_line_52w_high"] and row["dist_52w_high_pct"] < -1)
    return row


def main() -> None:
    DATA.mkdir(exist_ok=True)
    watch = load_watchlist()
    sp500 = load_sp500()
    universe = [t for t in sp500 if t not in watch]  # breites Screening, Watchlist nicht doppelt zaehlen

    groups = {**{t: "macro" for t in MACRO}, **{t: "sector" for t in SECTORS},
              **{t: "market" for t in MARKET}, **{t: "watchlist" for t in watch},
              **{t: "universe" for t in universe}}
    tickers = list(dict.fromkeys(MARKET + SECTORS + MACRO + watch + universe))

    data, failed = fetch(tickers)
    if BENCH not in data:
        print("SPY konnte nicht geladen werden, Abbruch.")
        sys.exit(1)
    bench = data[BENCH].Close

    rows, bars = [], []
    for t, df in data.items():
        try:
            rows.append(metrics(t, df, bench, groups[t]))
        except Exception as e:  # noqa: BLE001
            print(f"Kennzahlen-Fehler {t}: {e}")
            failed.append(t)
            continue
        tail = df.tail(OHLCV_BARS).copy()
        tail.insert(0, "ticker", t)
        bars.append(tail)

    summary = pd.DataFrame(rows)
    order = {"market": 0, "macro": 1, "sector": 2, "watchlist": 3, "universe": 4}
    summary = summary.sort_values(["group", "ticker"], key=lambda s: s.map(order) if s.name == "group" else s)

    # Nur fuers breite Screening: Score berechnen, Top-Kandidaten separat ausgeben (nicht in summary.csv, bleibt schlank)
    uni_mask = summary.group == "universe"
    summary.loc[uni_mask, "candidate_score"] = summary.loc[uni_mask].apply(score_candidate, axis=1)
    candidates = (summary.loc[uni_mask & (summary.candidate_score >= 0)]
                  .sort_values("candidate_score", ascending=False).head(20))
    keep_cols = ["ticker", "candidate_score", "close", "chg_1d_pct", "ma_stack_bull", "ext_21ema_atr",
                 "rs_1m_vs_spy", "rs_3m_vs_spy", "inside_day", "nr7", "dist_52w_high_pct",
                 "vol_ratio_50d", "dollar_vol_20d_m", "dist_days_25"]
    candidates[keep_cols].round(3).to_csv(DATA / "candidates.csv", index=False)
    summary = summary.drop(columns=["candidate_score"])
    summary.round(3).to_csv(DATA / "summary.csv", index=False)

    ohlcv = pd.concat(bars).reset_index().rename(columns={"index": "Date"})
    ohlcv["Date"] = pd.to_datetime(ohlcv["Date"]).dt.date
    ohlcv.round(4).to_csv(DATA / "ohlcv.csv", index=False)

    # News nur fuer auffaellige Mover ziehen (Watchlist + Top-Kandidaten), nicht fuers ganze Universum
    movers = summary[(summary.group == "watchlist") & (summary.chg_1d_pct.abs() >= NEWS_MOVE_THRESHOLD)].ticker.tolist()
    movers += candidates[candidates.chg_1d_pct.abs() >= NEWS_MOVE_THRESHOLD].ticker.tolist()
    news = fetch_news(sorted(set(movers)))
    (DATA / "news.json").write_text(json.dumps(news, indent=2, ensure_ascii=False), encoding="utf-8")

    vix, vix3m = data.get("^VIX"), data.get("^VIX3M")
    meta = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes"),
        "last_bar": bench.index[-1].date().isoformat(),
        "watchlist": watch,
        "universe_size": len(universe),
        "loaded": len(data),
        "failed": sorted(set(failed)),
        "vix_term_ratio": round(float(vix.Close.iloc[-1] / vix3m.Close.iloc[-1]), 3)
        if vix is not None and vix3m is not None else None,
        "rsp_spy_rs_1m": round(pct(data["RSP"].Close, 21) - pct(bench, 21), 2) if "RSP" in data else None,
        "news_tickers": sorted(news.keys()),
    }
    (DATA / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

