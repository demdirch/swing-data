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
