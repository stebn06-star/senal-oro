"""
Modelo de Trading Intradia para Oro (XAU/USD)
================================================
Combina analisis tecnico (EMA, RSI, MACD, VWAP, ATR, Bollinger, pivotes)
con analisis de noticias/sentimiento (diccionario de terminos relevantes
para el oro) para generar senales de COMPRA / VENTA / NEUTRAL, con
gestion de riesgo automatica (stop-loss, take-profit, tamano de posicion).

REQUISITOS:
    pip install yfinance pandas numpy requests python-dotenv

VARIABLES DE ENTORNO (crear archivo .env, ver .env.example):
    MARKETAUX_API_KEY=tu_api_key_de_marketaux.com   (gratis, 100 requests/dia)
    TELEGRAM_BOT_TOKEN=tu_token_de_bot               (opcional, para alertas)
    TELEGRAM_CHAT_ID=tu_chat_id                      (opcional, para alertas)

USO:
    python gold_trading_model.py --once             # reporte simple: accion, entrada, SL, TP
    python gold_trading_model.py --once --detallado  # + desglose tecnico y noticias
    python gold_trading_model.py --loop              # corre en bucle cada N minutos
    python gold_trading_model.py --backtest          # backtest simple con datos historicos

NOTA: marketaux.com es una API de noticias financieras (no generica como
NewsAPI) que ademas entrega su propio sentiment_score por entidad. Este
script sigue usando el diccionario GOLD_LEXICON propio sobre el titulo/
descripcion para mantener el criterio especifico "bueno/malo para el
oro" -- el sentiment_score de marketaux queda disponible en la respuesta
si mas adelante quieres incorporarlo tambien.

⚠️ Este script es una herramienta de apoyo EDUCATIVA. No garantiza
rentabilidad y no es asesoria financiera. Pruebalo en cuenta demo antes
de arriesgar capital real.
"""

import os
import time
import argparse
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import requests

try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# =========================================================================
# CONFIGURACION
# =========================================================================

CONFIG = {
    "symbol": "GC=F",          # Futuros de oro COMEX en Yahoo Finance
    "interval": "15m",         # Timeframe intradia
    "period": "5d",            # Ventana historica a descargar
    "peso_tecnico": 0.65,
    "peso_fundamental": 0.35,
    "umbral_compra": 40,
    "umbral_venta": -40,
    "riesgo_por_operacion_pct": 0.75,   # % de la cuenta por operacion
    "atr_multiplicador_sl": 1.5,
    "ratio_riesgo_beneficio": 2.0,
    "balance_cuenta": 10000,   # ejemplo, ajustar a tu cuenta real
    "minutos_entre_analisis": 15,
    "noticias_query": 'gold|XAUUSD|"Federal Reserve"|inflation|"interest rate"',
    "noticias_max": 30,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("gold_model")


# =========================================================================
# 1. OBTENCION DE DATOS DE PRECIO
# =========================================================================

def fetch_price_data(symbol=None, interval=None, period=None):
    """Descarga velas OHLCV desde Yahoo Finance."""
    if yf is None:
        raise ImportError("Instala yfinance: pip install yfinance")

    symbol = symbol or CONFIG["symbol"]
    interval = interval or CONFIG["interval"]
    period = period or CONFIG["period"]

    df = yf.download(symbol, interval=interval, period=period, progress=False)
    if df.empty:
        raise ValueError(f"No se recibieron datos para {symbol}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.rename(columns=str.lower)
    df.index.name = "datetime"
    return df


# =========================================================================
# 2. INDICADORES TECNICOS
# =========================================================================

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series, fast=12, slow=26, signal=9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def atr(df, period=14):
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def bollinger_bands(series, period=20, num_std=2):
    mid = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


def vwap_session(df):
    """VWAP reiniciado cada dia (aproximacion util para intradia)."""
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    day = df.index.date
    cum_vol = df.groupby(day)["volume"].cumsum()
    cum_vol_price = (typical_price * df["volume"]).groupby(day).cumsum()
    return cum_vol_price / cum_vol.replace(0, np.nan)


def daily_pivots(df):
    """Pivotes clasicos calculados con la vela diaria previa completa."""
    daily = df.resample("1D").agg({"high": "max", "low": "min", "close": "last"}).dropna()
    if len(daily) < 2:
        return None
    prev = daily.iloc[-2]
    pivot = (prev["high"] + prev["low"] + prev["close"]) / 3
    r1 = 2 * pivot - prev["low"]
    s1 = 2 * pivot - prev["high"]
    r2 = pivot + (prev["high"] - prev["low"])
    s2 = pivot - (prev["high"] - prev["low"])
    return {"pivot": pivot, "r1": r1, "s1": s1, "r2": r2, "s2": s2}


def add_indicators(df):
    df = df.copy()
    df["ema9"] = ema(df["close"], 9)
    df["ema21"] = ema(df["close"], 21)
    df["rsi14"] = rsi(df["close"], 14)
    df["macd"], df["macd_signal"], df["macd_hist"] = macd(df["close"])
    df["atr14"] = atr(df, 14)
    df["bb_upper"], df["bb_mid"], df["bb_lower"] = bollinger_bands(df["close"])
    if "volume" in df.columns and df["volume"].sum() > 0:
        df["vwap"] = vwap_session(df)
    else:
        df["vwap"] = np.nan
    return df


def technical_score(df):
    """
    Convierte el ultimo estado de los indicadores en un score de -100 a +100.
    """
    last = df.iloc[-1]
    score = 0.0
    detalles = []

    if last["ema9"] > last["ema21"]:
        score += 20
        detalles.append("EMA9 > EMA21 (tendencia alcista de corto plazo) +20")
    else:
        score -= 20
        detalles.append("EMA9 < EMA21 (tendencia bajista de corto plazo) -20")

    vwap_val = last.get("vwap", np.nan)
    if pd.notna(vwap_val):
        if last["close"] > vwap_val:
            score += 15
            detalles.append("Precio > VWAP (sesgo comprador intradia) +15")
        else:
            score -= 15
            detalles.append("Precio < VWAP (sesgo vendedor intradia) -15")

    rsi_val = last["rsi14"]
    if pd.notna(rsi_val):
        if rsi_val < 30:
            score += 15
            detalles.append(f"RSI={rsi_val:.1f} en sobreventa (posible rebote) +15")
        elif rsi_val > 70:
            score -= 15
            detalles.append(f"RSI={rsi_val:.1f} en sobrecompra (posible correccion) -15")
        else:
            contrib = (rsi_val - 50) / 50 * 10
            score += contrib
            detalles.append(f"RSI={rsi_val:.1f} zona neutral, contribucion {contrib:+.1f}")

    if pd.notna(last["macd"]) and pd.notna(last["macd_signal"]):
        if last["macd"] > last["macd_signal"]:
            score += 15
            detalles.append("MACD > senal (momentum alcista) +15")
        else:
            score -= 15
            detalles.append("MACD < senal (momentum bajista) -15")

    if pd.notna(last["bb_lower"]) and last["close"] <= last["bb_lower"]:
        score += 15
        detalles.append("Precio en banda inferior de Bollinger (posible rebote) +15")
    elif pd.notna(last["bb_upper"]) and last["close"] >= last["bb_upper"]:
        score -= 15
        detalles.append("Precio en banda superior de Bollinger (posible reversion) -15")

    score = max(-100, min(100, score))
    return score, detalles


# =========================================================================
# 3. NOTICIAS Y SENTIMIENTO
# =========================================================================

GOLD_LEXICON = {
    # Termino: peso (positivo = alcista para el oro, negativo = bajista)
    "rate cut": 20, "rate cuts": 20, "dovish": 18, "cuts rates": 20,
    "inflation surge": 15, "inflation rises": 12, "inflation jumps": 15,
    "safe haven": 15, "safe-haven": 15, "flight to safety": 18,
    "geopolitical tension": 10, "war": 8, "conflict escalates": 10,
    "recession fears": 15, "recession risk": 15,
    "dollar weakens": 18, "dollar falls": 18, "dollar slides": 15,
    "yields fall": 15, "yields drop": 15,
    "central bank buying": 18, "gold reserves": 10,
    "market uncertainty": 10, "stocks fall": 8, "stock market crash": 20,
    "rate hike": -20, "rate hikes": -20, "hawkish": -18, "hikes rates": -20,
    "inflation cools": -15, "inflation falls": -12, "inflation eases": -12,
    "dollar strengthens": -18, "dollar rises": -18, "dollar rallies": -15,
    "yields rise": -15, "yields jump": -15, "yields surge": -18,
    "risk-on": -12, "risk appetite": -10, "stocks rally": -8,
    "strong jobs report": -15, "strong payrolls": -15, "strong gdp": -12,
    "gold selloff": -15, "gold slides": -12, "gold falls": -10,
    "profit taking": -10, "reduced demand": -10, "ceasefire": -10,
}


def fetch_news(api_key=None, query=None, limit=None):
    """Trae titulares recientes desde marketaux.com. Marketaux tambien
    entrega su propio sentiment_score por entidad, pero aqui seguimos
    usando el diccionario GOLD_LEXICON sobre el titulo/descripcion para
    mantener el criterio especifico de "bueno/malo para el oro"."""
    api_key = api_key or os.environ.get("MARKETAUX_API_KEY")
    if not api_key:
        log.warning("MARKETAUX_API_KEY no configurada; se omite el analisis de noticias.")
        return []

    query = query or CONFIG["noticias_query"]
    limit = limit or CONFIG["noticias_max"]

    url = "https://api.marketaux.com/v1/news/all"
    params = {
        "search": query,
        "language": "en",
        "limit": limit,
        "api_token": api_key,
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    if "error" in data:
        log.error(f"Error de marketaux: {data['error'].get('message', data['error'])}")
        return []

    articles = data.get("data", [])
    # Normalizamos al mismo formato {title, description} que usa el resto del script
    return [{"title": a.get("title") or "", "description": a.get("description") or ""} for a in articles]


def score_headline(text):
    text_lower = text.lower()
    total = 0
    matched = []
    for term, weight in GOLD_LEXICON.items():
        if term in text_lower:
            total += weight
            matched.append((term, weight))
    return total, matched


def news_sentiment_score(articles):
    """Agrega el score de todos los titulares recientes. Devuelve un score
    normalizado -100..100 y las noticias que mas pesaron en el resultado."""
    if not articles:
        return 0, []

    total = 0
    razones = []
    for art in articles:
        titulo = art.get("title") or ""
        descripcion = art.get("description") or ""
        texto = f"{titulo}. {descripcion}"
        s, matched = score_headline(texto)
        if matched:
            total += s
            razones.append({"titular": titulo, "score": s, "terminos": matched})

    normalizado = max(-100, min(100, total))
    razones.sort(key=lambda r: abs(r["score"]), reverse=True)
    return normalizado, razones[:5]


# =========================================================================
# 4. MOTOR DE SENALES
# =========================================================================

def generate_signal(score_tecnico, score_fundamental):
    compuesto = (
        CONFIG["peso_tecnico"] * score_tecnico
        + CONFIG["peso_fundamental"] * score_fundamental
    )

    if compuesto > CONFIG["umbral_compra"]:
        accion = "COMPRA"
    elif compuesto < CONFIG["umbral_venta"]:
        accion = "VENTA"
    else:
        accion = "NEUTRAL"

    mismo_signo = (score_tecnico >= 0) == (score_fundamental >= 0)
    confianza = "ALTA" if mismo_signo else "BAJA"

    # --- Cuando NO se debe operar ---
    # Caso 1: la senal quedo en NEUTRAL (score entre los umbrales).
    # Caso 2: hay direccion (COMPRA/VENTA) pero tecnico y noticias se
    #         contradicen -- no se opera aunque el score cruce el umbral.
    if accion == "NEUTRAL":
        operar = False
        razon_no_operar = f"senal poco clara (score {round(compuesto, 1)}, entre -{CONFIG['umbral_venta']*-1:.0f} y +{CONFIG['umbral_compra']:.0f})"
    elif confianza == "BAJA":
        operar = False
        razon_no_operar = "tecnico y noticias estan en desacuerdo (mira el detalle antes de decidir)"
    else:
        operar = True
        razon_no_operar = None

    return {
        "accion": accion,
        "operar": operar,
        "razon_no_operar": razon_no_operar,
        "score_compuesto": round(compuesto, 1),
        "score_tecnico": round(score_tecnico, 1),
        "score_fundamental": round(score_fundamental, 1),
        "confianza": confianza,
    }


def risk_management(df, accion, balance=None):
    balance = balance or CONFIG["balance_cuenta"]
    last_close = df["close"].iloc[-1]
    last_atr = df["atr14"].iloc[-1]

    if pd.isna(last_atr) or last_atr <= 0:
        return None

    if accion == "COMPRA":
        stop_loss = last_close - CONFIG["atr_multiplicador_sl"] * last_atr
        take_profit = last_close + CONFIG["atr_multiplicador_sl"] * last_atr * CONFIG["ratio_riesgo_beneficio"]
    elif accion == "VENTA":
        stop_loss = last_close + CONFIG["atr_multiplicador_sl"] * last_atr
        take_profit = last_close - CONFIG["atr_multiplicador_sl"] * last_atr * CONFIG["ratio_riesgo_beneficio"]
    else:
        return None

    riesgo_usd = balance * (CONFIG["riesgo_por_operacion_pct"] / 100)
    riesgo_por_onza = abs(last_close - stop_loss)
    tamano_posicion_oz = riesgo_usd / riesgo_por_onza if riesgo_por_onza > 0 else 0

    return {
        "precio_entrada": round(last_close, 2),
        "stop_loss": round(stop_loss, 2),
        "take_profit": round(take_profit, 2),
        "riesgo_usd": round(riesgo_usd, 2),
        "tamano_posicion_oz": round(tamano_posicion_oz, 3),
    }


# =========================================================================
# 5. NOTIFICACIONES Y REPORTE
# =========================================================================

def send_telegram_alert(message, bot_token=None, chat_id=None):
    bot_token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID")
    if not bot_token or not chat_id:
        log.info("Telegram no configurado; la senal solo se muestra en consola.")
        return False

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    try:
        resp = requests.post(url, data={"chat_id": chat_id, "text": message}, timeout=10)
        resp.raise_for_status()
        return True
    except requests.RequestException as e:
        log.error(f"Error enviando alerta a Telegram: {e}")
        return False


def format_report(signal, gestion, symbol):
    """Reporte SIMPLE (por defecto): solo lo accionable -- accion, entrada,
    stop-loss, take-profit, y si NO se debe operar, por que."""
    lineas = []
    lineas.append(f"{symbol} -- {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lineas.append("")

    if signal["operar"] and gestion:
        lineas.append(f">>> {signal['accion']} <<<")
        lineas.append("")
        lineas.append(f"Entrada:     {gestion['precio_entrada']}")
        lineas.append(f"Stop-loss:   {gestion['stop_loss']}")
        lineas.append(f"Take-profit: {gestion['take_profit']}")
        lineas.append(f"Tamano:      {gestion['tamano_posicion_oz']} oz (riesgo ${gestion['riesgo_usd']})")
    else:
        lineas.append(">>> NO OPERAR <<<")
        lineas.append(f"Razon: {signal['razon_no_operar']}")

    lineas.append("")
    lineas.append(
        f"Score: {signal['score_compuesto']} "
        f"(tecnico {signal['score_tecnico']}, noticias {signal['score_fundamental']})"
        f"  |  Confianza: {signal['confianza']}"
    )
    lineas.append("Herramienta de apoyo educativa. No es asesoria financiera.")
    return "\n".join(lineas)


def format_report_detallado(signal, gestion, detalles_tecnicos, razones_noticias, pivots, symbol):
    """Reporte COMPLETO: incluye el desglose tecnico y las noticias que
    influyeron. Usar con --detallado."""
    lineas = [format_report(signal, gestion, symbol), ""]

    if pivots:
        lineas.append("Niveles del dia (pivotes clasicos):")
        lineas.append(f"  R2: {pivots['r2']:.2f}   R1: {pivots['r1']:.2f}")
        lineas.append(f"  Pivote: {pivots['pivot']:.2f}")
        lineas.append(f"  S1: {pivots['s1']:.2f}   S2: {pivots['s2']:.2f}")
        lineas.append("")

    lineas.append("Detalle tecnico:")
    for d in detalles_tecnicos:
        lineas.append(f"  - {d}")

    if razones_noticias:
        lineas.append("")
        lineas.append("Noticias que mas influyeron:")
        for r in razones_noticias:
            lineas.append(f"  - ({r['score']:+d}) {r['titular']}")

    return "\n".join(lineas)


# =========================================================================
# 6. EJECUCION PRINCIPAL
# =========================================================================

def run_once(detallado=False):
    log.info("Descargando datos de precio...")
    df = fetch_price_data()
    df = add_indicators(df)

    score_tec, detalles = technical_score(df)
    pivots = daily_pivots(df)

    log.info("Descargando noticias...")
    articles = fetch_news()
    score_news, razones = news_sentiment_score(articles)

    signal = generate_signal(score_tec, score_news)
    gestion = risk_management(df, signal["accion"])

    if detallado:
        reporte = format_report_detallado(signal, gestion, detalles, razones, pivots, CONFIG["symbol"])
    else:
        reporte = format_report(signal, gestion, CONFIG["symbol"])
    print(reporte)

    if signal["operar"]:
        send_telegram_alert(reporte)

    return signal


def run_loop(detallado=False):
    log.info(f"Iniciando bucle, analisis cada {CONFIG['minutos_entre_analisis']} minutos. Ctrl+C para detener.")
    while True:
        try:
            run_once(detallado=detallado)
        except Exception as e:
            log.error(f"Error durante el analisis: {e}")
        time.sleep(CONFIG["minutos_entre_analisis"] * 60)


def run_backtest():
    """Backtest simple: evalua si el cruce EMA9/EMA21 acerto la direccion
    del precio N velas despues. Es ILUSTRATIVO -- no incluye costos,
    spread, slippage ni gestion de posicion real. Sirve para tener una
    primera intuicion, no para decidir si el modelo es rentable."""
    df = fetch_price_data(period="60d", interval="1h")
    df = add_indicators(df)
    df["senal"] = np.where(df["ema9"] > df["ema21"], 1, -1)
    horizonte = 4  # velas hacia adelante
    df["retorno_futuro"] = df["close"].shift(-horizonte) / df["close"] - 1
    df["acierto"] = np.sign(df["retorno_futuro"]) == df["senal"]
    df = df.dropna(subset=["retorno_futuro"])

    win_rate = df["acierto"].mean() * 100
    retorno_promedio = df["retorno_futuro"].mean() * 100

    print(f"Backtest simple sobre {len(df)} velas de 1h ({CONFIG['symbol']})")
    print(f"  Win rate del cruce EMA9/EMA21 a {horizonte} velas: {win_rate:.1f}%")
    print(f"  Retorno promedio por senal: {retorno_promedio:.3f}%")
    print("  (Ilustrativo: no incluye costos, spread ni slippage)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modelo de trading intradia para oro")
    parser.add_argument("--once", action="store_true", help="Correr un solo analisis")
    parser.add_argument("--loop", action="store_true", help="Correr en bucle continuo")
    parser.add_argument("--backtest", action="store_true", help="Correr backtest simple")
    parser.add_argument("--detallado", action="store_true", help="Mostrar el desglose tecnico y las noticias, no solo la entrada/SL/TP")
    args = parser.parse_args()

    if args.backtest:
        run_backtest()
    elif args.loop:
        run_loop(detallado=args.detallado)
    else:
        run_once(detallado=args.detallado)
