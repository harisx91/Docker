# app.py
from flask import Flask, request, jsonify, render_template
import yfinance as yf
# Enable new scraper mode if available (fixes Yahoo blocking in 2025)
if hasattr(yf, "enable_scraper"):
    yf.enable_scraper(True)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import base64
from io import BytesIO
import time
import logging
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ----------------------------------------------------------------------
# Shared session with realistic headers (fixes Yahoo blocks in 2025)
# ----------------------------------------------------------------------
#session = requests.Session()
#session.headers.update({
#    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
#    "Accept": "application/json, text/plain, */*",
#    "Accept-Language": "en-US,en;q=0.9",
#    "Accept-Encoding": "gzip, deflate, br",
#    "Connection": "keep-alive",
#    "Referer": "https://finance.yahoo.com/"
#})

def download_one(ticker: str, period: str = "2y") -> pd.Series:
    """Download single ticker with retries, delays, and raw response logging."""
    fallback_period = "1y" if period == "2y" else "6mo"
    
    for attempt in range(1, 4):
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period=period, auto_adjust=True)
            if not hist.empty and len(hist) > 30:  # At least ~1 month of data
                logger.info(f"Success: {ticker} ({period}) -> {len(hist)} rows")
                return hist["Close"].rename(ticker)
            
            # Fallback to shorter period
            if attempt == 2:
                logger.warning(f"{ticker} ({period}) empty, trying {fallback_period}")
                period = fallback_period
                continue
                
        except Exception as e:
            if "JSONDecodeError" in str(e):
                logger.warning(f"{ticker} JSON error (attempt {attempt}): {e}")
                # Log raw response if possible (for debug)
                try:
                    resp = session.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}")
                    logger.warning(f"Raw response status: {resp.status_code}, content preview: {resp.text[:200]}")
                except:
                    pass
            else:
                logger.warning(f"{ticker} other error (attempt {attempt}): {e}")
        
        time.sleep(2 ** attempt)  # Back-off: 2s, 4s, 8s
    
    logger.error(f"{ticker} failed completely")
    return pd.Series(name=ticker)  # Empty fallback

def get_stock_data(tickers, period="2y"):
    if len(tickers) < 2:
        raise ValueError("Need ≥2 tickers")

    # Download with delays between tickers to avoid rate limits
    series_list = []
    for t in tickers:
        series_list.append(download_one(t, period))
        time.sleep(2)  # Delay between downloads

    data = pd.concat(series_list, axis=1)

    # Filter valid columns (non-empty)
    valid = [c for c in data.columns if data[c].notna().any()]
    if len(valid) < 2:
        raise ValueError(f"Only {len(valid)} ticker(s) returned data: {valid}")

    returns = data[valid].pct_change().dropna()
    if returns.empty:
        raise ValueError("No overlapping price history")
    
    logger.info(f"Valid data for {len(valid)} tickers, returns shape: {returns.shape}")
    return returns

# ----------------------------------------------------------------------
# Portfolio math (unchanged)
# ----------------------------------------------------------------------
def portfolio_stats(weights, returns):
    w = np.array(weights)
    ret = np.sum(returns.mean() * w) * 252
    vol = np.sqrt(np.dot(w.T, np.dot(returns.cov() * 252, w)))
    sharpe = ret / vol if vol > 0 else -np.inf
    return ret, vol, sharpe

def neg_sharpe(w, r): return -portfolio_stats(w, r)[2]

def optimize_portfolio(returns):
    n = len(returns.columns)
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},)
    bounds = tuple((0, 1) for _ in range(n))
    res = minimize(neg_sharpe, [1/n]*n, args=(returns,),
                   method='SLSQP', bounds=bounds, constraints=cons)
    if not res.success:
        raise RuntimeError("Optimization did not converge")
    return res.x

def generate_efficient_frontier(returns, n_ports=1000):
    n = len(returns.columns)
    results = np.zeros((3, n_ports))
    for i in range(n_ports):
        w = np.random.random(n)
        w /= w.sum()
        r, v, s = portfolio_stats(w, returns)
        results[:, i] = [r, v, s]
    return results

def plot_efficient_frontier(returns):
    results = generate_efficient_frontier(returns)
    if results.shape[1] == 0:
        raise RuntimeError("No portfolios generated")

    best = np.argmax(results[2])
    plt.figure(figsize=(10, 6))
    plt.scatter(results[1], results[0], c=results[2], cmap='viridis',
                s=15, alpha=0.7)
    plt.scatter(results[1, best], results[0, best], c='red',
                s=120, label='Max Sharpe', zorder=5)
    plt.colorbar(label='Sharpe Ratio')
    plt.xlabel('Annual Volatility')
    plt.ylabel('Annual Expected Return')
    plt.title('Efficient Frontier')
    plt.legend()

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=110)
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode()
    plt.close()
    return f"data:image/png;base64,{img}"

# ----------------------------------------------------------------------
# Flask routes (unchanged)
# ----------------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/optimize', methods=['POST'])
def optimize():
    raw = request.json.get('tickers', [])
    tickers = [t.strip().upper() for t in raw if t.strip()]
    if len(tickers) < 2:
        return jsonify({"error": "Provide at least 2 tickers"}), 400

    try:
        returns = get_stock_data(tickers)
        logger.info(f"Using tickers: {list(returns.columns)}")

        opt_w = optimize_portfolio(returns)
        ret, vol, shr = portfolio_stats(opt_w, returns)

        plot_url = plot_efficient_frontier(returns)

        weights = {returns.columns[i]: round(float(opt_w[i]), 4)
                   for i in range(len(opt_w))}

        return jsonify({
            "tickers_used": list(returns.columns),
            "optimal_weights": weights,
            "expected_return": round(ret, 4),
            "volatility": round(vol, 4),
            "sharpe_ratio": round(shr, 4),
            "plot": plot_url
        })

    except Exception as e:
        logger.exception("Optimization error")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
