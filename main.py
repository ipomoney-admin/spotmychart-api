from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import traceback

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/chart/{symbol}")
def get_chart(symbol: str):
    try:
        ticker = yf.Ticker(symbol + ".NS")
        df = ticker.history(start="2021-01-01", interval="1d")
        if df.empty:
            return {"error": "No data found for " + symbol}
        df = df.reset_index()
        df["Date"] = df["Date"].astype(str).str[:10]
        candles = df[["Date","Open","High","Low","Close","Volume"]].rename(
            columns={"Date":"date","Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"}
        ).to_dict("records")
        return {"symbol": symbol, "candles": candles}
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}

@app.get("/health")
def health():
    return {"status": "ok"}
