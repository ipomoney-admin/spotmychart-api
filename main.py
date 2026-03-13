from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/chart/{symbol}")
def get_chart(symbol: str):
    ticker = yf.Ticker(symbol + ".NS")
    df = ticker.history(start="2021-01-01", period="max", interval="1d")
    if df.empty:
        return {"error": "No data"}
    df = df.reset_index()
    df["Date"] = df["Date"].astype(str).str[:10]
    return {
        "symbol": symbol,
        "candles": df[["Date","Open","High","Low","Close","Volume"]].rename(columns={"Date":"date","Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"}).to_dict("records")
    }

@app.get("/health")
def health():
    return {"status": "ok"}
