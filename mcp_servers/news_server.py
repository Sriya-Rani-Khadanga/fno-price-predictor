"""
MCP News Sentiment Server — FastAPI app on port 8005.
Provides daily headlines and sentiment scores for the trading agent.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import datetime
from typing import Dict, List, Optional
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="PCP Arb News MCP Server", version="1.0.0")

# Mock database of news by date
_news_db: Dict[str, List[Dict]] = {
    "2024-06-28": [
        {"headline": "Reliance Industries announces 12-25% tariff hike for telecom", "sentiment": 0.8},
        {"headline": "Nifty hits record high of 24,174; Banks face profit booking", "sentiment": -0.2},
        {"headline": "Pharma stocks gain as Dr. Reddy's beats earnings estimates", "sentiment": 0.5}
    ],
    "2024-04-24": [
        {"headline": "Corporate earnings season kicks off with mixed results", "sentiment": 0.1},
        {"headline": "Crude oil prices steady as geopolitical tensions ease", "sentiment": 0.3},
        {"headline": "FIIs remain net sellers in Indian equities for 3rd day", "sentiment": -0.4}
    ]
}

class NewsRequest(BaseModel):
    date_iso: str
    symbol: Optional[str] = "NIFTY"

@app.post("/tools/get_news_summary")
async def get_news_summary(req: NewsRequest):
    news = _news_db.get(req.date_iso, [])
    if not news:
        return {"message": f"No news recorded for {req.date_iso}", "sentiment_score": 0.0}
    
    avg_sentiment = sum(n["sentiment"] for n in news) / len(news)
    headlines = [n["headline"] for n in news]
    
    return {
        "date": req.date_iso,
        "symbol": req.symbol,
        "avg_sentiment": round(avg_sentiment, 2),
        "headlines": headlines,
        "market_impact": "Positive" if avg_sentiment > 0.2 else "Negative" if avg_sentiment < -0.2 else "Neutral"
    }

@app.get("/health")
async def health():
    return {"status": "ok", "server": "news", "port": 8005}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8005)
