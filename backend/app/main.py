from fastapi import FastAPI

app = FastAPI(
    title="TradingArena API",
    description="Multi-Agent AI Trading Simulation Platform",
    version="0.1.0"
)

@app.get("/")
def home():
    return {
        "project": "TradingArena",
        "status": "running",
        "agents": 0
    }

@app.get("/health")
def health():
    return {"status": "healthy"}
