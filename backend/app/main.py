from fastapi import FastAPI

app = FastAPI(
    title="Multi-Agent Trading Simulator API",
    description="Multi-Agent AI Trading Simulation Platform",
    version="0.1.0"
)

@app.get("/")
def home():
    return {
        "project": "Multi-Agent-Trading-Simulator",
        "status": "running",
        "agents": 0
    }

@app.get("/health")
def health():
    return {"status": "healthy"}
