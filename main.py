import logging
import os

import uvicorn
from apscheduler.triggers.cron import CronTrigger
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from core.scheduler import IST, run_scan_job, scheduler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="SpotMyChart API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://spotmychart.com",
        "https://www.spotmychart.com",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
def health():
    return {"status": "ok", "service": "SpotMyChart API"}


@app.on_event("startup")
async def on_startup():
    scheduler.add_job(
        run_scan_job,
        CronTrigger(day_of_week="mon-fri", hour=15, minute=35, timezone=IST),
        id="daily_scan",
        replace_existing=True,
    )
    scheduler.start()
    logger.info("SpotMyChart API is running. APScheduler started — daily scan at 15:35 IST Mon-Fri.")


@app.on_event("shutdown")
async def on_shutdown():
    scheduler.shutdown()
    logger.info("APScheduler stopped.")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
