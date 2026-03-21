import os
import logging
import redis
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")

try:
    r = redis.from_url(REDIS_URL, decode_responses=True)
    r.ping()
except redis.RedisError as e:
    logger.warning(f"Redis unavailable — running without cache: {e}")
    r = None
