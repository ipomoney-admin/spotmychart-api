import os
from dotenv import load_dotenv
from supabase import create_client
load_dotenv()
sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])

rows = sb.table("backtest_signals").select("*").limit(3).execute().data
for r in rows:
    print(r)
print("\nColumns:", list(rows[0].keys()) if rows else "no rows")
