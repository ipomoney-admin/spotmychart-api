import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])

sig = sb.table("smc_signals").select("id", count="exact").execute()
trd = sb.table("smc_trades").select("id", count="exact").execute()
print(f"smc_signals rows: {sig.count}")
print(f"smc_trades rows:  {trd.count}")
