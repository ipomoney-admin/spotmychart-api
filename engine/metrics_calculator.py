def calculate_metrics(trades: list[dict]) -> dict:
    """
    Calculate performance metrics from a list of closed trades.

    Each trade dict must have:
        return_pct: float  (positive = gain, negative = loss)
        status:     str    ("closed_win" | "closed_loss")

    Returns a dict with 7 metrics. All float values rounded to 2 dp.
    """
    _EMPTY = {
        "win_rate":     0.0,
        "avg_win_pct":  0.0,
        "avg_loss_pct": 0.0,
        "risk_reward":  0.0,
        "expectancy":   0.0,
        "max_gain_pct": 0.0,
        "max_loss_pct": 0.0,
    }

    if not trades:
        return _EMPTY

    wins   = [t["return_pct"] for t in trades if t.get("status") == "closed_win"]
    losses = [t["return_pct"] for t in trades if t.get("status") == "closed_loss"]

    total = len(trades)

    win_rate     = round(len(wins) / total * 100, 2)
    avg_win_pct  = round(sum(wins)   / len(wins),   2) if wins   else 0.0
    avg_loss_pct = round(sum(losses) / len(losses), 2) if losses else 0.0

    # Risk/reward: avg_win / abs(avg_loss); 0 if no losses
    avg_loss_abs = abs(avg_loss_pct)
    risk_reward  = round(avg_win_pct / avg_loss_abs, 2) if avg_loss_abs > 0 else 0.0

    # Expectancy = (win_rate% * avg_win) + (loss_rate% * avg_loss) — in pct terms
    loss_rate  = 1 - (len(wins) / total)
    expectancy = round((len(wins) / total) * avg_win_pct + loss_rate * avg_loss_pct, 2)

    all_returns  = [t["return_pct"] for t in trades]
    max_gain_pct = round(max(all_returns), 2)
    max_loss_pct = round(min(all_returns), 2)

    return {
        "win_rate":     win_rate,
        "avg_win_pct":  avg_win_pct,
        "avg_loss_pct": avg_loss_pct,
        "risk_reward":  risk_reward,
        "expectancy":   expectancy,
        "max_gain_pct": max_gain_pct,
        "max_loss_pct": max_loss_pct,
    }
