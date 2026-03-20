def assign_tier(win_rate: float, occurrence_count: int) -> int:
    """
    Assign a reliability tier (1-5) to a pattern based on historical performance.

    Tier 1 — elite    (win_rate > 75%)
    Tier 2 — strong   (win_rate >= 50%)
    Tier 3 — moderate (win_rate >= 25%)
    Tier 4 — weak     (win_rate < 25%)
    Tier 5 — unrated  (< 3 occurrences, insufficient data)
    """
    if occurrence_count < 3:
        return 5
    if win_rate > 75:
        return 1
    if win_rate >= 50:
        return 2
    if win_rate >= 25:
        return 3
    return 4
