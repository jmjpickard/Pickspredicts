"""Shared feature-group definitions used by training, backtests, and autoresearch."""

from __future__ import annotations

from typing import Any

FEATURE_GROUPS: dict[str, list[str]] = {
    "race_context": [
        "field_size", "is_handicap", "race_type_encoded", "race_class_num",
        "is_grade1", "is_grade2", "is_grade3", "track_direction_encoded",
        "distance_band", "going_bucket",
    ],
    "ratings": [
        "or_current", "rpr_current", "ts_current",
        "or_best_last3", "or_best_last5", "rpr_best_last3", "rpr_best_last5",
        "or_rpr_diff", "or_trend_last5",
    ],
    "horse_form": [
        "age_at_race", "days_since_last_run", "career_runs", "career_wins", "career_places",
        "chase_starts", "hurdle_starts", "dnf_rate_last5", "avg_btn_last3", "btn_trend_last5",
        "headgear_changed", "first_time_headgear", "win_rate_overall", "place_rate_overall",
        "runs_last_90d", "runs_last_365d", "win_rate_going_bucket", "win_rate_dist_band",
        "win_rate_course", "win_rate_track_direction", "place_rate_track_direction",
    ],
    "connections": [
        "trainer_winpct_14d", "trainer_winpct_30d", "trainer_winpct_90d",
        "jockey_winpct_14d", "jockey_winpct_30d", "jockey_winpct_90d",
        "trainer_festival_winpct", "jockey_festival_winpct", "combo_winpct",
    ],
    "pedigree": [
        "sire_cheltenham_winpct", "sire_going_winpct", "sire_dist_winpct",
    ],
    "market_proxy": [
        "sp_rank",
    ],
    "runner_profile": [
        "weight_carried", "weight_vs_field_avg",
        "course_dist_winner", "days_since_last_win", "class_change",
    ],
    "market": [
        "market_implied_prob", "market_rank", "pre_price_move", "market_confidence",
    ],
    "comments": [
        "dominant_style_code", "pct_trouble", "pct_jumping_issues",
    ],
    "ratings_vs_field": [
        "or_vs_field", "rpr_vs_field", "ts_vs_field", "rpr_trend_last5",
    ],
    "enhanced": [
        "sex_encoded", "festival_starts", "class_drop_from_grade1",
        "days_since_last_win_capped",
    ],
    "connections_extended": [
        "trainer_winpct_class", "trainer_winpct_course", "jockey_winpct_course",
        "trainer_winpct_race_type", "jockey_winpct_race_type",
    ],
    "horse_context": [
        "first_time_chase", "first_time_course", "first_time_distance",
        "field_avg_or", "field_max_or", "or_change_last_run",
    ],
}

REQUIRED_FEATURE_GROUPS = {"race_context"}


def default_feature_group_flags() -> dict[str, bool]:
    return {name: True for name in FEATURE_GROUPS}


def active_feature_cols(
    all_feature_cols: list[str],
    feature_group_flags: dict[str, Any] | None = None,
) -> list[str]:
    """Return active feature columns after applying group-level disables."""
    flags = default_feature_group_flags()
    if feature_group_flags:
        for group_name, enabled in feature_group_flags.items():
            if group_name in flags:
                flags[group_name] = bool(enabled)

    disabled_cols: set[str] = set()
    for group_name, cols in FEATURE_GROUPS.items():
        if not flags.get(group_name, True):
            disabled_cols.update(cols)
    return [col for col in all_feature_cols if col not in disabled_cols]


def disabled_feature_groups(feature_group_flags: dict[str, Any] | None = None) -> list[str]:
    flags = default_feature_group_flags()
    if feature_group_flags:
        for group_name, enabled in feature_group_flags.items():
            if group_name in flags:
                flags[group_name] = bool(enabled)
    return [group_name for group_name, enabled in flags.items() if not enabled]
