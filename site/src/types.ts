export interface Runner {
  horse_id: string;
  horse_name: string;
  win_prob: number;
  place_prob: number;
  implied_prob: number | null;
  value_score: number | null;
  verdict: string;
  top_features: string;
  official_rating?: number | null;
  best_odds?: number | null;
  // Result fields (2025 retrospective)
  finish_position?: number | null;
  sp_decimal?: number | null;
  won?: boolean | null;
  placed?: boolean | null;
  is_bet?: boolean;
  win_pl?: number | null;
  ew_pl?: number | null;
}

export interface Race {
  race_id: string;
  date?: string;
  course?: string;
  race_name?: string;
  race_type?: string;
  race_class?: string;
  off_time?: string;
  distance_f?: number | null;
  is_handicap?: boolean;
  pattern?: string;
  field_size?: number | null;
  runners: Runner[];
  // P&L fields
  race_win_pl?: number;
  race_ew_pl?: number;
  cumulative_win_pl?: number;
  cumulative_ew_pl?: number;
  num_bets?: number;
  num_places_paid?: number;
  place_fraction?: string;
}

export interface RaceDay {
  date: string;
  label: string;
  slug: string;
}

function formatDayLabel(dateStr: string): string {
  const date = new Date(`${dateStr}T00:00:00`);
  if (Number.isNaN(date.getTime())) {
    return dateStr;
  }
  return new Intl.DateTimeFormat("en-GB", {
    weekday: "short",
    day: "numeric",
    month: "short",
  }).format(date);
}

export function getRaceDays(races: Race[]): RaceDay[] {
  const dates = Array.from(
    new Set(
      races
        .map((r) => (r.date ?? "").slice(0, 10))
        .filter((d) => d.length === 10),
    ),
  ).sort();

  return dates.map((date) => ({
    date,
    slug: date,
    label: formatDayLabel(date),
  }));
}
