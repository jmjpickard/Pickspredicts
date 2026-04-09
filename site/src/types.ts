export type NumericLike = number | string | null | undefined;
export type BooleanLike = boolean | number | string | null | undefined;

export interface Runner {
  horse_id: string;
  horse_name: string;
  win_prob: number;
  place_prob: number;
  implied_prob: number | null;
  value_score: number | null;
  verdict: string;
  top_features: string;
  official_rating?: NumericLike;
  best_odds?: NumericLike;
  plan_a_points?: NumericLike;
  // Result fields (2025 retrospective)
  finish_position?: NumericLike;
  sp_decimal?: NumericLike;
  won?: BooleanLike;
  placed?: BooleanLike;
  is_bet?: BooleanLike;
  win_pl?: NumericLike;
  ew_pl?: NumericLike;
}

export interface Race {
  race_id: string;
  date?: string;
  course?: string;
  race_name?: string;
  analysis?: string;
  plan_a_pick_horse_id?: string;
  plan_a_pick_horse_name?: string;
  plan_a_points?: NumericLike;
  race_type?: string;
  race_class?: string;
  off_time?: string;
  distance_f?: NumericLike;
  is_handicap?: BooleanLike;
  pattern?: string;
  field_size?: NumericLike;
  runners: Runner[];
  // P&L fields
  race_win_pl?: NumericLike;
  race_ew_pl?: NumericLike;
  cumulative_win_pl?: NumericLike;
  cumulative_ew_pl?: NumericLike;
  num_bets?: NumericLike;
  num_places_paid?: NumericLike;
  place_fraction?: string;
}

export interface RaceDay {
  date: string;
  label: string;
  slug: string;
}

export function toNumber(value: NumericLike): number | null {
  if (value === null || value === undefined || value === "") {
    return null;
  }

  const numeric = typeof value === "number" ? value : Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

export function toBoolean(value: BooleanLike): boolean {
  if (typeof value === "boolean") {
    return value;
  }
  if (typeof value === "number") {
    return value !== 0;
  }
  if (typeof value === "string") {
    const normalised = value.trim().toLowerCase();
    if (["1", "true", "yes", "y"].includes(normalised)) {
      return true;
    }
    if (["0", "false", "no", "n", ""].includes(normalised)) {
      return false;
    }
  }
  return Boolean(value);
}

export function raceHasResults(race: Race): boolean {
  return race.runners.some((runner) => toNumber(runner.finish_position) !== null);
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
