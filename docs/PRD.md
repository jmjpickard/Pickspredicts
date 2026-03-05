# PRD: Cheltenham Festival Value Finder

**Date**: 2026-03-04
**Target**: Cheltenham Festival 2026 (March 10-13)
**Status**: MVP — ship by March 9

---

## 1. Problem

Cheltenham Festival has 28 races over 4 days. Identifying genuine value bets requires comparing your own assessment of each horse's probability against what the market offers. Doing this manually is slow, inconsistent, and doesn't compound knowledge across races.

## 2. Goal

A live website that shows, for every Cheltenham 2026 race:
- Model-estimated **win probability** and **place probability** per runner
- Current **Betfair exchange odds** (and implied probability)
- **Value score**: where our model disagrees with the market (model prob > implied prob = value)
- Ranked by value, so the best opportunities surface first

Share a URL with mates. Look at it on your phone in the pub. Make better decisions.

## 3. Users

- Jack (primary) — engineering lead, places bets, wants edge
- Jack's friends — consume the output, share by race, not technical

## 4. Success criteria (MVP)

- [ ] All 28 races covered with probabilities before race day
- [ ] Win probability + place probability per runner
- [ ] Value overlay against live/recent Betfair exchange prices
- [ ] Accessible via a shareable URL (Cloudflare Pages)
- [ ] Predictions update when ground conditions change (manual input trigger)

### What "good" looks like
- Probabilities are directionally sensible (favourite has highest prob, long shots have low prob)
- Value flags surface non-obvious selections (not just the favourite every time)
- A friend can open the URL, pick a race, and immediately see "bet this horse EW at 8/1, model says 18% win chance vs 11% implied"

### What failure looks like
- Model outputs are nonsensical (100/1 shot flagged as top value with no rationale)
- Data gaps mean key runners are missing
- Site isn't up by March 10

## 5. Data sources

### 5.1 Racing Post (via rpscrape)

**Tool**: [rpscrape](https://github.com/joenano/rpscrape) — actively maintained CLI scraper (310 commits, last updated Feb 2026). Uses curl_cffi browser impersonation for anti-bot bypass. Outputs CSV (results) and JSON (racecards).

**Auth**: Free RP account (no paid subscription). Requires `EMAIL`, `AUTH_STATE` cookie, and `ACCESS_TOKEN` from browser dev tools. Stored in `.env`.

**What we get (results — CSV, one row per runner per race)**:
- Race: date, course, race_id, race_name, type, class, pattern (Grade), distance (multiple units), going, surface, field size
- Runner: finishing position, lengths beaten, horse_id, horse name, age, sex, weight, headgear
- Performance: time (secs), SP (fractional + decimal), OR, RPR (RP Rating), Topspeed
- Connections: jockey_id, jockey, trainer_id, trainer, owner_id, owner
- Pedigree: sire_id, sire, dam_id, dam, damsire_id, damsire
- Commentary: race comment per runner
- Betfair: BSP, WAP, morning WAP, pre/IP min-max prices, volume traded (optional flag)

**What we get (racecards — JSON)**:
- Full runner profiles: form, stats, medical history, wind surgery, breeding, quotes, jockey/trainer details

**How to invoke**:
```bash
# Historical results — Cheltenham (course 13), jumps, 2020-2024 season
./rpscrape.py -c 13 -y 2020-2024 -t jumps

# Upcoming racecards — next 7 days
./racecards.py --days 7 --region gb

# Single date results
./rpscrape.py -d 2025/03/11

# With Betfair SP data
./rpscrape.py -c 13 -y 2020-2024 -t jumps -b
```

**Integration**: Shell out to rpscrape CLI → read CSV output with pandas. Not importable as a library. Keep rpscrape as a git submodule or cloned dependency.

**Scope**:
- All Cheltenham results: 2020-2025 seasons (5 full festivals)
- All UK/IRE jumps results for broader training data (configurable)
- Racecards for all 28 Cheltenham 2026 races

**Risks**:
- RP changes page structure → rpscrape breaks. Mitigate: rpscrape is actively maintained; pin to a known-good commit.
- Rate limiting / blocking. Mitigate: rpscrape has built-in retry logic (7 retries, 1.4s delay). Cache raw CSV files; don't re-fetch.

### 5.2 BHA (official results) — FALLBACK ONLY

**Purpose**: Cross-reference if rpscrape data has gaps. Not a primary source since rpscrape already provides comprehensive official results.

**Status**: Deprioritised for MVP. Investigate post-festival if data quality issues emerge.

### 5.3 Betfair Exchange API

**What we get**:
- **Live market prices** — current back/lay odds for every runner in every race
- **Historical Betfair SP** — rpscrape already captures BSP, WAP, pre/IP prices, and volume for historical races (via `-b` flag)
- **Market depth** — how much money is matched (confidence signal)

**How**: Betfair API (Jack has an account). Free API access with developer app key.

**Scope**: All 28 Cheltenham 2026 races. Poll periodically (every 30 min? hourly?) from when markets are available.

**This is the highest-value data source.** Market prices encode the collective wisdom of thousands of punters. Our model's job is to find where the market is wrong.

### 5.4 Racecards (for current entries)

**What we get**: Declared runners, form, stats, weights, jockey bookings, OR, pedigree, quotes, medical history.

**How**: rpscrape's `racecards.py` — outputs structured JSON.

**Scope**: All 28 races. Re-fetch when declarations are confirmed and on race morning for final non-runner updates.

## 6. Data pipeline

```
[rpscrape results CSV] ──┐
                          ├──> [Normalise to Parquet/DuckDB] ──> [Build features] ──> [Train model]
[rpscrape racecards JSON]┘                                                                 │
                                                                                           v
[Betfair API: live odds] ─────────────────────────────────────> [Value calculation] <── [Predict]
                                                                       │
                                                                       v
                                                                [Static JSON] ──> [Cloudflare Pages site]
```

### Pipeline steps

1. **Fetch** — `rpscrape.py` for historical results (CSV) + `racecards.py` for entries (JSON) → `data/raw/`
2. **Normalise** — parse CSV/JSON into canonical schema (see section 8) → `data/staged/parquet/`
3. **Features** — compute per-runner features using only past data → `data/marts/features/`
4. **Train** — LightGBM on historical data with walk-forward validation
5. **Predict** — score Cheltenham 2026 runners → win prob + place prob per runner
6. **Value** — merge predictions with Betfair odds → value score
7. **Publish** — write JSON → deploy to Cloudflare Pages

Steps 1-5 run once (or re-run when ground changes).
Step 6 runs periodically to capture odds movement.
Step 7 deploys on each update.

## 7. Model

### Approach: OR baseline + LightGBM + market overlay

**Layer 1 — Rating baseline**
- OR, RPR, and Topspeed are all available from rpscrape — three independent rating systems
- Convert best available rating → probability using logistic model fitted to historical data
- This alone gives a reasonable baseline; three ratings together are stronger than one

**Layer 2 — LightGBM feature model**
- Captures non-linear interactions that ratings alone miss
- Features (computed using only pre-race data to avoid leakage):

  **Ratings (from rpscrape)**
  - Official Rating (OR)
  - RP Rating (RPR) — rpscrape provides this
  - Topspeed (TS) — rpscrape provides this
  - Rating differentials (OR vs RPR, trend across last N runs)

  **Horse form**
  - Age at race date
  - Days since last run
  - Runs in last 90 / 365 days
  - Best OR/RPR/TS in last N runs
  - OR trend (slope over last N)
  - Distance suitability (win/place rate by distance band)
  - Going suitability (win/place rate by going bucket)
  - Course form (Cheltenham-specific win/place %)
  - Left-handed track record (Cheltenham is left-handed)
  - DNF rate last N runs (falls/unseats — critical for chases)
  - Chase/hurdle experience (number of starts over fences/flights)
  - Lengths beaten trend (improving/worsening?)
  - Time performance (race time vs standard for distance/going)
  - Headgear changes (first time blinkers etc.)

  **Race comment features (LLM-extracted — see section 15)**
  - Running style (led / tracked leaders / held up / prominent)
  - Finishing effort (stayed on, weakened, ran on strongly)
  - Trouble in running (hampered, short of room, badly hampered)
  - Jumping quality (fluent, mistakes, fell, unseated)
  - Stamina signals (found nil, kept on, plugged on)
  - These are extracted from rpscrape's free-text `comment` field via Claude API batch processing during feature build

  **Pedigree (from rpscrape — sire/dam/damsire IDs)**
  - Sire win% at Cheltenham (where sample exists)
  - Sire win% by going bucket
  - Sire win% by distance band
  - (Useful for lightly-raced novices where form is thin)

  **Connections**
  - Trainer win% last 14/30/90 days
  - Jockey win% last 14/30/90 days
  - Trainer Cheltenham Festival record
  - Jockey Cheltenham Festival record
  - Trainer-jockey combo win%

  **Market (from rpscrape Betfair data — historical only)**
  - Historical BSP vs SP differential (was horse shorter/longer on exchange?)
  - Pre-race volume (market confidence)
  - Morning price vs BSP movement

  **Race context**
  - Field size
  - Handicap vs non-handicap
  - Race class/grade (Grade 1, Grade 2, handicap, etc.)
  - Race type (hurdle/chase/NH flat/cross-country)

  **Conditions (manual input)**
  - Going (good, good-to-soft, soft, heavy, etc.)
  - Rail movements / course configuration if known

- Output: raw score per runner → softmax within race → win probability

**Layer 3 — Value overlay**
- Betfair exchange back odds → implied win probability (1/odds)
- Value = model_prob - implied_prob
- Positive value = model thinks horse has better chance than market
- Rank runners within each race by value score

**Place probability**
- Derived from win probability using Harville formula
- Adjusts for field size and EW terms (1/4 odds, places 1-2-3 or 1-2-3-4)

### Validation

- Walk-forward: train on years 2021-2024, validate on 2025 Cheltenham
- Metrics: log loss, calibration plot, ROI if betting top value pick per race
- Sanity check: does the model's top pick beat "always bet the favourite"?

## 8. Canonical data schema

Derived from rpscrape's actual CSV output fields. Entity IDs (horse_id, jockey_id, trainer_id, sire_id, dam_id, damsire_id) come directly from rpscrape — no fuzzy matching needed for the primary dataset.

### races
| Field | Type | Source |
|-------|------|--------|
| race_id | string | rpscrape |
| date | date | rpscrape |
| region | string | rpscrape |
| course_id | int | rpscrape |
| course | string | rpscrape |
| off_time | time | rpscrape |
| race_name | string | rpscrape |
| race_type | enum (hurdle/chase/nh_flat/cross_country) | rpscrape (type field) |
| class | string | rpscrape |
| pattern | string (Grade 1/2/3, Listed, etc.) | rpscrape |
| distance_f | float | rpscrape (furlongs) |
| distance_m | int | rpscrape (metres) |
| distance_y | int | rpscrape (yards) |
| going | string | rpscrape / manual override |
| surface | string | rpscrape |
| field_size | int | rpscrape (ran) |
| age_band | string | rpscrape |
| rating_band | string | rpscrape |
| sex_rest | string | rpscrape |
| is_handicap | bool | derived from class/rating_band |

### runners (historical results)
| Field | Type | Source |
|-------|------|--------|
| race_id | FK | rpscrape |
| horse_id | int | rpscrape |
| horse_name | string | rpscrape |
| age | int | rpscrape |
| sex | string | rpscrape |
| finish_position | int | rpscrape (pos) |
| ovr_btn | float | rpscrape (total lengths beaten) |
| btn | float | rpscrape (lengths behind horse in front) |
| weight_lbs | int | rpscrape (lbs) |
| headgear | string | rpscrape (hg) |
| time_secs | float | rpscrape (secs) |
| sp_decimal | float | rpscrape (dec) |
| sp_fractional | string | rpscrape (sp) |
| official_rating | int | rpscrape (ofr) |
| rpr | int | rpscrape (RP Rating) |
| topspeed | int | rpscrape (ts) |
| jockey_id | int | rpscrape |
| jockey | string | rpscrape |
| trainer_id | int | rpscrape |
| trainer | string | rpscrape |
| owner_id | int | rpscrape |
| sire_id | int | rpscrape |
| sire | string | rpscrape |
| dam_id | int | rpscrape |
| dam | string | rpscrape |
| damsire_id | int | rpscrape |
| damsire | string | rpscrape |
| comment | string | rpscrape (race comment) |
| prize | float | rpscrape |

### betfair_historical (from rpscrape -b flag)
| Field | Type | Source |
|-------|------|--------|
| race_id | FK | matched to runners |
| horse_id | FK | matched |
| bsp | float | rpscrape |
| wap | float | rpscrape (weighted avg price) |
| morning_wap | float | rpscrape |
| pre_min | float | rpscrape (pre-race min price) |
| pre_max | float | rpscrape |
| ip_min | float | rpscrape (in-play min) |
| ip_max | float | rpscrape (in-play max) |
| morning_vol | float | rpscrape |
| pre_vol | float | rpscrape |
| ip_vol | float | rpscrape |

### betfair_live (from Betfair Exchange API)
| Field | Type | Source |
|-------|------|--------|
| race_id | FK | matched |
| runner_name | string | Betfair |
| selection_id | int | Betfair |
| back_odds | float | Betfair |
| lay_odds | float | Betfair |
| matched_amount | float | Betfair |
| fetched_at | timestamp | - |

### predictions (output)
| Field | Type |
|-------|------|
| race_id | FK |
| horse_id | FK |
| horse_name | string |
| win_prob | float |
| place_prob | float |
| betfair_implied_prob | float |
| value_score | float |
| top_features | json (top 3 reasons) |

## 9. Frontend (Cloudflare Pages)

### Tech
- Astro or plain HTML/JS static site (keep it dead simple)
- Tailwind for styling
- Reads from a `predictions.json` file deployed alongside the site

### Views

**Race list** (home page)
- 4 tabs: Tuesday / Wednesday / Thursday / Friday
- Each race: name, time, type, field size
- Badge on each race showing "X value picks"

**Race detail**
- All runners in a table, sorted by value score (desc)
- Columns: Horse | OR | Model Win% | Model Place% | Betfair Odds | Implied% | Value | Verdict
- Value column: green if positive (good value), red if negative (short price)
- "Verdict" column: short text like "Strong value" / "Fair price" / "Opposable"
- Going shown prominently at top (with last-updated timestamp)
- Optional: expandable row showing top 3 feature drivers ("why this horse?")

**Minimal UX**
- Mobile-first (used in the pub / at the course)
- No auth, no login — public URL
- Fast load — it's a static JSON, should be instant

## 10. Project structure

```
cheltenham/
  rpscrape/                      # git submodule or cloned repo
  src/
    ingest/
      fetch_results.py           # shell out to rpscrape.py, manage raw CSV output
      fetch_racecards.py         # shell out to racecards.py, manage raw JSON output
      fetch_betfair_odds.py      # Betfair Exchange API client (live odds)
    transform/
      normalise.py               # CSV/JSON → canonical Parquet tables
    features/
      build_features.py          # per-runner feature computation
      parse_comments.py          # LLM-based race comment extraction (Claude Haiku)
    model/
      train.py                   # LightGBM training + walk-forward CV
      predict.py                 # score Cheltenham 2026 runners
      value.py                   # merge with Betfair odds → value scores
    agents/                      # Phase 6 — agentic capabilities
      race_preview.py            # generate natural language race previews (Claude Sonnet)
      conditions_monitor.py      # parse going reports, trigger re-prediction
      post_race.py               # post-race analysis and model learning
    publish/
      generate_json.py           # predictions → predictions.json for frontend
  site/                          # Astro static site for Cloudflare Pages
    src/
      pages/
      components/
      layouts/
    public/
      predictions.json           # generated by pipeline
  data/
    raw/
      results/                   # rpscrape CSV output (gitignored)
      racecards/                 # rpscrape JSON output (gitignored)
      betfair/                   # Betfair API responses (gitignored)
    staged/                      # normalised Parquet tables
    marts/                       # feature tables
    analysis/                    # post-race analysis output (Phase 6)
  configs/
    pipeline.yaml                # date ranges, paths, toggles
  .env                           # RP + Betfair + Anthropic API keys (never committed)
  pyproject.toml
  .gitignore
```

## 11. Tech stack

| Layer | Choice | Reason |
|-------|--------|--------|
| Language (data) | Python 3.13 | rpscrape requirement + ML ecosystem |
| Scraping (RP) | rpscrape (CLI) | Proven, maintained, handles anti-bot, rich output |
| Data storage | DuckDB + Parquet | Fast, zero-infra, SQL interface |
| ML | LightGBM | Fast, handles small datasets well, good with tabular |
| Validation | scikit-learn | Walk-forward CV, metrics |
| LLM (features) | Claude Haiku via Anthropic SDK | Comment parsing, structured extraction, cost-efficient |
| LLM (analysis) | Claude Sonnet via Anthropic SDK | Race previews, post-race analysis (Phase 6) |
| API (odds) | Betfair Exchange API | Free, real-time, best odds source |
| Frontend | Astro (static) | Simple, fast, deploys to CF Pages |
| Styling | Tailwind CSS | Quick, mobile-first |
| Hosting | Cloudflare Pages | Free, fast, shareable URL |
| Language (site) | TypeScript | Jack's primary language |

## 12. Risks and mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| RP blocks scraping | No historical data | rpscrape has built-in anti-bot (curl_cffi) + retries. Cache all raw files. If blocked, use broader UK/IRE jumps data already fetched. |
| rpscrape breaks (RP page changes) | Can't fetch new data | Pin to known-good commit. Raw data already cached. Racecard data can be manually entered as last resort. |
| Insufficient historical data (5 yrs Cheltenham = ~140 races) | Weak model | Supplement with all UK/IRE Grade 1-3 NH races via rpscrape. Lean on OR + RPR + Topspeed as strong baseline features. |
| Model overfits small sample | Bad predictions | Strong regularisation; walk-forward CV; sanity check against OR-only baseline |
| Betfair API setup takes too long | No value overlay | Historical BSP already available via rpscrape `-b` flag. Fall back to manual odds entry for live races. |
| Going changes on race day | Stale predictions | Manual going input triggers re-prediction; timestamp prominently on UI |
| Entries change (non-runners) | Wrong runners in predictions | Re-run `racecards.py` morning of; flag non-runners in UI |
| Free RP account tokens expire | Can't authenticate | Document how to refresh tokens from browser dev tools. Quick manual step. |
| Time (6 days) | Don't ship | Cut scope ruthlessly — see section 13 |

## 13. Phased delivery

### Phase 1: Data (March 4-6) — MUST HAVE
- [ ] Clone rpscrape, configure `.env` with free RP account tokens
- [ ] Fetch Cheltenham results 2020-2025 via `rpscrape.py -c 13 -y 2020-2024 -t jumps -b`
- [ ] Fetch broader NH results (Grade 1-3 races, key courses) for training depth
- [ ] Normalise CSV output to DuckDB/Parquet
- [ ] Betfair API client (get app key, test connection)
- [ ] Fetch 2026 racecards via `racecards.py` when declarations available

### Phase 2: Model (March 6-7) — MUST HAVE
- [ ] LLM comment parsing — batch extract structured features from race comments via Claude API (see section 15)
- [ ] Feature engineering pipeline (including comment-derived features)
- [ ] Train LightGBM with walk-forward CV
- [ ] Generate win + place probabilities for 2026 entries
- [ ] Validate on 2025 Cheltenham (does it make sense?)

### Phase 3: Value + publish (March 7-8) — MUST HAVE
- [ ] Betfair odds integration (live or periodic)
- [ ] Value score calculation
- [ ] Generate predictions.json

### Phase 4: Frontend (March 8-9) — MUST HAVE
- [ ] Static site with race list + race detail views
- [ ] Mobile-friendly
- [ ] Deploy to Cloudflare Pages
- [ ] Share URL

### Phase 5: Race week (March 10-13) — NICE TO HAVE
- [ ] Update going daily → re-predict
- [ ] Update odds periodically
- [ ] Handle non-runners
- [ ] Post-race: track actual results vs predictions

### Phase 6: Agentic layer (post-festival) — NICE TO HAVE
- [ ] Race preview agent — Claude Sonnet generates natural language previews per race
- [ ] Conditions monitor — auto-parses going reports, triggers re-prediction
- [ ] Post-race analysis — compares predictions to results, generates learning notes
- [ ] Ad-hoc Q&A — natural language queries over the dataset via Claude + DuckDB

## 14. Agentic layer (Claude API)

LLM capabilities integrated at two levels: **feature extraction** (Phase 2, MVP) and **race analysis** (Phase 6, post-MVP).

### 14.1 Comment parsing — Phase 2 (MVP)

rpscrape provides free-text race comments per runner (e.g. "tracked leaders, hampered 3 out, stayed on well flat"). These contain signal that structured data misses.

**Approach**: Batch process all historical comments through Claude API during feature build.

**Input**: Raw comment string per runner per race
**Output**: Structured JSON per comment:
```json
{
  "running_style": "tracked_leaders",
  "finishing_effort": "stayed_on",
  "trouble_in_running": true,
  "trouble_detail": "hampered 3 out",
  "jumping": "clean",
  "stamina_signal": "positive",
  "distance_signal": null,
  "ground_signal": null
}
```

**Implementation**:
- `src/features/parse_comments.py` — sends batches to Claude API (Haiku for cost efficiency)
- Cache parsed results in Parquet alongside other features
- ~5,000-10,000 comments for Cheltenham + broader NH data — manageable cost with Haiku
- Structured output schema enforced via Claude's tool use / JSON mode

**Features derived**:
- Dominant running style (% of runs where led / prominent / held up)
- Trouble-in-running rate (% of runs with interference)
- Jumping error rate (% with mistakes noted)
- Finishing effort trend (improving language over last N runs?)

### 14.2 Race preview agent — Phase 6 (post-MVP)

The feature that turns a data table into something your mates actually want to read.

**What it does**: For each race, generates a 2-3 paragraph natural language preview synthesising the model's predictions, form data, and value signals.

**Example output**:
> **Champion Hurdle — Tuesday 1:30**
>
> The model strongly fancies Constitution Hill (42% win prob vs 28% market-implied — strong value at 5/2). His Cheltenham record is perfect (3/3), RPR consistently 170+, and he handles any ground. The one concern is 287 days since his last run, but trainer Nicky Henderson has a 34% strike rate with horses returning from 250+ day breaks at the Festival.
>
> State Man (28% win prob, 25% implied) looks fairly priced. His soft ground form is excellent but he's never beaten Constitution Hill. The value play underneath is Doyen Quest at 16/1 — the model gives him 11% vs 6% implied. His sire's progeny have a 22% place rate at Cheltenham on soft ground.

**Implementation**:
- `src/agents/race_preview.py`
- Input: predictions.json + full form data for each runner
- Uses Claude Sonnet — needs reasoning quality for narrative synthesis
- Generates one preview per race → added to predictions.json as `race_preview` field
- Displayed on the race detail page above the runner table

### 14.3 Conditions monitor agent — Phase 6 (post-MVP)

Watches going reports and triggers re-prediction.

**What it does**:
- Polls clerk of the course going reports (published on racing sites / X)
- Parses "Going changed from Good to Good to Soft on the chase course" → structured going update
- Triggers re-prediction pipeline with new going → updated predictions.json → redeploy

**Implementation**:
- `src/agents/conditions_monitor.py`
- Runs periodically during race week (cron or manual trigger)
- Scrapes going updates from racingpost.com or attheraces.com free pages
- Uses Claude Haiku to extract structured going from free-text reports
- Outputs going change events → feeds back into predict step

### 14.4 Post-race analysis agent — Phase 6 (post-MVP)

After each race, analyses what the model got right and wrong.

**What it does**:
- Compares predictions to actual results
- Generates analysis: "Model correctly identified the winner at value. Key features driving the pick were going suitability (3/3 on soft) and trainer Festival record."
- Or: "Model's top value pick fell at the 3rd last — DNF rate was flagged at 12% but the model still ranked them #1. Consider increasing DNF penalty weight."
- Feeds learnings into a running log for post-festival model improvement

**Implementation**:
- `src/agents/post_race.py`
- Input: predictions + actual results + feature importances
- Uses Claude Sonnet for nuanced analysis
- Output: markdown analysis per race, accumulated in `data/analysis/`
- Optional: displayed on the site as a "results" tab

### 14.5 Ad-hoc Q&A — future

Natural language queries over the dataset:
- "Which trainers have the best Cheltenham Festival record in novice hurdles?"
- "Show me all horses that have won over 3m on soft ground at left-handed tracks"
- Uses Claude with tool use to generate and execute DuckDB SQL queries
- Not for MVP — nice for exploration and building intuition post-festival

### Tech choices for agentic layer

| Capability | Model | Why | When |
|---|---|---|---|
| Comment parsing | Claude Haiku | High volume, structured extraction, cost-efficient | Phase 2 (MVP) |
| Race previews | Claude Sonnet | Needs reasoning + narrative quality | Phase 6 |
| Conditions monitor | Claude Haiku | Simple extraction task | Phase 6 |
| Post-race analysis | Claude Sonnet | Nuanced analysis, small volume | Phase 6 |
| Ad-hoc Q&A | Claude Sonnet + tool use | SQL generation, reasoning | Future |

All via the **Anthropic Python SDK** (`anthropic` package). No framework needed — direct API calls with structured output schemas.

## 16. Open questions

1. ~~**BHA data access**~~ — deprioritised. rpscrape provides comprehensive results data including official ratings.
2. **Betfair historical data** — rpscrape's `-b` flag gets BSP/WAP/volume for historical races. Check if this is sufficient or if we also need Betfair's free CSV downloads for additional market features.
3. **How to handle the cross-country race** (Glenfarclas) — very different form profile, may need special treatment or exclusion.
4. **Breeding/pedigree data** — rpscrape provides sire/dam/damsire IDs and names for free. Include as features for novice races where horses have thin form? Low effort to add given data is already available.
5. **Staking recommendations** — do we show "bet X units" or just "this is value"? For MVP: just flag value, no staking advice.
6. **RP token refresh** — how often do `AUTH_STATE` / `ACCESS_TOKEN` expire? Document the refresh process clearly.
7. **Training data breadth** — Cheltenham-only (140 races) vs all UK/IRE NH (thousands). Trade-off: more data improves model generality but Cheltenham has unique characteristics (uphill finish, left-handed, altitude). Likely solution: train on broad data with course-specific features.
