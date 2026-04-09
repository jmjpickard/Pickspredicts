# Aintree 2026 Grand National Festival — Prediction Plan

**Festival dates:** Thursday April 9 — Saturday April 11, 2026
**Model:** V5 Optuna (67 features, market off)
**Betting strategy:** Top-1 pick every race (primary) + E/W on strong value bets (secondary)

## Timeline

### ~April 4 (Sat) — Full cards declared
- Re-run full pipeline to pick up all ~21 races (currently only 3 headline races declared)
```bash
.venv/bin/python -m src.pipeline --step fetch-racecards
.venv/bin/python -m src.pipeline --step normalise
.venv/bin/python -m src.pipeline --step features
.venv/bin/python -m src.pipeline --step predict
```

### ~April 7 (Tue) — First actionable predictions
- Betfair exchange markets should be open for most races
- Fetch exchange odds, re-run predict to get value scores
```bash
.venv/bin/python -m src.pipeline --step fetch-odds
.venv/bin/python -m src.pipeline --step features
.venv/bin/python -m src.pipeline --step predict
```
- Review initial strong value bets and top-1 picks
- Check `sp_rank` coverage in logs — should be >80%

### April 9/10/11 — Race mornings (8-9am)
- Non-runners declared, odds firmed overnight — this is the betting run
```bash
.venv/bin/python -m src.pipeline --step fetch-racecards   # picks up NRs
.venv/bin/python -m src.pipeline --step fetch-odds        # latest exchange odds
.venv/bin/python -m src.pipeline --step features
.venv/bin/python -m src.pipeline --step predict
```
- Place bets based on output

### Optional: ~30min before first race
- One final predict pass if odds have moved significantly

## Betting Rules

### Top-1 (primary)
- Back the model's top-ranked horse in every race, 1pt win
- ~21 races across 3 days = 21pt total outlay
- Backtest evidence: +73.7% ROI (2023), +103% ROI (2024), -29.1% ROI (2025)

### Strong value E/W (secondary)
- Back any horse with `value_score >= 0.05`, 1pt E/W
- Skip horses with `best_odds > 30` (too much variance)
- Skip if `win_prob < 0.05` (model not confident enough)
- Backtest evidence: +70.5% E/W ROI (2023), +49.2% (2024), +16.4% (2025)

### Overlap handling
- If the top-1 pick is also a strong value bet, back it 1pt win + 1pt E/W (2pt total)
- If a strong value bet is NOT the top-1, back it E/W only

## Pipeline Config

The pipeline is currently configured for Aintree in `configs/pipeline.yaml`:
```yaml
racecards:
  dates: ['2026-04-09', '2026-04-10', '2026-04-11']
  scoring_courses: [Aintree]
```

No retraining needed between runs — the model is fixed. Only re-run `predict` (and upstream steps if data changed).

## Current State (March 26)

- 3/21 races declared (Foxhunters', Topham, Grand National)
- 151 runners loaded
- 0% exchange odds coverage (markets not open)
- 0% going coverage (not declared)
- Initial top-1 picks are directional only — will shift when real odds arrive

### Preliminary Top-1 Picks (will change)

| Day | Race | Top Pick | Win% |
|-----|------|----------|------|
| Thu | Foxhunters' Chase | Whats The Solution | 4.1% |
| Fri | Topham Chase | Viroflay | 2.1% |
| Sat | Grand National | Jagwar | 2.5% |
