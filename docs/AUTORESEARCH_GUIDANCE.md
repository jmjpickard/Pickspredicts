# Autoresearch Agent Guidance: Model Optimisation Best Practices

This document is the reference guide for the autoresearch agent (and human reviewers) when deciding what configuration changes to propose. It covers LightGBM tuning, feature selection, calibration, ensemble methods, and pitfalls specific to our problem: binary classification with ~8-10% positive rate, small temporal validation windows (~300-450 runners across 20-30 races per festival), and a composite objective (log loss + MRR).

---

## 1. LightGBM Tuning for Small Datasets with Class Imbalance

### 1.1 The Core Tension

With small validation sets (300-450 runners), any hyperparameter change that improves score by < 0.01 is likely noise. The agent must be sceptical of small improvements and prefer configurations that are robust across all 6 windows over those that spike on 1-2 windows.

### 1.2 Parameter Guidance (Specific to Our Problem)

**`num_leaves` [16, 96] — Current default: 31**
- This is the single most important regularisation lever. Each leaf partitions the data further; with ~8% positive rate and small datasets, deep trees memorise noise.
- **Recommended range: 16-40.** Values above 48 are almost certainly overfitting on our data size.
- Rule of thumb: `num_leaves` should be < `2^(max_depth)`. With our data, effective max_depth of 4-5 is plenty, so 16-32 leaves is the sweet spot.
- If the model is underfitting (high log loss, low MRR across all windows), try increasing toward 40. If overfitting (good on some windows, terrible on others), decrease toward 16-24.

**`learning_rate` [0.01, 0.15] — Default: 0.05**
- Lower learning rates + more iterations = better generalisation, but diminishing returns below 0.02 with early stopping at 50.
- **Sweet spot: 0.02-0.05.** Below 0.02, early stopping may fire before the model has learned enough. Above 0.08, the model overfits in few iterations.
- Learning rate and num_leaves interact: if you lower learning_rate, you may need to increase `num_iterations` or decrease `early_stopping_rounds` sensitivity. But our fixed 2000 iterations + 50-round early stopping is already generous.

**`feature_fraction` [0.5, 1.0] — Default: 0.8**
- Column subsampling per tree. Acts as regularisation AND decorrelation between trees.
- **Recommended: 0.6-0.8.** With 49 features and many correlated features (e.g., or_current / rpr_current / ts_current are highly correlated), lower feature_fraction forces the model to use diverse features.
- Below 0.5 risks excluding critical features from too many trees. Above 0.9 reduces the regularisation benefit.

**`bagging_fraction` [0.5, 1.0] — Default: 0.8**
- Row subsampling per tree. Critical for our imbalanced problem.
- **Recommended: 0.7-0.85.** Lower values (0.6-0.7) add more randomness, which helps when the model is overfitting to specific patterns in the training data.
- Must be paired with `bagging_freq > 0` (we use 5, which is fine).

**`min_child_samples` [5, 50] — Default: 20**
- Minimum number of samples in a leaf. The primary guard against overfitting on rare patterns.
- **Recommended: 15-35.** With ~8% positive rate, a leaf needs at least 12-15 samples to have a stable estimate of win probability.
- Lower values (5-10) allow the model to create highly specific leaves that may capture real signals OR noise — dangerous with small val sets.
- Higher values (30-50) force coarser splits, which improves generalisation but may miss real interactions.

### 1.3 Parameters NOT in Search Space (But Should Be Considered)

These are currently fixed but could yield improvements if added to the search space:

- **`scale_pos_weight`**: Set to `(n_negative / n_positive)` to upweight the minority class. For ~8% positive rate, this would be ~11.5. Alternatively, use `is_unbalance=True` which auto-calculates this. *Caution: this changes the raw probabilities and may hurt log loss even if it helps MRR. Requires recalibration.*
- **`lambda_l1` (L1 regularisation)**: Range [0, 5]. Encourages sparsity — effectively does feature selection during training. Start with 0.1-1.0.
- **`lambda_l2` (L2 regularisation)**: Range [0, 5]. Smooths predictions. Start with 0.1-2.0.
- **`max_depth`**: Explicitly cap tree depth. Range [-1 (unlimited), 3-6]. Setting to 4 or 5 with num_leaves=24-32 provides double regularisation.
- **`min_gain_to_split`**: Range [0, 1.0]. Prevents splits that don't improve the objective enough. A value of 0.01-0.1 can prune noisy splits.
- **`path_smooth`**: Range [0, 10]. Smooths leaf values, particularly useful for small datasets. Higher values = more smoothing. Try 1-5.
- **`extra_trees`** (bool): If True, uses Extra Trees algorithm which adds randomness to split point selection. Good for reducing overfitting on small data.

### 1.4 Class Imbalance Strategies

For ~8-10% positive rate, the imbalance is moderate (not extreme like fraud detection at 0.1%). Strategies in order of priority:

1. **Do nothing special** (current approach) — LightGBM handles moderate imbalance reasonably well with binary log loss. The softmax normalisation per race already adjusts for within-race ranking.
2. **`is_unbalance=True`** — simple, but changes raw probabilities. Test whether it helps MRR at the expense of log loss.
3. **`scale_pos_weight`** — more control than `is_unbalance`. Try values in [5, 15].
4. **Focal loss** (custom objective) — down-weights easy negatives, focuses on hard examples. Useful if the model is confidently wrong on certain losers. Implementation: `objective=None`, pass custom `fobj` function. `gamma` in [1, 3] for our imbalance level.

### 1.5 Anti-Patterns to Avoid

- **Chasing small score improvements**: With 300-450 validation runners, a score improvement of < 0.005 is within noise. Require improvement on at least 4/6 windows to accept a change.
- **Correlated parameter changes**: Don't propose num_leaves=64 + learning_rate=0.01 in quick succession. Change one, observe, then adjust the other.
- **Extremely low learning rates**: Below 0.015, the 50-round early stopping will fire too early. Either reduce early_stopping_rounds or increase patience.
- **num_iterations too low**: With early stopping, having 2000 max is fine — the model will stop early. Don't reduce this.

---

## 2. Hyperparameter Search Strategy

### 2.1 Random Search vs Bayesian Optimisation vs Agent-Guided

**Random Search (current `--provider random`)**:
- Pros: No overhead, embarrassingly parallel, surprisingly competitive for < 20 parameters.
- Cons: Doesn't learn from history, wastes evaluations on unpromising regions.
- When to use: As a baseline, or for initial exploration when you have no prior knowledge.

**Bayesian Optimisation (e.g., Optuna, SMAC)**:
- Pros: Builds a surrogate model of the objective surface, proposes promising points, handles noisy objectives.
- Cons: Overhead per iteration, surrogate model needs ~10-20 evaluations to be useful, struggles with discrete/boolean parameters.
- Best for: Continuous hyperparameters (learning_rate, feature_fraction, bagging_fraction). Less useful for feature group toggles.
- If integrating: Use Optuna with TPE sampler, set `n_startup_trials=10` for random exploration first.

**Agent-Guided Search (current `--provider openrouter`)**:
- Pros: Can reason about parameter interactions, learn from rejection patterns, avoid known bad regions.
- Cons: LLM may hallucinate patterns, limited by prompt context window, can get stuck in local optima.
- Best for: Our setup — small search space (5 continuous + 7 boolean = 12 parameters), expensive evaluation (~30-60s per iteration), human-interpretable reasoning.

### 2.2 Agent Search Strategy Recommendations

The agent should follow this exploration schedule:

**Phase 1 (iterations 0-10): Coarse grid exploration**
- Try extreme values of each continuous parameter to understand the response surface.
- Example: num_leaves=16, then 48, then 64 — to see if the objective is monotonic or has a minimum.
- Test each feature group toggle at least once.

**Phase 2 (iterations 10-30): Focus on winners**
- Narrow the search to parameters that showed sensitivity in Phase 1.
- If num_leaves mattered but feature_fraction didn't, spend more time on num_leaves.
- Try 2-3 points in the promising region of each sensitive parameter.

**Phase 3 (iterations 30+): Fine-tuning + interaction effects**
- Small adjustments to the best configuration.
- If num_leaves=24 was best, try 20 and 28.
- Consider interaction effects: low learning_rate + higher num_leaves vs high learning_rate + lower num_leaves.

**Meta-rules for the agent**:
- Never propose the exact same change that was previously rejected.
- If the last 5 iterations were all rejected, try a larger step (explore more aggressively).
- Track which feature groups have been toggled off successfully vs which degraded the model — this is strong signal.
- If the best score hasn't improved in 10 iterations, the model may be near-optimal for this search space. Consider stopping.

---

## 3. Feature Selection Strategies

### 3.1 Overview of Methods

Our current approach (feature group toggles) is a coarse-grained feature selection. Finer-grained methods exist:

**SHAP-based importance**:
- Compute SHAP values on validation set after training.
- Rank features by mean |SHAP value|.
- Remove features with near-zero SHAP importance.
- **Pros**: Accounts for feature interactions, theoretically grounded.
- **Cons**: Computationally expensive for tree models (O(TLD^2) per sample), can be noisy on small val sets.
- **Practical tip**: Use `model.predict(data, pred_contrib=True)` in LightGBM for fast TreeSHAP. This is already available in the codebase.

**Permutation importance**:
- Shuffle each feature column in the validation set, measure increase in loss.
- Features where shuffling doesn't increase loss are dispensable.
- **Pros**: Model-agnostic, directly measures predictive impact, fast.
- **Cons**: Correlated features share importance (shuffling one doesn't hurt if the correlated partner remains). This is relevant for our ratings features (or_current, rpr_current, ts_current).
- **Practical tip**: Run 5-10 repetitions and check standard deviation. Features where the importance confidence interval includes zero should be candidates for removal.

**Boruta**:
- Adds "shadow features" (random permutations of real features), trains model, keeps features that are consistently more important than the best shadow feature.
- **Pros**: Automated threshold, statistically principled.
- **Cons**: Requires many iterations (50-100), computationally expensive.
- **Practical tip**: Use `BorutaPy` with LightGBM as the estimator. Set `max_iter=100`, `perc=100`. Features marked "rejected" should be dropped.

**LightGBM native importance** (split count or gain):
- `feature_importance(importance_type='gain')` — total gain from splits on this feature.
- `feature_importance(importance_type='split')` — number of times feature was used for splitting.
- **Pros**: Free, fast.
- **Cons**: Biased toward high-cardinality features, doesn't account for interactions well.
- **Practical tip**: Use gain, not split. Features with zero gain across all training runs are safe to remove.

### 3.2 Recommendations for Our Setup

1. **Start with group-level toggles** (current approach). This is appropriate given our small val sets — individual feature removal is too noisy to evaluate reliably.
2. **Within accepted groups, prune zero-importance features**. After finding the best config, train the final model and check LightGBM gain importance. Remove any feature with zero gain — it's never used.
3. **Watch for the comments group**. The learnings file notes that comment-derived features (dominant_style, pct_trouble, pct_jumping_issues) previously degraded the model. The agent should try toggling this group off early in the search.
4. **Correlated feature groups**: The "ratings" group has highly correlated features. If this group is on, consider whether `feature_fraction < 0.7` helps by forcing the model to not always pick the same correlated feature.

### 3.3 Feature Engineering Signals (Beyond Selection)

If the agent notices that certain feature groups consistently help or hurt:
- **Market features helping a lot**: The model is partially learning to copy the market. This is fine for probability calibration (market is well-calibrated) but limits edge discovery.
- **Market features hurting**: Possible data leakage concern — ensure market_implied_prob is from pre-race prices, not in-running.
- **Pedigree features hurting**: Sire stats may be too sparse (few observations per sire at specific course/going/distance combos). Consider dropping if consistently rejected.
- **Connections features**: Trainer/jockey form windows (14d, 30d, 90d) may be redundant — the model often picks just one window. But the group as a whole tends to carry signal for festival specialists.

---

## 4. Calibration Techniques Post-Training

### 4.1 Why Calibration Matters

LightGBM's raw output with binary log loss is already reasonably well-calibrated (log loss explicitly optimises for it). However:
- The softmax normalisation per race (current approach) transforms probabilities into a within-race distribution that may not be well-calibrated.
- Small validation sets make calibration evaluation unreliable — reliability diagrams with 300 samples are very noisy.

### 4.2 Methods

**Platt Scaling (Logistic Calibration)**:
- Fits a logistic regression on raw model outputs: `P(y=1|f(x)) = 1 / (1 + exp(-(a*f(x) + b)))`.
- Two parameters (a, b), so very robust with small datasets.
- **Best for our case**: Only 2 free parameters, works well when the model's uncalibrated outputs are approximately sigmoidal.
- Implementation: `sklearn.calibration.CalibratedClassifierCV(method='sigmoid', cv='prefit')`.
- **Pitfall**: Must fit on a held-out calibration set, NOT the training set. With small val sets, use leave-one-window-out: fit Platt on 5 windows, evaluate on the 6th.

**Isotonic Regression**:
- Non-parametric monotonic mapping from raw scores to calibrated probabilities.
- **Avoid for our case**: Requires 1000+ samples for stable estimation. With 300-450 runners per window, isotonic regression will overfit severely.
- Only consider if pooling all validation data across windows (6 * ~350 = ~2100 samples).

**Temperature Scaling**:
- Single parameter T: `P(y=1) = sigmoid(logit(p) / T)`.
- **Good for our case**: Even simpler than Platt (1 parameter), less risk of overfitting.
- Optimise T by minimising log loss on a held-out set. Typical range: T in [0.5, 2.0].
- Implementation: Grid search T from 0.5 to 2.0 in steps of 0.05, pick the T that minimises pooled log loss across 5/6 windows.

**Beta Calibration**:
- Fits a beta distribution mapping, 3 parameters. More flexible than Platt, less prone to overfitting than isotonic.
- Worth trying if Platt doesn't improve calibration.

### 4.3 Practical Recommendations

1. **First check if calibration is actually needed**: Plot reliability diagrams on pooled validation data. If the model is already well-calibrated, adding a calibration layer adds complexity with no benefit.
2. **If calibrating, use temperature scaling first** (1 parameter, minimal overfitting risk).
3. **If temperature scaling helps, try Platt** (2 parameters) to see if the additional degree of freedom helps.
4. **Never use isotonic regression** on individual festival windows. Only on pooled data with 2000+ samples.
5. **Calibrate BEFORE softmax normalisation**, not after. The softmax step should receive well-calibrated probabilities as input.

### 4.4 Integration with the Autoresearch Loop

Calibration is a post-processing step, not a hyperparameter. The autoresearch loop should:
1. Train and evaluate with the standard pipeline (no calibration) to get the best hyperparameters.
2. After the loop completes, apply calibration as a final polishing step.
3. Do NOT tune calibration parameters inside the autoresearch loop — it would add noise to an already noisy evaluation.

---

## 5. Ensemble Methods for Small Validation Sets

### 5.1 Why Ensembles Help

Ensembles reduce variance, which is exactly what we need with noisy small validation sets. A single model's predictions on 300 runners have high variance; averaging 5 models reduces this.

### 5.2 Approaches

**Seed Ensemble (Simplest, Recommended First)**:
- Train 3-5 models with different `seed` values, average their predictions.
- Reduces variance with zero additional complexity.
- Typical improvement: 0.5-2% in log loss.
- Implementation: In `_eval_window`, train 5 models with seeds [42, 43, 44, 45, 46], average `raw_prob`.
- **Cost**: 5x training time per evaluation. For autoresearch iterations this may be too slow, but for the final model it's worthwhile.

**Bagging Ensemble**:
- Train models on different bootstrap samples of the training data.
- Essentially what `bagging_fraction < 1.0` already does within LightGBM, but more explicit.
- Not recommended on top of LightGBM's internal bagging — diminishing returns.

**Window-Weighted Ensemble**:
- Train a separate model for each festival window (or a sliding window), weight predictions by recency.
- For predictions on Cheltenham 2026, weight the Cheltenham-trained models higher than Aintree-trained models.
- Risk: With only 3 Cheltenham windows, this is very sample-starved.

**Stacking (Avoid)**:
- Train a meta-model on out-of-fold predictions from base models.
- Requires a separate held-out set for the meta-model, which we don't have.
- With 300-450 validation samples, the meta-model will overfit.

### 5.3 Practical Recommendation

For the autoresearch loop: **Don't ensemble** — it masks the signal from individual hyperparameter changes and makes it harder to attribute improvements.

For the final production model: **Seed ensemble of 5 models**. Train 5 LightGBM models with different seeds, average their raw probabilities, then apply softmax normalisation per race.

---

## 6. Walk-Forward Validation: Pitfalls and Best Practices

### 6.1 Current Setup Assessment

The current walk-forward setup (train on all data before festival start, evaluate on that festival) is sound. Key strengths:
- No future data leakage: training strictly before validation.
- 6 independent windows provide some robustness.
- Mix of courses (Cheltenham + Aintree) tests generalisation.

### 6.2 Common Pitfalls

**Pitfall 1: Temporal feature leakage**
- Features computed from the full dataset (e.g., career win rates) might include post-validation data if not carefully windowed.
- **Check**: Ensure all features for a validation runner are computed only from data available before the festival start date. This includes trainer_winpct, jockey_winpct, sire stats, etc.
- **Common mistake**: Using the entire dataset to compute global statistics (like overall sire win rate) and then using that for both training and validation.

**Pitfall 2: Look-ahead bias in feature engineering**
- If the feature pipeline processes all data at once, rolling window features (e.g., `or_trend_last5`, `btn_trend_last5`) might inadvertently include future data.
- **Check**: For each validation window, verify that the feature values for validation runners are identical whether computed on the full dataset or only on data available at prediction time.

**Pitfall 3: Small sample size instability**
- 20-30 races with ~12-15 runners each = 300-450 validation samples. MRR is averaged over only 20-30 races.
- A single race where the model happens to rank the winner first vs third changes MRR by ~0.02-0.03.
- **Mitigation**: Pool metrics across all 6 windows (current approach is good). Also consider bootstrap confidence intervals on the pooled metrics.

**Pitfall 4: Non-stationarity**
- Horse racing has trends: going conditions vary by year, star horses retire, training methods evolve.
- Older training data may contain patterns that no longer apply.
- **Mitigation**: Consider down-weighting old data (e.g., exponential decay on sample weights) or simply limiting training to the most recent 3-4 years.

**Pitfall 5: Overfitting the validation protocol itself**
- Running 100+ iterations of autoresearch against the same 6 windows is itself a form of overfitting — you're optimising for those specific windows.
- **Mitigation**: Hold out 1-2 windows as a "test set" that the agent never sees. Only evaluate the final config on these held-out windows.
- Practical: Use 4 windows for the autoresearch loop, hold out 2 (e.g., cheltenham_2025 + aintree_2025) for final evaluation.

**Pitfall 6: Survivorship bias in training data**
- Non-runners are excluded from training, but their absence affects field dynamics.
- Last-minute non-runners change the competitive landscape for remaining runners.
- **Not easily fixable**, but worth noting — don't overfit to small field races where one withdrawal changes everything.

### 6.3 Expanding the Validation Protocol

To get more robust estimates:

1. **Add more festival windows**: If data is available, add Punchestown and Leopardstown festivals. Even though the target is Cheltenham, more validation windows = more robust tuning.
2. **Synthetic validation windows**: Use non-festival race days at Cheltenham as additional validation windows. The distribution is different (smaller fields, lower class), but it adds sample size.
3. **Leave-one-window-out cross-validation**: Instead of pooling all 6 windows, use 5 for training hyperparameter evaluation and 1 for testing. Rotate. This gives 6 estimates of generalisation performance.

---

## 7. Learning-to-Rank vs Pointwise Binary Classification

### 7.1 The Fundamental Choice

Our scoring function values both calibration (log loss) and ranking (MRR). These objectives can conflict:

- **Pointwise (current)**: Binary classification — predicts P(win) independently for each runner. Good for calibration, indirectly good for ranking.
- **Pairwise (LambdaRank)**: Directly optimises the ordering of runners within a race. Better for ranking metrics (MRR, NDCG), but probabilities need calibration post-hoc.
- **Listwise (LambdaMART)**: Optimises the entire list ordering. Most sophisticated, best for ranking, but requires careful setup.

### 7.2 LightGBM's Ranking Modes

LightGBM supports ranking natively:

```
objective: lambdarank
metric: ndcg
eval_at: [1, 3]
label_gain: [0, 1]  # for binary relevance (0=loser, 1=winner)
```

The training data needs a `group` column specifying how many runners are in each race (for grouping).

**Pros of switching to lambdarank**:
- Directly optimises the ranking — may improve MRR significantly.
- Naturally handles the within-race structure.
- Can use NDCG@1 as the metric, which is equivalent to "did we rank the winner first?".

**Cons of switching to lambdarank**:
- Raw outputs are scores, not probabilities. Needs Platt/isotonic calibration for log loss.
- The calibration step adds noise on small validation sets.
- More complex to set up (group column, relevance labels).

### 7.3 Recommendation

**Short-term (current autoresearch loop): Stay with pointwise binary classification.**
- It's simpler, already well-calibrated, and the 0.5*MRR term in the score partially captures ranking quality.
- The current search space is focused on the right things.

**Medium-term experiment: Try lambdarank as a separate branch.**
- Set up a parallel evaluation with `objective=lambdarank`, `metric=ndcg`, `eval_at=[1]`.
- Compare MRR (should improve) and log loss (will likely degrade without calibration).
- If MRR improves enough, the composite score may still be better.

**Hybrid approach (worth trying)**:
- Train two models: one binary (for calibration), one lambdarank (for ranking).
- Final prediction: weighted average of the two models' outputs (after calibrating the lambdarank model).
- Weight tuning: start with 0.5/0.5, then optimise the weight on the composite score.

### 7.4 LambdaRank-Specific Parameters

If experimenting with ranking:
- `lambdarank_truncation_level`: How many positions to consider. Set to field_size or slightly less.
- `lambdarank_norm`: Normalise lambda gradients. Set to `True` for more stable training.
- `label_gain`: For binary (win/no-win), use `[0, 1]`. For place relevance, use `[0, 0.5, 1]` (3rd, 2nd, 1st).

---

## 8. Avoiding Overfitting on Small Temporal Validation Windows

### 8.1 The Meta-Overfitting Problem

The autoresearch loop evaluates ~100 configs against 6 fixed windows. This is equivalent to running 100 hypothesis tests — by chance, some configs will appear better. This is **multiple comparisons overfitting** — you're overfitting the hyperparameters to the validation set.

### 8.2 Concrete Mitigations

**1. Require consistent improvement across windows**

Don't just check pooled score. Track per-window metrics. A good change should improve at least 4/6 windows. If a change improves 2 windows dramatically but hurts the other 4, it's likely overfitting to those 2.

Implementation hint for the scoring function:
```
# Instead of just: score = pooled_ll - 0.5 * pooled_mrr
# Also check: improved_windows >= 4 out of 6
```

**2. Use a significance threshold**

Don't accept changes that improve score by < 0.005. With our sample sizes, improvements smaller than this are noise.

**3. Limit total iterations**

More iterations = more chances to find spurious improvements. Cap at 50-80 iterations for the agent. If no improvement in the last 15-20 iterations, stop early.

**4. Hold-out window**

Remove 1-2 windows from the autoresearch loop. After the loop finishes, evaluate the best config on the held-out windows. If performance degrades significantly, the loop overfit.

Recommended hold-out: `cheltenham_2025` and `aintree_2025` (most recent, most representative of future conditions).

**5. Regularisation bias**

When in doubt, prefer the more regularised configuration. Between two configs with similar scores:
- Prefer lower num_leaves
- Prefer higher min_child_samples
- Prefer more feature groups enabled (less information discarded)
- Prefer parameter values closer to defaults

**6. Bootstrap validation**

For a more robust evaluation, bootstrap the validation set:
- For each window, resample races with replacement 100 times.
- Compute score on each bootstrap sample.
- Use the mean and 95% CI of the bootstrap scores.
- A change is "real" only if the bootstrap CIs don't overlap.

This is computationally expensive (100x evaluation), so use it only for final model validation, not during the search loop.

### 8.3 Practical Decision Framework for the Agent

When proposing a change, the agent should classify it as:

1. **Exploration** (trying something new to gather information): Accept even if it doesn't improve score. The goal is to map the response surface.
2. **Exploitation** (refining a parameter that showed promise): Require improvement > 0.005 to accept.
3. **Validation** (confirming a previously accepted change is robust): Re-test with a different seed or slightly different value. If the nearby region also performs well, the signal is real.

### 8.4 Signs of Overfitting to Watch For

- **Window variance increasing**: If the standard deviation of per-window scores is increasing with iteration count, the model is specialising to a subset of windows.
- **Score improving but per-window consistency degrading**: Pooled score can improve even when individual windows get worse (the pool is an average).
- **Feature group toggling instability**: If the optimal set of feature groups changes with every iteration, the features are adding noise rather than signal.
- **Very low num_leaves being best**: If the model keeps preferring fewer leaves (< 16), it's a sign that more complex models overfit — the signal-to-noise ratio in the data is low.

---

## 9. Additional Search Space Recommendations

### 9.1 Parameters to Add to the Search Space

Based on the analysis above, these parameters would be high-value additions:

| Parameter | Type | Range | Rationale |
|-----------|------|-------|-----------|
| `lgbm.lambda_l1` | log_float | [0.001, 5.0] | L1 regularisation, encourages sparsity |
| `lgbm.lambda_l2` | log_float | [0.001, 5.0] | L2 regularisation, smooths predictions |
| `lgbm.max_depth` | int | [3, 8] | Hard cap on tree depth, complements num_leaves |
| `lgbm.min_gain_to_split` | log_float | [0.001, 1.0] | Prevents noisy splits |
| `lgbm.path_smooth` | float | [0, 10.0] | Leaf value smoothing, strong for small data |
| `lgbm.is_unbalance` | bool | - | Auto-handles class imbalance |
| `lgbm.bagging_freq` | int | [1, 10] | How often to bag |

### 9.2 Composite Score Tuning

The current score formula is:
```
score = pooled_log_loss - 0.5 * pooled_mrr
```

The 0.5 weight on MRR is a design choice. Consider:
- **Higher MRR weight (0.7-1.0)**: If the primary use case is picking winners (e.g., for betting), ranking quality matters more than calibration.
- **Lower MRR weight (0.2-0.3)**: If the primary use case is value betting (comparing model probability to market odds), calibration is paramount.
- The current 0.5 is a balanced default.

### 9.3 Early Stopping Refinements

The current `early_stopping_rounds=50` is reasonable. Considerations:
- With `learning_rate=0.03`, 50 rounds of no improvement represents ~1.5 units of "learning" (50 * 0.03). This is appropriate.
- If learning_rate is reduced to 0.01, consider increasing early_stopping_rounds to 100 to avoid premature stopping.
- Rule of thumb: `early_stopping_rounds * learning_rate` should be roughly 1.0-2.0.

---

## 10. Summary: Priority-Ordered Action List for the Agent

1. **Toggle off comments feature group** — previously shown to degrade model.
2. **Reduce num_leaves to 20-28 range** — most impactful regularisation lever.
3. **Reduce learning_rate to 0.02-0.04** — paired with sufficient iterations.
4. **Reduce feature_fraction to 0.6-0.7** — decorrelates trees, helps with correlated ratings features.
5. **Try min_child_samples in 20-35 range** — stabilises leaf predictions.
6. **Test bagging_fraction at 0.7** — slight additional regularisation.
7. **Consider adding lambda_l1/l2 to search space** — strong regularisation, especially lambda_l1 for sparsity.
8. **Toggle pedigree group** — potentially sparse, test whether it helps.
9. **After finding best config, apply seed ensemble** (5 seeds) for final model.
10. **After finding best config, try temperature scaling** as a final calibration step.
