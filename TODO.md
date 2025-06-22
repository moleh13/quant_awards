# TODO

A concise, task-by-task checklist to implement the Graph-Guided, Regime-Aware RL framework
for crypto-portfolio optimization (Python scripts, local execution, Stable-Baselines3).

---

## 1. Repository & Environment
- [x] Initialize Git repository and `.gitignore`.
- [x] Create virtual environment (`venv` / `conda`) and install core libraries  
  (`pandas`, `numpy`, `torch`, `stable-baselines3`, `gymnasium`, `networkx`,  
  `torch_geometric`, `hmmlearn`, `pykalman`, `matplotlib`, `pytest`).
- [x] Set up base project tree:

graph_rl_portfolio/
├── agents/
├── data_pipeline/
├── environment/
├── graph/
├── models/
├── utils/
├── tests/
└── configs/

- [x] Add `config.py` (or `configs/default.yaml`) with global settings  
(asset list, date range, fees, learning rates, seeds, paths).

---

## 2. Data Pipeline
- [x] `data_loader.py` — fetch or load OHLCV for all assets, unify to DataFrame.
- [x] `preprocessor.py` — compute returns, log-prices, technical indicators.
- [x] `splitter.py` — create train / val / test date ranges w/out look-ahead bias.
- [x] CLI / script to dump preprocessed data to `data/` cache.
- [x] `tests/test_data_pipeline.py` — sanity-check shapes, NaNs, ranges.

---

## 3. Custom Trading Environment
- [x] `portfolio_env.py` (`gym.Env` subclass):
- [x] Define `observation_space` (features per asset + cash).
- [x] Define `action_space` (continuous weights; internal normalization).
- [x] Implement `reset()` (init capital, holdings, first obs).
- [x] Implement `step(action)`:
      * portfolio rebalancing & transaction cost
      * price update, reward (log return)
      * next observation, `done`, `info`
- [x] Support seeding and vectorized env compatibility (optional).
- [x] Quick smoke test with random actions.

---

## 4. Graph Construction & Features
- [x] `graph_builder.py` — build static correlation graph (`networkx`).
- [x] `graph_features.py` — Node stats: degree, centrality, cluster ID.
- [x] GNN embedding pipeline (GraphSAGE via `torch_geometric`).
- [x] Persist graph and embeddings for reuse.
- [x] Unit test to validate edge weights and embedding dims.

---

## 5. Regime & Signal Models
- [x] `regime_hmm.py` — fit HMM on market returns, output state labels / probs.
- [x] `kalman_filter.py` — Kalman smoother for latent trend / volatility.
- [x] Interface to query current regime + Kalman signals per timestep.
- [x] Ensure no future leakage (only past data used at inference).

---

## 6. RL Agents
- [x] Baseline MLP policy:
- [x] `train_baseline.py` — PPO/SAC with simple MLP `policy_kwargs`.
- [x] `evaluate_baseline.py` — test-set run, store metrics.
- [ ] Graph-aware policy:
- [ ] `graph_policy.py` — custom `ActorCriticPolicy` with GraphSAGE encoder.
- [ ] `train_graph_agent.py` — use custom policy; load graph embeddings.
- [ ] Adaptive (regime + Kalman) policy:
- [ ] Extend env to append regime/Kalman features to observation.
- [ ] `train_final_agent.py` — train agent on full feature set.
- [x] Model checkpoint saving / loading utilities.

---

## 7. Evaluation & Benchmarking
- [ ] `metrics.py` — ROI, Sharpe, Sortino, Max Drawdown, Calmar.
- [ ] `compare_strategies.py` — run baseline, graph, final adaptive, plus  
    equal-weight & buy-and-hold; produce table of metrics.
- [ ] `plots.py` — equity curves, bar charts, regime vs exposure overlay.
- [ ] CSV / JSON logging of experiment results.

---

## 8. Utilities & Helpers
- [ ] `seed_everything(seed)` function (numpy, torch, random).
- [ ] Action normalization / clipping helper.
- [ ] Transaction-cost calculator.
- [ ] Config loader (YAML → dict).

---

## 9. Testing & CI
- [ ] Add `pytest` unit tests for:
    * Data loader & preprocessor
    * Env step reward correctness
    * Graph feature dimensions
    * HMM state-count integrity
- [ ] Lightweight integration test:  
    run 100 env steps with random policy → no exceptions.
- [ ] Optional: set up GitHub Actions workflow for lint + tests.

---

## 10. Documentation & Wrap-Up
- [ ] Inline docstrings for all public classes / functions.
- [ ] `README.md` — quick-start, dependency list, run commands.
- [ ] `run_full_pipeline.py` — one-click script: preprocess → train final → evaluate.
- [ ] Final refactor for clarity, remove unused code, ensure reproducibility.

---