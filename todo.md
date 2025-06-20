# TODO – Graph-Guided RL Crypto Portfolio Optimization

> **Goal:** Build and evaluate a regime-aware, graph-guided deep reinforcement-learning framework that dynamically allocates between traditional cryptocurrencies and DeFi tokens, then package results for Quant Awards submission.

---

## 0. Project Setup
- [ ] Create project repo & directory structure (`data/`, `src/`, `notebooks/`, `reports/`, `models/`, `figures/`)
- [ ] Define Python environment (conda/poetry) with required libraries  
  - core: `numpy`, `pandas`, `scipy`, `scikit-learn`  
  - RL: `stable-baselines3`, `gymnasium` / `gym`  
  - deep learning: `torch`  
  - graph: `networkx`, `torch-geometric` (optional)  
  - plotting: `matplotlib`, `seaborn`, `plotly`  
  - API: `python-binance`
- [ ] Configure `.env` for Binance API keys
- [ ] Set up experiment tracking (Weights & Biases / MLflow)  

## 1. Literature Review & Conceptual Design
- [ ] Survey RL portfolio-allocation papers (esp. crypto)
- [ ] Review regime-switching & volatility-state models for crypto
- [ ] Study graph/network approaches in finance
- [ ] Analyse existing Quant Awards papers to ensure originality
- [ ] Finalize conceptual architecture & key hypotheses

## 2. Data Pipeline
- [ ] **Acquisition**  
  - [ ] Write Binance API wrapper to download OHLCV for ≥ 30 major assets (BTC, ETH, etc.) + DeFi tokens (UNI, AAVE, SUSHI, COMP, …)
  - [ ] Fetch at uniform resolution (e.g., 1 h / 4 h)
  - [ ] Store raw data in `data/raw/`
- [ ] **Pre-processing**  
  - [ ] Clean & align timestamps, handle missing candles
  - [ ] Compute returns, log-returns, vol, volume features
  - [ ] Resample if multiple granularities needed
  - [ ] Save processed data (`data/processed/`)

## 3. Market-Regime Detection Module
- [ ] Engineer candidate regime indicators (rolling vol, drawdown, momentum)
- [ ] Train/fit regime-switching model  
  - Option A: Hidden Markov Model (HMM) on volatility states  
  - Option B: LSTM classifier predicting bull/bear/turbulent
- [ ] Validate regime labels & visualise timeline
- [ ] Expose callable function `get_current_regime(t)`

## 4. Graph Construction & Feature Extraction
- [ ] Build dynamic correlation / mutual-information network per rolling window
- [ ] Apply community detection (Louvain / Leiden) to tag tokens
- [ ] Compute node centrality metrics (degree, betweenness, eigenvector)
- [ ] Serialize graph features for each timestep
- [ ] Benchmark graph stability over regimes

## 5. RL Environment
- [ ] Implement custom `gym` environment  
  - Observation = concatenated market features + regime label + graph features  
  - Action = portfolio weight vector (long-only / leverage cap)  
  - Reward = risk-adjusted return (e.g., return – λ·CVaR)
- [ ] Add transaction-cost & turnover penalties
- [ ] Unit-test environment for edge cases (NaNs, weight normalization)

## 6. Agent Development & Training
- [ ] Select base algorithm (PPO / SAC) with risk-aware objective
- [ ] Integrate graph-guided attention (optional: GNN encoder)
- [ ] Configure hyper-parameter search (Optuna / wandb sweep)
- [ ] Train on rolling walk-forward splits
- [ ] Log metrics, checkpoints

## 7. Backtesting & Evaluation
- [ ] Compare RL strategy vs benchmarks  
  - Static BTC/ETH 60/40  
  - Equal-weight Top N  
  - Mean-variance optimal (ex-ante)  
  - Previous RL w/o regimes or graph
- [ ] Metrics: CAGR, Sharpe, Sortino, Max DD, CVaR₉₅, turnover
- [ ] Perform Diebold-Mariano tests for significance
- [ ] Stress-test on extreme periods (e.g., Mar-2020, Nov-2022 FTX)

## 8. Hypothesis Testing & Ablation Studies
- [ ] H1: regime-aware > single-regime  
- [ ] H2: DeFi inclusion improves mean-variance frontier  
- [ ] H3: graph constraints reduce tail risk
- [ ] Run ablations: remove graph features, remove regimes, remove DeFi tokens
- [ ] Summarise statistical outcomes

## 9. Documentation & Reporting
- [ ] Draft paper structure (Abstract, Intro, Methodology, Experiments, Results, Conclusion)
- [ ] Generate figures: equity curves, heatmaps, network diagrams
- [ ] Create tables of performance & significance tests
- [ ] Write reproducibility instructions & provide GitHub repo
- [ ] Prepare Quant Awards submission (PDF + code link + abstract)

## 10. Timeline & Milestones
| Week | Milestone |
|------|-----------|
| 1    | Env setup & literature matrix |
| 2-3  | Data pipeline finished |
| 4    | Regime detection prototype |
| 5    | Graph module complete |
| 6-7  | RL env + baseline agent |
| 8-9  | Full training & sweeps |
| 10   | Backtesting & evaluation |
| 11   | Paper drafting |
| 12   | Final edits & submission |

---

_Last updated: {20.06.2025}_