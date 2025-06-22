10-Week Roadmap:
 Graph-Guided RL for Adaptive Crypto Portfolio Optimization
Week 1: Preliminary Research & Project Setup
Literature Review: Survey recent research on deep reinforcement learning (DRL) in portfolio management, regime-switching models in finance, and graph-based market analysis. Note findings such as the benefit of adding graph neural network features to a PPO trading agent and the promise of regime-aware RL strategies for cryptocurrencies . Collect relevant papers (e.g., on HMM regimes, Kalman filters in trading) to guide methodology.


Project Scope & Plan: Define the project objectives and components: an RL agent (PPO/SAC) that adjusts a crypto portfolio, augmented by regime dynamics (market state shifts) and graph-guided features (relationships among DeFi tokens). Break down the approach (data -> features -> models -> backtest -> report) and map these to a timeline. Decide on the crypto assets to include (e.g., a selection of major DeFi tokens plus BTC/ETH for context) and the frequency of data (e.g., daily closing prices and volumes from Binance).


Environment Setup: Configure the Python development environment. Install key libraries: data access (e.g. python-binance or CCXT for Binance API), data handling (pandas, numpy), modeling (hmmlearn for HMM, pykalman for Kalman filter, networkx or PyG for graph features, stable-baselines3 for RL algorithms, etc.). Set up version control (Git) and create a GitHub repository for the project to track code and ensure reproducibility.


Quant Awards Requirements: Read the Quant Awards 2025 guidelines to internalize report expectations. Note that the final paper must emphasize importance and practical application over technical detail , be 5–7 pages long (excluding appendices) in PDF, and have an anonymous cover page (title only, no name or university) . Plan to align the project with judging criteria: applicability, innovation, completeness, and presentation . This will shape how results are evaluated and presented later.


Week 2: Data Collection and Preprocessing
Historical Data Gathering: Use the Binance API (or stored CSV datasets if available) to collect historical price and volume data for the chosen crypto portfolio. Include multiple DeFi tokens (e.g., UNI, AAVE, MKR, COMP, etc.) along with reference assets (BTC, ETH) to allow diverse regime behaviors. Fetch data over a substantial period (e.g., 3-5 years if available) to capture different market cycles.


Data Cleaning: Process the raw data into a consistent format (e.g., daily time series of OHLCV – Open, High, Low, Close, Volume). Handle missing entries (fill or drop), adjust for splits or outliers if any, and align time indices across assets (ensure synchronous dates). Compute derived metrics like daily returns and volatilities for each asset.


Exploratory Analysis: Perform initial analysis to understand the data. Plot price trajectories and log-returns; examine correlations between assets and basic statistics (mean, variance, max drawdowns) to identify any obvious regimes (e.g., bull vs. bear periods). This helps in defining regimes later. Document any observations (for example, high correlations during crashes, or DeFi tokens’ behavior differing from broader market).


Data Pipeline & Reproducibility: Create scripts or notebooks for data downloading and preprocessing. Save cleaned datasets to disk (CSV or HDF5) for reuse. Ensure this pipeline is reproducible – for instance, use config files to specify asset list, date range, and data frequency so that others can re-run it easily. By end of Week 2, have a well-organized dataset and code that can regenerate it as needed.


Milestone Checkpoint: Completed Data Collection & Cleaning. All required historical data is available in a usable format and the preprocessing code is validated (e.g., spot-check that returns are computed correctly).


Week 3: Regime Modeling & Market Dynamics
Define Market Regimes: Decide on a regime schema to capture market regime dynamics (market conditions that affect strategy). Common regimes might be bull vs. bear, high-volatility vs. low-volatility, or more finely, “boom”, “steady”, “crash”. For crypto, consider at least two or three regimes – for example, rally, stable, and drawdown periods defined by return and volatility thresholds . Past research in crypto suggests defining regimes via volatility/return quantiles ; you can adopt a similar approach to ensure regimes are economically meaningful.


Implement Regime Detection (HMM): Develop an unsupervised Hidden Markov Model on the returns to statistically infer regimes. Use an HMM with 2–3 hidden states, feeding it features like daily returns or volatility estimates for a representative index (or BTC) to learn distinct state distributions. Train the HMM and interpret the states (e.g., one state might correspond to low volatility, another to high volatility). If the HMM converges, label each time step in the data with its most likely regime (and/or regime probability). Validate that the inferred regimes make sense (e.g., one state should align with known bear markets, etc.).


Alternative/Complementary Approach: As a cross-check, implement a simpler regime classification method. For example, use a rolling volatility or return threshold to classify days as volatile vs. calm, or train a basic LSTM classifier using a moving window of returns to predict a regime label (if you define labels by some heuristic). This can serve as a baseline or to generate an initial label set. Compare these with HMM results for consistency.


Integrate Regime Signals: Prepare to incorporate regime information into the RL framework. Decide how the agent will use regime knowledge: options include including the current regime label or probabilities as part of the state vector for the RL agent, or using separate RL models for each regime (e.g., a switching policy approach). Given project scope, the simplest path is to feed a numerical regime indicator (or the probability of bull vs bear) into the agent’s observation each time step . This way, the agent can condition its actions on the prevailing market state. Ensure the regime identification step (HMM or otherwise) is coded and reproducible (store model parameters, set random seeds).


Milestone Checkpoint: Regime Model Ready. You have a working regime classification for each time step (e.g., an array of regime labels or probabilities over the historical period) and a plan for how this will enhance the RL agent’s decision-making.


Week 4: Graph-Based Feature Engineering (Asset Relationships)
Construct Asset Graph: Build a graph that captures relationships among the crypto assets, especially focusing on DeFi tokens. One approach is to use the correlation matrix of asset returns: compute rolling or full-period correlations and create a graph where each node is a token and edge weights reflect correlation strength. Alternatively, consider fundamental relationships (if data available, e.g., tokens in the same DeFi category or protocol interactions) to inform graph connections. For practicality, start with a correlation network: e.g., link assets that have correlation above a certain threshold or use all pairwise correlations as weighted edges.


Graph Clustering: Apply graph analysis to identify clusters of assets. Using the correlation network, perform community detection or hierarchical clustering to see if assets naturally group (perhaps DeFi tokens cluster together, large-cap vs small-cap, etc.). These clusters can indicate sectors or themes. For each asset, derive a cluster membership feature (categorical or one-hot) and possibly cluster-level metrics (like average return or volatility of that cluster over time) as additional features for the RL agent.


Graph Feature Extraction: Create quantitative features from the graph. Examples: centrality measures (degree or eigenvector centrality to indicate an asset’s influence in the network) or cluster-relative performance (asset return minus average return of its cluster, as a mean-reversion signal). If feasible, explore using a graph neural network (e.g., GraphSAGE) to automatically learn an embedding for each asset that captures complex relationships . For instance, implement a small GraphSAGE model on the static graph to generate node embeddings, which then serve as part of the state input to the RL. (This is an advanced sub-task; if time is constrained, stick to simpler graph metrics).


Dynamic vs Static Graph: Decide whether to update the graph over time. A static graph (e.g., based on full-period correlations) is simpler and can still inform relationships. A dynamic graph (rolling window correlation) would allow the agent to see evolving relationships (regime-dependent correlations), but it’s more complex to implement. Perhaps start with a static approach, and note in the report that dynamic graph updates are a potential extension. Implement the code to calculate all chosen graph features for each asset and each time point (if time-varying).


Feature Assembly: By the end of this week, augment your dataset with the new features: for each time step and asset, you should now have not just raw prices/returns, but also a regime label/probability, and one or more graph-derived features (cluster ID, centrality, etc.). Verify that these features make intuitive sense (e.g., an asset’s cluster assignment doesn’t wildly fluctuate if using dynamic clustering).


Milestone Checkpoint: Graph-Guided Features Complete. The project now has a set of engineered features capturing inter-asset structure, ready to be fed into the learning model. You have code that, given the price data, outputs these features reproducibly.


Week 5: Integrating Econometric Techniques (Kalman Filters & Markov Processes)
Kalman Filter for Signal Smoothing: Integrate a Kalman filter to enhance the data signals. The Kalman filter can be used to estimate latent trends or remove noise from price series . For example, implement a Kalman filter on each asset’s price to estimate a smooth latent price trend (essentially a dynamically updated moving average) and the noise (deviation from trend). This gives a trend indicator that adapts over time, which can serve as a feature. Another use-case is to apply Kalman filtering to the portfolio’s P/L or volatility estimate to smooth out short-term fluctuations. Decide on a specific application (e.g., use pykalman to model each asset’s price as an unobserved true value plus noise, updating daily).


Apply Kalman Features: Once the Kalman filter is implemented, extract the outputs as features. Possible features: the filtered price (which is a de-noised price), or the residual (actual price minus filtered price, perhaps normalized as a z-score). The residual indicates if an asset is above or below its short-term fundamental value – similar to a mean-reversion signal that can be incorporated into the RL state or even as part of the reward (to encourage buying undervalued assets and selling overvalued). This leverages quantitative finance insight that smoothing out short-term volatility can help identify true trends .


Refine Regime Modeling (Markov Processes): Ensure the regime model from Week 3 is well-calibrated. If using an HMM, review the state transition probabilities and adjust the number of states if needed. Hidden Markov Models are a classic approach to regime switching and provide a principled way to model state transitions in markets . Consider if a Markov regime-switching econometric model (like a Hamiltonian switching model) could supplement the analysis: for example, fit a 2-state Markov switching model on returns to validate the HMM regimes. Although a full econometric estimation might be complex, discussing it in the report will show theoretical rigor. At minimum, confirm that regime persistence (states aren’t switching every day erratically) looks reasonable.


Feature Set Consolidation: At this point, compile all inputs that will feed the RL agent: technical features (like recent returns or moving averages if any), regime indicator/probabilities, graph-based features, and any Kalman-filtered signals. Standardize or normalize these features as needed (e.g., scale numerical features to similar ranges to stabilize neural network training). Split the data into training and testing periods (e.g., train on an earlier span and reserve the last portion for out-of-sample testing) to prepare for model training.


Milestone Checkpoint: Advanced Features Integrated. The dataset now includes advanced signals from econometrics (Kalman-filtered trends) and the regime model is finalized. The full feature vector for the RL agent at each time step is defined and ready for use in the environment.


Week 6: RL Environment Design and Algorithm Selection
Define the RL Environment: Formulate the portfolio optimization problem as a reinforcement learning environment (Markov Decision Process). Define state s_t to include all relevant information at time t: e.g., recent returns or prices of each asset, the regime indicator (from Week 3), graph features of assets, and possibly portfolio-related info (like current holdings or last action, if needed). The action a_t will be the portfolio weight allocation across assets (likely a vector of weights for each asset, which could be continuous). Determine if you include a cash asset for uninvested portion or restrict to full investment among cryptos.


Reward Function: Design a reward that aligns with portfolio performance. A common choice is the portfolio return (or log return) at each step, so that maximizing cumulative reward equates to maximizing total return. You may also incorporate a risk-adjustment: for example, use Sharpe ratio or include a penalty for volatility or large drawdowns. To keep the RL single-step reward dense, one approach is to subtract a penalty term like -\lambda \times \text{(portfolio variance)} or use a utility function. Decide on including transaction costs in the reward (for realism, e.g., deduct a small cost when weights change significantly to discourage excessive trading). The reward formulation should encourage the agent to seek high returns with controlled risk.


Portfolio Constraints: Program any necessary constraints in the environment: e.g., weights must sum to 1 (if fully invested) or ≤1 if cash is allowed; no short selling if not allowed (weights ≥0); perhaps limit allocation per asset if desired. Such constraints can be handled by action space design (e.g., using a softmax over logits to ensure positivity and sum=1).


Select RL Algorithm: Choose between PPO and SAC (or another DRL algorithm) based on the environment’s characteristics. Proximal Policy Optimization (PPO) is a stable on-policy method known to work well for continuous control and has been used in portfolio contexts . Soft Actor-Critic (SAC) is off-policy and can be sample-efficient for continuous action spaces. For initial implementation, PPO (via Stable-Baselines3) is a solid choice given its robustness. Set up the RL agent with a suitable neural network architecture (e.g., an MLP that takes the state inputs; you might include an LSTM layer if you want the agent to account for temporal dependencies).


Implement and Test Environment: Write the environment class (following OpenAI Gym interface, if possible). Include functions for reset() (starting a new episode, e.g., at the beginning of the training data or a random point) and step(action) (apply an action, compute portfolio return and new state). Test the environment with trivial policies (random actions or constant allocation) to ensure it behaves as expected (check that rewards are computed correctly, constraints hold, etc.). Fix any bugs in state assembly (especially ensuring that the next state’s features align with the action taken). By the end of Week 6, you should be able to run a single episode of the environment end-to-end with dummy actions.


Milestone Checkpoint: RL Environment Ready. The custom environment for crypto portfolio RL is implemented and producing sensible rewards. You have chosen an RL algorithm (PPO/SAC) and configured it to interface with the environment, laying groundwork for training.


Week 7: Model Training and Hyperparameter Tuning
Train the RL Agent: Begin training the agent on historical data using the selected DRL algorithm. This may involve episodes that cycle through the training period data multiple times (since unlike a game, data is fixed – consider shuffling start points or sampling mini-batches of time segments to improve learning). Monitor training progress: track reward trajectory, losses, etc., over training iterations. Ensure that the training is stable (especially with PPO, watch the policy loss and entropy; with SAC, monitor Q-value estimates). If the agent is not learning (flat reward) or diverging (huge variations), pause to adjust hyperparameters.


Hyperparameter Tuning: Adjust key hyperparameters for effective learning. This includes learning rate, batch size, discount factor (γ), PPO clip ratio (if PPO), or SAC temperature, etc. Given the complexity of the problem, the agent might need a slower learning rate or more training timesteps. Also experiment with different neural network sizes or architectures (e.g., include LSTM if the agent benefits from memory of previous states beyond what’s in the state vector). Use reproducible configurations for each experiment (log the parameters and random seed for each run).


Incorporate Features Gradually (if needed): If training with the full feature set (regimes + graph + Kalman signals) is too complex initially, consider a curriculum: start with a simpler state (e.g., just prices/returns) to ensure the agent can learn basic behavior, then incrementally add the regime indicator, then graph features. This can help troubleshoot which component might be causing issues if the agent fails. For example, confirm the agent can at least learn to beat a random allocation on a small subset of assets before adding all complexities.


Avoid Overfitting: Since the agent is being trained on historical data, guard against overfitting to that sample. Use techniques like early stopping or evaluate on a validation set periodically. Check if the agent’s performance is consistent across different training subsets. You can train multiple agents on different splits of data or using different random seeds and see if a general pattern of strategy emerges (rather than one-off luck).


Model Checkpointing: Save the model (network weights) at regular intervals or when performance seems good, so you can revert to the best version if needed. By the end of Week 7, aim to have at least one trained RL model that achieves sensible results (e.g., positive returns or a reasonable Sharpe ratio on the training period).


Milestone Checkpoint: RL Agent Trained (Preliminary). The DRL agent has been successfully trained on the in-sample data. You have a set of tuned hyperparameters and a saved model/policy that will be used for backtesting on unseen data.


Week 8: Backtesting and Performance Evaluation
Out-of-Sample Backtest: Evaluate the trained model on a test dataset (holdout period not seen during training, e.g., the most recent 6-12 months of data). Simulate the portfolio decisions of the RL agent on this period: start with an initial capital (or just an index of 1.0) and apply the model’s actions sequentially, recording the portfolio value over time. Do the same for baseline strategies for comparison – e.g., benchmark 1: equal-weighted portfolio of the same assets, benchmark 2: a simple momentum or mean-reversion strategy, benchmark 3: buy-and-hold Bitcoin or a crypto index. This will show whether the RL approach adds value.


Performance Metrics: Calculate key metrics on the backtest results: total return, annualized return, volatility, Sharpe ratio, Sortino ratio, maximum drawdown, and Calmar ratio. Also compute turnover/trading frequency (to assess if the strategy is practical given transaction costs). If possible, incorporate a modest transaction cost in the backtest to see the impact on performance. The goal is to demonstrate superior risk-adjusted returns or other advantages (like lower drawdown) from the RL model. Plot the equity curve of the portfolio versus benchmarks for a visual comparison.


Analyze Regime Behavior: Dive deeper into results by relating performance to regimes. For example, break down the RL agent’s performance in each identified regime (did it perform better in bull markets and protect capital in bear markets?). Examine a timeline of regime labels alongside the portfolio value – this can provide insight and a compelling narrative (e.g., “the agent cut exposure during high-volatility regimes, avoiding losses”). Similarly, inspect if the agent’s actions correspond to cluster behaviors (perhaps increasing allocation to a particular cluster of tokens during certain periods).


Model Diagnostics: Evaluate if the agent’s learned policy makes intuitive sense. Look at a few example periods and see how it reallocates: Does it follow trends or contrarian signals? Does it reduce exposure when volatility spikes (showing a form of risk management learned)? Also, consider if any single asset dominates the portfolio – if so, is that because that asset had historically the best performance (risk of overfitting)? Use this analysis to discuss strengths and potential weaknesses of the model.


Iterate if Needed: If the out-of-sample performance is poor or a particular weakness is identified, consider iterative improvements. For instance, if the agent took on excessive risk, you might introduce a stronger risk penalty in the reward and retrain. If it failed to adapt to a market regime shift, perhaps refine the regime model or give the agent a longer state history. Since time is limited at this stage, prioritize the most critical tweaks that could improve practical performance. Re-run backtests if changes are made.


Finalize Results: By the end of Week 8, converge on a final model configuration whose performance is documented and reproducible. Freeze this model for reporting. Save all backtest results, charts, and metrics in an organized manner, as they will be used in the final report.


Milestone Checkpoint: Strategy Validated. You have a complete set of results for the RL strategy vs. benchmarks, showing how the approach performs. These results will form the core of the “Results & Discussion” in your report, and they demonstrate the impact and practicality of your method (key judging criteria).


Week 9: Code Refinement, Reproducibility & Drafting the Report
Code Cleanup and Documentation: Refactor and clean the codebase. Organize code into modules or well-documented notebooks (e.g., a notebook for data prep, one for feature engineering, one for training, one for evaluation). Remove any hard-coded hacks used during experimentation. Ensure all parameters (like asset list, date ranges, model hyperparams) are configurable via a single configuration file or clearly explained constants. This makes it easy for others (e.g., judges) to run or inspect the code. Write a clear README for the GitHub repository, explaining how to set up the environment and run each stage of the project. Emphasize instructions for reproducibility: for example, specify random seeds and library versions used.


Reproducibility Check: Do a clean run of the entire pipeline on a fresh environment if possible. From data download to final backtest, verify that the code produces the same results/figures as you intend to report. This acts as a rehearsal for the submission and ensures no missing pieces. Make use of Jupyter notebooks for an interactive demo, but also have pure script versions for automated runs. Package any trained model artifacts (e.g., saved neural network weights) and data splits if needed for quick reuse.


Begin Report Draft: Start writing the Quant Awards report (5-7 pages). Create an outline following a typical research paper structure:


Introduction: Describe the problem (crypto portfolio optimization) and why it’s challenging (high volatility, regime shifts, many assets). Highlight the novelty of your approach: combining reinforcement learning with regime switching and graph-based features. Emphasize the importance and practical application (e.g., helping investors adapt to crypto market regimes) – aligning with the guidelines to focus on application over pure technique .


Methodology: Detail your approach in subsections: Data (describe dataset and any preprocessing), Regime Modeling (HMM or other method used), Graph-Guided Features (how the graph was constructed and what features extracted), RL Model (PPO/SAC details, state/action/reward design), and any other econometric integration (Kalman filter usage). Keep descriptions precise but accessible, focusing on what and why rather than low-level code. You can cite known methods but ensure the narrative is about how these choices add value (innovation and completeness are key judging criteria ).


Results: Summarize the backtest outcomes. Include a table of performance metrics and one or two well-chosen figures (e.g., an equity curve plot, or a bar chart of Sharpe ratios for your model vs benchmarks). Describe results in words: e.g., “The DRL agent achieved X% annual return vs Y% for benchmark, with lower drawdown, demonstrating superior risk-adjusted performance.” If any regime-specific behavior was observed, report that insight. Address any failure cases or caveats here too (honesty adds to completeness).


Conclusion: Reinforce the significance of the work. For instance: “This project demonstrates a feasible approach to dynamically managing a crypto portfolio by combining modern AI (RL) with financial regime modeling and graph analytics. The improved performance over static strategies highlights the practical potential for investors.” Suggest future improvements (perhaps mention how transaction costs or live testing could be explored, or extensions like dynamic graphs, more regimes, etc.). Keep the tone confident about impact but realistic about limitations.


Incorporate Judging Criteria: As you write, continuously ensure the applicability/relevance is clear (why the work matters in real-world terms), innovation is highlighted (what new combination or technique you brought in), accuracy/completeness is evident (methodology is sound, results are credible and all claims supported by data), and presentation is polished (figures are clear, paper is well-structured) . For example, explicitly mention practical use-cases (like institutional crypto fund management) to hit applicability, and mention how integrating regimes and graphs is a novel angle for crypto trading to hit innovation.


Milestone Checkpoint: By the end of Week 9, you should have a first draft of the paper and a nearly final codebase. The draft will likely be over the page limit initially – that’s okay for now. You also have a fully reproducible setup ready to share if needed (e.g., GitHub repository updated).


Week 10: Report Polishing and Submission Preparation
Editing and Proofreading: Refine the report draft into a polished final document (5-7 pages). Tighten the writing to stay within page limit without losing clarity. Ensure the narrative flows logically and tells a story: from problem statement to solution to results. Double-check that you have not included any identifying personal information. The cover page should contain only the project title (no name or university) , and the writing should be in impersonal academic style (to maintain anonymity for judging). Have a peer or advisor (if available) review the report for clarity, grammar, and impact of explanations.


Finalize Figures and Tables: Make sure all figures are legible and captions are informative. Every figure or table should be referenced in the text and discussed. For instance, include a figure of the strategy’s performance and explicitly describe its key takeaways. Use consistent formatting for tables (properly labeled columns, etc.). If you have more results than fit in 7 pages, move them to an Appendix (since appendices are allowed and not counted in the page limit) – for example, detailed hyperparameter tables or extended result charts can go there, with a reference in the main text.


Consistency and Rigor: Verify that all assertions in the paper are backed by either your results or a citation. Ensure you cite any sources you referred (e.g., if you mention prior work or methodologies). However, do not include any reference on the cover page or any self-revealing acknowledgments. Check that the terminology is consistent (e.g., if you call a regime “bull” in one section, don’t call it “Type 1” elsewhere). Small details like this improve presentation score.


Checklist vs. Guidelines: Do a final check against the Quant Awards guidelines and judging criteria. Make sure the PDF is named correctly and all formatting rules are followed. Confirm that the content emphasizes practical importance (the judges want to see why your work matters, not just how clever it is ). For example, in the intro or conclusion, explicitly state how a practitioner could use this system (maybe through a live trading bot or as a tool for allocating between crypto sectors). Also highlight the novelty (maybe first to combine all these elements in crypto). This alignment will maximize the scores on relevance and innovation.


Submission Package: Prepare the final submission materials. This includes the PDF report and possibly the code repository link or attachment if allowed. (The competition typically judges primarily the report, but being ready to share code on request is good.) If required or allowed, include a brief abstract or cover letter (focusing on key contributions). Ensure the report PDF is finalized with no last-minute typos and the file is not excessively large (optimize embedded images if needed).


Submit by Deadline: Submit the report to the Quant Awards organizers before the August 31 deadline (e.g., via email to the given address) . Double-check you receive a confirmation of submission. Once submitted, tag the final commit in your code repo as the version corresponding to the submission.


Milestone Checkpoint: Project Completed & Submitted. By end of Week 10, you have a complete, polished 5-7 page report and a fully reproducible codebase. The submission meets all requirements and showcases a rigorous, impactful piece of quantitative finance research, positioning you strongly for the Quant Awards.



