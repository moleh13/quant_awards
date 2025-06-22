"""
Evaluate Baseline RL Agent (PPO) on PortfolioEnv

This script loads a trained PPO model and evaluates its performance on test data,
calculating key metrics like ROI, Sharpe ratio, Sortino ratio, max drawdown, etc.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from environment.portfolio_env import PortfolioEnv


def calculate_metrics(portfolio_values, risk_free_rate=0.02):
    """
    Calculate key portfolio performance metrics.
    
    Args:
        portfolio_values: List of portfolio values over time
        risk_free_rate: Annual risk-free rate (default 2%)
    
    Returns:
        dict: Dictionary containing all metrics
    """
    portfolio_values = np.array(portfolio_values)
    
    # Calculate returns
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    
    # Basic metrics
    total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
    annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
    
    # Volatility
    volatility = np.std(returns) * np.sqrt(252)
    
    # Sharpe ratio
    excess_returns = returns - risk_free_rate / 252
    sharpe_ratio = np.mean(excess_returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
    
    # Sortino ratio (using downside deviation)
    downside_returns = returns[returns < 0]
    downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
    sortino_ratio = np.mean(excess_returns) / downside_deviation if downside_deviation > 0 else 0
    
    # Maximum drawdown
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - peak) / peak
    max_drawdown = np.min(drawdown)
    
    # Calmar ratio
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Win rate
    win_rate = np.sum(returns > 0) / len(returns) if len(returns) > 0 else 0
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio,
        'win_rate': win_rate,
        'final_value': portfolio_values[-1],
        'initial_value': portfolio_values[0]
    }


def evaluate_model(model_path, env, num_episodes=1, render=False):
    """
    Evaluate a trained model on the environment.
    
    Args:
        model_path: Path to the trained model
        env: Environment to evaluate on
        num_episodes: Number of episodes to run
        render: Whether to render the environment
    
    Returns:
        dict: Evaluation results
    """
    # Load the trained model
    model = PPO.load(model_path)
    
    all_episode_rewards = []
    all_portfolio_values = []
    all_actions = []
    
    for episode in range(num_episodes):
        print(f"Running episode {episode + 1}/{num_episodes}")
        
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        portfolio_values = []
        actions = []
        
        while not done:
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            actions.append(action)
            
            # Take step in environment
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            
            # Store portfolio value
            portfolio_values.append(info['portfolio_value'])
            
            if render:
                env.render()
        
        all_episode_rewards.append(episode_reward)
        all_portfolio_values.append(portfolio_values)
        all_actions.append(actions)
        
        print(f"Episode {episode + 1} completed. Total reward: {episode_reward:.6f}")
        print(f"Final portfolio value: ${portfolio_values[-1]:,.2f}")
    
    return {
        'episode_rewards': all_episode_rewards,
        'portfolio_values': all_portfolio_values,
        'actions': all_actions
    }


def compare_with_baselines(env, portfolio_values):
    """
    Compare RL agent performance with baseline strategies.
    
    Args:
        env: Environment instance
        portfolio_values: Portfolio values from RL agent
    
    Returns:
        dict: Comparison results
    """
    # Equal-weight strategy (rebalanced daily)
    print("Running equal-weight baseline...")
    obs, _ = env.reset()
    done = False
    equal_weight_values = []
    
    while not done:
        # Equal weight allocation (1/n for each asset)
        action = np.ones(env.n_assets) / env.n_assets
        obs, reward, done, truncated, info = env.step(action)
        equal_weight_values.append(info['portfolio_value'])
    
    # Buy-and-hold strategy (equal weight, buy once)
    print("Running buy-and-hold baseline...")
    obs, _ = env.reset()
    done = False
    buy_hold_values = []
    
    # First step: buy and hold
    action = np.ones(env.n_assets) / env.n_assets  # Equal weight
    obs, reward, done, truncated, info = env.step(action)
    buy_hold_values.append(info['portfolio_value'])
    
    # Subsequent steps: hold (by sending current weights as target)
    while not done:
        # Pass current weights (from obs) as action to hold positions
        current_weights = obs['positions']
        obs, reward, done, truncated, info = env.step(current_weights)
        buy_hold_values.append(info['portfolio_value'])
    
    # Calculate metrics for baselines
    equal_weight_metrics = calculate_metrics(equal_weight_values)
    buy_hold_metrics = calculate_metrics(buy_hold_values)
    rl_metrics = calculate_metrics(portfolio_values[0])  # First episode
    
    return {
        'equal_weight': equal_weight_metrics,
        'buy_hold': buy_hold_metrics,
        'rl_agent': rl_metrics,
        'equal_weight_values': equal_weight_values,
        'buy_hold_values': buy_hold_values
    }


def plot_results(portfolio_values, equal_weight_values, buy_hold_values, save_path=None):
    """
    Plot portfolio value comparisons.
    
    Args:
        portfolio_values: RL agent portfolio values
        equal_weight_values: Equal-weight strategy values
        buy_hold_values: Buy-and-hold strategy values
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Plot portfolio values
    plt.subplot(2, 1, 1)
    plt.plot(portfolio_values[0], label='RL Agent', linewidth=2)
    plt.plot(equal_weight_values, label='Equal Weight', linewidth=2)
    plt.plot(buy_hold_values, label='Buy & Hold', linewidth=2)
    plt.title('Portfolio Value Comparison')
    plt.xlabel('Time Step')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    
    # Plot cumulative returns
    plt.subplot(2, 1, 2)
    rl_returns = np.array(portfolio_values[0]) / portfolio_values[0][0] - 1
    ew_returns = np.array(equal_weight_values) / equal_weight_values[0] - 1
    bh_returns = np.array(buy_hold_values) / buy_hold_values[0] - 1
    
    plt.plot(rl_returns, label='RL Agent', linewidth=2)
    plt.plot(ew_returns, label='Equal Weight', linewidth=2)
    plt.plot(bh_returns, label='Buy & Hold', linewidth=2)
    plt.title('Cumulative Returns')
    plt.xlabel('Time Step')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def main():
    # Configuration
    model_path = '../models/baseline_ppo/ppo_baseline_final.zip'
    num_episodes = 1
    save_plots = True
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please train a model first using train_baseline.py")
        return
    
    # Create environment (same as training)
    env = PortfolioEnv(asset_list_path='data_cache/assets_2019.csv')
    
    print("=== Evaluating Baseline PPO Model ===")
    print(f"Model: {model_path}")
    print(f"Assets: {env.n_assets}")
    print(f"Assets: {env.assets}")
    print(f"Start date: {env.sorted_dates[env.start_step]}")
    print(f"End date: {env.sorted_dates[-1]}")
    
    # Evaluate the model
    results = evaluate_model(model_path, env, num_episodes=num_episodes)
    
    # Compare with baselines
    print("\n=== Running Baseline Comparisons ===")
    comparison = compare_with_baselines(env, results['portfolio_values'])
    
    # Print results
    print("\n=== Performance Metrics ===")
    print("\nRL Agent:")
    for metric, value in comparison['rl_agent'].items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")
    
    print("\nEqual Weight:")
    for metric, value in comparison['equal_weight'].items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")
    
    print("\nBuy & Hold:")
    for metric, value in comparison['buy_hold'].items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")
    
    # Create comparison table
    print("\n=== Performance Comparison Table ===")
    metrics_df = pd.DataFrame({
        'RL Agent': comparison['rl_agent'],
        'Equal Weight': comparison['equal_weight'],
        'Buy & Hold': comparison['buy_hold']
    }).T
    
    print(metrics_df.round(4))
    
    # Save results
    results_dir = '../results/baseline_evaluation'
    os.makedirs(results_dir, exist_ok=True)
    
    # Save metrics to CSV
    metrics_df.to_csv(f'{results_dir}/baseline_metrics.csv')
    print(f"\nMetrics saved to {results_dir}/baseline_metrics.csv")
    
    # Plot results
    if save_plots:
        plot_path = f'{results_dir}/baseline_comparison.png'
        plot_results(
            results['portfolio_values'],
            comparison['equal_weight_values'],
            comparison['buy_hold_values'],
            plot_path
        )
    
    print("\n=== Evaluation Complete ===")


if __name__ == "__main__":
    main() 