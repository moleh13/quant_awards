"""
Simple test script to verify the environment works correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.portfolio_env import PortfolioEnv

def test_env():
    print("Testing PortfolioEnv...")
    
    # Create environment
    env = PortfolioEnv()
    
    # Test reset
    print("Testing reset...")
    obs, info = env.reset()
    print(f"Reset successful. Observation keys: {obs.keys()}")
    
    # Test step
    print("Testing step...")
    action = env.action_space.sample()
    print(f"Action shape: {action.shape}")
    
    try:
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step successful!")
        print(f"Reward: {reward}")
        print(f"Done: {done}")
        print(f"Truncated: {truncated}")
        print(f"Info keys: {info.keys()}")
    except Exception as e:
        print(f"Step failed: {e}")
        return False
    
    # Test multiple steps
    print("Testing multiple steps...")
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {i}: reward={reward:.4f}, done={done}")
        if done:
            break
    
    print("Environment test completed successfully!")
    return True

if __name__ == "__main__":
    test_env() 