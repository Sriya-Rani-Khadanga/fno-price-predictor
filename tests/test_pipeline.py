"""
Smoke tests for the full F&O pipeline.

Tests:
  1. Bhavcopy download (single day)
  2. Gymnasium environment (reset/step)
  3. Feature engineer (feature extraction)
  4. Training smoke test (100 PPO steps)
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestBhavCopyDownloader:
    """Test Layer 2: Bhavcopy downloads."""

    def test_download_single_day(self):
        """Download bhavcopy for one known trading day."""
        from nse_data_collector import BhavCopyDownloader

        dl = BhavCopyDownloader()
        # Use a known trading day
        dt = date(2024, 1, 3)
        df = dl.download_day(dt)

        # May fail if NSE servers are down — skip gracefully
        if df.empty:
            pytest.skip("NSE servers unreachable or holiday")

        assert not df.empty
        assert len(df) > 0
        assert "date" in df.columns
        print(f"Downloaded {len(df)} rows for {dt}")

    def test_pcp_deviation_calculator(self):
        """Test PCP deviation calculation on synthetic data."""
        import pandas as pd
        from nse_data_collector import compute_pcp_deviation

        # Create minimal synthetic chain data
        rows = []
        for strike in [21900, 21950, 22000, 22050, 22100]:
            for side in ["CE", "PE"]:
                rows.append({
                    "timestamp": "2024-01-03T12:00:00",
                    "symbol": "NIFTY",
                    "spot": 22000,
                    "expiry": "2024-01-25",
                    "strike": strike,
                    "type": side,
                    "ltp": 200 if side == "CE" else 180,
                    "bid": 199 if side == "CE" else 179,
                    "ask": 201 if side == "CE" else 181,
                    "oi": 50000,
                    "volume": 10000,
                    "iv": 0.15,
                })
        df = pd.DataFrame(rows)
        result = compute_pcp_deviation(df)
        assert not result.empty
        assert "pcp_deviation_pct" in result.columns
        print(f"PCP deviations computed: {len(result)} rows")


class TestGymnasiumEnv:
    """Test the NiftyOptionsEnv Gymnasium environment."""

    def test_env_creation(self):
        """Environment can be created."""
        from rl.gym_env import NiftyOptionsEnv, FEATURE_DIM
        env = NiftyOptionsEnv(use_synthetic=True, seed=42)
        assert env.observation_space.shape == (FEATURE_DIM,)
        assert env.action_space.n == 5

    def test_reset(self):
        """Environment reset returns valid observation."""
        from rl.gym_env import NiftyOptionsEnv, FEATURE_DIM
        env = NiftyOptionsEnv(use_synthetic=True, seed=42)
        obs, info = env.reset()

        assert isinstance(obs, np.ndarray)
        assert obs.shape == (FEATURE_DIM,)
        assert not np.any(np.isnan(obs))
        assert "step" in info
        print(f"Reset obs shape: {obs.shape}, info keys: {list(info.keys())}")

    def test_step(self):
        """Environment step returns valid outputs."""
        from rl.gym_env import NiftyOptionsEnv
        env = NiftyOptionsEnv(use_synthetic=True, seed=42)
        obs, _ = env.reset()

        # Take a random action
        action = env.action_space.sample()
        obs2, reward, terminated, truncated, info = env.step(action)

        assert isinstance(obs2, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        print(f"Step result: reward={reward:.4f}, terminated={terminated}, "
              f"truncated={truncated}")

    def test_full_episode(self):
        """Run a complete episode with random actions."""
        from rl.gym_env import NiftyOptionsEnv
        env = NiftyOptionsEnv(
            use_synthetic=True, seed=42,
            session_snapshots=50,  # shorter episodes for testing
        )
        obs, info = env.reset()
        total_reward = 0.0
        steps = 0

        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            if terminated or truncated:
                break

        assert steps > 0
        assert "total_pnl" in info
        print(f"Episode: {steps} steps, total_reward={total_reward:.4f}, "
              f"PnL=₹{info['total_pnl']:,.0f}")


class TestFeatureEngineer:
    """Test feature extraction."""

    def test_feature_extraction(self):
        """Extract features from a synthetic chain."""
        from rl.feature_engineer import FeatureEngineer, PortfolioState, FEATURE_DIM
        from data.historical.generator import SyntheticGenerator

        gen = SyntheticGenerator(seed=42)
        chains = gen.generate_session("NIFTY", date(2024, 3, 15), n_snapshots=5)
        assert len(chains) > 0

        fe = FeatureEngineer()
        portfolio = PortfolioState()

        for chain in chains:
            features = fe.extract(chain, T=15/365, minutes_to_close=200,
                                  days_to_expiry=15, portfolio=portfolio)
            assert features.shape == (FEATURE_DIM,)
            assert not np.any(np.isnan(features))

        print(f"Extracted {len(chains)} feature vectors of dim {FEATURE_DIM}")

    def test_feature_names(self):
        """Feature names match feature dimension."""
        from rl.feature_engineer import FeatureEngineer, FEATURE_DIM
        names = FeatureEngineer.feature_names()
        assert len(names) == FEATURE_DIM


class TestTrainingSmoke:
    """Smoke test for RL training (runs very briefly)."""

    def test_ppo_smoke(self):
        """Run 200 PPO timesteps to verify training loop works."""
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        from rl.gym_env import NiftyOptionsEnv

        env = DummyVecEnv([
            lambda: NiftyOptionsEnv(
                use_synthetic=True, seed=42, session_snapshots=30,
            )
        ])

        model = PPO(
            "MlpPolicy", env,
            n_steps=32,
            batch_size=16,
            n_epochs=2,
            learning_rate=1e-3,
            verbose=0,
        )

        model.learn(total_timesteps=200)
        print("PPO smoke test passed (200 steps)")

        # Verify model can predict
        obs = env.reset()
        action, _ = model.predict(obs)
        assert action is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
