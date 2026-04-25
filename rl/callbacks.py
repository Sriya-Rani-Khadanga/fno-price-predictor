"""
Custom Stable-Baselines3 callbacks for RL training.

- TradingMetricsCallback: Logs Sharpe, win rate, avg PnL every N episodes
- BestModelCallback: Saves model with best Sharpe ratio
"""
from __future__ import annotations

import json
import os
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class TradingMetricsCallback(BaseCallback):
    """
    Tracks trading-specific metrics and logs them periodically.

    Metrics logged:
    - Episode PnL (realized + unrealized)
    - Win rate (% of episodes with positive PnL)
    - Sharpe ratio (rolling window)
    - Average number of trades per episode
    - Average transaction cost per episode
    """

    def __init__(
        self,
        log_dir: str = "logs/rl",
        log_freq: int = 10,
        window_size: int = 100,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_freq = log_freq
        self.window_size = window_size

        # Rolling buffers
        self.episode_pnls: deque = deque(maxlen=window_size)
        self.episode_costs: deque = deque(maxlen=window_size)
        self.episode_trades: deque = deque(maxlen=window_size)
        self.episode_lengths: deque = deque(maxlen=window_size)
        self.episode_rewards: deque = deque(maxlen=window_size)
        self._episode_count = 0
        self._metrics_log: List[Dict] = []

    def _on_step(self) -> bool:
        """Called after each step. Check for episode completion."""
        # Check if any env finished an episode
        if self.locals.get("dones") is not None:
            for i, done in enumerate(self.locals["dones"]):
                if done:
                    self._on_episode_end(i)
        return True

    def _on_episode_end(self, env_idx: int = 0):
        """Process metrics at end of episode."""
        self._episode_count += 1

        # Extract info from the last info dict
        infos = self.locals.get("infos", [])
        if env_idx < len(infos):
            info = infos[env_idx]
            pnl = info.get("total_pnl", 0.0)
            cost = info.get("total_cost", 0.0)
            n_trades = len(info.get("trade_log", []))
        else:
            pnl = 0.0
            cost = 0.0
            n_trades = 0

        self.episode_pnls.append(pnl)
        self.episode_costs.append(cost)
        self.episode_trades.append(n_trades)

        # Log periodically
        if self._episode_count % self.log_freq == 0:
            metrics = self._compute_metrics()
            self._metrics_log.append(metrics)

            if self.verbose >= 1:
                print(
                    f"[RL] Ep {self._episode_count} | "
                    f"Sharpe: {metrics['sharpe']:.3f} | "
                    f"Win%: {metrics['win_rate']:.1f} | "
                    f"Avg PnL: ₹{metrics['avg_pnl']:,.0f} | "
                    f"Avg Trades: {metrics['avg_trades']:.1f}"
                )

            # Log to TensorBoard
            if self.logger:
                for key, val in metrics.items():
                    self.logger.record(f"trading/{key}", val)

    def _compute_metrics(self) -> Dict[str, float]:
        """Compute rolling trading metrics."""
        pnls = np.array(self.episode_pnls)

        if len(pnls) == 0:
            return {
                "sharpe": 0.0, "win_rate": 0.0, "avg_pnl": 0.0,
                "max_pnl": 0.0, "min_pnl": 0.0, "avg_trades": 0.0,
                "avg_cost": 0.0, "episode": self._episode_count,
            }

        avg_pnl = float(np.mean(pnls))
        std_pnl = float(np.std(pnls))
        sharpe = avg_pnl / std_pnl * np.sqrt(252) if std_pnl > 0 else 0.0
        win_rate = float(np.mean(pnls > 0)) * 100

        return {
            "sharpe": sharpe,
            "win_rate": win_rate,
            "avg_pnl": avg_pnl,
            "max_pnl": float(np.max(pnls)),
            "min_pnl": float(np.min(pnls)),
            "avg_trades": float(np.mean(list(self.episode_trades))),
            "avg_cost": float(np.mean(list(self.episode_costs))),
            "episode": self._episode_count,
            "timestamp": datetime.now().isoformat(),
        }

    def _on_training_end(self):
        """Save metrics log at end of training."""
        log_path = self.log_dir / "training_metrics.json"
        with open(log_path, "w") as f:
            json.dump(self._metrics_log, f, indent=2)
        if self.verbose >= 1:
            print(f"[RL] Metrics saved to {log_path}")


class BestModelCallback(BaseCallback):
    """
    Saves the model whenever the rolling Sharpe ratio improves.
    """

    def __init__(
        self,
        save_dir: str = "checkpoints/rl",
        check_freq: int = 50,
        min_episodes: int = 20,
        metric: str = "sharpe",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.check_freq = check_freq
        self.min_episodes = min_episodes
        self.metric = metric

        self.best_metric = -np.inf
        self._step_count = 0
        self.episode_pnls: deque = deque(maxlen=100)
        self._episode_count = 0

    def _on_step(self) -> bool:
        self._step_count += 1

        # Track episodes
        if self.locals.get("dones") is not None:
            for i, done in enumerate(self.locals["dones"]):
                if done:
                    self._episode_count += 1
                    infos = self.locals.get("infos", [])
                    if i < len(infos):
                        pnl = infos[i].get("total_pnl", 0.0)
                        self.episode_pnls.append(pnl)

        # Check if we should evaluate
        if (self._step_count % self.check_freq == 0
                and self._episode_count >= self.min_episodes):
            current_metric = self._compute_metric()

            if current_metric > self.best_metric:
                self.best_metric = current_metric
                save_path = str(self.save_dir / "best_model")
                self.model.save(save_path)
                if self.verbose >= 1:
                    print(
                        f"[RL] New best {self.metric}: {current_metric:.4f} "
                        f"(step {self._step_count}). Saved to {save_path}"
                    )

        return True

    def _compute_metric(self) -> float:
        """Compute current metric value."""
        pnls = np.array(self.episode_pnls)
        if len(pnls) < 5:
            return -np.inf

        if self.metric == "sharpe":
            avg = float(np.mean(pnls))
            std = float(np.std(pnls))
            return avg / std * np.sqrt(252) if std > 0 else 0.0
        elif self.metric == "avg_pnl":
            return float(np.mean(pnls))
        elif self.metric == "win_rate":
            return float(np.mean(pnls > 0))
        return 0.0
