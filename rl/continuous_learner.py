"""
Continuous Learning Pipeline — Online RL adaptation.

After initial training, this module:
1. Loads a trained model
2. Runs it on new/test data episodes
3. Collects trajectories and fine-tunes the policy periodically
4. Evaluates on held-out test set
5. Logs performance progression

Usage:
    python -m rl.continuous_learner --model checkpoints/rl/best_model --episodes 200
    python -m rl.continuous_learner --model checkpoints/rl/best_model --eval-only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class ContinuousLearner:
    """
    Online learning loop that adapts a trained RL policy to new data.

    The learner alternates between:
    - Evaluation phase: run policy on test episodes, collect metrics
    - Adaptation phase: fine-tune policy on collected trajectories
    - Checkpoint phase: save improved models
    """

    def __init__(
        self,
        model_path: str,
        underlying: str = "NIFTY",
        adapt_interval: int = 50,    # fine-tune every N episodes
        adapt_timesteps: int = 5000, # timesteps per adaptation
        eval_episodes: int = 20,     # episodes per evaluation
        save_dir: str = "checkpoints/rl/continuous",
        log_dir: str = "logs/rl/continuous",
        seed: int = 42,
    ):
        self.model_path = model_path
        self.underlying = underlying
        self.adapt_interval = adapt_interval
        self.adapt_timesteps = adapt_timesteps
        self.eval_episodes = eval_episodes
        self.save_dir = Path(save_dir)
        self.log_dir = Path(log_dir)
        self.seed = seed

        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Performance tracking
        self.performance_log: List[Dict] = []
        self.best_sharpe = -np.inf
        self.total_episodes = 0
        self.adaptation_count = 0

    def run(
        self,
        total_episodes: int = 200,
        eval_only: bool = False,
    ):
        """
        Main continuous learning loop.

        Args:
            total_episodes: total episodes to run
            eval_only: if True, only evaluate without weight updates
        """
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        from rl.gym_env import NiftyOptionsEnv

        print(f"[CL] Loading model from {self.model_path}...")
        model = PPO.load(self.model_path)

        # Create environment
        env = NiftyOptionsEnv(
            underlying=self.underlying,
            use_synthetic=True,
            seed=self.seed,
        )

        # Create vectorized env for adaptation
        if not eval_only:
            from rl.train_rl import make_env
            adapt_env = DummyVecEnv([
                make_env(rank=0, seed=self.seed + 1000,
                         underlying=self.underlying)
            ])
            model.set_env(adapt_env)

        mode = "EVAL-ONLY" if eval_only else "CONTINUOUS LEARNING"
        print(f"\n{'═' * 60}")
        print(f"  {mode}")
        print(f"  Model: {self.model_path}")
        print(f"  Episodes: {total_episodes}")
        if not eval_only:
            print(f"  Adapt every: {self.adapt_interval} episodes")
            print(f"  Adapt timesteps: {self.adapt_timesteps}")
        print(f"{'═' * 60}\n")

        episode_pnls = []
        batch_pnls = []

        for ep in range(total_episodes):
            self.total_episodes += 1

            # ── Run episode ───────────────────────────────────────────────
            obs, info = env.reset()
            done = False
            episode_reward = 0.0
            steps = 0

            while not done:
                action, _ = model.predict(obs, deterministic=False)
                obs, reward, terminated, truncated, info = env.step(int(action))
                episode_reward += reward
                steps += 1
                done = terminated or truncated

            pnl = info.get("total_pnl", 0.0)
            cost = info.get("total_cost", 0.0)
            n_trades = len(info.get("trade_log", []))

            episode_pnls.append(pnl)
            batch_pnls.append(pnl)

            # Log individual episode
            if (ep + 1) % 10 == 0:
                recent_pnls = np.array(episode_pnls[-10:])
                avg = float(np.mean(recent_pnls))
                win = float(np.mean(recent_pnls > 0)) * 100
                print(
                    f"  [{ep + 1}/{total_episodes}] "
                    f"PnL=₹{pnl:,.0f} | "
                    f"Avg(10)=₹{avg:,.0f} | "
                    f"Win%(10)={win:.0f}% | "
                    f"Trades={n_trades}"
                )

            # ── Periodic adaptation ───────────────────────────────────────
            if (not eval_only
                    and (ep + 1) % self.adapt_interval == 0
                    and len(batch_pnls) >= self.adapt_interval):

                self.adaptation_count += 1
                metrics = self._compute_batch_metrics(batch_pnls)
                self.performance_log.append(metrics)

                print(f"\n  ┌── Adaptation #{self.adaptation_count} ──┐")
                print(f"  │ Batch Sharpe: {metrics['sharpe']:.3f}")
                print(f"  │ Batch Win%:   {metrics['win_rate']:.1f}%")
                print(f"  │ Batch Avg:    ₹{metrics['avg_pnl']:,.0f}")

                # Fine-tune
                print(f"  │ Fine-tuning for {self.adapt_timesteps} steps...")
                model.learn(
                    total_timesteps=self.adapt_timesteps,
                    reset_num_timesteps=False,
                    progress_bar=False,
                )

                # Save if improved
                if metrics["sharpe"] > self.best_sharpe:
                    self.best_sharpe = metrics["sharpe"]
                    save_path = str(self.save_dir / f"adapted_{self.adaptation_count}")
                    model.save(save_path)
                    print(f"  │ ★ New best! Saved to {save_path}")

                print(f"  └{'─' * 25}┘\n")
                batch_pnls = []

        # ── Final summary ─────────────────────────────────────────────────
        all_pnls = np.array(episode_pnls)
        final_metrics = self._compute_batch_metrics(episode_pnls)

        print(f"\n{'═' * 60}")
        print(f"  CONTINUOUS LEARNING COMPLETE")
        print(f"  Total episodes: {total_episodes}")
        print(f"  Adaptations: {self.adaptation_count}")
        print(f"  Overall Sharpe: {final_metrics['sharpe']:.3f}")
        print(f"  Overall Win%:   {final_metrics['win_rate']:.1f}%")
        print(f"  Overall Avg PnL: ₹{final_metrics['avg_pnl']:,.0f}")
        print(f"  Best Sharpe: {self.best_sharpe:.3f}")
        print(f"{'═' * 60}")

        # Save final model and performance log
        if not eval_only:
            final_path = str(self.save_dir / "final_adapted")
            model.save(final_path)
            print(f"  Final model: {final_path}")

        log_path = self.log_dir / "continuous_learning_log.json"
        with open(log_path, "w") as f:
            json.dump({
                "summary": final_metrics,
                "performance_log": self.performance_log,
                "episode_pnls": [float(p) for p in episode_pnls],
            }, f, indent=2)
        print(f"  Log saved to: {log_path}")

        return final_metrics

    def _compute_batch_metrics(self, pnls: List[float]) -> Dict:
        """Compute metrics for a batch of episodes."""
        arr = np.array(pnls)
        if len(arr) == 0:
            return {"sharpe": 0, "win_rate": 0, "avg_pnl": 0,
                    "std_pnl": 0, "max_pnl": 0, "min_pnl": 0}

        avg = float(np.mean(arr))
        std = float(np.std(arr))
        sharpe = avg / std * np.sqrt(252) if std > 0 else 0.0

        return {
            "sharpe": sharpe,
            "win_rate": float(np.mean(arr > 0)) * 100,
            "avg_pnl": avg,
            "std_pnl": std,
            "max_pnl": float(np.max(arr)),
            "min_pnl": float(np.min(arr)),
            "n_episodes": len(arr),
            "adaptation": self.adaptation_count,
            "timestamp": datetime.now().isoformat(),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continuous RL learning pipeline")
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--underlying", default="NIFTY")
    parser.add_argument("--adapt-interval", type=int, default=50)
    parser.add_argument("--adapt-timesteps", type=int, default=5000)
    parser.add_argument("--eval-only", action="store_true",
                        help="Evaluate without weight updates")
    parser.add_argument("--save-dir", default="checkpoints/rl/continuous")
    parser.add_argument("--log-dir", default="logs/rl/continuous")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    learner = ContinuousLearner(
        model_path=args.model,
        underlying=args.underlying,
        adapt_interval=args.adapt_interval,
        adapt_timesteps=args.adapt_timesteps,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        seed=args.seed,
    )
    learner.run(
        total_episodes=args.episodes,
        eval_only=args.eval_only,
    )
