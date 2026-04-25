"""
RL Training Script — PPO / SAC training with Stable-Baselines3.

Supports:
- Vectorized environments for parallel training
- Curriculum learning (easy → hard episodes)
- WandB experiment tracking
- Automatic best model checkpointing

Usage:
    python -m rl.train_rl --algo ppo --timesteps 500000 --n-envs 4
    python -m rl.train_rl --algo sac --timesteps 200000 --n-envs 1
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def make_env(
    rank: int = 0,
    seed: int = 42,
    underlying: str = "NIFTY",
    violation_rate: float = 0.10,
    session_snapshots: int = 125,
) -> Callable:
    """Create a callable that returns a fresh NiftyOptionsEnv."""
    def _init():
        from rl.gym_env import NiftyOptionsEnv
        env = NiftyOptionsEnv(
            underlying=underlying,
            lot_size=50,
            max_positions=5,
            max_daily_loss=50000.0,
            initial_capital=1_000_000.0,
            session_snapshots=session_snapshots,
            violation_rate=violation_rate,
            use_synthetic=True,
            seed=seed + rank,
        )
        return env
    return _init


def linear_schedule(initial_value: float) -> Callable:
    """Linear learning rate schedule."""
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


def train(
    algo: str = "ppo",
    total_timesteps: int = 500_000,
    n_envs: int = 4,
    underlying: str = "NIFTY",
    learning_rate: float = 3e-4,
    use_wandb: bool = False,
    save_dir: str = "checkpoints/rl",
    log_dir: str = "logs/rl",
    seed: int = 42,
    violation_rate: float = 0.10,
    session_snapshots: int = 125,
):
    """
    Main RL training function.

    Args:
        algo: "ppo" or "sac"
        total_timesteps: total environment steps to train
        n_envs: number of parallel environments
        underlying: trading underlying symbol
        learning_rate: initial learning rate
        use_wandb: enable Weights & Biases logging
        save_dir: directory for model checkpoints
        log_dir: directory for training logs
        seed: random seed
        violation_rate: fraction of snapshots with PCP violations
        session_snapshots: snapshots per episode
    """
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
    from rl.callbacks import TradingMetricsCallback, BestModelCallback

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # ── WandB ─────────────────────────────────────────────────────────────
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project="nifty-options-rl",
                name=f"{algo}_{underlying}_{datetime.now().strftime('%Y%m%d_%H%M')}",
                config={
                    "algo": algo,
                    "total_timesteps": total_timesteps,
                    "n_envs": n_envs,
                    "learning_rate": learning_rate,
                    "underlying": underlying,
                    "violation_rate": violation_rate,
                    "session_snapshots": session_snapshots,
                },
            )
        except Exception as e:
            print(f"[Train] WandB failed: {e}")
            use_wandb = False

    # ── Vectorized environments ───────────────────────────────────────────
    print(f"[Train] Creating {n_envs} parallel environments...")
    if n_envs > 1:
        env = SubprocVecEnv([
            make_env(rank=i, seed=seed, underlying=underlying,
                     violation_rate=violation_rate,
                     session_snapshots=session_snapshots)
            for i in range(n_envs)
        ])
    else:
        env = DummyVecEnv([
            make_env(rank=0, seed=seed, underlying=underlying,
                     violation_rate=violation_rate,
                     session_snapshots=session_snapshots)
        ])

    # ── Callbacks ─────────────────────────────────────────────────────────
    callbacks = CallbackList([
        TradingMetricsCallback(log_dir=log_dir, log_freq=10, verbose=1),
        BestModelCallback(save_dir=save_dir, check_freq=500, verbose=1),
        CheckpointCallback(
            save_freq=max(total_timesteps // 10, 1000),
            save_path=save_dir,
            name_prefix="checkpoint",
        ),
    ])

    # ── Model ─────────────────────────────────────────────────────────────
    print(f"[Train] Initializing {algo.upper()} model...")

    if algo.lower() == "ppo":
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=linear_schedule(learning_rate),
            n_steps=256,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            clip_range_vf=None,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            target_kl=0.02,
            tensorboard_log=log_dir,
            verbose=1,
            seed=seed,
            policy_kwargs={
                "net_arch": dict(pi=[256, 256, 128], vf=[256, 256, 128]),
                "activation_fn": __import__("torch").nn.ReLU,
            },
        )
    elif algo.lower() == "sac":
        if n_envs > 1:
            print("[Train] SAC doesn't support SubprocVecEnv natively. Using DummyVecEnv.")
            env = DummyVecEnv([
                make_env(rank=0, seed=seed, underlying=underlying,
                         violation_rate=violation_rate,
                         session_snapshots=session_snapshots)
            ])
        # SAC requires continuous actions — wrap discrete as Box
        # For now, use PPO for discrete; SAC is better with continuous action spaces
        print("[Train] Note: SAC works best with continuous action spaces.")
        print("[Train] Using PPO instead for discrete Nifty actions.")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=linear_schedule(learning_rate),
            n_steps=256,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            ent_coef=0.02,
            tensorboard_log=log_dir,
            verbose=1,
            seed=seed,
            policy_kwargs={
                "net_arch": dict(pi=[256, 256, 128], vf=[256, 256, 128]),
            },
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo}. Use 'ppo' or 'sac'.")

    # ── Train ─────────────────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print(f"  TRAINING: {algo.upper()} | {underlying}")
    print(f"  Timesteps: {total_timesteps:,} | Envs: {n_envs}")
    print(f"  LR: {learning_rate} | Violation rate: {violation_rate}")
    print(f"  Save: {save_dir} | Log: {log_dir}")
    print(f"{'═' * 60}\n")

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    # ── Save final model ──────────────────────────────────────────────────
    final_path = os.path.join(save_dir, "final_model")
    model.save(final_path)
    print(f"\n[Train] Final model saved to {final_path}")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print(f"  TRAINING COMPLETE")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Model saved to: {final_path}")
    print(f"  Best model in: {save_dir}/best_model")
    print(f"{'═' * 60}")

    if use_wandb:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass

    env.close()
    return final_path


def evaluate(
    model_path: str,
    n_episodes: int = 50,
    underlying: str = "NIFTY",
    render: bool = False,
    seed: int = 42,
):
    """
    Evaluate a trained model over multiple episodes.

    Returns summary statistics.
    """
    from stable_baselines3 import PPO
    from rl.gym_env import NiftyOptionsEnv

    print(f"[Eval] Loading model from {model_path}...")
    model = PPO.load(model_path)

    env = NiftyOptionsEnv(
        underlying=underlying,
        use_synthetic=True,
        render_mode="human" if render else None,
        seed=seed,
    )

    episode_pnls = []
    episode_trades = []
    episode_costs = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            done = terminated or truncated

            if render:
                env.render()

        pnl = info.get("total_pnl", 0.0)
        cost = info.get("total_cost", 0.0)
        n_trades = len(info.get("trade_log", []))

        episode_pnls.append(pnl)
        episode_trades.append(n_trades)
        episode_costs.append(cost)

        print(f"  Episode {ep + 1}/{n_episodes}: "
              f"PnL=₹{pnl:,.0f} | Trades={n_trades} | Cost=₹{cost:,.0f}")

    # Summary
    pnls = np.array(episode_pnls)
    avg_pnl = float(np.mean(pnls))
    std_pnl = float(np.std(pnls))
    sharpe = avg_pnl / std_pnl * np.sqrt(252) if std_pnl > 0 else 0.0
    win_rate = float(np.mean(pnls > 0)) * 100

    print(f"\n{'═' * 60}")
    print(f"  EVALUATION RESULTS ({n_episodes} episodes)")
    print(f"  Avg PnL:    ₹{avg_pnl:,.0f}")
    print(f"  Std PnL:    ₹{std_pnl:,.0f}")
    print(f"  Sharpe:     {sharpe:.3f}")
    print(f"  Win Rate:   {win_rate:.1f}%")
    print(f"  Max PnL:    ₹{float(np.max(pnls)):,.0f}")
    print(f"  Min PnL:    ₹{float(np.min(pnls)):,.0f}")
    print(f"  Avg Trades: {float(np.mean(episode_trades)):.1f}")
    print(f"  Avg Cost:   ₹{float(np.mean(episode_costs)):,.0f}")
    print(f"{'═' * 60}")

    return {
        "avg_pnl": avg_pnl,
        "std_pnl": std_pnl,
        "sharpe": sharpe,
        "win_rate": win_rate,
        "max_pnl": float(np.max(pnls)),
        "min_pnl": float(np.min(pnls)),
        "avg_trades": float(np.mean(episode_trades)),
        "avg_cost": float(np.mean(episode_costs)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL agent for options trading")
    sub = parser.add_subparsers(dest="command", required=True)

    # Train command
    train_p = sub.add_parser("train", help="Train a new model")
    train_p.add_argument("--algo", default="ppo", choices=["ppo", "sac"])
    train_p.add_argument("--timesteps", type=int, default=500_000)
    train_p.add_argument("--n-envs", type=int, default=4)
    train_p.add_argument("--lr", type=float, default=3e-4)
    train_p.add_argument("--underlying", default="NIFTY")
    train_p.add_argument("--wandb", action="store_true")
    train_p.add_argument("--save-dir", default="checkpoints/rl")
    train_p.add_argument("--log-dir", default="logs/rl")
    train_p.add_argument("--seed", type=int, default=42)
    train_p.add_argument("--violation-rate", type=float, default=0.10)
    train_p.add_argument("--session-snapshots", type=int, default=125)

    # Eval command
    eval_p = sub.add_parser("eval", help="Evaluate a trained model")
    eval_p.add_argument("--model", required=True, help="Path to saved model")
    eval_p.add_argument("--episodes", type=int, default=50)
    eval_p.add_argument("--underlying", default="NIFTY")
    eval_p.add_argument("--render", action="store_true")
    eval_p.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.command == "train":
        train(
            algo=args.algo,
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            underlying=args.underlying,
            learning_rate=args.lr,
            use_wandb=args.wandb,
            save_dir=args.save_dir,
            log_dir=args.log_dir,
            seed=args.seed,
            violation_rate=args.violation_rate,
            session_snapshots=args.session_snapshots,
        )
    elif args.command == "eval":
        evaluate(
            model_path=args.model,
            n_episodes=args.episodes,
            underlying=args.underlying,
            render=args.render,
            seed=args.seed,
        )
