"""
NiftyOptionsEnv — Gymnasium-compatible environment for NIFTY50 options trading.

Observation: Box(shape=(47,)) — features from FeatureEngineer
Action: Discrete(5) — HOLD, BUY_CALL, SELL_CALL, BUY_PUT, SELL_PUT
Reward: Realized PnL (cost-adjusted) + shaping for Greeks-aware decisions

Uses the project's existing data infrastructure (synthetic generator, historical
store, option chain processors) to replay market sessions.
"""
from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.processors.options_chain import (
    OptionChain, StrikeData, black_scholes_call, black_scholes_put,
    compute_greeks,
)
from data.historical.generator import SyntheticGenerator
from data.historical.store import HistoricalStore
from rl.feature_engineer import FeatureEngineer, PortfolioState, FEATURE_DIM


# ── Action Constants ──────────────────────────────────────────────────────────
HOLD = 0
BUY_CALL = 1
SELL_CALL = 2
BUY_PUT = 3
SELL_PUT = 4

ACTION_NAMES = {
    HOLD: "HOLD",
    BUY_CALL: "BUY_CALL",
    SELL_CALL: "SELL_CALL",
    BUY_PUT: "BUY_PUT",
    SELL_PUT: "SELL_PUT",
}


# ── Position Tracking ────────────────────────────────────────────────────────

class Position:
    """Tracks a single open option position."""

    def __init__(
        self,
        side: str,       # "call" or "put"
        direction: int,  # +1 = long, -1 = short
        strike: float,
        entry_price: float,
        lot_size: int,
        entry_time: datetime,
    ):
        self.side = side
        self.direction = direction
        self.strike = strike
        self.entry_price = entry_price
        self.lot_size = lot_size
        self.entry_time = entry_time
        self.pnl = 0.0

    def mark_to_market(self, current_price: float) -> float:
        """Calculate unrealized PnL."""
        self.pnl = (current_price - self.entry_price) * self.direction * self.lot_size
        return self.pnl

    def close(self, exit_price: float) -> float:
        """Close position and return realized PnL."""
        pnl = (exit_price - self.entry_price) * self.direction * self.lot_size
        return pnl


# ── Transaction Cost Model ───────────────────────────────────────────────────

class CostModel:
    """Realistic Indian F&O transaction cost model."""

    def __init__(self):
        self.brokerage = 20.0  # flat per order (discount broker)
        self.stt_rate = 0.000625  # STT on sell side for options
        self.exchange_txn = 0.0019  # exchange transaction charges
        self.sebi = 0.000001  # SEBI charges
        self.gst = 0.18  # GST on brokerage + exchange charges
        self.stamp_duty = 0.00003  # stamp duty on buy side
        self.slippage_bps = 5  # 5 bps slippage

    def total_cost(self, price: float, qty: int, lot_size: int,
                   is_sell: bool = False) -> float:
        """Calculate total transaction cost for an order."""
        turnover = price * qty * lot_size
        if turnover < 1:
            return 0.0

        cost = self.brokerage
        cost += turnover * self.exchange_txn
        cost += turnover * self.sebi
        cost += turnover * self.stamp_duty

        if is_sell:
            cost += turnover * self.stt_rate

        # GST on brokerage + exchange charges
        taxable = self.brokerage + turnover * self.exchange_txn
        cost += taxable * self.gst

        # Slippage
        cost += turnover * self.slippage_bps / 10000

        return cost


# ═══════════════════════════════════════════════════════════════════════════════
# GYMNASIUM ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════════════════

class NiftyOptionsEnv(gym.Env):
    """
    Gymnasium environment for NIFTY50 options trading.

    The agent observes market features and decides whether to buy/sell
    calls or puts at the ATM strike. Positions are automatically managed
    with realistic transaction costs.

    Configuration:
        underlying: "NIFTY" or "BANKNIFTY"
        lot_size: 25 for NIFTY (post July 2024), 50 for older data
        max_positions: maximum concurrent positions
        max_daily_loss: daily loss limit to trigger episode termination
        initial_capital: starting capital for the session
        use_synthetic: if True, generate synthetic data; else use historical store
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        underlying: str = "NIFTY",
        lot_size: int = 50,
        max_positions: int = 5,
        max_daily_loss: float = 50000.0,
        initial_capital: float = 1_000_000.0,
        session_snapshots: int = 125,
        violation_rate: float = 0.10,
        use_synthetic: bool = True,
        render_mode: Optional[str] = None,
        seed: int = 42,
    ):
        super().__init__()

        self.underlying = underlying
        self.lot_size = lot_size
        self.max_positions = max_positions
        self.max_daily_loss = max_daily_loss
        self.initial_capital = initial_capital
        self.session_snapshots = session_snapshots
        self.violation_rate = violation_rate
        self.use_synthetic = use_synthetic
        self.render_mode = render_mode
        self._seed = seed

        # Spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(FEATURE_DIM,), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(5)

        # Internal state
        self.feature_eng = FeatureEngineer(history_len=20)
        self.cost_model = CostModel()
        self.generator = SyntheticGenerator(seed=seed)
        self.store = HistoricalStore()
        self.rng = np.random.RandomState(seed)

        # Session data
        self._chains: List[OptionChain] = []
        self._step: int = 0
        self._positions: List[Position] = []
        self._realized_pnl: float = 0.0
        self._unrealized_pnl: float = 0.0
        self._trade_log: List[Dict] = []
        self._episode_count: int = 0
        self._total_cost: float = 0.0

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment to a new trading session."""
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.RandomState(seed)
            self._seed = seed

        self._step = 0
        self._positions = []
        self._realized_pnl = 0.0
        self._unrealized_pnl = 0.0
        self._total_cost = 0.0
        self._trade_log = []
        self.feature_eng.reset()
        self._episode_count += 1

        # Generate or load session data
        replay_date = options.get("date") if options else None
        self._chains = self._load_session(replay_date)

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.

        Args:
            action: integer action from action space

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        chain = self._chains[self._step]
        atm_strike = chain.atm_strike
        spot = chain.spot_price

        reward = 0.0
        trade_executed = False

        # ── Execute action ────────────────────────────────────────────────
        if action == BUY_CALL and len(self._positions) < self.max_positions:
            reward += self._open_position("call", +1, chain, atm_strike)
            trade_executed = True

        elif action == SELL_CALL:
            reward += self._close_positions("call", chain)
            if not any(p.side == "call" for p in self._positions):
                # No calls to sell — small penalty for invalid action
                if reward == 0:
                    reward = -0.01

        elif action == BUY_PUT and len(self._positions) < self.max_positions:
            reward += self._open_position("put", +1, chain, atm_strike)
            trade_executed = True

        elif action == SELL_PUT:
            reward += self._close_positions("put", chain)
            if not any(p.side == "put" for p in self._positions):
                if reward == 0:
                    reward = -0.01

        elif action == HOLD:
            pass  # no action

        # ── Mark to market all positions ──────────────────────────────────
        self._unrealized_pnl = 0.0
        for pos in self._positions:
            sd = chain.get_strike(pos.strike)
            if sd:
                current_price = sd.call_mid if pos.side == "call" else sd.put_mid
                pos.mark_to_market(current_price)
                self._unrealized_pnl += pos.pnl

        # ── Reward shaping ────────────────────────────────────────────────
        # Small reward for unrealized gains
        reward += self._unrealized_pnl / 100000.0

        # Penalize if near daily loss limit
        total_pnl = self._realized_pnl + self._unrealized_pnl
        if total_pnl < -self.max_daily_loss * 0.8:
            reward -= 0.5

        # Greeks-aware bonus: reward closing positions when theta is large negative
        T = self._get_T()
        if action in (SELL_CALL, SELL_PUT) and T < 2 / 365:
            reward += 0.1  # good to close near expiry

        # ── Advance step ──────────────────────────────────────────────────
        self._step += 1

        # ── Check termination ─────────────────────────────────────────────
        terminated = False
        truncated = False

        if total_pnl < -self.max_daily_loss:
            terminated = True
            reward -= 1.0  # loss limit breach penalty

        if self._step >= len(self._chains) - 1:
            truncated = True
            # Force close all positions at session end
            if self._positions:
                last_chain = self._chains[self._step]
                close_reward = self._close_all_positions(last_chain)
                reward += close_reward

        # Check for 3:20 PM (force close)
        if not terminated and not truncated:
            ts = chain.timestamp
            if ts.hour >= 15 and ts.minute >= 20:
                truncated = True
                if self._positions:
                    close_chain = self._chains[min(self._step, len(self._chains) - 1)]
                    reward += self._close_all_positions(close_chain)

        obs = self._get_obs() if not (terminated or truncated) else np.zeros(FEATURE_DIM, dtype=np.float32)
        info = self._get_info()

        # Log trade
        self._trade_log.append({
            "step": self._step,
            "action": ACTION_NAMES[action],
            "spot": spot,
            "reward": reward,
            "realized_pnl": self._realized_pnl,
            "unrealized_pnl": self._unrealized_pnl,
            "n_positions": len(self._positions),
            "timestamp": chain.timestamp.isoformat(),
        })

        return obs, float(reward), terminated, truncated, info

    # ── Internal Methods ──────────────────────────────────────────────────────

    def _open_position(self, side: str, direction: int,
                       chain: OptionChain, strike: float) -> float:
        """Open a new position and return immediate reward."""
        sd = chain.get_strike(strike)
        if sd is None:
            return -0.05  # invalid strike

        price = sd.call_mid if side == "call" else sd.put_mid
        if price <= 0:
            return -0.05

        cost = self.cost_model.total_cost(price, 1, self.lot_size, is_sell=False)
        self._total_cost += cost

        pos = Position(
            side=side, direction=direction, strike=strike,
            entry_price=price, lot_size=self.lot_size,
            entry_time=chain.timestamp,
        )
        self._positions.append(pos)

        # Small negative reward for cost
        return -cost / 10000.0

    def _close_positions(self, side: str, chain: OptionChain) -> float:
        """Close all positions of a given side. Returns realized PnL reward."""
        to_close = [p for p in self._positions if p.side == side]
        reward = 0.0

        for pos in to_close:
            sd = chain.get_strike(pos.strike)
            if sd is None:
                continue
            exit_price = sd.call_mid if side == "call" else sd.put_mid
            pnl = pos.close(exit_price)
            cost = self.cost_model.total_cost(exit_price, 1, self.lot_size, is_sell=True)
            self._total_cost += cost
            net_pnl = pnl - cost
            self._realized_pnl += net_pnl
            reward += net_pnl / 1000.0  # scale reward
            self._positions.remove(pos)

        return reward

    def _close_all_positions(self, chain: OptionChain) -> float:
        """Force close all open positions."""
        reward = 0.0
        reward += self._close_positions("call", chain)
        reward += self._close_positions("put", chain)
        return reward

    def _load_session(self, replay_date: Optional[date] = None) -> List[OptionChain]:
        """Load or generate a session's worth of option chain snapshots."""
        if not self.use_synthetic and replay_date:
            # Try historical store first
            try:
                chains = self.store.load_session(replay_date, self.underlying)
                if chains and len(chains) > 10:
                    return chains
            except Exception:
                pass

        # Fall back to synthetic generation
        dt = replay_date or date(2024, 1, 2) + timedelta(days=self.rng.randint(0, 500))
        # Skip weekends
        while dt.weekday() >= 5:
            dt += timedelta(days=1)

        base_spot = self.rng.uniform(20000, 25000) if "BANK" not in self.underlying else self.rng.uniform(45000, 52000)
        chains = self.generator.generate_session(
            self.underlying, dt,
            n_snapshots=self.session_snapshots,
            violation_rate=self.violation_rate,
            base_spot=base_spot,
        )
        return chains

    def _get_obs(self) -> np.ndarray:
        """Build observation vector from current chain."""
        if self._step >= len(self._chains):
            return np.zeros(FEATURE_DIM, dtype=np.float32)

        chain = self._chains[self._step]
        T = self._get_T()
        minutes_to_close = self._get_minutes_to_close(chain)
        days_to_expiry = max(1, int(T * 365))

        portfolio = PortfolioState(
            unrealized_pnl=self._unrealized_pnl,
            realized_pnl=self._realized_pnl,
            n_positions=len(self._positions),
            max_positions=self.max_positions,
            capital_used_pct=self._total_cost / self.initial_capital,
            holding_seconds=self._avg_holding_time(chain),
        )

        return self.feature_eng.extract(
            chain, T, minutes_to_close, days_to_expiry, portfolio)

    def _get_info(self) -> Dict[str, Any]:
        """Build info dict for current state."""
        return {
            "step": self._step,
            "realized_pnl": self._realized_pnl,
            "unrealized_pnl": self._unrealized_pnl,
            "total_pnl": self._realized_pnl + self._unrealized_pnl,
            "n_positions": len(self._positions),
            "total_cost": self._total_cost,
            "episode_index": self._episode_count,
            "trade_log": self._trade_log[-5:],
        }

    def _get_T(self) -> float:
        """Get time to expiry in years (approximate)."""
        return 15.0 / 365.0  # synthetic data uses ~15 days

    def _get_minutes_to_close(self, chain: OptionChain) -> int:
        """Calculate minutes remaining until market close (15:30)."""
        close = chain.timestamp.replace(hour=15, minute=30, second=0)
        delta = (close - chain.timestamp).total_seconds() / 60
        return max(0, int(delta))

    def _avg_holding_time(self, chain: OptionChain) -> float:
        """Average holding time in seconds across open positions."""
        if not self._positions:
            return 0.0
        total = sum(
            (chain.timestamp - p.entry_time).total_seconds()
            for p in self._positions
        )
        return total / len(self._positions)

    def render(self) -> Optional[str]:
        """Render the environment state."""
        if self._step >= len(self._chains):
            return None

        chain = self._chains[self._step]
        lines = [
            f"═══ Step {self._step}/{len(self._chains)} ═══",
            f"Spot: {chain.spot_price:.2f}  ATM: {chain.atm_strike}  IV: {chain.atm_iv:.4f}",
            f"Time: {chain.timestamp.strftime('%H:%M')}  "
            f"Mins to close: {self._get_minutes_to_close(chain)}",
            f"Positions: {len(self._positions)}/{self.max_positions}",
            f"Realized PnL: ₹{self._realized_pnl:,.2f}  "
            f"Unrealized: ₹{self._unrealized_pnl:,.2f}",
            f"Total Cost: ₹{self._total_cost:,.2f}",
        ]
        if self._positions:
            lines.append("Open positions:")
            for p in self._positions:
                lines.append(
                    f"  {p.side.upper()} @ {p.strike} "
                    f"entry={p.entry_price:.2f} PnL=₹{p.pnl:,.2f}"
                )

        output = "\n".join(lines)
        if self.render_mode == "human":
            print(output)
        return output


# ── Register with Gymnasium ───────────────────────────────────────────────────

gym.register(
    id="NiftyOptions-v0",
    entry_point="rl.gym_env:NiftyOptionsEnv",
    max_episode_steps=200,
)
