"""
GRPO training script using TRL + Unsloth for PCP arbitrage agent.
"""
from __future__ import annotations
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

def train(total_steps: int = 3000, checkpoint_path: str = None,
          wandb_enabled: bool = True):
    """
    Main GRPO training loop with curriculum progression.
    
    Uses Unsloth for memory-efficient 4-bit quantized training
    and TRL's GRPOTrainer for RL optimization.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config.settings import get_settings, CHECKPOINTS_DIR
    settings = get_settings()
    tc = settings.training

    print("[Train] Loading model via Unsloth...")
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=tc.model_name,
        max_seq_length=tc.max_prompt_length + tc.max_completion_length,
        dtype=None,
        load_in_4bit=True)

    model = FastLanguageModel.get_peft_model(
        model, r=tc.lora_r, target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                              "gate_proj", "up_proj", "down_proj"],
        lora_alpha=tc.lora_alpha, lora_dropout=0, bias="none",
        use_gradient_checkpointing="unsloth", random_state=42)

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"[Train] Loading checkpoint from {checkpoint_path}")
        from peft import PeftModel
        model.load_adapter(checkpoint_path)

    from mcp_servers.mcp_client import MCPClient
    from pcp_arb_env.environment import PCPArbEnv
    from pcp_arb_env.curriculum import CurriculumManager
    from training.curriculum_scheduler import CurriculumScheduler
    from training.rollout import SYSTEM_PROMPT

    curriculum = CurriculumManager()
    scheduler = CurriculumScheduler(curriculum)
    mcp_client = MCPClient(timeout=5.0)  # Local in-process, no actual HTTP needed

    # Build shared environment
    feed = scheduler.get_feed()
    env = PCPArbEnv(feed=feed, mcp_client=mcp_client)

    if wandb_enabled:
        try:
            import wandb
            wandb.init(project=tc.wandb_project, name=f"pcp_arb_{datetime.now().strftime('%Y%m%d_%H%M')}")
        except Exception as e:
            print(f"[Train] WandB init failed: {e}")
            wandb_enabled = False

    # Build prompts for GRPO
    def build_prompt_dataset(n: int):

        prompts = []
        for i in range(min(n, 210)):
            prompts.append({
                "prompt": f"Market state {i}: PCP deviation=0.8, strike=23000, expiry_days=5. What action?",
                "answer": "BUY_CALL"
            })





        return prompts

    print("[Train] Generating training prompts...")
    prompts = build_prompt_dataset(210)

    # GRPO reward function
    step_counter = [0]
    metrics_log = []

    def _parse_completion_action(completion: str) -> tuple[Dict, bool]:
        """Parse completion leniently into the environment's expected action dict."""
        valid_keywords = {"BUY_CALL", "SELL_CALL", "BUY_PUT", "SELL_PUT", "HOLD"}
        keyword_to_action_type = {
            "BUY_CALL": "enter_long_call_short_put",
            "SELL_PUT": "enter_long_call_short_put",
            "SELL_CALL": "enter_short_call_long_put",
            "BUY_PUT": "enter_short_call_long_put",
            "HOLD": "hold",
        }

        clean = completion.strip()
        payload = None
        parsed_ok = False

        try:
            payload = json.loads(clean)
            parsed_ok = True
        except (json.JSONDecodeError, TypeError, ValueError):
            match = re.search(r"\{.*\}", clean, re.DOTALL)
            if match:
                try:
                    payload = json.loads(match.group(0))
                    parsed_ok = True
                except (json.JSONDecodeError, TypeError, ValueError):
                    payload = None

        if isinstance(payload, dict):
            action_value = str(
                payload.get("action_type", payload.get("action", "HOLD"))
            ).strip()
            normalized = action_value.upper()
            action_type = payload.get("action_type")
            if not action_type and normalized in keyword_to_action_type:
                action_type = keyword_to_action_type[normalized]
            return {
                "action_type": action_type or "hold",
                "tool_calls": payload.get("tool_calls", []),
                "strike": payload.get("strike"),
                "qty": payload.get("qty", 1),
            }, parsed_ok

        upper_completion = clean.upper()
        for keyword in valid_keywords:
            if keyword in upper_completion:
                return {
                    "action_type": keyword_to_action_type[keyword],
                    "tool_calls": [],
                    "strike": None,
                    "qty": 1,
                }, False

        return {"action_type": "hold", "tool_calls": [], "strike": None, "qty": 1}, False

    def format_reward(completions: List[str], **kwargs) -> List[float]:
        """Award partial credit for increasingly well-formed action JSON."""
        valid_keywords = {"BUY_CALL", "SELL_CALL", "BUY_PUT", "SELL_PUT", "HOLD"}
        rewards = []
        for completion in completions:
            reward = 0.0
            clean = completion.strip()

            if "{" in clean and "}" in clean:
                reward += 0.2

            payload = None
            parsed_ok = False
            try:
                payload = json.loads(clean)
                parsed_ok = True
            except (json.JSONDecodeError, TypeError, ValueError):
                match = re.search(r"\{.*\}", clean, re.DOTALL)
                if match:
                    try:
                        payload = json.loads(match.group(0))
                        parsed_ok = True
                    except (json.JSONDecodeError, TypeError, ValueError):
                        payload = None

            if isinstance(payload, dict):
                if "action" in payload:
                    reward += 0.2
                    action_value = str(payload.get("action", "")).strip().upper()
                    if action_value in valid_keywords:
                        reward += 0.3
                if "strike" in payload:
                    reward += 0.15
                if parsed_ok:
                    reward += 0.15

            rewards.append(reward)
        return rewards

    def reward_fn(completions: List[str], **kwargs) -> List[float]:
        """GRPO reward function — evaluates each completion in the environment."""
        rewards = []
        for completion in completions:
            action, parsed_ok = _parse_completion_action(completion)
            # Simulate a step with this action
            try:
                result = env.step(action)
                reward = result.reward.total
            except Exception:
                reward = -1.0
                parsed_ok = False
            rewards.append(reward)
            step_counter[0] += 1
            if step_counter[0] % 50 == 0:
                curriculum.advance(50)
                new_feed = scheduler.get_feed()
                env.feed = new_feed
                env.reset()
                print(f"[Train] Step {step_counter[0]}, Stage: {curriculum.stage_name}")
        if wandb_enabled:
            try:
                import wandb
                wandb.log({
                    "reward_mean": sum(rewards) / len(rewards),
                    "reward_max": max(rewards),
                    "reward_min": min(rewards),
                    "curriculum_stage": curriculum.stage_name,
                    "step": step_counter[0],
                })
            except Exception:
                pass
        return rewards

    # Configure GRPO
    from trl import GRPOConfig, GRPOTrainer

    training_args = GRPOConfig(
        output_dir=str(CHECKPOINTS_DIR),
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=tc.learning_rate,
        max_grad_norm=tc.gradient_clip,
        lr_scheduler_type=tc.schedule,
        logging_steps=10,
        save_steps=tc.checkpoint_interval,
        num_generations=tc.num_generations,
        max_prompt_length=tc.max_prompt_length,
        max_completion_length=tc.max_completion_length,
        temperature=0.7,
        report_to="wandb" if wandb_enabled else "none",
    )

    # Format prompts for trainer
    from datasets import Dataset
    train_data = Dataset.from_dict({"prompt": prompts})

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        processing_class=tokenizer,
        reward_funcs=[format_reward, reward_fn],
    )

    print(f"[Train] Starting GRPO training for {total_steps} steps...")
    print(f"[Train] Model: {tc.model_name}, Generations: {tc.num_generations}")
    print(f"[Train] Curriculum stages: {[s.name for s in settings.curriculum.stages]}")

    trainer.train()

    # Save final checkpoint
    final_path = CHECKPOINTS_DIR / "final"
    model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"[Train] Final checkpoint saved to {final_path}")

    if wandb_enabled:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass

    return str(final_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    train(args.steps, args.checkpoint, not args.no_wandb)
