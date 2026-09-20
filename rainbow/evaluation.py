# -*- coding: utf-8 -*-

import os
import plotly
from plotly.graph_objs import Scatter
from plotly.graph_objs.scatter import Line
import torch
from datetime import datetime

from .logging_utils import EvaluationLogger, StreamingLogger
from .env import Env


def evaluate_agent(args, step, dqn, metrics, results_dir, save_best=False, env=None):
    """Evaluate a policy without taking unrelated exploratory actions first."""

    owns_environment = env is None
    if env is None:
        env = Env(args)
    env.eval()
    metrics["steps"].append(step)
    episode_returns, q_values = [], []

    evaluation_data = []

    evaluation_logger = EvaluationLogger(results_dir)
    streaming_logger = StreamingLogger(results_dir)

    try:
        for episode_idx in range(args.evaluation_episodes):
            state = env.reset()
            reward_sum = 0.0
            episode_rewards = []
            episode_steps = 0

            while True:
                q_values.append(dqn.evaluate_q(state))
                action = dqn.act(state)
                state, reward, done = env.step(action)
                reward_sum += reward
                episode_rewards.append(reward)
                episode_steps += 1

                if args.render:
                    env.render()

                if done:
                    episode_returns.append(reward_sum)
                    episode_info = {
                        "evaluation_step": step,
                        "episode_idx": episode_idx,
                        "total_reward": reward_sum,
                        "episode_steps": episode_steps,
                        "step_rewards": episode_rewards,
                        "timestamp": datetime.now().isoformat(),
                    }
                    evaluation_data.append(episode_info)
                    streaming_logger.log_evaluation_episode(episode_info)
                    break
    finally:
        streaming_logger.close()
        if owns_environment:
            env.close()

    evaluation_logger.log_evaluation(step, evaluation_data)
    avg_reward = sum(episode_returns) / len(episode_returns)
    avg_q = sum(q_values) / len(q_values)
    if save_best:
        if avg_reward > metrics["best_avg_reward"]:
            metrics["best_avg_reward"] = avg_reward
            dqn.save(results_dir)

        metrics["rewards"].append(episode_returns)
        metrics["Qs"].append(q_values)
        torch.save(metrics, os.path.join(results_dir, "metrics.pth"))

        _plot_line(metrics["steps"], metrics["rewards"], "Reward", path=results_dir)
        _plot_line(metrics["steps"], metrics["Qs"], "Q", path=results_dir)

    return avg_reward, avg_q


def _plot_line(xs, ys_population, title, path=""):
    max_colour, mean_colour, std_colour, transparent = (
        "rgb(0, 132, 180)",
        "rgb(0, 172, 237)",
        "rgba(29, 202, 255, 0.2)",
        "rgba(0, 0, 0, 0)",
    )

    ys = torch.tensor(ys_population, dtype=torch.float32)
    ys_min, ys_max, ys_mean, ys_std = (
        ys.min(1)[0].squeeze(),
        ys.max(1)[0].squeeze(),
        ys.mean(1).squeeze(),
        ys.std(1).squeeze(),
    )
    ys_upper, ys_lower = ys_mean + ys_std, ys_mean - ys_std

    trace_max = Scatter(
        x=xs, y=ys_max.numpy(), line=Line(color=max_colour, dash="dash"), name="Max"
    )
    trace_upper = Scatter(
        x=xs,
        y=ys_upper.numpy(),
        line=Line(color=transparent),
        name="+1 Std. Dev.",
        showlegend=False,
    )
    trace_mean = Scatter(
        x=xs,
        y=ys_mean.numpy(),
        fill="tonexty",
        fillcolor=std_colour,
        line=Line(color=mean_colour),
        name="Mean",
    )
    trace_lower = Scatter(
        x=xs,
        y=ys_lower.numpy(),
        fill="tonexty",
        fillcolor=std_colour,
        line=Line(color=transparent),
        name="-1 Std. Dev.",
        showlegend=False,
    )
    trace_min = Scatter(
        x=xs, y=ys_min.numpy(), line=Line(color=max_colour, dash="dash"), name="Min"
    )

    plotly.offline.plot(
        {
            "data": [trace_upper, trace_mean, trace_lower, trace_min, trace_max],
            "layout": dict(
                title=title, xaxis={"title": "Step"}, yaxis={"title": title}
            ),
        },
        filename=os.path.join(path, title + ".html"),
        auto_open=False,
    )
