import unittest
from unittest import mock

import numpy as np
import pytest
import statistics

from me_layer.experiments import evaluator


def reward_phi(x):
    return np.clip(x, -1, 1)


@pytest.mark.parametrize("n_episodes", [None, 1])
@pytest.mark.parametrize("n_steps", [2, 5, 7])
def test_run_evaluation_episodes_with_n_steps(n_episodes, n_steps):
    # MagicMock can mock eval_mode while Mock cannot
    # TODO: Check that last untruncated episode is good.
    agent = mock.MagicMock()
    env = mock.Mock()
    # First episode: 0 -> 1 -> 2 -> 3 (reset)
    # Second episode: 4 -> 5 -> 6 -> 7 (done)
    env.reset.side_effect = [("state", 0), ("state", 4), ("state", 8)]
    env.step.side_effect = [
        (("state", 1), 1.3, False, False, {}),
        (("state", 2), 0.2, False, False, {}),
        (("state", 3), -5.0, False, True, {}),
        (("state", 5), -0.5, False, False, {}),
        (("state", 6), 0.0, False, False, {}),
        (("state", 7), 1.0, True, False, {}),
        (("state", 9), 1.0, False, False, {}),
    ]
    q_sequence = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    agent.compute_q.side_effect = q_sequence

    if n_episodes:
        with pytest.raises(AssertionError):
            (
                scores,
                returns,
                episode_undiscounted_returns,
                episode_discounted_returns,
                lengths,
                online_q_values,
                std_online_q_values,
                overestimations,
                std_overestimations,
                normalized_overestimations,
                std_normalized_overestimations,
                truncated_episode_count,
            ) = evaluator.run_evaluation_episodes(
                env,
                agent,
                n_steps=n_steps,
                n_episodes=n_episodes,
                discount=0.9,
                reward_phi=reward_phi,
            )
    else:
        (
            scores,
            returns,
            episode_undiscounted_returns,
            episode_discounted_returns,
            lengths,
            online_q_values,
            std_online_q_values,
            overestimations,
            std_overestimations,
            normalized_overestimations,
            std_normalized_overestimations,
            truncated_episode_count,
        ) = evaluator.run_evaluation_episodes(
            env,
            agent,
            n_steps=n_steps,
            n_episodes=n_episodes,
            discount=0.9,
            reward_phi=reward_phi,
        )
        num_truncations = 1 if n_steps > 2 else 0
        assert agent.act.call_count == n_steps + num_truncations
        assert agent.observe.call_count == n_steps
        assert agent.compute_q.call_count == n_steps + num_truncations
        if n_steps == 2:
            assert len(scores) == 1
            assert len(lengths) == 1
            assert len(returns) == 0
            assert len(online_q_values) == 0
            assert len(overestimations) == 0
            assert len(std_overestimations) == 0
            np.testing.assert_allclose(scores[0], 1.5)
            np.testing.assert_allclose(lengths[0], 2)
        elif n_steps == 5:
            assert len(scores) == 1
            assert len(lengths) == 1
            assert len(returns) == 1
            assert len(online_q_values) == 1
            assert len(overestimations) == 1
            assert len(std_overestimations) == 1
            np.testing.assert_allclose(scores[0], -3.5)
            np.testing.assert_allclose(lengths[0], 3)
            return_1 = 1.0 + 0.9 * 0.2 + 0.81 * (-1.0 + 0.9 * 4.0)
            return_2 = 0.2 + 0.9 * (-1.0 + 0.9 * 4.0)
            return_3 = -1.0 + 0.9 * 4.0
            pointwise_overestimations = [
                q_sequence[0] - return_1,
                q_sequence[1] - return_2,
                q_sequence[2] - return_3,
            ]
            np.testing.assert_allclose(returns[0], (return_1 + return_2 + return_3) / 3)
            np.testing.assert_allclose(online_q_values[0], 2.0)
            np.testing.assert_allclose(
                overestimations[0], online_q_values[0] - returns[0]
            )
            np.testing.assert_allclose(
                overestimations[0], sum(pointwise_overestimations) / 3
            )
            true_stdev_overestimations = statistics.stdev(pointwise_overestimations)
            np.testing.assert_allclose(
                std_overestimations[0], true_stdev_overestimations
            )
        else:
            assert len(scores) == 2
            assert len(lengths) == 2
            assert len(returns) == 2
            assert len(online_q_values) == 2
            assert len(overestimations) == 2
            assert len(std_overestimations) == 2
            np.testing.assert_allclose(scores[0], -3.5)
            np.testing.assert_allclose(scores[1], 0.5)
            np.testing.assert_allclose(lengths[0], 3)
            np.testing.assert_allclose(lengths[1], 3)
            return_4 = -0.5 + 0.9 * 0 + 0.81 * 1.0
            return_5 = 0 + 0.9 * 1.0
            return_6 = 1.0
            pointwise_overestimations = [
                q_sequence[4] - return_4,
                q_sequence[5] - return_5,
                q_sequence[6] - return_6,
            ]
            np.testing.assert_allclose(returns[1], (return_4 + return_5 + return_6) / 3)
            np.testing.assert_allclose(online_q_values[1], 6.0)
            np.testing.assert_allclose(
                overestimations[1], online_q_values[1] - returns[1]
            )
            np.testing.assert_allclose(
                overestimations[1], sum(pointwise_overestimations) / 3
            )
            true_stdev_overestimations = statistics.stdev(pointwise_overestimations)
            np.testing.assert_allclose(
                std_overestimations[1], true_stdev_overestimations
            )


@pytest.mark.parametrize("n_episodes", [None, 1])
@pytest.mark.parametrize("n_steps", [5, 7])
def test_eval_performance(n_episodes, n_steps):
    # MagicMock can mock eval_mode while Mock cannot
    agent = mock.MagicMock()
    env = mock.Mock()
    # First episode: 0 -> 1 -> 2 -> 3 (reset)
    # Second episode: 4 -> 5 -> 6 -> 7 (done)
    env.reset.side_effect = [("state", 0), ("state", 4), ("state", 8)]
    env.step.side_effect = [
        (("state", 1), 1.3, False, False, {}),
        (("state", 2), 0.2, False, False, {}),
        (("state", 3), -5.0, False, True, {}),
        (("state", 5), -0.5, False, False, {}),
        (("state", 6), 0.0, False, False, {}),
        (("state", 7), 1.0, True, False, {}),
        (("state", 9), 1.0, False, False, {}),
    ]
    q_sequence = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    agent.compute_q.side_effect = q_sequence

    if n_episodes:
        with pytest.raises(AssertionError):
            eval_stats = evaluator.eval_performance(
                env,
                agent,
                n_steps=n_steps,
                n_episodes=n_episodes,
                discount=0.9,
                reward_phi=reward_phi,
            )
    else:
        eval_stats = evaluator.eval_performance(
            env,
            agent,
            n_steps=n_steps,
            n_episodes=n_episodes,
            discount=0.9,
            reward_phi=reward_phi,
        )
        num_truncations = 1 if n_steps > 2 else 0
        assert agent.act.call_count == n_steps + num_truncations
        assert agent.observe.call_count == n_steps
        assert agent.compute_q.call_count == n_steps + num_truncations
        return_1 = 1.0 + 0.9 * 0.2 + 0.81 * (-1.0 + 0.9 * 4.0)
        return_2 = 0.2 + 0.9 * (-1.0 + 0.9 * 4.0)
        return_3 = -1.0 + 0.9 * 4.0
        average_return_1 = (return_1 + return_2 + return_3) / 3
        return_4 = -0.5 + 0.9 * 0 + 0.81 * 1.0
        return_5 = 0 + 0.9 * 1.0
        return_6 = 1.0
        average_return_2 = (return_4 + return_5 + return_6) / 3
        if n_steps == 5:
            np.testing.assert_allclose(eval_stats["mean_return"], average_return_1)
            np.testing.assert_allclose(eval_stats["mean_q_value"], 2.0)
        else:
            np.testing.assert_allclose(
                eval_stats["mean_return"], (average_return_1 + average_return_2) / 2.0
            )
            np.testing.assert_allclose(eval_stats["mean_q_value"], (2.0 + 6.0) / 2.0)


class TestRunEvaluationEpisode(unittest.TestCase):
    def test_needs_reset(self):
        # MagicMock can mock eval_mode while Mock cannot
        agent = mock.MagicMock()
        env = mock.Mock()
        # First episode: 0 -> 1 -> 2 -> 3 (reset)
        # Second episode: 4 -> 5 -> 6 -> 7 (done)
        env.reset.side_effect = [("state", 0), ("state", 4)]
        env.step.side_effect = [
            (("state", 1), 0, False, False, {}),
            (("state", 2), 0, False, False, {}),
            (("state", 3), 0, False, True, {"needs_reset": True}),
            (("state", 5), -0.5, False, False, {}),
            (("state", 6), 0, False, False, {}),
            (("state", 7), 1, True, False, {}),
        ]
        (
            scores,
            returns,
            episode_undiscounted_returns,
            episode_discounted_returns,
            lengths,
            online_q_values,
            std_online_q_values,
            overestimations,
            std_overestimations,
            normalized_overestimations,
            std_normalized_overestimations,
            truncated_episode_count,
        ) = evaluator.run_evaluation_episodes(
            env, agent, n_steps=None, n_episodes=2, discount=0.9, reward_phi=reward_phi
        )
        assert len(scores) == 2
        assert len(lengths) == 2

        np.testing.assert_allclose(scores[0], 0)
        np.testing.assert_allclose(scores[1], 0.5)
        np.testing.assert_allclose(lengths[0], 3)
        np.testing.assert_allclose(lengths[1], 3)
        assert agent.act.call_count == 7
        assert agent.observe.call_count == 6
