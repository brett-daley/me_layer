from unittest import mock

import numpy as np
import pytest

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
    agent.compute_q.side_effect = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

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
            assert len(episode_undiscounted_returns) == 0
            assert len(episode_discounted_returns) == 0
        elif n_steps == 5:
            assert len(episode_undiscounted_returns) == 1
            assert len(episode_discounted_returns) == 1
            discounted_return = 1.0 + 0.9 * 0.2 + 0.81 * (-1.0)
            undiscounted_return = 1.0 + 0.2 - 1.0
            np.testing.assert_allclose(episode_discounted_returns[0], discounted_return)
            np.testing.assert_allclose(
                episode_undiscounted_returns[0], undiscounted_return
            )
        else:
            assert len(episode_undiscounted_returns) == 2
            assert len(episode_discounted_returns) == 2
            undiscounted_return_1 = 1.0 + 0.2 - 1.0
            undiscounted_return_2 = -0.5 + 0 + 1.0
            discounted_return_1 = 1.0 + 0.9 * 0.2 + 0.81 * (-1.0)
            discounted_return_2 = -0.5 + 0.9 * 0 + 0.81 * 1.0
            np.testing.assert_allclose(
                episode_discounted_returns[0], discounted_return_1
            )
            np.testing.assert_allclose(
                episode_discounted_returns[1], discounted_return_2
            )
            np.testing.assert_allclose(
                episode_undiscounted_returns[0], undiscounted_return_1
            )
            np.testing.assert_allclose(
                episode_undiscounted_returns[1], undiscounted_return_2
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
    agent.compute_q.side_effect = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

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
        if n_steps == 5:
            np.testing.assert_allclose(eval_stats["mean"], -3.5)
            discounted_return = 1.0 + 0.9 * 0.2 + 0.81 * (-1.0)
            undiscounted_return = 1.0 + 0.2 - 1.0
            np.testing.assert_allclose(
                eval_stats["mean_discounted_return"], discounted_return
            )
            np.testing.assert_allclose(
                eval_stats["mean_undiscounted_return"], undiscounted_return
            )
        else:
            discounted_return_1 = 1.0 + 0.9 * 0.2 + 0.81 * (-1.0)
            discounted_return_2 = -0.5 + 0.9 * 0 + 0.81 * 1.0
            undiscounted_return_1 = 1.0 + 0.2 + (-1.0)
            undiscounted_return_2 = -0.5 + 0 + 1.0
            np.testing.assert_allclose(
                eval_stats["mean_discounted_return"],
                (discounted_return_1 + discounted_return_2) / 2.0,
            )
            np.testing.assert_allclose(
                eval_stats["mean_undiscounted_return"],
                (undiscounted_return_1 + undiscounted_return_2) / 2.0,
            )
