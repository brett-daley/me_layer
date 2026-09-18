# coding=utf-8
# Copyright 2021 The Rliable Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ---------------------------------------------------------------------------
# This file is derived from google-research/rliable
# (https://github.com/google-research/rliable), and has been MODIFIED by
# Prabhat Nagarajan (2026). Changes from the original:
#   * `aggregate_iqm` (from rliable/metrics.py) and `StratifiedBootstrap` /
#     `get_interval_estimates` (from rliable/library.py) were extracted into
#     this single module; all other upstream code was dropped.
#   * `StratifiedBootstrap.__init__` now forwards `seed=` rather than
#     `random_state=` to `arch.bootstrap.IIDBootstrap`, for compatibility with
#     arch >= 5 (this repo pins arch==8.0.0).
#   * `get_interval_estimates` dropped the `absl.logging` progress logging and
#     the `StratifiedIndependentBootstrap` branch for list-valued scores; it
#     now asserts that scores are a single `np.ndarray`.
# A copy of the Apache License 2.0 is included at licenses/Apache-2.0.txt.
# ---------------------------------------------------------------------------
"""Interquartile mean and stratified bootstrap CIs, vendored from rliable."""

import numpy as np
from numpy import random
from typing import Callable, Dict, List, Optional, Tuple, Union, Mapping
import arch.bootstrap as arch_bs
import scipy.stats
Float = Union[float, np.float32, np.float64]

def aggregate_iqm(scores: np.ndarray):
  """Computes the interquartile mean across runs and tasks.

  Args:
    scores: A matrix of size (`num_runs` x `num_tasks`) where scores[n][m]
      represent the score on run `n` of task `m`.
  Returns:
    IQM (25% trimmed mean) of scores.
  """
  return scipy.stats.trim_mean(scores, proportiontocut=0.25, axis=None)


####################### Stratified Bootstrap #######################
class StratifiedBootstrap(arch_bs.IIDBootstrap):
  """Bootstrap using stratified resampling.

  Supports numpy arrays. Data returned has the same type as the input data.
  Data entered using keyword arguments is directly accessibly as an attribute.

  To ensure a reproducible bootstrap, you must set the `random_state`
  attribute after the bootstrap has been created. See the example below.
  Note that `random_state` is a reserved keyword and any variable
  passed using this keyword must be an instance of `RandomState`.

  Examples
  --------
  Data can be accessed in a number of ways.  Positional data is retained in
  the same order as it was entered when the bootstrap was initialized.
  Keyword data is available both as an attribute or using a dictionary syntax
  on kw_data.

  >>> from rliable.library import StratifiedBootstrap
  >>> from numpy.random import standard_normal
  >>> x = standard_normal((5, 50))
  >>> bs = StratifiedBootstrap(x)
  >>> for data in bs.bootstrap(100):
  ...     bs_x = data[0][0]
  >>> bs.conf_int(np.mean, method='percentile', reps=50000)  # 95% CIs for mean

  Set the random_state if reproducibility is required.

  >>> from numpy.random import RandomState
  >>> rs = RandomState(1234)
  >>> bs = StratifiedBootstrap(x, random_state=rs)

  See also: `arch.bootstrap.IIDBootstrap`

  Attributes:
    data: tuple, Two-element tuple with the pos_data in the first position and
      kw_data in the second (pos_data, kw_data). Derived from `IIDBootstrap`.
    pos_data: tuple, Tuple containing the positional arguments (in the order
      entered). Derived from `IIDBootstrap`.
    kw_data: dict, Dictionary containing the keyword arguments. Derived from
      `IIDBootstrap`.
  """

  _name = 'Stratified Bootstrap'

  def __init__(
      self,
      *args: np.ndarray,
      random_state: Optional[random.RandomState] = None,
      task_bootstrap: bool = False,
      **kwargs: np.ndarray,
  ) -> None:
    """Initializes StratifiedBootstrap.

    Args:
      *args: Positional arguments to bootstrap. Typically used for the
        performance on a suite of tasks with multiple runs/episodes. The inputs
        are assumed to be of the shape `(num_runs, num_tasks, ..)`.
      random_state: If specified, ensures reproducibility in uncertainty
        estimates.
      task_bootstrap: Whether to perform bootstrapping (a) over runs or (b) over
        both runs and tasks. Defaults to False which corresponds to (a). (a)
        captures the statistical uncertainty in the aggregate performance if the
        experiment is repeated using a different set of runs (e.g., changing
        seeds) on the same set of tasks. (b) captures the sensitivity of the
        aggregate performance to a given task and provides the performance
        estimate if we had used a larger unknown population of tasks.
      **kwargs: Keyword arguments, passed directly to `IIDBootstrap`.
    """

    super().__init__(*args, seed=random_state, **kwargs)
    self._args_shape = args[0].shape
    self._num_tasks = self._args_shape[1]
    self._parameters = [self._num_tasks, task_bootstrap]
    self._task_bootstrap = task_bootstrap
    self._strata_indices = self._get_strata_indices()

  def _get_strata_indices(self) -> List[np.ndarray]:
    """Samples partial indices for bootstrap resamples.

    Returns:
      A list of arrays of size N x 1 x 1 x .., 1 x M x 1 x ..,
      1 x 1 x L x .. and so on, where the `args_shape` is `N x M x L x ..`.
    """
    ogrid_indices = tuple(slice(x) for x in (0, *self._args_shape[1:]))
    strata_indices = np.ogrid[ogrid_indices]
    return strata_indices[1:]

  def update_indices(self,) -> Tuple[np.ndarray, ...]:
    """Selects the indices to sample from the bootstrap distribution."""
    # `self._num_items` corresponds to the number of runs
    indices = np.random.choice(self._num_items, self._args_shape, replace=True)
    if self._task_bootstrap:
      task_indices = np.random.choice(
          self._num_tasks, self._strata_indices[0].shape, replace=True)
      return (indices, task_indices, *self._strata_indices[1:])
    return (indices, *self._strata_indices)


####################### Interval Estimates #######################
def get_interval_estimates(
    score_dict: Union[Mapping[str, np.ndarray], Mapping[str, List[np.ndarray]]],
    func: Callable[..., np.ndarray],
    method: str = 'percentile',
    task_bootstrap: bool = False,
    reps: int = 50000,
    confidence_interval_size: Float = 0.95,
    random_state: Optional[random.RandomState] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
  """Computes interval estimates via stratified bootstrap confidence intervals.

  Args:
    score_dict: A dictionary of scores for each method where scores are arranged
      as a matrix of the shape (`num_runs` x `num_tasks` x ..). For example, the
      scores could be 2D matrix containing final scores of the algorithm or a 3D
      matrix containing evaluation scores at multiple points during training.
    func: Function that computes the aggregate performance, which outputs a 1D
      numpy array. See Notes for requirements. For example, if computing
      estimates for interquartile mean across all runs, pass the function as
      `lambda x: np.array([metrics.aggregate_IQM])`.
    method:  One of `basic`, `percentile`, `bc` (identical to `debiased`,
      `bias-corrected’), or ‘bca`.
    task_bootstrap:  Whether to perform bootstrapping over tasks in addition to
      runs. Defaults to False. See `StratifiedBoostrap` for more details.
    reps: Number of bootstrap replications.
    confidence_interval_size: Coverage of confidence interval. Defaults to 95%.
    random_state: If specified, ensures reproducibility in uncertainty
      estimates.

  Returns:
    point_estimates: A dictionary of point estimates obtained by applying `func`
      on score data corresponding to each key in `data_dict`.
    interval_estimates: Confidence intervals~(CIs) for point estimates. Default
      is to return 95% CIs. Returns a np array of size (2 x ..) where the first
      row contains the lower bounds while the second row contains the upper
      bound of the 95% CIs.
  Notes:
    When there are no extra keyword arguments, the function is called

    .. code:: python

        func(*args, **kwargs)

    where args and kwargs are the bootstrap version of the data provided
    when setting up the bootstrap.  When extra keyword arguments are used,
    these are appended to kwargs before calling func.

    The bootstraps are:

    * 'basic' - Basic confidence using the estimated parameter and
      difference between the estimated parameter and the bootstrap
      parameters.
    * 'percentile' - Direct use of bootstrap percentiles.
    * 'bc' - Bias corrected using estimate bootstrap bias correction.
    * 'bca' - Bias corrected and accelerated, adding acceleration parameter
      to 'bc' method.
  """
  interval_estimates, point_estimates = {}, {}
  for key, scores in score_dict.items():
    assert isinstance(scores, np.ndarray)
    stratified_bs = StratifiedBootstrap(
        scores, task_bootstrap=task_bootstrap, random_state=random_state)
    point_estimates[key] = func(scores)
    interval_estimates[key] = stratified_bs.conf_int(
        func, reps=reps, size=confidence_interval_size, method=method)
  return point_estimates, interval_estimates