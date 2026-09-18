import pfrl
from pfrl.utils.contexts import evaluating
import torch
from typing import Any, Sequence


class IQN(pfrl.agents.IQN):
    def compute_q(
        self, batch_obs: Sequence[Any], batch_action: Sequence[Any]
    ) -> Sequence[Any]:
        with torch.no_grad(), evaluating(self.model):
            batch_av = self._evaluate_model_and_update_recurrent_states(batch_obs)
            q_values = batch_av.q_values
            batch_q_values = q_values[torch.arange(q_values.shape[0]), batch_action]
            return batch_q_values
