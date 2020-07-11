import torch
import torch.nn as nn
import torch.nn.functional as F

from policies.base import FF_Base, LSTM_Base

class Stochastic_Actor:
  """
  The base class for stochastic actors.
  """
  def __init__(self, latent, action_dim, dynamics_randomization, fixed_std):

    self.action_dim             = action_dim
    self.dynamics_randomization = dynamics_randomization
    self.means                  = nn.Linear(latent, action_dim)

    self.fixed_std = fixed_std

  def _get_dist_params(self, state, update=False):
    state = self.normalize_state(state, update=update)
    x = self._base_forward(state)

    mu = self.means(x)

    std = self.fixed_std

    return mu, std

  def stochastic_forward(self, state, deterministic=True, update=False, log_probs=False):
    mu, sd = self._get_dist_params(state, update=update)

    if not deterministic or log_probs:
      dist = torch.distributions.Normal(mu, sd)
      sample = dist.rsample()

    action = mu if deterministic else sample

    return action

  def pdf(self, state):
    mu, sd = self._get_dist_params(state)
    return torch.distributions.Normal(mu, sd)


class FF_Stochastic_Actor(FF_Base, Stochastic_Actor):
  """
  A class inheriting from FF_Base and Stochastic_Actor
  which implements a feedforward stochastic policy.
  """
  def __init__(self, input_dim, action_dim, layers=(256, 256), dynamics_randomization=False, nonlinearity=torch.tanh, fixed_std=None):

    FF_Base.__init__(self, input_dim, layers, nonlinearity)
    Stochastic_Actor.__init__(self, layers[-1], action_dim, dynamics_randomization, fixed_std)

  def forward(self, x, deterministic=True, update_norm=False, return_log_probs=False):
    return self.stochastic_forward(x, deterministic=deterministic, update=update_norm, log_probs=return_log_probs)


class LSTM_Stochastic_Actor(LSTM_Base, Stochastic_Actor):
  """
  A class inheriting from LSTM_Base and Stochastic_Actor
  which implements a recurrent stochastic policy.
  """
  def __init__(self, input_dim, action_dim, layers=(128, 128), dynamics_randomization=False, fixed_std=None):

    LSTM_Base.__init__(self, input_dim, layers)
    Stochastic_Actor.__init__(self, layers[-1], action_dim, dynamics_randomization, fixed_std)

    self.is_recurrent = True
    self.init_hidden_state()

  def forward(self, x, deterministic=True, update_norm=False, return_log_probs=False):
    return self.stochastic_forward(x, deterministic=deterministic, update=update_norm, log_probs=return_log_probs)
