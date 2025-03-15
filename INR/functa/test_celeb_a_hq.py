# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Quick script to test that celeb_a_hq experiment can import and run."""
from absl import app
import jax
import jax.numpy as jnp
import experiment_meta_learning as exp


def main(_):
  """Tests the meta learning experiment on celeba."""
  config = exp.get_config()
  exp_config = config.experiment_kwargs.config
  exp_config.dataset.name = 'celeb_a_hq_custom'
  exp_config.training.per_device_batch_size = 2
  exp_config.evaluation.batch_size = 2
  exp_config.model.width = 16
  exp_config.model.depth = 2
  exp_config.model.latent_dim = 16
  print(exp_config)

  xp = exp.Experiment('train', jax.random.PRNGKey(0), exp_config)
  bcast = jax.pmap(lambda x: x)
  global_step = bcast(jnp.zeros(jax.local_device_count()))
  rng = bcast(jnp.stack([jax.random.PRNGKey(0)] * jax.local_device_count()))
  print('Taking a single experiment step for test purposes.')
  result = xp.step(global_step, rng)
  print(f'Step successfully taken, resulting metrics are {result}')


if __name__ == '__main__':
  app.run(main)
