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

"""Utils for loading and processing datasets."""

from typing import Mapping, Optional
import jax.numpy as jnp
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


Array = jnp.ndarray
Batch = Mapping[str, np.ndarray]
DATASET_ATTRIBUTES = {
    # roughly split the dataset into 80-20
    'tumor_custom': {
      'num_channels': 1,
      'resolution': 100,
      'type': 'volume',
      'train_size': 1411,
      'test_size': 353,
    },
}


def load_dataset(dataset_name: str,
                 subset: str,
                 batch_size: Optional[int] = None,
                 shuffle: bool = False,
                 repeat: bool = False,
                 num_examples: Optional[int] = None,
                 shuffle_buffer_size: int = 10000):
  """Tensorflow dataset loaders.

  Args:
    dataset_name (string): One of elements of DATASET_NAMES.
    subset (string): One of 'train', 'test'.
    batch_size (int):
    shuffle (bool): Whether to shuffle dataset.
    repeat (bool): Whether to repeat dataset.
    num_examples (int): If not -1, returns only the first num_examples of the
      dataset.
    shuffle_buffer_size (int): Buffer size to use for shuffling dataset.

  Returns:
    Tensorflow dataset iterator.
  """

  # Load dataset
  if dataset_name.startswith('tumor'):
    # tumor does not have a test dataset, so do a 80/20
    # split on training data to create train and test sets
    if subset == 'train':
      subset = 'train[:80%]'
    elif subset == 'test':
      subset = 'train[80%:]'
    ds = tfds.load(dataset_name, split=subset)
    ds = ds.map(process_tumor, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  # Optionally subsample dataset
  if num_examples is not None:
    ds = ds.take(num_examples)

  # Optionally shuffle dataset
  if shuffle:
    ds = ds.shuffle(shuffle_buffer_size)

  # Optionally repeat dataset if repeat
  if repeat:
    ds = ds.repeat()

  if batch_size is not None:
    ds = ds.batch(batch_size)

  # Convert from tf.Tensor to numpy arrays for use with Jax
  return iter(tfds.as_numpy(ds))


def process_tumor(batch: Batch):
  volume = tf.cast(batch['volume'], tf.float32) / 65535.
  return {'array': volume}





