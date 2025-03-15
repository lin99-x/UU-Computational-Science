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

"""Customized tensorflow dataset (tfds) for tumor cubes."""
import tensorflow as tf
import tensorflow_datasets as tfds
from pathlib import Path
from skimage import io
import numpy as np


class TumorCustom(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for tumor dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return tfds.core.DatasetInfo(
        builder=self,
        features=tfds.features.FeaturesDict({
            'volume':
                tfds.features.Tensor(shape=(100, 100, 100), dtype=tf.float32),
        }),
        supervised_keys=None,  # Set to `None` to disable
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    path = dl_manager.download_and_extract(
        'https://drive.google.com/uc?id=1TKtoOZ5LnLKuVzwm5vzcGpXSw1lceJfc')
    return {
        'train': self._generate_examples(path),
    }

  def _generate_examples(self, path):
    data_path = Path(path) / 'small_cubes_dataset'
    # print(f'Data path: {data_path}')
    for volume_path in data_path.glob('*.tif'):
      # print(f'Processing volume: {volume_path.name}')
      # Load 3D volume
      try:
        volume = io.imread(str(volume_path)).astype(np.float32)
        if volume.shape != (100, 100, 100):
          raise ValueError(f'Invalid volume shape for {volume_path.name}. Expected (100, 100, 100), got {volume.shape}')
        yield volume_path.name, {'volume': volume}
      except Exception as e:
        print(f'Error loading {volume_path.name}: {str(e)}')
