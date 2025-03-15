import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
import math
from skimage import io

from PIL import Image
import numpy as np
import skimage
import matplotlib
import matplotlib.pyplot as plt
from dataclasses import dataclass
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import argparse
import scipy.ndimage
from typing import List, Union

import time

# generate flattened coords
def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
    elif dim == 3:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)

    pixel_coords -= 0.5
    pixel_coords *= 2.
    pixel_coords = torch.Tensor(pixel_coords).view(-1, dim)
    return pixel_coords

"""sine layer, basic siren block"""

class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                             1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False,
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()

        self.net = []
        # First layer
        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        # coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output

    def forward_with_activations(self, coords, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)

                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()

                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else:
                x = layer(x)

                if retain_grad:
                    x.retain_grad()

            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations

def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)

def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div

def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad

def l2_loss(gt, predicted) -> torch.Tensor:
    loss = F.mse_loss(predicted, gt, reduction="none")
    # loss = loss * weight_map
    loss = loss.mean()
    return loss  

def normalize(
    data: np.ndarray
) -> np.ndarray:
    """
    use minmax normalization to scale and offset the data range to [normalized_min,normalized_max]
    """
    normalized_min, normalized_max = 0, 100
    original_min = float(data.min())
    original_max = float(data.max())
    data = (data - original_min) / (original_max - original_min)
    data *= normalized_max - normalized_min
    data += normalized_min
    return data

def inv_normalize(
    data: np.ndarray
) -> np.ndarray:
    dtype = "float32"
    if dtype == "uint8":
        dtype = np.uint8
    elif dtype == "uint12":
        dtype = np.uint12
    elif dtype == "uint16":
        dtype = np.uint16
    elif dtype == "float32":
        dtype = np.float32
    elif dtype == "float64":
        dtype = np.float64
    else:
        raise NotImplementedError
    data -= 0
    data /= 100 - 0
    data = np.clip(data, 0, 1)
    data = (
        data * (100 - 0)
        + 0
    )
    data = np.array(data, dtype=dtype)
    return data

def denoise(
    data: np.ndarray,
    denoise_level: int,
    denoise_close: Union[bool, List[int]],
) -> np.ndarray:
    denoised_data = np.copy(data)
    if denoise_close == False:
        # using 'denoise_level' as a hard threshold,
        # the pixel with instensity below this threshold will be set to zero
        denoised_data[data <= denoise_level] = 0
    else:
        # using 'denoise_level' as a soft threshold,
        # only the pixel with itself and neighbors instensities below this threshold will be set to zero
        denoised_data[
            ndimage.binary_opening(
                data <= denoise_level,
                structure=np.ones(tuple(list(denoise_close))),
                iterations=1,
            )
        ] = 0
    return denoised_data

def get_type_max(data):
    dtype = data.dtype.name
    if dtype == "uint8":
        max = 255
    elif dtype == "uint12":
        max = 4098
    elif dtype == "uint16":
        max = 65535
    elif dtype == "float32":
        max = 65535
    elif dtype == "float64":
        max = 65535
    else:
        raise NotImplementedError
    return max

def calc_psnr(gt: np.ndarray, predicted: np.ndarray):
    data_range = get_type_max(gt)
    mse = np.mean(np.power(predicted / data_range - gt / data_range, 2))
    psnr = -10 * np.log10(mse)
    return psnr

"""# **Experiments**"""

class Image3d(Dataset):
    def __init__(self, path_to_image):
        super().__init__()
        self.img = io.imread(path_to_image).astype(np.float32)
        self.channels = 1
        self.shape = self.img.shape
        self.max = float(self.img.max())
        self.min = float(self.img.min())

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        data = normalize(self.img)
        return data

class Implicit3DWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, sidelength=None, compute_diff=None, sample_fraction=1.):
        if isinstance(sidelength, int):
            sidelength = 3 * (sidelength,)

        self.compute_diff = compute_diff
        self.dataset = dataset
        self.mgrid = get_mgrid(sidelength, dim=3)

        self.sample_fraction = sample_fraction
        self.N_samples = int(self.sample_fraction * self.mgrid.shape[0])

        self.transform = Compose([ToTensor()])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.sample_fraction < 1.:
            coord_idx = torch.randint(0, self.data.shape[0], (self.N_samples,))
            data = self.dataset[coord_idx, ...]
            coords = self.mgrid[coord_idx, ...]
        else:
            coords = self.mgrid
            # transfrom ground truth data into tensor
            data = self.transform(self.dataset[idx])
            data = data.permute(1, 2, 0).contiguous().view(-1, self.dataset.channels)

        in_dict = {'idx': idx, 'coords': coords}
        gt_dict = {'img': data}

        return coords, data

def calc_features(param_count, coords_channel, data_channel, layers, **kwargs):
    a = layers - 2
    b = coords_channel + 1 + layers - 2 + data_channel
    c = -param_count + data_channel

    if a == 0:
        features = round(-c / b)
    else:
        features = round((-b + math.sqrt(b**2 - 4 * a * c)) / (2 * a))
    return features

def calc_param_count(coords_channel, data_channel, features, layers, **kwargs):
    param_count = (
        coords_channel * features
        + features
        + (layers - 2) * (features**2 + features)
        + features * data_channel
        + data_channel
    )
    return int(param_count)

def get_nnmodule_param_count(module: nn.Module):
    param_count = 0
    for param in module.state_dict().values():
        param_count += int(np.prod(param.shape))
    return param_count

"""Run the Experiment"""

# Prepare data
img = Image3d('./data/small_cube_100.tif')
coord = Implicit3DWrapper(img, sidelength=100, sample_fraction=1.)
n_samples = coord.N_samples
dataloader = DataLoader(coord, batch_size=1, pin_memory=True, num_workers=0)

# Define experiments
compression_rate = 5

# Calculate network size
ideal_network_size_bytes = os.path.getsize('./data/small_cube_100.tif') / compression_rate
print('Ideal network size in bytes: ', ideal_network_size_bytes)
ideal_network_parameters_count = ideal_network_size_bytes / 4.0
n_network_features = calc_features(
    param_count=ideal_network_parameters_count, coords_channel=3, data_channel=1, layers=5
)
actual_network_parameters_count = calc_param_count(
    features=n_network_features, coords_channel=3, data_channel=1,layers=5
)
print("Actual network parameters: ", actual_network_parameters_count)
actual_network_size_bytes = actual_network_parameters_count * 4.0
print("Actual network size in bytes: ", actual_network_size_bytes)

# prepare network
img_3d_siren = Siren(in_features=3, hidden_features=n_network_features, hidden_layers=3, out_features=1, outermost_linear=True, first_omega_0=30)
# Check actual network size
assert (
    get_nnmodule_param_count(img_3d_siren) == actual_network_parameters_count
), "The calculated network structure mismatch the actual_network_parameters_count!"
img_3d_siren.cuda()

total_steps = 80000
steps_til_summary = 100

# Assuming your Siren3D model takes 3D coordinates as input
model_input, ground_truth = next(iter(dataloader))
model_input, ground_truth = model_input.cuda(), ground_truth.cuda()
torch.cuda.empty_cache()
optim = torch.optim.Adamax(lr=0.001, params=img_3d_siren.parameters(), betas=[0.9, 0.999], eps=1e-8, weight_decay=0)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[50000,60000,70000], gamma=0.2)

for step in range(total_steps):
    optim.zero_grad()
    model_output = img_3d_siren(model_input)    
    loss = l2_loss(ground_truth, model_output)
    if not step % steps_til_summary:
        print("Step %d, Total loss %0.6f" % (step, loss))
        # img_grad = gradient(model_output, coords)
        # img_laplacian = laplace(model_output, coords)

    loss.backward()
    optim.step()
    lr_scheduler.step()

# save model
torch.save(img_3d_siren.state_dict(), f'./siren_compress_{compression_rate}.pt')


# Show the model output and ground truth
original_shape = (100, 100, 100)
decompressed_data = model_output.view(*original_shape).cpu().detach().numpy()
decompressed_data = inv_normalize(decompressed_data)
ground_truth_3d = ground_truth.view(*original_shape).cpu().detach().numpy()

psnr = calc_psnr(ground_truth_3d[..., 0], decompressed_data[..., 0])
print(f"psnr of {compression_rate} is: ", psnr)

# Choose the slice index along the z-axis (e.g., slice_index = 50)
slice_index = 50

# Extract the 2D slice along the z-axis
slice_2d = decompressed_data[:, :, slice_index]
gt_2d = ground_truth_3d[:, :, slice_index]
fig, axes = plt.subplots(1,2, figsize=(18,6))
# Create a figure and display the 2D slice
axes[0].imshow(slice_2d, cmap='gray')  
axes[1].imshow(gt_2d, cmap='gray')
plt.show()
plt.savefig(f'test_compare_{compression_rate}.tif')

plt.figure()
plt.imshow(slice_2d, cmap='gray')
plt.show()
plt.savefig(f'test_result_{compression_rate}.tif')

print("Train complete!")
