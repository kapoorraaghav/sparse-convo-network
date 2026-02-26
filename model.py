import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class H5Dataset(Dataset):
    def __init__(self, file_path, x_key="jet", threshold=0.0):
        self.file      = h5py.File(file_path, "r")
        self.X         = self.file[x_key]
        self.threshold = threshold

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x    = torch.tensor(self.X[idx], dtype=torch.float32)  # (H, W, C)
        x    = x.permute(2, 0, 1)                              # (C, H, W)
        mask = (x.abs().sum(dim=0, keepdim=True) > self.threshold)  # (1, H, W)
        return x, mask


def count_active_sites(mask):
    """Count active sites per layer — paper tracks this in Tables 2-4"""
    return mask.sum().item()

# In forward pass, track per layer:
def forward(self, x, mask):
    active_counts = {}

    z, m = self.enc1(x, mask);  z = self.relu(z)
    active_counts['enc1'] = count_active_sites(m)

    z, m = self.enc2(z, m);     z = self.relu(z)
    active_counts['enc2'] = count_active_sites(m)

    z, m = self.down1(z, m);    z = self.relu(z)
    active_counts['down1'] = count_active_sites(m)

    z, m = self.enc3(z, m);     z = self.relu(z)
    active_counts['enc3'] = count_active_sites(m)

    z, m = self.down2(z, m);    z = self.relu(z)
    active_counts['down2'] = count_active_sites(m)

    ...
    return x_hat, latent, active_counts

class SubmanifoldSparseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        padding   = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              padding=padding, bias=True)   # allow bias
        self.bn   = nn.BatchNorm2d(out_channels)

    def forward(self, x, mask):
        # Compute ground state once (what zero input produces)
        ground_state = self.bn(self.conv(torch.zeros_like(x[:1])))

        # Zero inactive sites, apply conv
        x   = x * mask.float()
        out = self.bn(self.conv(x))

        # Subtract ground state at inactive sites, zero them out
        out = out - ground_state                # center around ground state
        out = out * mask.float()               # zero inactive sites
        return out, mask


class StridedSparseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2):
        super().__init__()
        padding     = kernel_size // 2
        self.conv   = nn.Conv2d(in_channels, out_channels, kernel_size,
                                stride=stride, padding=padding, bias=False)
        self.bn     = nn.BatchNorm2d(out_channels)
        self.stride = stride

    def forward(self, x, mask):
        x   = x * mask.float()
        out = self.bn(self.conv(x))

        # match mask size to conv output size exactly
        out_h, out_w = out.shape[2], out.shape[3]
        new_mask = F.adaptive_max_pool2d(mask.float(), (out_h, out_w)) > 0

        out = out * new_mask.float()
        return out, new_mask
class SparseAutoencoder(nn.Module):
    def __init__(self, in_channels=8):  # only in_channels, no out_channels
        super().__init__()

        # Encoder
        self.enc1  = SubmanifoldSparseConv2d(in_channels, 32,  3)
        self.enc2  = SubmanifoldSparseConv2d(32,          64,  3)
        self.down1 = StridedSparseConv2d    (64,          64,  3, stride=2)
        self.enc3  = SubmanifoldSparseConv2d(64,          128, 3)
        self.down2 = StridedSparseConv2d    (128,         128, 3, stride=2)

        # Decoder
        self.up1   = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.dec1  = SubmanifoldSparseConv2d(64,  64,          3)
        self.up2   = nn.ConvTranspose2d(64,  32, 3, stride=2, padding=1, output_padding=1)
        self.dec2  = SubmanifoldSparseConv2d(32,  in_channels, 3)

        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x, mask):
        H, W = x.shape[2], x.shape[3]

        # Encoder
        z, m  = self.enc1(x, mask);  z = self.relu(z)
        z, m  = self.enc2(z, m);     z = self.relu(z)
        z, m  = self.down1(z, m);    z = self.relu(z)
        z, m  = self.enc3(z, m);     z = self.relu(z)
        z, m  = self.down2(z, m);    z = self.relu(z)

        latent = z

        # Decoder
        z     = self.relu(self.up1(z))
        m_up1 = F.interpolate(m.float(), scale_factor=2, mode='nearest') > 0
        z, _  = self.dec1(z, m_up1);  z = self.relu(z)

        z     = self.relu(self.up2(z))
        m_up2 = F.interpolate(m_up1.float(), scale_factor=2, mode='nearest') > 0
        x_hat, _ = self.dec2(z, m_up2)

        x_hat = x_hat[:, :, :H, :W]

        return x_hat, latent

