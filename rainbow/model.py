# -*- coding: utf-8 -*-
from __future__ import division
import math
import torch
from torch import nn
from torch.nn import functional as F
from .interface import SENSOR_DIM


# Factorised NoisyLinear layer with bias
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, input):
        if self.training:
            return F.linear(
                input,
                self.weight_mu + self.weight_sigma * self.weight_epsilon,
                self.bias_mu + self.bias_sigma * self.bias_epsilon,
            )
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)


class DQN(nn.Module):
    def __init__(self, args, action_space):
        super(DQN, self).__init__()
        self.atoms = args.atoms
        self.action_space = action_space

        self.convs = nn.Sequential(
            nn.Conv2d(args.history_length, 32, kernel_size=5, stride=2, padding=2),  # 128 -> 64
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 32 -> 16
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),  # 16 -> 8
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((14, 14)),
        )
        self.conv_output_size = 128 * 14 * 14

        self.sensor_fc = nn.Sequential(
            nn.Linear(SENSOR_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
        )
        self.image_only = args.image_only

        if self.image_only:
            fused_input_size = self.conv_output_size
        else:
            fused_input_size = self.conv_output_size + 32
        self.fc_h_v = NoisyLinear(
            fused_input_size, args.hidden_size, std_init=args.noisy_std
        )
        self.fc_h_a = NoisyLinear(
            fused_input_size, args.hidden_size, std_init=args.noisy_std
        )
        self.fc_z_v = NoisyLinear(args.hidden_size, self.atoms, std_init=args.noisy_std)
        self.fc_z_a = NoisyLinear(
            args.hidden_size, action_space * self.atoms, std_init=args.noisy_std
        )

    def forward(self, x, log=False):
        if self.image_only:
            image = x["image"]
            conv_out = self.convs(image)
            conv_out = conv_out.view(-1, self.conv_output_size)
            fused = conv_out
        else:
            image = x["image"]
            sensor = x["sensor"]

            if sensor.dim() == 3:
                sensor = sensor[:, -1, :]
            conv_out = self.convs(image)
            conv_out = conv_out.view(-1, self.conv_output_size)
            sensor_out = self.sensor_fc(sensor)
            fused = torch.cat([conv_out, sensor_out], dim=1)

        v = self.fc_z_v(F.relu(self.fc_h_v(fused)))
        a = self.fc_z_a(F.relu(self.fc_h_a(fused)))
        v, a = v.view(-1, 1, self.atoms), a.view(-1, self.action_space, self.atoms)
        q = v + a - a.mean(1, keepdim=True)
        q = F.log_softmax(q, dim=2) if log else F.softmax(q, dim=2)
        return q

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.fc(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)


class ImprovedDQN(nn.Module):
    def __init__(self, args, action_space):
        super().__init__()
        self.atoms = args.atoms
        self.action_space = action_space
        H = args.history_length

        # 1) Initial Conv + Pool
        self.conv1 = nn.Sequential(
            nn.Conv2d(H, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # 128 -> 64
        )
        # Residual block (32 -> 32)
        self.res1 = ResidualBlock(32)

        # 2) Middle Conv
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 64 -> 32
            nn.ReLU()
        )
        self.res2 = ResidualBlock(64)

        # 3) Dilated Conv + SE attention
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=2, dilation=2),  # 32 -> 32
            nn.ReLU(),
            SEBlock(128)
        )

        # 4) Final Conv + Pool
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),  # 32 -> 16
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)  # 16x16 -> 1x1
        )

        conv_out_size = 128  # 128 * 1 * 1

        self.sensor_fc = nn.Sequential(
            nn.Linear(SENSOR_DIM, 64), nn.ReLU(),
            nn.Linear(64, 128),         nn.ReLU(),
            nn.Linear(128, 128),        nn.ReLU(),
            nn.Linear(128, 64),         nn.ReLU(),
            nn.Linear(64, 32),          nn.ReLU(),
            nn.LayerNorm(32),
        )
        self.image_only = args.image_only
        if self.image_only:
            fused_size = conv_out_size
        else:
            fused_size = conv_out_size + 32

        self.fc_h_v = NoisyLinear(fused_size, args.hidden_size, std_init=args.noisy_std)
        self.fc_h_a = NoisyLinear(fused_size, args.hidden_size, std_init=args.noisy_std)
        self.fc_z_v = NoisyLinear(args.hidden_size, self.atoms, std_init=args.noisy_std)
        self.fc_z_a = NoisyLinear(args.hidden_size, action_space * self.atoms, std_init=args.noisy_std)

    def forward(self, x, log=False):
        img = x["image"]  # [B, H, 128, 128]

        out = self.conv1(img)
        out = self.res1(out)
        out = self.conv2(out)
        out = self.res2(out)
        out = self.conv3(out)
        out = self.conv4(out)    # [B, 128, 1, 1]
        conv_out = out.view(out.size(0), -1)  # [B, 128]

        if self.image_only:
            fused = conv_out
        else:
            sensor = x["sensor"]
            if sensor.dim() == 3:
                sensor = sensor[:, -1, :]
            sensor_out = self.sensor_fc(sensor)
            fused = torch.cat([conv_out, sensor_out], dim=1)

        # Noisy linear & Dueling
        v = self.fc_z_v(F.relu(self.fc_h_v(fused)))
        a = self.fc_z_a(F.relu(self.fc_h_a(fused)))
        v, a = v.view(-1, 1, self.atoms), a.view(-1, self.action_space, self.atoms)
        q = v + a - a.mean(1, keepdim=True)
        return F.log_softmax(q, dim=2) if log else F.softmax(q, dim=2)

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()
