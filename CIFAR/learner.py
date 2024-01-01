import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class Conv_Standard_ANIL(nn.Module):
    def __init__(self, args, x_dim, hid_dim, z_dim, final_layer_size, stride=1):
        super(Conv_Standard_ANIL, self).__init__()
        self.args = args
        self.stride = stride
        if not hasattr(args, "meta_block_num"):
            self.net = nn.Sequential(self.conv_block(x_dim, hid_dim), self.conv_block(hid_dim, hid_dim),
                                    self.conv_block(hid_dim, hid_dim), self.conv_block(hid_dim, z_dim), Flatten())
        else:
            meta_block_num = args.meta_block_num
            assert meta_block_num > 4
            block_list = [self.conv_block(x_dim, hid_dim)]
            for _ in range(meta_block_num - 4):
                block_list.append(self.intermediate_conv_block(hid_dim, hid_dim))
            block_list.append(self.conv_block(hid_dim, hid_dim))
            block_list.append(self.conv_block(hid_dim, hid_dim))
            block_list.append(self.conv_block(hid_dim, z_dim))
            block_list.append(Flatten())
            #
            self.net = nn.Sequential(*block_list)

        self.dist = Beta(torch.FloatTensor([2]), torch.FloatTensor([2]))
        self.hid_dim = hid_dim

        self.logits = nn.Linear(final_layer_size, self.args.num_classes)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=self.stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def intermediate_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=1, padding=1)
        )

    def functional_conv_block(self, x, weights, biases,
                              bn_weights, bn_biases, is_training):
        """Performs 3x3 convolution, ReLu activation, 2x2 max pooling in a functional fashion.

        # Arguments:
            x: Input Tensor for the conv block
            weights: Weights for the convolutional block
            biases: Biases for the convolutional block
            bn_weights:
            bn_biases:
        """
        x = F.conv2d(x, weights, biases, padding=1, stride=self.stride)
        x = F.batch_norm(x, running_mean=None, running_var=None, weight=bn_weights, bias=bn_biases,
                         training=is_training)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        return x

    def intermediate_functional_conv_block(self, x, weights, biases,
                              bn_weights, bn_biases, is_training):
        """Performs 3x3 convolution, ReLu activation, 2x2 max pooling in a functional fashion.

        # Arguments:
            x: Input Tensor for the conv block
            weights: Weights for the convolutional block
            biases: Biases for the convolutional block
            bn_weights:
            bn_biases:
        """
        x = F.conv2d(x, weights, biases, padding=1, stride=self.stride)
        x = F.batch_norm(x, running_mean=None, running_var=None, weight=bn_weights, bias=bn_biases,
                         training=is_training)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        return x

    def forward(self, x):
        x = self.net(x)

        x = x.view(x.size(0), -1)

        return self.logits(x)

    def functional_forward(self, x, weights, is_training=True):
        x = self.net(x)

        x = x.view(x.size(0), -1)

        x = F.linear(x, weights['weight'], weights['bias'])

        return x

    def functional_forward_val(self, x, weights, weights_logits, is_training=True):
        if not hasattr(self.args, "meta_block_num"):
            for block in range(4):
                x = self.functional_conv_block(x, weights[f'net.{block}.0.weight'], weights[f'net.{block}.0.bias'],
                                            weights.get(f'net.{block}.1.weight'), weights.get(f'net.{block}.1.bias'),
                                            is_training)

            x = x.view(x.size(0), -1)

            x = F.linear(x, weights_logits['weight'], weights_logits['bias'])
        else:
            meta_block_num = self.args.meta_block_num
            #
            x = self.functional_conv_block(x, weights[f'net.{0}.0.weight'], weights[f'net.{0}.0.bias'],
                                            weights.get(f'net.{0}.1.weight'), weights.get(f'net.{0}.1.bias'),
                                            is_training)
            for block in range(1, (meta_block_num - 3)):
                x = self.intermediate_functional_conv_block(x, weights[f'net.{block}.0.weight'], weights[f'net.{block}.0.bias'],
                                                            weights.get(f'net.{block}.1.weight'), weights.get(f'net.{block}.1.bias'),
                                                            is_training)
            for block in range((meta_block_num - 3), meta_block_num):
                x = self.functional_conv_block(x, weights[f'net.{block}.0.weight'], weights[f'net.{block}.0.bias'],
                                            weights.get(f'net.{block}.1.weight'), weights.get(f'net.{block}.1.bias'),
                                            is_training)

            x = x.view(x.size(0), -1)

            x = F.linear(x, weights_logits['weight'], weights_logits['bias'])

        return x
