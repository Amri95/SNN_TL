# -*-coding:utf-8-*-
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# import slayer_cuda
from torch.nn.modules.batchnorm import BatchNorm2d, BatchNorm3d

# snn主题结构和核心类
####################################################################################


class SlayerNet(nn.Module):

    def __init__(self, net_params, weights_init=[4, 4], device=torch.device('cuda')):
        super(SlayerNet, self).__init__()
        self.net_params = net_params
        self.trainer = SlayerTrainer(net_params, device)
        self.srm = self.trainer.calculate_srm_kernel()
        self.ref = self.trainer.calculate_ref_kernel()
        # Emulate a fully connected 250 -> 25
        self.fc1 = SpikeLinear(250, 25).to(device)
        nn.init.normal_(self.fc1.weight, mean=0, std=weights_init[0])
        # Emulate a fully connected 25 -> 1
        self.fc2 = SpikeLinear(25, 1).to(device)
        nn.init.normal_(self.fc2.weight, mean=0, std=weights_init[1])
        self.device=device

    def forward(self, x):
        # Apply srm to input spikes
        x = self.trainer.apply_srm_kernel(x, self.srm)
        # Linear + activation
        x = self.fc1(x)
        x = SpikeFunc.apply(x, self.net_params, self.ref, self.net_params['af_params']['sigma'][0], self.device)
        # Apply srm to middle layer spikes
        x = self.trainer.apply_srm_kernel(x.view(1,1,1,25,-1), self.srm)
        # # Apply second layer
        x = SpikeFunc.apply(self.fc2(x), self.net_params, self.ref, self.net_params['af_params']['sigma'][1], self.device)
        return x


# snn计算梯度下降的核心类定义 #
class SpikeFunc(torch.autograd.Function):
    # 如果你想要添加一个新的 Operation的话，你的Operation需要继承 class Function
    # torch.autograd.Function中的class Function是自动计算梯度的核心类，
    # autograd使用Function计算结果和梯度，同时编码 operation的历史

    @staticmethod
    def forward(ctx, multiplied_activations, net_params, ref, sigma, device=torch.device('cuda')):
        # Calculate membrane potentials

        (multiplied_activations, spikes) = SpikeFunc.get_spikes_cuda(multiplied_activations, ref, net_params,
                                                                     device=device)
        scale = torch.autograd.Variable(torch.tensor(net_params['pdf_params']['scale'],
                                                     device=device, dtype=torch.float32), requires_grad=False)
        tau = torch.autograd.Variable(torch.tensor(net_params['pdf_params']['tau'],
                                                   device=device, dtype=torch.float32), requires_grad=False)
        theta = torch.autograd.Variable(torch.tensor(net_params['af_params']['theta'],
                                                     device=device, dtype=torch.float32), requires_grad=False)
        ctx.save_for_backward(multiplied_activations, theta, tau, scale)
        return spikes

    @staticmethod
    def backward(ctx, grad_output):
        (membrane_potentials, theta, tau, scale) = ctx.saved_tensors
        # Don't return any gradient for parameters
        return grad_output * SpikeFunc.calculate_pdf(membrane_potentials, theta, tau, scale), None, None, None, None

    @staticmethod
    def apply_weights(activations, weights):
        applied = F.conv3d(activations, weights)
        return applied

    # Input is potentials after matrix multiplication
    @staticmethod
    def calculate_membrane_potentials(potentials, net_params, ref, sigma):
        # Need float32 to do convolution later
        spikes = torch.empty_like(potentials)
        ref_length = len(ref)
        # Iterate over timestamps, NOTE, check if iterating in this dimension is a bottleneck
        for p in range(potentials.shape[-1]):
            ts_pots = potentials[:,:,:,:,p]
            spike_positions = (ts_pots > net_params['af_params']['theta']).to(dtype=torch.float32)
            spike_values = spike_positions / net_params['t_s']
            # Assign output spikes
            spikes[:,:,:,:,p] = spike_values
            num_spikes = torch.sum(spike_positions, dim=(1,2,3))
            have_spike_response = ref * (1 + sigma * (num_spikes - 1))
            no_spike_response = ref * (sigma * num_spikes)
            # Now iterate over neurons, apply refractory response
            for n_id in range(spikes.shape[1]):
                # Make sure our refractory response doesn't overshoot the total time
                resp_length = min(potentials.shape[-1] - p, ref_length)
                if spike_positions[:,n_id,0,0] > 0:
                    # Have spike here
                    potentials[:,n_id,0,0,p:p+resp_length] += have_spike_response[0:resp_length]
                else:
                    # Didn't have a spike
                    potentials[:,n_id,0,0,p:p+resp_length] += no_spike_response[0:resp_length]
            # print(num_spikes)
        return potentials, spikes

    # Call the cuda wrapper, Note! sigma is not implemented
    @staticmethod
    def get_spikes_cuda(potentials, ref, net_params, device=torch.device('cuda'), thetas=[]):
        # thetas = torch.ones([net_params['batch_size'], 10, 1, 1, 1])
        # multiplied_activations_for_sts = potentials.clone().detach()
        theta_shape = torch.tensor(potentials.shape, device=device)

        theta_shape[-1] = 1  # time dimension

        if len(thetas) == 0:
            # thetas = torch.eyes((net_params['batch_size'], net_params['num_classes'], 1, 1,
            #                      350))*net_params['af_params']['theta']

            thetas = torch.zeros(tuple(theta_shape), dtype=torch.float32, device=device) + net_params['af_params'][
                'theta']
        # theta2 = torch.tensor(potentials)
        else:
            thetas = torch.zeros(tuple(theta_shape), dtype=torch.float32, device=device) + thetas

        # return slayer_cuda.get_spikes_cuda(multiplied_activations_for_sts,
        #                                    torch.empty_like(potentials), thetas, ref, net_params['t_s'])
        return SpikeFunc.get_spikes(thetas, potentials, net_params, ref, device=device)

    @staticmethod
    def get_spikes(theta, potentials, net_params, ref, device):
        # Need float32 to do convolution later
        spikes = torch.zeros(potentials.shape, dtype=torch.float, device=device)
        ref_length = len(ref)
        # print('ref length:', len(ref))
        # Iterate over timestamps, NOTE, check if iterating in this dimension is a bottleneck
        sha = torch.tensor(potentials.shape, device=device)
        sha[-1] = 1
        theta_p = theta.reshape(potentials[:, :, :, :, 0].shape)  # (10,10,1,1)
        for p in range(net_params['t_valid']):
            # ts_pots = potentials[:, :, :, :, p].reshape(tuple(sha)) # align with the shape of theta: (10,10,1,1,1)
            ts_pots = potentials[:, :, :, :, p]

            spike_positions = (ts_pots > theta_p).to(dtype=torch.float32)  # binary, (10,10,1,1)
            ind_theta = ts_pots > theta_p
            theta_fired = theta_p[ind_theta]  # (10,10,1,1)
            num_s = (spike_positions).nonzero()

            if len(num_s) > 0:  # have spike emited
                spike_values = spike_positions / net_params['t_s']
                # Assign output spikes
                spikes[:, :, :, :, p] = spike_values

                resp_length = min(potentials.shape[-1] - p - 1, ref_length)

                temp_potential_new = potentials[ind_theta]

                for j in range(temp_potential_new.shape[0]):
                    temp_potential_part = temp_potential_new[j, p + 1:p + resp_length]
                    Ref = ref[0:resp_length - 1] * theta_fired[j] / net_params['af_params']['theta']
                    temp_potential_part = temp_potential_part + Ref
                    temp_potential_new[j, p + 1:p + resp_length] = temp_potential_part

                potentials[ind_theta] = temp_potential_new

        return potentials , spikes

    @staticmethod
    def calculate_pdf(membrane_potentials, theta, tau, scale):
        return scale / tau * torch.exp(-torch.abs(membrane_potentials - theta) / tau)


# 定义各种layer
###########################################################################

# Helper module to emulate fully connected layers in Spiking Networks #
class SpikeLinear(nn.Conv3d):

    def __init__(self, in_features, out_features):
        kernel = (1, in_features, 1)
        super(SpikeLinear, self).__init__(1, out_features, kernel, bias=False)

    # 这里forward函数没有必要，直接使用父类中的forward函数即可
    # def forward(self, input):
    #    out = F.conv3d(input, self.weight, self.bias, self.stride, self.padding,
    #                    self.dilation, self.groups)
    #    return out


# Helper module to emulate conv2d layer in Spiking Networks #
class SpikeConv2d(nn.Conv3d):
    """
    input尺寸为（N, C_in,H,W,T）
        N:batch
        C_in: in_channels
        H: height
        W: width
        T: spike中为时间T
    与之对应，卷积核kernel、padding、stride、dilation的shape也是{H，W，T}的

    attention：
        由于继承了nn.Conv3d类，
        kernel_size、stride、padding、dilation都不能在super前传入为不可迭代对象，
        否则都会被父类__init__()中的_triple（x）函数变成（x，x，x）形式的正3维元组,
        不适用SpikeConv2d
    """

    # out_channels is the number of convolving kernel
    def __init__(self, in_channels, out_channels, kernel_size, stride=None, padding=None, dilation=1, bias=False):
        """
        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param dilation: 膨胀系数
        膨胀后的卷积核尺寸 - 1= 膨胀系数 * (原始卷积核尺寸 - 1)
        https://blog.csdn.net/wangyuxi__/article/details/83003357
        """
        if type(kernel_size) is int:
            kernel = (kernel_size, kernel_size, 1)
        elif type(kernel_size) is tuple and len(kernel_size) is 3:
            if kernel_size[2] != 1:
                print("the third dimension should be 1")
                raise ValueError
            kernel = kernel_size  # the convolving kernel shape is {H,W,T},
        else:
            print("the parameter type of kernel_size is error")
            raise ValueError
        super(SpikeConv2d, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                          kernel_size=kernel, stride=1,
                                          padding=(int((kernel[0] - 1) / 2), int((kernel[1] - 1) / 2), 0),
                                          dilation=1, bias=False)


class SpikePool2d(nn.Conv3d):
    # SpikePool2d的方式，由于SNN的特殊性，此处我们定义了一个完全不同于常规pool的操作，更近似于SpikeConv2d操作
    def __init__(self, in_channels, kernel_size, theta, stride=None):
        """
        input尺寸为(N, C_in,H, W, T)

        :param in_channels: in_channels == out_channels
        :param kernel_size: 池化核尺寸
        :param stride: 池化核步长
        :param theta: spike脉冲点火阈值，我们的SpikePool2d的filter中的weights均为1.1*theta
        """
        if type(kernel_size) is int:
            kernel = (kernel_size, kernel_size, 1)
        elif type(kernel_size) is tuple and len(kernel_size) is 3:
            if kernel_size[2] != 1:
                print("the third dimension should be 1")
                raise ValueError
            kernel = kernel_size  # the maxpooling kernel shape is {T,H,W}
        else:
            print("the parameter type of kernel_size is error")
            raise ValueError

        if stride is None:
            # stride = (kernel_size, kernel_size, 1) #stride = None 时默认stride == kernel_size
            stride = kernel
        elif type(stride) is int:
            stride = (stride, stride, 1)
        elif type(stride) is tuple and len(stride) is 3:
            if stride[2] != 1:
                print("the third dimension should be 1")
                raise ValueError
            stride = stride  # the maxpooling kernel shape is {T,H,W}
        else:
            print("param stride should be int or tuple(3)")
            raise ValueError

        # 调用父类的__init__()，pool不改变channel数
        super(SpikePool2d, self).__init__(in_channels=in_channels, out_channels=in_channels,
                                          kernel_size=kernel, stride=stride, dilation=1,
                                          padding=(0, 0, 0),
                                          bias=False)
        # SpikePool2d的filter权重全部设为1.1theta
        torch.nn.init.constant(self.weight, 1.1 * theta)


class SpikeBatchNorm2d(BatchNorm2d):
    '''
    SpikeBatchNormal的方法
    继承的BatchNorm2d类输入为{N, C, H, W}
    此类要求输入为{N, C, H, W, T}，仅对H,W即2d层面进行batchNormal操作

    父类中_check_input_dim方法验证输入是否为{N, C, H, W}维度，而我们额外加上了T维度，因此需要 \
    在输入时对Input暂时移除T维度，计算SpikeBatchNorm2d之后加上T维度
    '''

    # 重写forward方法,torch中所有layer的最终父类Module中默认了__call__方法都调用的是forward函数
    def forward(self, input):
        # 先将T维度拆掉
        # input = in
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)


class SpikeBatchNorm3d(BatchNorm3d):
    '''
    SpikeBatchNorm3d的方法
    此类要求输入为{N, C, H, W, T},完全和BatchNorm3d所需维度一模一样。
    '''

# snn的kernel、snn计算loss
####################################################################################


class SlayerTrainer(object):

    def __init__(self, net_params, device=torch.device('cuda'), data_type=np.float32):
        self.device = device
        self.net_params = net_params
        # Data type used for membrane potentials, weights
        self.data_type = data_type

    def calculate_srm_kernel(self, num_channels=None):
        if num_channels is None:
            num_channels = self.net_params['input_channels']
        single_kernel = self._calculate_srm_kernel(self.net_params['sr_params']['mult'],
                                                   self.net_params['sr_params']['tau'],
                                                   self.net_params['sr_params']['epsilon'],
                                                   self.net_params['t_end'], self.net_params['t_s'])
        concatenated_srm = self._concatenate_srm_kernel(single_kernel, num_channels)
        return torch.tensor(concatenated_srm).cuda()

    # Generate kernels that will act on a single channel (0 outside of diagonals)
    def _concatenate_srm_kernel(self, kernel, n_channels):
        eye_tensor = np.reshape(np.eye(n_channels, dtype=self.data_type), (n_channels, n_channels, 1, 1, 1))
        return kernel * eye_tensor

    def _calculate_srm_kernel(self, mult, tau, epsilon, t_end, t_s):
        srm_kernel = self._calculate_eps_func(mult, tau, epsilon, t_end, t_s)
        # Make sure the kernel is odd size (for convolution)
        if (len(srm_kernel) % 2) == 0:
            srm_kernel.append(0)
        # Convert to numpy array and reshape in a shape compatible for 3d convolution
        srm_kernel = np.asarray(srm_kernel, dtype=self.data_type)
        # Prepend length-1 zeros to make the convolution filter causal
        prepended_zeros = np.zeros((len(srm_kernel)-1,), dtype=self.data_type)
        srm_kernel = np.flip(np.concatenate((prepended_zeros, srm_kernel)))
        return srm_kernel.reshape((1, 1, len(srm_kernel)))
        # Convert to pytorch tensor

    def _calculate_eps_func(self, mult, tau, epsilon, t_end, t_s):
        eps_func = []
        for t in np.arange(0, t_end, t_s):
            srm_val = mult * t / tau * math.exp(1 - t / tau)
            # Make sure we break after the peak
            if abs(srm_val) < abs(epsilon) and t > tau:
                break
            eps_func.append(srm_val)
        return eps_func

    def calculate_ref_kernel(self):
        ref_kernel = self._calculate_eps_func(self.net_params['ref_params']['mult'],
                                              self.net_params['ref_params']['tau'],
                                              self.net_params['ref_params']['epsilon'],
                                              self.net_params['t_end'], self.net_params['t_s'])
        return torch.tensor(ref_kernel, device=self.device)

    def apply_srm_kernel(self, input_spikes, srm):
        return F.conv3d(input_spikes, srm, padding=(0, 0, int(srm.shape[4]/2))) * self.net_params['t_s']

    def calculate_error_spiketrain(self, a, des_a):
        return a - des_a

    def calculate_l2_loss_spiketrain(self, a, des_a):
        return torch.sum(self.calculate_error_spiketrain(a, des_a) ** 2) / 2 * self.net_params['t_s']

    def calculate_error_classification(self, spikes, des_spikes):
        err = spikes.detach()
        t_valid = self.net_params['t_valid']
        err[:, :, :, :, 0:t_valid] = (torch.sum(spikes[:, :, :, :, 0:t_valid], 4, keepdim=True) -
                                      des_spikes) / (t_valid/ self.net_params['t_s'])
        return err

    def calculate_l2_loss_classification(self, spikes, des_spikes):
        return torch.sum(self.calculate_error_classification(spikes, des_spikes) ** 2) / 2 * self.net_params['t_s']

    def get_accurate_classifications(self, out_spikes, des_labels):
        output_labels = torch.argmax(torch.sum(out_spikes, 4, keepdim=True), 1)
        correct_labels = sum([1 for (classified, gtruth) in zip(output_labels.flatten(),
                                                                des_labels.flatten()) if (classified == int(gtruth))])
        return correct_labels
