import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from slayer_train import SlayerTrainer
from dataloader import SlayerParams
import numpy as np
import os
from args import get_parser

parser = get_parser()
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#  mix the original spike and laplace spike
def smooth(data, lap_data):
    result = args.original_ratio*data + (1-args.original_ratio)*lap_data
    result = result > torch.rand(result.size(), device=device)
    return result


# define approximate firing function
class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(args.thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        # if torch.is_tensor(grad_output):
        #     input, = ctx.saved_tensors
        # else:
        #     input, = ctx.saved_variables

        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - args.thresh) < args.lens
        return grad_input * temp.float()


act_fun = ActFun.apply


# membrane potential update
def mem_update(ops, x, mem, spike):
    mem = mem * args.decay * (1. - spike) + ops(x)
    spike = act_fun(mem)  # act_fun : approximation firing function
    return mem, spike

# 原DDC结构
# # cnn_layer(in_planes, out_planes, stride, padding, kernel_size)
# cfg_cnn = [(3, 64, 4, 2, 11),
#            (64, 192, 1, 2, 5),
#            (192, 384, 1, 1, 3),
#            (384, 256, 1, 1, 3),
#            (256, 256, 1, 1, 3),]
# # kernel size
# cfg_kernel = [56, 27, 13, 13, 13]
# # fc layer
# cfg_fc = [9216, 4096, 256, args.num_classes]


# 裁剪后的DDC结构
# cnn_layer(in_planes, out_planes, stride, padding, kernel_size)
cfg_cnn = [(3, 64, 4, 2, 11),
           (64, 192, 1, 2, 5),
           (192, 256, 1, 1, 3)]
# kernel size
cfg_kernel = [56, 27, 13]
# fc layer
cfg_fc = [9216, 2048, 256, args.num_classes]


class SDDC(nn.Module):
    def __init__(self):
        super(SDDC, self).__init__()
        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[0]
        self.conv0 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[1]
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[2]
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)

        self.classifier_fc1 = nn.Linear(cfg_fc[0], cfg_fc[1])
        self.classifier_fc2 = nn.Linear(cfg_fc[1], cfg_fc[1])

        self.bottleneck_fc = nn.Linear(cfg_fc[1], cfg_fc[2])

        self.final_classifier_fc = nn.Linear(cfg_fc[2], cfg_fc[-1])

        self.net_params = SlayerParams("./parameters.yaml")
        self.trainer = SlayerTrainer(self.net_params, device)
        self.input_srm = self.trainer.calculate_srm_kernel(1)

    def forward(self, source, source_lap, target, target_lap, epoch, batch):
        source_c0_mem = source_c0_spike = Variable(
            torch.zeros(args.base_batch_size, cfg_cnn[0][1], cfg_kernel[0], cfg_kernel[0], device=device))
        source_c1_mem = source_c1_spike = Variable(
            torch.zeros(args.base_batch_size, cfg_cnn[1][1], cfg_kernel[1], cfg_kernel[1], device=device))
        source_c2_mem = source_c2_spike = Variable(
            torch.zeros(args.base_batch_size, cfg_cnn[2][1], cfg_kernel[2], cfg_kernel[2], device=device))

        source_h1_mem = source_h1_spike = source_h1_sumspike = Variable(
            torch.zeros(args.base_batch_size, cfg_fc[1], device=device))
        source_h2_mem = source_h2_spike = source_h2_sumspike = Variable(
            torch.zeros(args.base_batch_size, cfg_fc[1], device=device))
        source_h3_mem = source_h3_spike = source_h3_sumspike = Variable(
            torch.zeros(args.base_batch_size, cfg_fc[2], device=device))
        source_h4_mem = source_h4_spike = source_h4_sumspike = Variable(
            torch.zeros(args.base_batch_size, cfg_fc[-1], device=device))

        target_c0_mem = target_c0_spike = Variable(
            torch.zeros(args.base_batch_size, cfg_cnn[0][1], cfg_kernel[0], cfg_kernel[0], device=device))
        target_c1_mem = target_c1_spike = Variable(
            torch.zeros(args.base_batch_size, cfg_cnn[1][1], cfg_kernel[1], cfg_kernel[1], device=device))
        target_c2_mem = target_c2_spike = Variable(
            torch.zeros(args.base_batch_size, cfg_cnn[2][1], cfg_kernel[2], cfg_kernel[2], device=device))

        target_h1_mem = target_h1_spike = target_h1_sumspike = Variable(
            torch.zeros(args.base_batch_size, cfg_fc[1], device=device))
        target_h2_mem = target_h2_spike = target_h2_sumspike = Variable(
            torch.zeros(args.base_batch_size, cfg_fc[1], device=device))
        target_h3_mem = target_h3_spike = target_h3_sumspike = Variable(
            torch.zeros(args.base_batch_size, cfg_fc[2], device=device))

        source_psp = []
        target_psp = []

        for step in range(args.time_window):  # simulation time steps
            """
            source feature
            """
            source_lap_x = source_lap > torch.ones(source_lap.size(), device=device) * args.lap_threshold
            source_x = source < torch.rand(source.size(), device=device)  # prob. firing
            if args.is_mix == 1:
                source_x = smooth(source_x.float(), source_lap_x.float())
            # out_source = source_x.clone()

            # conv2d(3, 64, 11, 4, 2) + act_fun + avg_poo2d(3, 2)
            # print(x.shape)
            source_c0_mem, source_c0_spike = mem_update(self.conv0, source_x.float(),
                                                        source_c0_mem, source_c0_spike)
            source_x = F.avg_pool2d(source_c0_spike, kernel_size=3, stride=2)
            # print(x.shape)
            # conv2d(64, 192, 5, None, 2) + act_fun + avg_poo2d(3, 2)
            source_c1_mem, source_c1_spike = mem_update(self.conv1, source_x.float(),
                                                        source_c1_mem, source_c1_spike)
            source_x = F.avg_pool2d(source_c1_spike, kernel_size=3, stride=2)
            # print(x.shape)
            # conv2d(192, 256, 3, None, 1) + act_fun
            source_c2_mem, source_c2_spike = mem_update(self.conv2, source_x.float(),
                                                        source_c2_mem, source_c2_spike)
            # print(c2_spike.shape)

            source_x = F.avg_pool2d(source_c2_spike, kernel_size=3, stride=2)
            # print(x.shape)
            source_x = source_x.view(args.base_batch_size, -1)

            # print(x.shape)
            source_h1_mem, source_h1_spike = mem_update(self.classifier_fc1, source_x,
                                                        source_h1_mem, source_h1_spike)
            source_h1_sumspike += source_h1_spike
            source_h2_mem, source_h2_spike = mem_update(self.classifier_fc2, source_h1_spike,
                                                        source_h2_mem, source_h2_spike)
            source_h2_sumspike += source_h2_spike
            source_h3_mem, source_h3_spike = mem_update(self.bottleneck_fc, source_h2_spike,
                                                        source_h3_mem, source_h3_spike)
            source_h3_sumspike += source_h3_spike
            source_psp.append(source_h3_spike.view(args.base_batch_size, 256, 1))

            source_h4_mem, source_h4_spike = mem_update(self.final_classifier_fc, source_h3_spike,
                                                        source_h4_mem, source_h4_spike)
            source_h4_sumspike += source_h4_spike

            """
            target feature
            """
            if self.training:
                target_lap_x = target_lap > torch.ones(target_lap.size(), device=device) * args.lap_threshold
                target_x = target < torch.rand(target.size(), device=device)  # prob. firing
                if args.is_mix == 1:
                    target_x = smooth(target_x.float(), target_lap_x.float())
                # out_target = target_x.clone()

                # conv2d(3, 64, 11, 4, 2) + act_fun + avg_poo2d(3, 2)
                # print(x.shape)
                target_c0_mem, target_c0_spike = mem_update(self.conv0, target_x.float(),
                                                            target_c0_mem, target_c0_spike)
                target_x = F.avg_pool2d(target_c0_spike, kernel_size=3, stride=2)
                # print(x.shape)
                # conv2d(64, 192, 5, None, 2) + act_fun + avg_poo2d(3, 2)
                target_c1_mem, target_c1_spike = mem_update(self.conv1, target_x.float(),
                                                            target_c1_mem, target_c1_spike)
                target_x = F.avg_pool2d(target_c1_spike, kernel_size=3, stride=2)
                # print(x.shape)
                # conv2d(192, 256, 3, None, 1) + act_fun
                target_c2_mem, target_c2_spike = mem_update(self.conv2, target_x.float(),
                                                            target_c2_mem, target_c2_spike)
                # print(c2_spike.shape)

                target_x = F.avg_pool2d(target_c2_spike, kernel_size=3, stride=2)
                # print(x.shape)
                target_x = target_x.view(args.base_batch_size, -1)

                # print(x.shape)
                target_h1_mem, target_h1_spike = mem_update(self.classifier_fc1, target_x,
                                                            target_h1_mem, target_h1_spike)
                target_h1_sumspike += target_h1_spike
                target_h2_mem, target_h2_spike = mem_update(self.classifier_fc2, target_h1_spike,
                                                            target_h2_mem, target_h2_spike)
                target_h2_sumspike += target_h2_spike
                target_h3_mem, target_h3_spike = mem_update(self.bottleneck_fc, target_h2_spike,
                                                            target_h3_mem, target_h3_spike)
                target_h3_sumspike += target_h3_spike
                target_psp.append(target_h3_spike.view(args.base_batch_size, 256, 1))

        result = source_h4_sumspike / args.time_window

        if self.training and args.psp == 1:
            source_psp = torch.cat(source_psp, dim=-1)
            source_psp = self.trainer.apply_srm_kernel(source_psp.view(args.base_batch_size, 1, 1, 256, -1),
                                                       self.input_srm)
            source_psp = source_psp.view(args.base_batch_size, -1)
            target_psp = torch.cat(target_psp, dim=-1)
            target_psp = self.trainer.apply_srm_kernel(target_psp.view(args.base_batch_size, 1, 1, 256, -1),
                                                       self.input_srm)
            target_psp = target_psp.view(args.base_batch_size, -1)
            return result, source_psp, target_psp

        return result, source_h3_sumspike, target_h3_sumspike



# class SDDC(nn.Module):
#     def __init__(self):
#         super(SDDC, self).__init__()
#         in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[0]
#         self.conv0 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
#         in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[1]
#         self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
#         in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[2]
#         self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
#         in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[3]
#         self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
#         in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[4]
#         self.conv4 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
#
#         self.classifier_fc1 = nn.Linear(cfg_fc[0], cfg_fc[1])
#         self.classifier_fc2 = nn.Linear(cfg_fc[1], cfg_fc[1])
#
#         self.bottleneck_fc = nn.Linear(cfg_fc[1], cfg_fc[2])
#
#         self.final_classifier_fc = nn.Linear(cfg_fc[2], cfg_fc[-1])
#
#         self.net_params = SlayerParams("./parameters.yaml")
#         self.trainer = SlayerTrainer(self.net_params, device)
#         self.input_srm = self.trainer.calculate_srm_kernel(1)
#
#     def forward(self, source, source_lap, target, target_lap, epoch, batch):
#         source_c0_mem = source_c0_spike = Variable(
#             torch.zeros(args.base_batch_size, cfg_cnn[0][1], cfg_kernel[0], cfg_kernel[0], device=device))
#         source_c1_mem = source_c1_spike = Variable(
#             torch.zeros(args.base_batch_size, cfg_cnn[1][1], cfg_kernel[1], cfg_kernel[1], device=device))
#         source_c2_mem = source_c2_spike = Variable(
#             torch.zeros(args.base_batch_size, cfg_cnn[2][1], cfg_kernel[2], cfg_kernel[2], device=device))
#         source_c3_mem = source_c3_spike = Variable(
#             torch.zeros(args.base_batch_size, cfg_cnn[3][1], cfg_kernel[3], cfg_kernel[3], device=device))
#         source_c4_mem = source_c4_spike = Variable(
#             torch.zeros(args.base_batch_size, cfg_cnn[4][1], cfg_kernel[4], cfg_kernel[4], device=device))
#
#         source_h1_mem = source_h1_spike = source_h1_sumspike = Variable(
#             torch.zeros(args.base_batch_size, cfg_fc[1], device=device))
#         source_h2_mem = source_h2_spike = source_h2_sumspike = Variable(
#             torch.zeros(args.base_batch_size, cfg_fc[1], device=device))
#         source_h3_mem = source_h3_spike = source_h3_sumspike = Variable(
#             torch.zeros(args.base_batch_size, cfg_fc[2], device=device))
#         source_h4_mem = source_h4_spike = source_h4_sumspike = Variable(
#             torch.zeros(args.base_batch_size, cfg_fc[-1], device=device))
#
#         target_c0_mem = target_c0_spike = Variable(
#             torch.zeros(args.base_batch_size, cfg_cnn[0][1], cfg_kernel[0], cfg_kernel[0], device=device))
#         target_c1_mem = target_c1_spike = Variable(
#             torch.zeros(args.base_batch_size, cfg_cnn[1][1], cfg_kernel[1], cfg_kernel[1], device=device))
#         target_c2_mem = target_c2_spike = Variable(
#             torch.zeros(args.base_batch_size, cfg_cnn[2][1], cfg_kernel[2], cfg_kernel[2], device=device))
#         target_c3_mem = target_c3_spike = Variable(
#             torch.zeros(args.base_batch_size, cfg_cnn[3][1], cfg_kernel[3], cfg_kernel[3], device=device))
#         target_c4_mem = target_c4_spike = Variable(
#             torch.zeros(args.base_batch_size, cfg_cnn[4][1], cfg_kernel[4], cfg_kernel[4], device=device))
#
#         target_h1_mem = target_h1_spike = target_h1_sumspike = Variable(
#             torch.zeros(args.base_batch_size, cfg_fc[1], device=device))
#         target_h2_mem = target_h2_spike = target_h2_sumspike = Variable(
#             torch.zeros(args.base_batch_size, cfg_fc[1], device=device))
#         target_h3_mem = target_h3_spike = target_h3_sumspike = Variable(
#             torch.zeros(args.base_batch_size, cfg_fc[2], device=device))
#
#         source_psp = []
#         target_psp = []
#
#         for step in range(args.time_window):  # simulation time steps
#             """
#             source feature
#             """
#             source_lap_x = source_lap > torch.ones(source_lap.size(), device=device) * args.lap_threshold
#             source_x = source < torch.rand(source.size(), device=device)  # prob. firing
#             if args.is_mix == 1:
#                 source_x = smooth(source_x.float(), source_lap_x.float())
#             out_source = source_x.clone()
#
#             # conv2d(3, 64, 11, 4, 2) + act_fun + avg_poo2d(3, 2)
#             # print(x.shape)
#             source_c0_mem, source_c0_spike = mem_update(self.conv0, source_x.float(),
#                                                         source_c0_mem, source_c0_spike)
#             source_x = F.avg_pool2d(source_c0_spike, kernel_size=3, stride=2)
#             # print(x.shape)
#             # conv2d(64, 192, 5, None, 2) + act_fun + avg_poo2d(3, 2)
#             source_c1_mem, source_c1_spike = mem_update(self.conv1, source_x.float(),
#                                                         source_c1_mem, source_c1_spike)
#             source_x = F.avg_pool2d(source_c1_spike, kernel_size=3, stride=2)
#             # print(x.shape)
#             # conv2d(192, 384, 3, None, 1) + act_fun
#             source_c2_mem, source_c2_spike = mem_update(self.conv2, source_x.float(),
#                                                         source_c2_mem, source_c2_spike)
#             # print(c2_spike.shape)
#             # conv2d(384, 256, 3, None, 1) + act_fun
#             source_c3_mem, source_c3_spike = mem_update(self.conv3, source_c2_spike,
#                                                         source_c3_mem, source_c3_spike)
#             # print(c3_spike.shape)
#             # conv2d(256, 256, 3, None, 1) + act_fun + avg_poo2d(3, 2)
#             source_c4_mem, source_c4_spike = mem_update(self.conv4, source_c3_spike,
#                                                         source_c4_mem, source_c4_spike)
#             source_x = F.avg_pool2d(source_c4_spike, kernel_size=3, stride=2)
#             # print(x.shape)
#             source_x = source_x.view(args.base_batch_size, -1)
#
#             # print(x.shape)
#             source_h1_mem, source_h1_spike = mem_update(self.classifier_fc1, source_x,
#                                                         source_h1_mem, source_h1_spike)
#             source_h1_sumspike += source_h1_spike
#             source_h2_mem, source_h2_spike = mem_update(self.classifier_fc2, source_h1_spike,
#                                                         source_h2_mem, source_h2_spike)
#             source_h2_sumspike += source_h2_spike
#             source_h3_mem, source_h3_spike = mem_update(self.bottleneck_fc, source_h2_spike,
#                                                         source_h3_mem, source_h3_spike)
#             source_h3_sumspike += source_h3_spike
#             source_psp.append(source_h3_spike.view(args.base_batch_size, 256, 1))
#
#             """
#             target feature
#             """
#             if self.training:
#                 target_lap_x = target_lap > torch.ones(target_lap.size(), device=device) * args.lap_threshold
#                 target_x = target < torch.rand(target.size(), device=device)  # prob. firing
#                 if args.is_mix == 1:
#                     target_x = smooth(target_x.float(), target_lap_x.float())
#                 out_target = target_x.clone()
#
#                 # conv2d(3, 64, 11, 4, 2) + act_fun + avg_poo2d(3, 2)
#                 # print(x.shape)
#                 target_c0_mem, target_c0_spike = mem_update(self.conv0, target_x.float(),
#                                                             target_c0_mem, target_c0_spike)
#                 target_x = F.avg_pool2d(target_c0_spike, kernel_size=3, stride=2)
#                 # print(x.shape)
#                 # conv2d(64, 192, 5, None, 2) + act_fun + avg_poo2d(3, 2)
#                 target_c1_mem, target_c1_spike = mem_update(self.conv1, target_x.float(),
#                                                             target_c1_mem, target_c1_spike)
#                 target_x = F.avg_pool2d(target_c1_spike, kernel_size=3, stride=2)
#                 # print(x.shape)
#                 # conv2d(192, 384, 3, None, 1) + act_fun
#                 target_c2_mem, target_c2_spike = mem_update(self.conv2, target_x.float(),
#                                                             target_c2_mem, target_c2_spike)
#                 # print(c2_spike.shape)
#                 # conv2d(384, 256, 3, None, 1) + act_fun
#                 target_c3_mem, target_c3_spike = mem_update(self.conv3, target_c2_spike,
#                                                             target_c3_mem, target_c3_spike)
#                 # print(c3_spike.shape)
#                 # conv2d(256, 256, 3, None, 1) + act_fun + avg_poo2d(3, 2)
#                 target_c4_mem, target_c4_spike = mem_update(self.conv4, target_c3_spike,
#                                                             target_c4_mem, target_c4_spike)
#                 target_x = F.avg_pool2d(target_c4_spike, kernel_size=3, stride=2)
#                 # print(x.shape)
#                 target_x = target_x.view(args.base_batch_size, -1)
#
#                 # print(x.shape)
#                 target_h1_mem, target_h1_spike = mem_update(self.classifier_fc1, target_x,
#                                                             target_h1_mem, target_h1_spike)
#                 target_h1_sumspike += target_h1_spike
#                 target_h2_mem, target_h2_spike = mem_update(self.classifier_fc2, target_h1_spike,
#                                                             target_h2_mem, target_h2_spike)
#                 target_h2_sumspike += target_h2_spike
#                 target_h3_mem, target_h3_spike = mem_update(self.bottleneck_fc, target_h2_spike,
#                                                             target_h3_mem, target_h3_spike)
#                 target_h3_sumspike += target_h3_spike
#                 target_psp.append(target_h3_spike.view(args.base_batch_size, 256, 1))
#
#             source_h4_mem, source_h4_spike = mem_update(self.final_classifier_fc, source_h3_spike,
#                                                         source_h4_mem, source_h4_spike)
#             source_h4_sumspike += source_h4_spike
#
#         result = source_h4_sumspike / args.time_window
#         if self.training and args.psp == 1:
#             source_psp = torch.cat(source_psp, dim=-1)
#             source_psp = self.trainer.apply_srm_kernel(source_psp.view(args.base_batch_size, 1, 1, 256, -1),
#                                                        self.input_srm)
#             source_psp = source_psp.view(args.base_batch_size, -1)
#             target_psp = torch.cat(target_psp, dim=-1)
#             target_psp = self.trainer.apply_srm_kernel(target_psp.view(args.base_batch_size, 1, 1, 256, -1),
#                                                        self.input_srm)
#             target_psp = target_psp.view(args.base_batch_size, -1)
#             return result, source_psp, target_psp
#         return result, source_h3_sumspike, target_h3_sumspike
#
#
# class SCNN(nn.Module):
#     def __init__(self):
#         super(SCNN, self).__init__()
#         in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[0]
#         self.conv0 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
#         in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[1]
#         self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
#         in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[2]
#         self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
#         in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[3]
#         self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
#         in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[4]
#         self.conv4 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
#
#         self.classifier_fc1 = nn.Linear(cfg_fc[0], cfg_fc[1])
#         self.classifier_fc2 = nn.Linear(cfg_fc[1], cfg_fc[1])
#
#         self.bottleneck_fc = nn.Linear(cfg_fc[1], cfg_fc[2])
#
#         self.final_classifier_fc = nn.Linear(cfg_fc[2], cfg_fc[-1])
#
#     def forward(self, input, time_window = 20):
#         c0_mem = c0_spike = torch.zeros(args.base_batch_size, cfg_cnn[0][1], cfg_kernel[0], cfg_kernel[0], device=device)
#         c1_mem = c1_spike = torch.zeros(args.base_batch_size, cfg_cnn[1][1], cfg_kernel[1], cfg_kernel[1], device=device)
#         c2_mem = c2_spike = torch.zeros(args.base_batch_size, cfg_cnn[2][1], cfg_kernel[2], cfg_kernel[2], device=device)
#         c3_mem = c3_spike = torch.zeros(args.base_batch_size, cfg_cnn[3][1], cfg_kernel[3], cfg_kernel[3], device=device)
#         c4_mem = c4_spike = torch.zeros(args.base_batch_size, cfg_cnn[4][1], cfg_kernel[4], cfg_kernel[4], device=device)
#
#         h1_mem = h1_spike = h1_sumspike = torch.zeros(args.base_batch_size, cfg_fc[1], device=device)
#         h2_mem = h2_spike = h2_sumspike = torch.zeros(args.base_batch_size, cfg_fc[1], device=device)
#         h3_mem = h3_spike = h3_sumspike = torch.zeros(args.base_batch_size, cfg_fc[2], device=device)
#         h4_mem = h4_spike = h4_sumspike = torch.zeros(args.base_batch_size, cfg_fc[-1], device=device)
#
#         for step in range(args.time_window): # simulation time steps
#             # print(step)
#             """
#             source feature
#             """
#             x = input > torch.rand(input.size(), device=device)  # prob. firing
#
#             # conv2d(3, 64, 11, 4, 2) + act_fun + avg_poo2d(3, 2)
#             # print(x.shape)
#             c0_mem, c0_spike = mem_update(self.conv0, x. float(), c0_mem, c0_spike)
#             x = F.avg_pool2d(c0_spike, kernel_size=3, stride=2)
#             # print(x.shape)
#             # conv2d(64, 192, 5, None, 2) + act_fun + avg_poo2d(3, 2)
#             c1_mem, c1_spike = mem_update(self.conv1, x.float(), c1_mem, c1_spike)
#             x = F.avg_pool2d(c1_spike, kernel_size=3, stride=2)
#             # print(x.shape)
#             # conv2d(192, 384, 3, None, 1) + act_fun
#             c2_mem, c2_spike = mem_update(self.conv2, x.float(), c2_mem, c2_spike)
#             # print(c2_spike.shape)
#             # conv2d(384, 256, 3, None, 1) + act_fun
#             c3_mem, c3_spike = mem_update(self.conv3, c2_spike, c3_mem, c3_spike)
#             # print(c3_spike.shape)
#             # conv2d(256, 256, 3, None, 1) + act_fun + avg_poo2d(3, 2)
#             c4_mem, c4_spike = mem_update(self.conv4, c3_spike, c4_mem, c4_spike)
#             x = F.avg_pool2d(c4_spike, kernel_size=3, stride=2)
#             # print(x.shape)
#             x = x.view(args.base_batch_size, -1)
#             # print(x.shape)
#             h1_mem, h1_spike = mem_update(self.classifier_fc1, x, h1_mem, h1_spike)
#             h1_sumspike += h1_spike
#             h2_mem, h2_spike = mem_update(self.classifier_fc2, h1_spike, h2_mem, h2_spike)
#             h2_sumspike += h2_spike
#             h3_mem, h3_spike = mem_update(self.bottleneck_fc, h2_spike, h3_mem, h3_spike)
#             h3_sumspike += h3_spike
#             h4_mem, h4_spike = mem_update(self.final_classifier_fc, h3_spike, h4_mem, h4_spike)
#             h4_sumspike += h4_spike
#
#         print(h4_sumspike.grad)
#
#         outputs = h4_sumspike / args.time_window
#         return outputs


class OldSCNN(nn.Module):
    def __init__(self):
        super(OldSCNN, self).__init__()
        in_planes, out_planes, stride, padding, kernel_size = (1, 32, 1, 1, 3)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        in_planes, out_planes, stride, padding, kernel_size = (32, 32, 1, 1, 3)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)

        self.fc1 = nn.Linear(7 * 7 * 32, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, input):
        c1_mem = c1_spike = torch.zeros(args.base_batch_size, 1, 28, 28, device=device)
        c2_mem = c2_spike = torch.zeros(args.base_batch_size, 32, 14, 14, device=device)

        h1_mem = h1_spike = h1_sumspike = torch.zeros(args.base_batch_size, 128, device=device)
        h2_mem = h2_spike = h2_sumspike = torch.zeros(args.base_batch_size, 10, device=device)

        for step in range(args.time_window): # simulation time steps
            x = input > torch.rand(input.size(), device=device) # prob. firing

            c1_mem, c1_spike = mem_update(self.conv1, x.float(), c1_mem, c1_spike)

            x = F.avg_pool2d(c1_spike, 2)

            c2_mem, c2_spike = mem_update(self.conv2,x, c2_mem,c2_spike)

            x = F.avg_pool2d(c2_spike, 2)
            x = x.view(args.base_batch_size, -1)

            h1_mem, h1_spike = mem_update(self.fc1, x, h1_mem, h1_spike)
            h1_sumspike += h1_spike
            h2_mem, h2_spike = mem_update(self.fc2, h1_spike, h2_mem,h2_spike)
            h2_sumspike += h2_spike

        outputs = h2_sumspike / args.time_window
        return outputs

# if args.mmd_level == 8:
#     # mmd_loss = torch.nn.KLDivLoss()(source_h3_sumspike.detach(), target_h3_sumspike.detach())
#     mmd_loss = CKA.linear_CKA(source_h3_sumspike, target_h3_sumspike)
# elif args.mmd_level == 7:
#     # mmd_loss = torch.nn.KLDivLoss()(source_h2_sumspike.detach(), target_h2_sumspike.detach())
#     mmd_loss = CKA.linear_CKA(source_h2_sumspike.detach(), target_h2_sumspike.detach())
# elif args.mmd_level == 6:
#     # mmd_loss = torch.nn.KLDivLoss()(source_h1_sumspike.detach(), target_h1_sumspike.detach())
#     mmd_loss = CKA.linear_CKA(source_h1_sumspike.detach(), target_h1_sumspike.detach())
# mmd_loss = torch.abs(mmd_loss)
# mmd_loss = mmd.mmd_linear(source_h3_sumspike, target_h3_sumspike)

# if self.training and batch % 20 == 0:
#     perfix = "/home/zsc/spiking/BP-for-SpikingNN-master/feature/" + str(epoch) + "/"
#     out_source_np = out_source.data.cpu().numpy()
#     if not os.path.exists(perfix + "source/out/"):
#         os.mkdir(perfix + "source/out/")
#     np.save(perfix + "source/out/" + str(batch), out_source_np)
#     source_c0_np = source_c0_spike.data.cpu().numpy()
#     np.save(perfix + "source/c0/" + str(batch), source_c0_np)
#     source_c1_np = source_c1_spike.data.cpu().numpy()
#     np.save(perfix + "source/c1/" + str(batch), source_c1_np)
#     source_c2_np = source_c2_spike.data.cpu().numpy()
#     np.save(perfix + "source/c2/" + str(batch), source_c2_np)
#     source_c3_np = source_c3_spike.data.cpu().numpy()
#     np.save(perfix + "source/c3/" + str(batch), source_c3_np)
#     source_c4_np = source_c4_spike.data.cpu().numpy()
#     np.save(perfix + "source/c4/" + str(batch), source_c4_np)
#
#     if not os.path.exists(perfix + "target/out/"):
#         os.mkdir(perfix + "target/out/")
#     out_target_np = out_target.data.cpu().numpy()
#     np.save(perfix + "target/out/" + str(batch), out_target_np)
#     target_c0_np = target_c0_spike.data.cpu().numpy()
#     np.save(perfix + "target/c0/" + str(batch), target_c0_np)
#     target_c1_np = target_c1_spike.data.cpu().numpy()
#     np.save(perfix + "target/c1/" + str(batch), target_c1_np)
#     target_c2_np = target_c2_spike.data.cpu().numpy()
#     np.save(perfix + "target/c2/" + str(batch), target_c2_np)
#     target_c3_np = target_c3_spike.data.cpu().numpy()
#     np.save(perfix + "target/c3/" + str(batch), target_c3_np)
#     target_c4_np = target_c4_spike.data.cpu().numpy()
#     np.save(perfix + "target/c4/" + str(batch), target_c4_np)