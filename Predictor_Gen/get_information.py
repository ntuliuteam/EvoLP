from torchsummary import summary
import math
import torch
import torch.nn as nn
from thop import profile


def get_conv_info(model_in, input_size):
    # Linear operation is not used! Just for an example to show how to add other operations

    support_operations = ['Conv2d', 'Linear']
    model_arch1 = []
    for m in model_in.modules():
        for i in support_operations:
            if i == m.__str__()[0:len(i)]:
                model_arch1.append({'op': i, 'para': m.__str__()[len(i) + 1:-1]})
    tmp_a = summary(model_in, (3, input_size, input_size), col_names=("input_size", "output_size", "kernel_size"),
                    verbose=0, depth=10)
    model_arch2 = []

    for line in tmp_a.__str__().split('\n'):
        tmp1 = line.replace('|', '').replace('└─', '').replace(' ', '').replace('├', '').replace('─', '')
        for i in support_operations:
            if i == tmp1[0:len(i)]:
                tmp2 = tmp1[len(i) + 1:-1].replace('[', ' ').replace(']', '').split()
                model_arch2.append({'op': i, 'in_size': tmp2[1], 'out_size': tmp2[2], 'kernel': tmp2[3]})

    assert len(model_arch1) == len(model_arch2)

    conv_range_1 = []
    for i in range(len(model_arch2)):
        arch1 = model_arch1[i]
        arch2 = model_arch2[i]

        assert arch1['op'] == arch2['op']

        if arch1['op'] == 'Conv2d':
            output = ['Conv2d']
            para = arch1['para'].replace(')', '').replace('(', '').replace(' ', '').replace('=', ',').split(',')
            input_size = int(arch2['in_size'].split(',')[2])
            output.append(input_size)
            in_channel = int(para[0])
            assert in_channel == int(arch2['in_size'].split(',')[1])
            output.append(in_channel)
            out_channel = int(para[1])
            assert out_channel == int(arch2['out_size'].split(',')[1])
            output.append(out_channel)

            t = para.index('kernel_size')
            kernel_size_1 = int(para[t + 1])
            kernel_size_2 = int(para[t + 2])
            assert kernel_size_1 == int(arch2['kernel'].split(',')[2])
            assert kernel_size_2 == int(arch2['kernel'].split(',')[3])

            output.append(kernel_size_1)
            output.append(kernel_size_2)
            if 'stride' in para:
                t = para.index('stride')
                stride_1 = int(para[t + 1])
                stride_2 = int(para[t + 2])
            else:
                stride_1 = stride_2 = 1
            output.append(stride_1)
            output.append(stride_2)
            if 'padding' in para:
                t = para.index('padding')
                padding_1 = int(para[t + 1])
                padding_2 = int(para[t + 2])
            else:
                padding_1 = padding_2 = 0
            output.append(padding_1)
            output.append(padding_2)
            if 'bias' in para:
                t = para.index('bias')
                bias = para[t + 1]
            else:
                bias = True
            output.append(bias)

            if 'groups' in para:
                t = para.index('groups')
                group = para[t + 1]
            else:
                group = 1
            output.append(group)

            output = output.__str__().replace('\'', '')
            conv_range_1.append(output)

        if arch1['op'] == 'Linear':
            output = ['Linear']
            para = arch1['para'].replace(')', '').replace('(', '').replace(' ', '').replace('=', ',').split(',')
            t = para.index('in_features')
            in_features = para[t + 1]
            assert in_features == arch2['in_size'].split(',')[1]
            output.append(in_features)
            t = para.index('out_features')
            out_features = para[t + 1]
            output.append(out_features)
            assert out_features == arch2['out_size'].split(',')[1]
            if 'bias' in para:
                t = para.index('bias')
                bias = para[t + 1]
            else:
                bias = None
            output.append(bias)
    return conv_range_1


def get_model_information(model, input_size, input_channel, padding_1, padding_2, kernel_size_1, kernel_size_2,
                          stride_1, stride_2, output_channel):
    model.eval()

    with torch.no_grad():
        image = torch.rand(1, input_channel, input_size, input_size)  # N* C*H*W
        flops, params = profile(model, inputs=(image,))

        allins = input_size * input_size * input_channel
        output_size_1 = math.floor((input_size + 2 * padding_1 - kernel_size_1) / stride_1 + 1)
        output_size_2 = math.floor((input_size + 2 * padding_2 - kernel_size_2) / stride_2 + 1)
        allouts = output_size_2 * output_size_1 * output_channel

        return flops, params, allins, allouts
