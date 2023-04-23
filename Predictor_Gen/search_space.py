import sys

sys.path.append('../')

import Models as models
import argparse
from torchsummary import summary
import math
import csv
from get_information import get_conv_info

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
print(model_names)
parser = argparse.ArgumentParser(description='Search Space of A PyTorch Model for Latency Predictor')
parser.add_argument('--arch', default='test', type=str, help='architecture to use')
parser.add_argument('--input_size', type=int, default=224, help='The size of the input')
parser.add_argument('--sample', type=int, default=5000, help='The number of samples')
parser.add_argument('--save', default='test.csv', type=str, help='path to save the search space file')

args = parser.parse_args()

model = models.__dict__[args.arch]()

conv_info = get_conv_info(model, args.input_size)
conv_range = list(set(conv_info))

range_num = []
for conv_tmp in conv_range:
    tmp = conv_tmp.split(',')
    range_num.append(int(tmp[2]) * int(tmp[3]))  # input channel = 3, 3 times as it is important

total_range_num = sum(range_num)

with open(args.save, 'w+') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(
        ["input_size", "input_channel", "output_channel", "kernel_size_1", "kernel_size_2", "stride_1", "stride_2",
         "padding_1", "padding_2", "bias", "groups"])
    for i in range(len(conv_range)):
        tmp = conv_range[i].replace(']', '').replace(' ', '').split(',')
        current_range_num = 1.0 * range_num[i] / total_range_num
        current_num = current_range_num * args.sample

        t1 = int(tmp[2])
        t2 = int(tmp[3])

        groups = int(tmp[-1])
        if 'mobilenet' not in args.arch:
            assert groups == 1

        c1 = round(math.sqrt(current_num * t1 / t2))
        c2 = round(math.sqrt(current_num * t2 / t1))

        if c1 < 1:
            c1 = 1
        if c2 < 1:
            c2 = 1

        for k in range(t1, 0, -int(t1 / c1)):
            for j in range(t2, 0, -int(t2 / c2)):
                # print(tmp)
                if 'mobilenet' not in args.arch:
                    writer.writerow([tmp[1], str(k), str(j)] + tmp[4:])
                else:
                    if groups != 1:
                        if k != j:
                            continue # less than samples
                        else:
                            g = k
                    else:
                        g = 1
                    writer.writerow([tmp[1], str(k), str(j)] + tmp[4:-1] + [str(g)])
