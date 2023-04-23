import time
import torch
import torch.nn as nn
import argparse
import csv

parser = argparse.ArgumentParser(description='Latency measure of all models in the search space on the device')
parser.add_argument('--load', default='./logs', type=str, help='path to the search space file')
parser.add_argument('--save', default='./logs', type=str, help='path to save the latency file')
parser.add_argument('--start_line', default=0, type=int, help='the start of line')
parser.add_argument('--end_line', default=5000, type=int, help='the start of line')
parser.add_argument('--cuda', default=1, type=int, help='if use cuda')


# Since device temperature affects latency, --start_line and --end_line can be used to split on-device measurements
# into smaller portions. Note that you will need to manually aggregate all the portions' results into one .csv file.

args = parser.parse_args()


class OneLayerX20(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias, groups=1):
        super(OneLayerX20, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias, groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        for tt in range(20):
            self.layer(x)
        return x


def validate(input_size_v, model_v):
    model_v.eval()

    with torch.no_grad():
        image = torch.rand(1, input_size_v[0], input_size_v[1], input_size_v[2])  # N* C*H*W
        if args.cuda == 1:
            image = image.cuda()
            model_v(image)

            torch.cuda.synchronize(torch.cuda.current_device())
            s = time.time()
            for i in range(20):
                model_v(image)
            torch.cuda.synchronize(torch.cuda.current_device())
            return time.time() - s
        else:
            model_v(image)
            s = time.time()
            for i in range(20):
                model_v(image)
            return time.time() - s


with open(args.load) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = -1
    column_names = ''
    for row in csv_reader:
        # print(row)
        line_count += 1
        if line_count < args.start_line:
            continue
        if line_count >= args.end_line:
            break
        if line_count == 0:
            if 'groups' in row:
                column_names = row + ['latency']
            else:
                column_names = row + ['groups', 'latency']

            with open(args.save, 'w+') as csv_save:
                writer = csv.writer(csv_save)
                writer.writerow(column_names)
        else:
            input_size = int(row[0])
            input_channel = int(row[1])
            output_channel = int(row[2])
            kernel_size_1 = int(row[3])
            kernel_size_2 = int(row[4])
            stride_1 = int(row[5])
            stride_2 = int(row[6])
            padding_1 = int(row[7])
            padding_2 = int(row[8])
            bias = bool(row[9] == 'True')
            if len(row) > 10:
                groups = int(row[10])
            else:
                groups = 1

            model = OneLayerX20(input_channel, output_channel, (kernel_size_1, kernel_size_2), (stride_1, stride_2),
                                (padding_1, padding_2), bias, groups)

            if args.cuda == 1:
                model.cuda()

            t = validate([input_channel, input_size, input_size], model)  # C*H*W
            t = t * 2.5
            del model

            result = [str(i) for i in
                      [input_size, input_channel, output_channel, kernel_size_1, kernel_size_2, stride_1, stride_2,
                       padding_1, padding_2, bias, groups, t]]

            print(result)
            with open(args.save, 'a+') as csv_save:
                writer = csv.writer(csv_save)
                writer.writerow(result)
                time.sleep(0.01)
