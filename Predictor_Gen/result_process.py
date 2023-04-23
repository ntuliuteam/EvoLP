import argparse
import csv

parser = argparse.ArgumentParser(description='Search Space of A PyTorch Model for Latency Predictor')
parser.add_argument('--load', default='./logs', type=str, help='path to the saved result file')
args = parser.parse_args()

result_process = {}
with open(args.load) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = -1
    for row in csv_reader:
        line_count += 1
        if line_count == 0:
            column_names = row
        else:
            input_size = row[0]
            input_channel = row[1]
            output_channel = row[2]
            kernel_size_1 = row[3]
            kernel_size_2 = row[4]
            stride_1 = row[5]
            stride_2 = row[6]
            padding_1 = row[7]
            padding_2 = row[8]
            bias = row[9]
            groups = row[10]
            latency = row[11]
            if int(groups) == 1:
                key = '_'.join([input_size, kernel_size_1, kernel_size_2, stride_1, stride_2, padding_1, padding_2, bias, groups])
            elif int(groups) == int(input_channel):
                key = '_'.join(
                    [input_size, kernel_size_1, kernel_size_2, stride_1, stride_2, padding_1, padding_2, bias, 'g'])
            else:
                exit('no void groups')
            if key in result_process.keys():
                result_process[key].append('_'.join([input_channel, output_channel, latency]))
            else:
                result_process[key] = ['_'.join([input_channel, output_channel, latency])]

for key in result_process.keys():
    with open(args.load[:-4] + '_'+key + '_final.csv', 'w+') as csv_file:
        writer = csv.writer(csv_file)
        for j in result_process[key]:
            writer.writerow(j.split('_'))

        if len(result_process[key]) < 2:
            print(key, "\t only includes one data, copy is applied")
            for j in result_process[key]:
                writer.writerow(j.split('_'))
