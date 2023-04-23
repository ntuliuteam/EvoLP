import math
import scipy.io


def clean(input_array):
    output_array = []
    for i in input_array:
        for j in i:
            if min(j.shape) > 0:
                output_array.append(j)
    return output_array


def load_mat(path_to_mat):
    mat = scipy.io.loadmat(path_to_mat)

    mint = mat['mint']
    maxt = mat['maxt']
    minp = mat['minp']
    maxp = mat['maxp']

    net = mat['net']
    net = list(net)

    net_IW = net[0][0][35]
    net_IW = clean(net_IW)

    net_LW = net[0][0][36]
    net_LW = clean(net_LW)

    net_b = net[0][0][37]

    net_b = clean(net_b)

    return mint, maxt, minp, maxp, net_IW, net_LW, net_b


def tansig(n):
    try:
        tmp = n[0]
    except IndexError:
        tmp = n

    try:
        return 2 / (1 + math.exp(-2 * tmp)) - 1

    except OverflowError:
        if (-2) * tmp > 0:
            return -1
        else:
            return 1


class net_sim:

    def __init__(self, mat_path):
        self.mat_path = mat_path
        self.mint, self.maxt, self.minp, self.maxp, self.net_IW, self.net_LW, self.net_b = load_mat(self.mat_path)
    def sim(self, input_data):

        # normalization
        input_nor = []
        for i in range(len(input_data)):

            if self.maxp[i][0] == self.minp[i][0]:
                input_nor.append(1.0*input_data[i])
            else:
                tmp = 1.0 * (input_data[i] - self.minp[i][0]) / (
                        self.maxp[i][0] - self.minp[i][0]) * 2 - 1
                input_nor.append(tmp)

        # input layer
        IW = self.net_IW[0]
        IW_b = self.net_b[0]

        output = []
        for i in range(len(IW)):
            tmp = 0
            for j in range(len(IW[0])):
                tmp = tmp + input_nor[j] * IW[i][j]
            tmp = tmp + IW_b[i]
            output.append(tansig(tmp))

        # hidden layer(s)
        layer_num = len(self.net_LW)
        for layer in range(layer_num):
            output_tmp = []
            for i in range(len(self.net_LW[layer])):
                tmp = 0
                for j in range(len(self.net_LW[layer][0])):
                    tmp = tmp + output[j] * self.net_LW[layer][i][j]
                tmp = tmp + self.net_b[layer + 1][i]

                if layer == layer_num - 1:
                    output_tmp.append(tmp)
                else:
                    output_tmp.append(tansig(tmp))
            output = output_tmp

        out = (output[0] + 1) / 2 * (self.maxt[0][0] - self.mint[0][0]) + self.mint[0][0]
        return math.pow(out[0], -8)
