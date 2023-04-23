import sys
import time

sys.path.append('../')

import Models as models
import torch
import torch.nn as nn
import copy
import os
from Predictor import net_sim
import numpy as np


class BpPredictor:
    def __init__(self, mat_folder, device_name, model_name):
        self.all_netsim = {}
        self.device_name = device_name
        self.model_name = model_name
        all_files = os.listdir(mat_folder)
        for file in all_files:
            mat_file = os.path.join(mat_folder, file)
            self.all_netsim[file] = net_sim(mat_file)

    def predict(self, conv_info):
        latency = 0.0
        for info_i in conv_info:
            tmp = info_i.replace(' ', '').split(',')
            tmp[-1] = tmp[-1][:-1]
            # print(tmp)
            input_size = tmp[1]

            in_channel = int(tmp[2])
            out_channel = int(tmp[3])

            groups = int(tmp[-1])

            if groups == 1:
                mat_name = self.model_name + '_result_' + self.device_name + '_' + str(input_size) + '_' + '_'.join(
                    str(j) for j in tmp[4:]) + '_final.mat'
            elif groups == in_channel:
                mat_name = self.model_name + '_result_' + self.device_name + '_' + str(input_size) + '_' + '_'.join(
                    str(j) for j in tmp[4:-1]) + '_g_final.mat'
            else:
                exit('no valid groups')
            # print(mat_name)

            if mat_name not in self.all_netsim.keys():
                mat_name = self.model_name + '_result_' + self.device_name + '_' + str(input_size) + '_' + '_'.join(
                    str(j) for j in tmp[4:-1]) + '_g_final.mat'

            assert mat_name in self.all_netsim.keys()
            cur_lat = self.all_netsim[mat_name].sim([in_channel, out_channel])
            latency = latency + cur_lat
        return latency

    def predict_with_layer(self, conv_info):
        latency = 0.0
        layer_latency = []
        for info_i in conv_info:
            tmp = info_i.replace(' ', '').split(',')
            tmp[-1] = tmp[-1][:-1]
            # print(tmp)
            input_size = tmp[1]

            in_channel = int(tmp[2])
            out_channel = int(tmp[3])

            groups = int(tmp[-1])

            if groups == 1:
                mat_name = self.model_name + '_result_' + self.device_name + '_' + str(input_size) + '_' + '_'.join(
                    str(j) for j in tmp[4:]) + '_final.mat'
            elif groups == in_channel:
                mat_name = self.model_name + '_result_' + self.device_name + '_' + str(input_size) + '_' + '_'.join(
                    str(j) for j in tmp[4:-1]) + '_g_final.mat'
            else:
                exit('no valid groups')
            # print(mat_name)
            if mat_name not in self.all_netsim.keys():
                mat_name = self.model_name + '_result_' + self.device_name + '_' + str(input_size) + '_' + '_'.join(
                    str(j) for j in tmp[4:-1]) + '_g_final.mat'
            assert mat_name in self.all_netsim.keys()
            cur_lat = self.all_netsim[mat_name].sim([in_channel, out_channel])
            layer_latency.append(cur_lat)
            latency = latency + cur_lat
        return latency, copy.deepcopy(layer_latency)


class RlLatency:
    def __init__(self, ip='192.168.0.2', device='tx2', username='nvidia', password='nvidia',
                 script_folder='~/EvoLp/Predictor_EDLAB', model_name='vgg16'):
        m = np.array(model_name)
        self.device = device
        if self.device == 'tx2':
            np.save('../Predictor_EDLAB/m_n.npy', m)
            self.py_path = '/usr/bin/python3'
        elif self.device == 'nano':
            np.save('../Predictor_EDLAB/m_n_nano.npy', m)
            self.py_path = '/usr/bin/python3'
        # elif self.device == 'pi':
        #     np.save('../Predictor_EDLAB/m_n_pi.npy', m)
        #     self.py_path = '/home/pi/.virtualenvs/python3/bin/python3'
        else:
            exit('no device')
        self.ip = ip
        self.username = username
        self.password = password
        self.script_folder = script_folder

        if 'inception' in model_name:
            self.script = '_inception'
        else:
            self.script = ''

    def run_model(self, model_cfg, args=None):
        if self.device == 'tx2':
            np.save('../Predictor_EDLAB/m_c.npy', np.array(model_cfg))
            time.sleep(3)
            command = 'sshpass -p \'' + self.password + '\' ssh ' + self.username + '@' + self.ip + ' \"cd ' + self.script_folder + '; ' + self.py_path + ' client' + self.script + '.py;\"'
        elif self.device == 'nano':
            np.save('../Predictor_EDLAB/m_c_nano.npy', np.array(model_cfg))
            time.sleep(3)
            command = 'sshpass -p \'' + self.password + '\' ssh ' + self.username + '@' + self.ip + ' \"cd ' + self.script_folder + '; ' + self.py_path + ' client' + self.script + '_nano.py;\"'
        # elif self.device == 'pi':
        #     np.save('../Predictor_EDLAB/m_c_pi.npy', np.array(model_cfg))
        #     time.sleep(3)
        #     command = 'sshpass -p \'' + self.password + '\' ssh ' + self.username + '@' + self.ip + ' \"cd ' + self.script_folder + '; ' + self.py_path + ' client' + self.script + '_pi.py;\"'
        else:
            exit('no device')
        t = os.popen(command).readlines()
        lat = float(t[0].replace('\n', ''))
        if args is not None:
            args.lat = lat * 1000
        return lat * 1000


