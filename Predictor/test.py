# to see if the predicted latency of the trained MLP network exceed a threshold.
# If yes, use matlab_train_every.m to retrain this specific .mat.


from bp_python import net_sim
import os
import random

THRESHOLD = 50
mat_folder = './mobilenet_v1_cpu'

all_files = os.listdir(mat_folder)
for j in range(100):
    a = random.random() * 9 + 1
    for file in all_files:
        mat_file = os.path.join(mat_folder, file)

        trans_sim = net_sim(mat_file)
        x = []
        y = []

        for i in range(1, trans_sim.maxp[1][0], 1):
            input_data = [int(trans_sim.maxp[0][0] / a), i]
            output = trans_sim.sim(input_data)
            x.append(i)
            y.append(output)
            if output > THRESHOLD:
                print(mat_file, int(trans_sim.maxp[0][0] / a), i)
