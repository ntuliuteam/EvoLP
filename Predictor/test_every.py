from bp_python import net_sim
import matplotlib.pyplot as plt
import numpy as np

mat_file = './mobilenet_v1_cpu/mobilenet_v1_result_cpu_7_3_3_1_1_1_1_False_g_final.mat'

trans_sim = net_sim(mat_file)

print(trans_sim.maxp)
x = []
y = []
for i in range(1, trans_sim.maxp[1][0], 1):
    input_data = [793, i]
    output = trans_sim.sim(input_data)
    x.append(i)
    y.append(output)


x = np.array(x)

y = np.array(y)
plt.plot(x, y, color='#A96C02')
plt.show()
