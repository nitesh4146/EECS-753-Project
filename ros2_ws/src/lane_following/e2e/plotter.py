import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# plt.style.use('ggplot')
sns.set() 
# /home/slickmind/Documents/ppt_content/inference/

folders = ["inference_gtx", "inference_nano"]
models = ["nvidia", "resnet", "rnn"]
rpi_folders = ["1", "2", "3", "4"]
############### gtx

avg_times = []
for model in models:

    temp = []
    
    for f in folders:
        df = pd.read_csv("/home/slickmind/Documents/ppt_content/inference/" + f + "/inference_time_" + model + ".csv", header=None)
        # print(df.mean())
        temp.append(df[0].mean()*1000)

    avg_times.append(temp)

avg_times = np.array(avg_times)
print(avg_times)

rpi_avg_times = []

for rpi_f in rpi_folders:
    temp = []
    for model in models:

        df = pd.read_csv("/home/slickmind/Documents/ppt_content/inference/inference_rpi/" + rpi_f + "/inference_time_" + model + ".csv", header=None)
        # print(df.mean())
        temp.append(df[0].mean()*1000)
    rpi_avg_times.append(temp)

rpi_avg_times = np.array(rpi_avg_times)
print(rpi_avg_times)
print(np.array([rpi_avg_times[-1, :]]))


avg_times = np.concatenate((avg_times, np.array([rpi_avg_times[-1, :]]).T), axis=1)
print(avg_times)


labels = ['GTX1080', 'Jetson Nano', 'RPi3b (4 cores)']


# x = np.arange(len(labels))  # the label locations
# print(x)
# width = 0.35  # the width of the bars

ind = np.arange(len(labels)) 
width = 0.10

fig, ax = plt.subplots()

# fig.figure(figsize=(20, 3))  # width:20, height:add_3

rects1 = ax.bar(ind, avg_times[0], width, label='NVIDIA')
rects2 = ax.bar(ind + width, avg_times[1], width, label='ResNet')
rects3 = ax.bar(ind + width*2, avg_times[2], width, label='RNN')


# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Inference Time (ms)')
# ax.set_title('Scores by group and gender')
ax.set_xticks(ind)
ax.set_yticks(np.arange(0, np.amax(avg_times), 10))

ax.set_xticklabels(labels)
ax.legend()

# ax.bar_label(rects1, padding=3)
# ax.bar_label(rects2, padding=3)
# ax.bar_label(rects3, padding=3)


fig.tight_layout()

plt.show()

########################################

rpi_avg_times = rpi_avg_times.T

labels = ['1 core', '2 cores', '3 cores', '4 cores']

ind = np.arange(len(labels)) 
width = 0.10

# plt.figure(figsize=(20, 3))  # width:20, height:add_3
fig, ax = plt.subplots()


rects1 = ax.bar(ind, rpi_avg_times[0], width, label='NVIDIA')
rects2 = ax.bar(ind + width, rpi_avg_times[1], width, label='ResNet')
rects3 = ax.bar(ind + width*2, rpi_avg_times[2], width, label='RNN')


# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Inference Time (ms)')
# ax.set_title('Scores by group and gender')
ax.set_xticks(ind)
ax.set_xticklabels(labels)
ax.set_yticks(np.arange(0, np.amax(rpi_avg_times), 50))

ax.legend()

# ax.bar_label(rects1, rotation=90, padding=3)
# ax.bar_label(rects2, rotation=90, padding=3)
# ax.bar_label(rects3, rotation=90, padding=3)


fig.tight_layout()

plt.show()

################

# import glob
# import cv2

# grid = []

# for i in glob.glob("/home/slickmind/Documents/ppt_content/evaluation/mse/**/*.png"):
#     img = cv2.imread(i)[:50,:]
#     cv2.imwrite(i, img)
    # cv2.imshow("", img[:50,:])
    # cv2.waitKey()

# df = pd.read_csv("/Users/nitish/Desktop/plots/inference_time_nvidia_sim.csv")
# df1 = pd.read_csv("/Users/nitish/Desktop/plots/inference_time_nvidia.csv")
# # df = df[df["NVIDIA"] < 7]
# # df = df[df["ResNet"] > 35]
# # df = df[df["RNN"] > 58]

# # df1 = df1[df1["NVIDIA"] < 7]
# # df1 = df1[df1["ResNet"] < 12]
# # df1 = df1[df1["RNN"] < 30]

# fig, axes = plt.subplots(1, 2, figsize=(16,8))
# fig.suptitle('Inference time on GTX 1080 GPU')

# sns.boxplot(ax=axes[0], x="variable", y="value", data=pd.melt(df1)).set(
#     xlabel='', 
#     ylabel='Inference time (ms)'
# )
# axes[0].set_title("Without SVL")
# # plt.show()

# sns.boxplot(ax=axes[1], x="variable", y="value", data=pd.melt(df)).set(
#     xlabel='', 
#     ylabel='Inference time (ms)'
# )
# axes[1].set_title("With SVL")

# axes[0].grid()
# axes[1].grid()

# plt.show()

# # NV_m1
# train_mse = [0.006482416274957358, 0.0004969343114498769, 0.00046050299402850215, 0.00043117922308738343, 0.0004227167474164162, 0.0014340462523978203, 0.0012024881894467398, 0.0011331632392830215, 0.0011023758753435685, 0.0010644857620354742, 0.000740045795100741, 0.0007204935324261896, 0.0006923536848626099, 0.0006796094769379124, 0.0006641955440863967]
# val_mse = [0.0073696293337852694, 0.0012722601392306388, 0.0012021838233340532, 0.0011657690687570722, 0.0011465228119050153, 0.0007732555584516376, 0.0006705401826184242, 0.0006870398489991203, 0.0006610314751742407, 0.0006395928916754202, 0.0009159197937697172, 0.0007696521317120641, 0.000902618282707408, 0.0007685031607979908, 0.000806573819136247]

# plt.plot(train_mse,label = "Train MSE")
# plt.plot(val_mse,label = "Validation MSE")
# plt.xlabel("Epochs")
# plt.ylabel("Mean Square Error")
# plt.legend()
# plt.grid()
# plt.show()

# #Res_m1
# train_mse = [0.0073836383150191978, 0.0008125602168729529, 0.0007945024719811045, 0.0007611968196579255, 0.0006393315629975404, 0.0020181573228910564, 0.0018405074102338403, 0.001743621508940123, 0.0016125926468521356, 0.001445683476049453, 0.0008793265608255752, 0.0008543144771829248, 0.0008404978626640514, 0.0008081232203403488, 0.000781108028604649]
# val_mse = [0.007726220113923773, 0.0017263280146289617, 0.001726523496909067, 0.0017597557394765317, 0.0018772762827575207, 0.0025286269839853047, 0.0018643776857061312, 0.003429774937685579, 0.0014348653238266706, 0.0017207954579498618, 0.0009714873018674552, 0.0010039163776673376, 0.0015806125476956368, 0.002270551109686494, 0.0007717314106412232]

# plt.plot(train_mse,label = "Train MSE")
# plt.plot(val_mse,label = "Validation MSE")
# plt.xlabel("Epochs")
# plt.ylabel("Mean Square Error")
# plt.legend()
# plt.grid()
# plt.show()

# #Rnn_m1
# train_mse = [0.007643435077625327, 0.0005268211170914583, 0.0004687700653448701, 0.00046445749103440903, 0.0004370949791336898, 0.001737217343179509, 0.0014116953068878501, 0.0012691085546975955, 0.0011560407176148147, 0.001166671866667457, 0.0007606379425851628, 0.0006655680743278935, 0.0006294043792877346, 0.0006244035146664828, 0.0006158524774946272]
# val_mse = [0.0078764011837542057, 0.0014292757958173751, 0.0013288106652908026, 0.0012748271925374865, 0.0012567942962050439, 0.000999363255687058, 0.000981688394676894, 0.0008227505837567151, 0.0008028230397030712, 0.0007636475551407784, 0.0007644135737791657, 0.000787290942389518, 0.0007520809187553823, 0.000773074897006154, 0.0007484378525987268]

# plt.plot(train_mse,label = "Train MSE")
# plt.plot(val_mse,label = "Validation MSE")
# plt.xlabel("Epochs")
# plt.ylabel("Mean Square Error")
# plt.legend()
# plt.grid()
# plt.show()
