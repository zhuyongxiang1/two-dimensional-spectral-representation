import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import hdf5storage
import time
from random import sample
from pyts.image import GramianAngularField, MarkovTransitionField, RecurrencePlot
from matplotlib import image
import torch

# 引入API
image_size = 100
gasf = GramianAngularField(image_size=image_size, method='summation')
# gadf = GramianAngularField(image_size=image_size, method='difference')
mtf = MarkovTransitionField(image_size=image_size,n_bins=8)
rep = RecurrencePlot(dimension=2, time_delay=3)

data1 = scio.loadmat('./Datasets/PaviaU/PaviaU.mat')
data2 = scio.loadmat('./Datasets/PaviaU/PaviaU_gt.mat')

data_hsi = data1['paviaU']
data_gt = data2['paviaU_gt']


print("data_gt:", data_gt.shape)
print("data_hsi:", data_hsi.shape)

all_time_gasf=0
all_time_mtf=0
all_time_rep=0
all_time_gmr=0
for j in range(1, 10):
    class_index = np.where(data_gt == j)  # 输出该类别所有像素点的位置给 class_index
    print("{}类别总共有".format(j), len(class_index[1]))

    for i in range(0, len(class_index[1])):

        tem_1 = data_hsi[class_index[0][i], class_index[1][i],].reshape(1, -1)  # 读取一个像素点的光谱 选择波段数



        tem_gt = data_gt[class_index[0][i], class_index[1][i]] # 读取一个像素点的光谱 选择波段数


        # gasf
        time_start1 = time.time()  # 记录开始时间
        image_gasf = gasf.fit_transform(tem_1)  # image_gasf 范围[-1,1]
        time_end1 = time.time()  # 记录结束时间
        time_sum1 = time_end1 - time_start1  # 计算的时间差为程序的执行时间，单位为秒/s
        #print("gasf_time:",time_sum)
        all_time_gasf = all_time_gasf + time_sum1


        time_start2 = time.time()
        image_gasf_resize = np.array((image_gasf - np.min(image_gasf))
                                     / (np.max(image_gasf) - np.min(image_gasf)))

        time_end2 = time.time()  # 记录结束时间
        time_sum2 = time_end2 - time_start2  # 计算的时间差为程序的执行时间，单位为秒/s
        # print("gasf_time:",time_sum)
        all_time_gmr = all_time_gmr + time_sum2




        # mtf
        time_start3 = time.time()  # 记录开始时间
        image_mft = mtf.fit_transform(tem_1)
        time_end3 = time.time()  # 记录结束时间
        time_sum3 = time_end3 - time_start3  # 计算的时间差为程序的执行时间，单位为秒/s
        all_time_mtf = all_time_mtf + time_sum3
        #print("mtf_time:",time_sum)

        # rep
        time_start4 = time.time()  # 记录开始时间
        image_rep = rep.fit_transform(tem_1)
        time_end4 = time.time()  # 记录结束时间
        time_sum4 = time_end4 - time_start4  # 计算的时间差为程序的执行时间，单位为秒/s
        #print("rep_time:",time_sum)
        all_time_rep = all_time_rep + time_sum4


        time_start5 = time.time()
        image_rep_resize = np.array((image_rep - np.min(image_rep))
                                    / (np.max(image_rep) - np.min(image_rep)))

        time_end5 = time.time()  # 记录结束时间
        time_sum5 = time_end5 - time_start5  # 计算的时间差为程序的执行时间，单位为秒/s
        # print("rep_time:",time_sum)
        all_time_gmr = all_time_gmr + time_sum5
        #  print(image_rep_resize.shape)
        # 三种变换结合成一起
        all_image = np.concatenate((image_gasf_resize, image_mft, image_rep_resize), axis=0)  # all_image (3,100,100)

        all_image_resize = np.transpose(all_image, (1, 2, 0))

        image.imsave("./Datasets_2d/PaviaU/{}_{}_{}.png".format(class_index[0][i], class_index[1][i],tem_gt),all_image_resize)
#
print("all_time_gasf:",all_time_gasf)
print("all_time_mtf:",all_time_mtf)
print("all_time_rep:",all_time_rep)
print("all_time_gmr:",all_time_gmr)

# text += "class  Accuracy:\n"
# if agregated:
#     for label, score, std in zip(label_values, F1_scores_mean,
#                                  F1_scores_std):
#         text += "\t{}: {:.04f} +- {:.04f}\n".format(label, score, std)
# else:
#     for label, score in zip(label_values, F1scores):
#         text += "\t{}: {:.04f}\n".format(label, score)
# text += "---\n"
#
# if agregated:
#     text += ("Accuracy: {:.02f} +- {:.02f}\n".format(np.mean(accuracies),
#                                                      np.std(accuracies)))
# else:
#     text += "Accuracy : {:.04f}%\n".format(accuracy)
# text += "---\n"
#
# if agregated:
#     total_sum = np.nansum(F1_scores_mean)
#     total_std = np.nansum(F1_scores_std)
#     average_accuracy = total_sum / (int(len(F1_scores_mean)) - 1)
#     average_accuracy_std = total_std / (int(len(F1_scores_std)) - 1)
# else:
#     total_sum = np.nansum(F1scores)
#     average_accuracy = total_sum / (int(len(F1scores)) - 1)
#
# if agregated:
#     text += ("average Accuracy:{:.04f} +- {:.04f}\n".format(average_accuracy, average_accuracy_std))
# else:
#     text += ("average Accuracy:{:.04f}".format(average_accuracy))
#
# if agregated:
#     text += ("Kappa: {:.04f} +- {:.04f}\n".format(np.mean(kappas),
#                                                   np.std(kappas)))
# else:
#     text += "Kappa: {:.04f}\n".format(kappa)
# vis.text(text.replace('\n', '<br/>'))
#
# text += "time: {:.04f}\n".format(time)
#
# if agregated:
#     folder_data = "./experiment_data/{}_tr{}_{}.txt".format(dataset, tr, model)
# else:
#     folder_data = "./experiment_data/{}_tr{}_{}_{}.txt".format(dataset, tr, model, run + 1)
#
# with open(folder_data, 'w') as x_file:
#     #  x_file.write('{:.04f} Overall accuracy (%)'.format(accuracy))
#     x_file.write(text)
#     x_file.write('\n')