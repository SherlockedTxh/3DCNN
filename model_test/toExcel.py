# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


data = np.load("/mnt/md1/a503tongxueheng/SoftmaxResults/result.npy")

print(data.shape)
data = data.reshape((-1,20))
print(data.shape)
data_df = pd.DataFrame(data)


data1 = np.load("/mnt/md1/a503tongxueheng/SoftmaxResults/lossresult.npy")
print(data1.shape)
data1_df = pd.DataFrame(data1)
result = pd.concat([data_df,data1_df],axis=1)

writer = pd.ExcelWriter('ResultExcel.xlsx')
result.to_excel(writer,'page_1',float_format='%.5f') # float_format 控制精度
writer.save()