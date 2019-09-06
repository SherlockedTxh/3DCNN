#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import numpy
import os
import random
import json
import scipy.ndimage
from scipy import spatial
import sys

DEFAULT_NODE_NAME = "defaultNode";
windows_dir_pre='/mnt/md1/a503tongxueheng/test_data_process'

label_res_dict={0:'HIS',1:'LYS',2:'ARG',3:'ASP',4:'GLU',5:'SER',6:'THR',7:'ASN',8:'GLN',9:'ALA',10:'VAL',11:'LEU',12:'ILE',13:'MET',14:'PHE',15:'TYR',16:'TRP',17:'PRO',18:'GLY',19:'CYS'}

resiName_to_label={'ILE': 12, 'GLN': 8, 'GLY': 18, 'GLU': 4, 'CYS': 19, 'HIS': 0, 'SER': 5, 'LYS': 1, 'PRO': 17, 'ASN': 7, 'VAL': 10, 'THR': 6, 'ASP': 3, 'TRP': 16, 'PHE': 14, 'ALA': 9, 'MET': 13, 'LEU': 11, 'ARG': 2, 'TYR': 15}



def integrate_20_AA_numpy(dict_name,in_dir, out_dir, num_3d_pixel, num_of_channels, num_of_parts):
    '''
    # 不需要获取各种氨基酸的个数
    if os.path.isfile(os.path.join(windows_dir_pre+'/data/DICT',dict_name)):
        with open(os.path.join(windows_dir_pre+'/data/DICT',dict_name)) as f:
            tmp_dict = json.load(f)
        res_count_dict={}
        for i in range (0,20):
            res_count_dict[i]=tmp_dict[str(i)]
    else:
        print ("dictionary not exist!")
        res_count_dict={0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0,11:0,12:0,13:0,14:0,15:0,16:0,17:0,18:0,19:0}
        
        files = [ f for f in os.listdir(in_dir) if os.path.isfile(os.path.join(in_dir,f))]
        for f in files:
            file_name=f.strip('\n')
            parts=file_name.split('_')
            res=parts[0]
            label=resiName_to_label[res]
            res_count_dict[label]+=1

    print ("res_count_dict content:")
    for key in res_count_dict:
        print (label_res_dict[key]+" "+str(res_count_dict[key]))
    '''

    '''
    res_files_dict={}
    
    # Identify the rarest amino acid microenvironment type, and the number of examples
    min_ind = min(res_count_dict, key=res_count_dict.get)
    min_data=res_count_dict[min_ind]
    
    print ("integrating data ... ")
    
    # 随机取min_data个下标存入mask，测试时不需要平衡
    for label in range (0,20):
        mask=random.sample(xrange(res_count_dict[label]), min_data)
        res_files_dict[label]=mask
    '''    
    
    for part in range(0,num_of_parts):
        #print "part:"+str(part)
        

        '''
        for label in range (0,20):
            res_files = res_files_dict[label]
            for i in range (part*(min_data/num_of_parts),(part+1)*(min_data/num_of_parts)):
                num = res_files[i]
                X = numpy.load(in_dir+label_res_dict[label]+"_"+str(num)+'.dat')
                y = label*numpy.ones((1000,1))
                equal_examples.append(X)
                equal_labels.append(y)
        '''

        for filename in os.listdir(in_dir):
            equal_examples=[]
            equal_labels=[]
            X = numpy.load(in_dir+filename,allow_pickle=True)
            print filename
            print X[0][0].shape
            for tmp in X:
                y = tmp[1]*numpy.ones((X.shape[0],1))
                equal_examples.append(tmp[0])
                equal_labels.append(y)

            equal_examples=numpy.array(equal_examples)
            equal_labels=numpy.array(equal_labels)

            #print equal_examples.shape
            #print equal_labels.shape
        
            #equal_examples=numpy.reshape(equal_examples,(X.shape[0], num_of_channels, num_3d_pixel, num_3d_pixel, num_3d_pixel))
            #equal_labels=numpy.reshape(equal_labels,X.shape[0])

            equal_examples.dump(windows_dir_pre+"/data/Sampled_Numpy/X_smooth_"+filename)
            equal_labels.dump(windows_dir_pre+"/data/Sampled_Numpy/y_"+filename)
        
        '''
        # 分训练集和验证集
        partition=equal_examples.shape[0]/20

        num_of_train=int(19*float(equal_examples.shape[0])/20)
        num_of_val=int(1*float(equal_examples.shape[0])/20)

        mask_train=random.sample(xrange(equal_examples.shape[0]), num_of_train)
        X_smooth=equal_examples[mask_train]
        y=equal_labels[mask_train]
        equal_examples=numpy.delete(equal_examples, mask_train, 0)
        equal_labels=numpy.delete(equal_labels, mask_train, 0)

        mask_val=random.sample(xrange(equal_examples.shape[0]), num_of_val)
        Xv_smooth=equal_examples[mask_val]
        yv=equal_labels[mask_val]
        equal_examples=numpy.delete(equal_examples, mask_val, 0)
        equal_labels=numpy.delete(equal_labels, mask_val, 0)

        # Dumping validation dataset as numpy array
        Xv_smooth.dump(windows_dir_pre+"/data/Sampled_Numpy/Xv_smooth_"+str(part+1)+".dat")
        yv.dump(windows_dir_pre+"/data/Sampled_Numpy/yv_"+str(part+1)+".dat")

        # Dumping training dataset as numpy array 
        for i in range (1,20):
            mask = range(partition*(i-1),partition*i)
            X_tmp = X_smooth[mask]
            y_tmp = y[mask]
            X_tmp.dump(windows_dir_pre+"/data/Sampled_Numpy/X_smooth"+str(i)+"_"+str(part+1)+".dat")
            y_tmp.dump(windows_dir_pre+"/data/Sampled_Numpy/y"+str(i)+"_"+str(part+1)+".dat")
        '''

        
if __name__ == '__main__':
    

    atom_density=0.01 # defalut = 0.01, desired threshold of atom density of boxes defined by num_of_atom / box volume
    pixel_size = 1
    mode = 'S'
    num_of_channels=4
    num_3d_pixel=20/pixel_size
    num_of_parts=1

    in_dir = windows_dir_pre+'/data/RAW_DATA/'
    out_dir = windows_dir_pre+'/data/Sampled_Numpy'
    dict_name = 'train_20AA_boxes.json'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    integrate_20_AA_numpy(dict_name,in_dir, out_dir, num_3d_pixel, num_of_channels, num_of_parts)

