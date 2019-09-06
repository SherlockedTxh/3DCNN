 #!/usr/bin/env python
from __future__ import division;
from __future__ import print_function;
import tables;
import numpy
import sys
import os
windows_dir_pre='/mnt/md1/a503tongxueheng/test_data_process'
DEFAULT_NODE_NAME = "defaultNode";


def init_h5_file(toDiskName, groupName=DEFAULT_NODE_NAME, groupDescription=DEFAULT_NODE_NAME):

    import tables;
    h5file = tables.open_file(toDiskName, mode="w", title="Dataset")
    gcolumns = h5file.create_group(h5file.root, groupName, groupDescription)
    return h5file;

class InfoToInitArrayOnH5File(object):
    def __init__(self, name, shape, atomicType):

        self.name = name;
        self.shape = shape;
        self.atomicType = atomicType;

def writeToDisk(h5file,theH5Column, whatToWrite, batch_size=5000):

    data_size = len(whatToWrite);
    last = int(data_size / float(batch_size)) * batch_size
    for i in xrange(0, data_size, batch_size):
        stop = (i + data_size%batch_size if i >= last
                else i + batch_size)
        theH5Column.append(whatToWrite[i:stop]);
        h5file.flush()
    
def getH5column(h5file, columnName, nodeName=DEFAULT_NODE_NAME):
    node = h5file.get_node('/', DEFAULT_NODE_NAME);
    return getattr(node, columnName);


def initColumnsOnH5File(h5file, infoToInitArraysOnH5File, expectedRows, nodeName=DEFAULT_NODE_NAME, complib='blosc', complevel=5):
    gcolumns = h5file.get_node(h5file.root, nodeName);
    filters = tables.Filters(complib=complib, complevel=complevel);
    for infoToInitArrayOnH5File in infoToInitArraysOnH5File:
        finalShape = [0]; #in an eArray, the extendable dimension is set to have len 0
        finalShape.extend(infoToInitArrayOnH5File.shape);
        h5file.create_earray(gcolumns, infoToInitArrayOnH5File.name, atom=infoToInitArrayOnH5File.atomicType
                            , shape=finalShape, title=infoToInitArrayOnH5File.name #idk what title does...
                            , filters=filters, expectedrows=expectedRows);
    
def performScikitFit(predictors, outcomes):
    import sklearn.linear_model;
    model = sklearn.linear_model.LinearRegression(predictors, outcomes);
    model.fit(predictors, outcomes);
    print(model.predict([2.0,2.0]));
    
if __name__ == "__main__":

    pixel_size = 1
    mode = 'S'
    atom_density=0.01
    num_of_channels=4
    num_of_parts = 1
    num_3d_pixel=20
    
    #intiialise the columns going on the file
    dataName = "data";
    dataShape = [num_of_channels,num_3d_pixel,num_3d_pixel,num_3d_pixel]; #arr describing the dimensions other than the extendable dim.
    labelName = "label";
    labelShape = []; #the outcome is a vector, so there's only one dimension, the extendable one.
    dataInfo = InfoToInitArrayOnH5File(dataName, dataShape, tables.Float32Atom());
    labelInfo = InfoToInitArrayOnH5File(labelName, labelShape, tables.Float32Atom());

    '''
    for part in range (1,num_of_parts+1):
        Xv_smooth = numpy.load(windows_dir_pre+"/data/Sampled_Numpy/Xv_smooth_"+str(part)+".dat")
        yv = numpy.load(windows_dir_pre+"/data/Sampled_Numpy/yv_"+str(part)+".dat")

        for i in range (1,20):
            X = numpy.load(windows_dir_pre+"/data/Sampled_Numpy/X_smooth"+str(i)+'_'+str(part)+".dat")
            y = numpy.load(windows_dir_pre+"/data/Sampled_Numpy/y"+str(i)+"_"+str(part)+".dat")
            
            if i==1:
                X_smooth = X
                labels = y[:,numpy.newaxis]
            else:
                X_smooth = numpy.concatenate((X_smooth,X), axis=0)
                labels = numpy.concatenate((labels,y[:,numpy.newaxis]), axis=0)

        labels = numpy.ravel(labels)
    '''

    # new
    for filename in os.listdir(windows_dir_pre+"/data/Sampled_Numpy"):
        if filename[0:8] == "X_smooth":
            X_smooth = numpy.load(windows_dir_pre+"/data/Sampled_Numpy/"+filename)
            y_filename = "y"+filename[8:]
            y = numpy.load(windows_dir_pre+"/data/Sampled_Numpy/"+y_filename)
        
        filename_test = windows_dir_pre+"/data/ATOM_CHANNEL_dataset/"+(filename[8:])[:-4]+".pytables"
        h5file = init_h5_file(filename_test)
        numSamples = X_smooth.shape[0]
        
        initColumnsOnH5File(h5file, [dataInfo,labelInfo], numSamples)
        dataColumn = getH5column(h5file, dataName)
        labelColumn = getH5column(h5file, labelName)
        writeToDisk(h5file,dataColumn, X_smooth)
        writeToDisk(h5file,labelColumn, y)
        h5file.close()

        '''
        # Writing Train pytables
        filename_train = windows_dir_pre+"/data/train_data_"+str(part)+".pytables";
        h5file = init_h5_file(filename_train);
        numSamples = X_smooth.shape[0];
        
        initColumnsOnH5File(h5file, [dataInfo,labelInfo], numSamples);
        dataColumn = getH5column(h5file, dataName);
        labelColumn = getH5column(h5file, labelName); 
        writeToDisk(h5file, dataColumn, X_smooth);
        writeToDisk(h5file, labelColumn, labels);
        h5file.close();

        # Writing Val pytables
        filename_val = windows_dir_pre+"/data/test_data_"+str(part)+".pytables";
        h5file = init_h5_file(filename_val);
        numSamples = Xv_smooth.shape[0];
        
        initColumnsOnH5File(h5file, [dataInfo,labelInfo], numSamples);
        dataColumn = getH5column(h5file, dataName);
        labelColumn = getH5column(h5file, labelName); 
        writeToDisk(h5file,dataColumn, Xv_smooth);
        writeToDisk(h5file,labelColumn, yv);
        h5file.close();
        '''