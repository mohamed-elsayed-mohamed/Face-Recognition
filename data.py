#!/usr/bin/env python

# Created by Mohamed Elsayed
import numpy as np, os

from scipy import misc
from sklearn.utils import shuffle

(TrainingData, TrainingLables, start) = ([], [], 0)
(TestingData, TestingLables, startT) = ([], [], 0)

def LoadTrainingData(Dir, Img_Shape):
    (Images, Lbls, Labels, ID, NClasses) = ([], [], [], 0, 0)

    for(_, Dirs, _) in os.walk(Dir):
        Dirs = sorted(Dirs)
        for SubDir in Dirs:
            SubjectPath = os.path.join(Dir, SubDir)
            for FileName in os.listdir(SubjectPath):
                path = SubjectPath + "/" + FileName
                Img = misc.imread(path, mode='L')

                #print Img.shape
                (height, width) = Img.shape

                if(width != Img_Shape[0] or height != Img_Shape[1]):
                    Img = Img.resize((Img_Shape[0], Img_Shape[1]))

                Images.append(Img)

                Lbls.append(int(ID))

            NClasses += 1

            ID += 1

    Images, Lbls = shuffle(Images, Lbls)

    Images = np.asarray(Images, dtype='float32').reshape([-1, Img_Shape[0], Img_Shape[1], 1]) /255

    #print "Classes: " + str(NClasses)

    for label in Lbls:
        Labels.append(Categorical([label], NClasses)[0])

    return (Images, np.asarray(Labels))


def LoadTestingData(Dir, Img_Shape):
    (Images, Labels, Names, Classes, Paths, ID, NClasses) = ([], [], [], [], [], 0, 0)

    for(_, Dirs, _) in os.walk(Dir):
        Dirs = sorted(Dirs)
        for SubDir in Dirs:
            SubjectPath = os.path.join(Dir, SubDir)
            for FileName in os.listdir(SubjectPath):
                path = SubjectPath + "/" + FileName
                Img = misc.imread(path, mode='L')
                Paths.append(path)

                #print Img.shape
                (height, width) = Img.shape

                if(width != Img_Shape[0] or height != Img_Shape[1]):
                    Img = Img.resize((Img_Shape[0], Img_Shape[1]))

                Images.append(Img)

                Labels.append(int(ID))
                Names.append(SubDir)
            Classes.append(SubDir)

            NClasses += 1

            ID += 1


    Images = np.asarray(Images, dtype='float32').reshape([-1, Img_Shape[0], Img_Shape[1], 1]) /255

    lbls = []
    for label in Labels:
        lbls.append(Categorical([label], NClasses)[0])

    return (Images, lbls, np.asarray(Names), np.asarray(Classes), np.asarray(Paths))


def Categorical(y, NClasses):
    y = np.asarray(y, dtype='int32')
    if not NClasses:
        NClasses = np.max(y)+1
    Y = np.zeros((len(y), NClasses))
    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y

def nextBatch(batchSize):
    global start
    end = start + batchSize

    #print "Start: " + str(start)
    #print "End: " + str(end)
    if(end > len(TrainingData)):
        X,Y = TrainingData[start:], TrainingLables[start:]
        start = 0
        return X, Y

    (X, Y) = (TrainingData[start: end], TrainingLables[start: end])
    start = end
    #X = X.reshape(len(X), X[0].size)
    if (end == len(TrainingData)):
        start = 0

    return X, Y

def nextTestBatch(batchSize):
    global startT
    end = startT + batchSize

    #print "Start: " + str(start)
    #print "End: " + str(end)
    if(end > len(TestingData)):
        X,Y = TestingData[startT:], TestingLables[startT:]
        startT = 0
        return X, Y

    (X, Y) = (TestingData[startT: end], TestingLables[startT: end])
    startT = end
    if(end == len(TestingData)):
        startT = 0
    #X = X.reshape(len(X), X[0].size)


    return X, Y
