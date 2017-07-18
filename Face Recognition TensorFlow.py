#!/usr/bin/env python
import tensorflow as tf, time, math, os, data # get custom dataset
from PIL import Image, ImageDraw, ImageFont

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

training_dir = 'Datasets/att_faces/Training'
testing_dir = 'Datasets/att_faces/Testing'

DATASET = 'ATT' # Please Change ATT to YALE if you want to train Yale Dataset

filterSize1 = 5
numFilters1 = 32
maxPooling1 = 2

filterSize2 = 5
numFilters2 = 64
maxPooling2 = 2

fullyConn1 = 2000
fullyConn2 = 750

dropout = 0.75

imageSize = 92*112
imageWidth = 92
imageHeight = 112
NChannels = 1

NClasses = 40
BatchSize = 5

NEpochs = 10
learningRate = 0.001

# Change some parameters if we use Yale dataset
if(DATASET == 'YALE'):
    training_dir = 'Datasets/yalefaces/Training'
    testing_dir = 'Datasets/yalefaces/Testing'

    maxPooling1 = 5
    maxPooling2 = 5

    imageSize = 320 * 243
    imageWidth = 320
    imageHeight = 243

    NClasses = 15

    NEpochs = 15

x = tf.placeholder(tf.float32, [None, imageSize])
y = tf.placeholder(tf.float32, [None, NClasses])
keepRatio = tf.placeholder(tf.float32)

X, Y= data.LoadTrainingData(training_dir, (imageWidth, imageHeight))
data.TrainingData = X
data.TrainingLables = Y

XT, YT, NamesT, _, Paths = data.LoadTestingData(testing_dir, (imageWidth, imageHeight))
data.TestingData = XT
data.TestingLables = YT
print (len(X), len(Y), len(XT), len(YT), len(NamesT), len(Paths))


########################################
################ Omar ##################
########################################

weights = {
    'wc1': tf.Variable(tf.random_normal([filterSize1, filterSize1, NChannels, numFilters1])),
    'wc2': tf.Variable(tf.random_normal([filterSize2, filterSize2, numFilters1, numFilters2])),
    'wf1': tf.Variable(tf.random_normal([int(math.ceil(imageWidth / float(maxPooling1 * maxPooling2))*math.ceil(imageHeight / float(maxPooling1*maxPooling2)))*numFilters2, fullyConn1])), # updated
    'wf2': tf.Variable(tf.random_normal([fullyConn1, fullyConn2])),
    'out': tf.Variable(tf.random_normal([fullyConn2, NClasses]))
}

########################################
################ Omar ##################
########################################

biases = {
    'bc1': tf.Variable(tf.random_normal([numFilters1])),
    'bc2': tf.Variable(tf.random_normal([numFilters2])),
    'bf1': tf.Variable(tf.random_normal([fullyConn1])),
    'bf2': tf.Variable(tf.random_normal([fullyConn2])),
    'out': tf.Variable(tf.random_normal([NClasses]))
}

########################################
############### Karima #################
########################################

def conv2d(layer, W):
    return tf.nn.conv2d(input=layer, filter=W, strides=[1, 1, 1, 1], padding='SAME')

########################################
############### Karima #################
########################################

def maxpool2d(layer, filterSize):
    return tf.nn.max_pool(value=layer, ksize=[1, filterSize, filterSize, 1], strides=[1, filterSize, filterSize, 1], padding='SAME')

########################################
############### Karima #################
########################################

def newConvLayer(input, weights, biases, activation='relu', usePooling=True, poolingFilter = 2):
    layer=conv2d(layer=input,W=weights)
    layer = tf.nn.bias_add(layer, biases)

    if(activation=='relu'):
        layer=tf.nn.relu(layer)
    else:
        layer=tf.nn.tanh(layer)

    if usePooling:
        layer= maxpool2d(layer=layer,filterSize=poolingFilter)

    return layer

########################################
################ Omar ##################
########################################

def flattenLayer(input):
    layerShape = input.get_shape() # [num_images, height, width, num_channels]

    num_features =  layerShape[1:4].num_elements()

    Layer = tf.reshape(input, [-1, num_features])

    return Layer

########################################
################ Marwa #################
########################################

def newFCLayer(input, weights, biases, isOut = False, activation='tanh', dropout=0.75):
    layer = tf.add(tf.matmul(input, weights), biases)

    if(isOut == True):
        return layer

    if(activation=='relu'):
        layer=tf.nn.relu(layer)
    else:
        layer=tf.nn.tanh(layer)

    layer = tf.nn.dropout(layer, dropout)

    return layer

########################################
################ Hend ##################
########################################

def CNN(input, weights, biases, keepratio):
    network = tf.reshape(input, [-1, imageWidth, imageHeight, NChannels])

    network = newConvLayer(input=network, weights=weights['wc1'], biases=biases['bc1'], activation='relu', usePooling=True, poolingFilter=maxPooling1)
    network = tf.nn.lrn(input=network, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75)
    network = newConvLayer(input=network, weights=weights['wc2'], biases=biases['bc2'], activation='relu', usePooling=True, poolingFilter=maxPooling2)

    network = flattenLayer(network)
    network = newFCLayer(input=network, weights=weights['wf1'], biases=biases['bf1'], activation='relu', dropout=keepratio)
    network = newFCLayer(input=network, weights=weights['wf2'], biases=biases['bf2'], activation='relu', dropout=keepratio)
    network = newFCLayer(input=network, weights=weights['out'], biases=biases['out'], isOut=True)

    return network

########################################
############### Mohamed ################
########################################

def main():

    Prediction = CNN(x, weights, biases, keepRatio)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Prediction, labels=y))

    optimizer = tf.train.AdamOptimizer(learningRate).minimize(cost)

    correct = tf.equal(tf.argmax(Prediction, 1), tf.argmax(y, 1))

    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        rang = int(len(X) / BatchSize)
        if len(X) % BatchSize != 0:
            rang += 1

        TestData = XT.reshape(len(XT), imageSize)

        for epoch in range(1, NEpochs+1):
            avg_loss = 0.0
            avg_acc = 0.0
            for i in range(rang):
                epochX, epochY = data.nextBatch(BatchSize)

                epochX = epochX.reshape(len(epochX), imageSize)

                feeds = {x: epochX, y:epochY, keepRatio: dropout}

                sess.run(optimizer, feed_dict=feeds)

                loss, acc = sess.run([cost, accuracy], feed_dict={x:epochX, y:epochY, keepRatio:1.})

                avg_acc += (acc / (rang))
                avg_loss += (loss / (rang))

                print("Epoch: %01d/%01d loss: %.4f Accuracy: %.2f" % (epoch, NEpochs, avg_loss, (avg_acc*100.0)) + str(' %'))

            print "Epoch " + str(epoch) + " Finished !"


        print("Testing Accuracy: " + str(sess.run(accuracy, feed_dict={x: TestData, y: YT, keepRatio: 1.}) * 100) + str(' %'))

        Predictions = sess.run(tf.argmax(Prediction, 1), feed_dict={x: TestData, keepRatio:1.})
        print (Predictions)

        i = 0
        for p in Predictions:
            validName = str(NamesT[i])
            predictedName = str(NamesT[p])

            print(str(p) + "-PreTest: " + validName + " --> Test: " + NamesT[p])

            TestImg = Image.open(Paths[i]).convert('RGBA')
            draw = ImageDraw.Draw(TestImg)
            font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 12)

            txt = Image.new('RGBA', TestImg.size, (255, 255, 255, 0))

            draw = ImageDraw.Draw(txt)

            draw.text((0, 0), validName, font=font, fill=(0, 255, 0, 255))

            if (validName == predictedName):
                draw.text((imageWidth - 25, 0), predictedName, font=font, fill=(0, 255, 0, 255))
            else:
                draw.text((imageWidth - 25, 0), predictedName, font=font, fill=(255, 0, 0, 255))

            TestImg = Image.alpha_composite(TestImg, txt)

            TestImg.show()

            i += 1
            time.sleep(.5)

if __name__ == '__main__':
    main()