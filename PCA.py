# -*- coding: utf-8 -*
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import misc

# covert image to vector
def img2vector(filename, scale):
    imgVector = misc.imresize(plt.imread(filename), scale).flatten()
    #imgVector = plt.imread(filename).flatten()
    return imgVector.astype(np.float)

# load image from diretion
def loadimage(dataSetDir, k, scale):
    train_face = np.zeros((40 * k, int(112 * scale) * int(92 * scale)))  # image size:112*92
    train_face_number = np.zeros(40 * k).astype(np.int8)
    test_face = np.zeros((40 * (10 - k), int(112 * scale) * int(92 * scale)))
    test_face_number = np.zeros(40 * (10 - k)).astype(np.int8)
    for i in np.linspace(1, 40, 40).astype(np.int8): #40 sample people
            people_num = i
            for j in np.linspace(1, 10, 10).astype(np.int8): #everyone has 10 different face
                if j <= k:
                    filename = dataSetDir+'/s'+str(people_num)+'/'+str(j)+'.pgm'
                    img = img2vector(filename, scale)
                    train_face[(i-1)*k+(j-1),:] = img
                    train_face_number[(i-1)*k+(j-1)] = people_num
                else:
                    filename = dataSetDir+'/s'+str(people_num)+'/'+str(j)+'.pgm'
                    img = img2vector(filename, scale)
                    test_face[(i-1)*(10-k)+(j-k)-1,:] = img
                    test_face_number[(i-1)*(10-k)+(j-k)-1] = people_num

    return train_face,train_face_number,test_face,test_face_number

# subtract mean data
def submean(target_data, mean_data):
    mean_data_4target = np.repeat(mean_data,target_data.shape[0],axis = 0)
    target_data = target_data - mean_data_4target
    return target_data

print("Project Start...")
scale = 0.5
k = 4
train_face,train_face_number,test_face,test_face_number = loadimage(os.getcwd()+'/att_faces',k,scale)
img_mean = train_face.mean(axis = 0).reshape((1, train_face.shape[1]))
train_face = submean(train_face, img_mean)
test_face = submean(test_face, img_mean)

cov = np.zeros((train_face.shape[1],train_face.shape[1]))
cov = np.dot(train_face.T, train_face)
l = np.zeros(train_face.shape[1])
v = np.zeros((train_face.shape[1],train_face.shape[1]))
#l, v = la.eig(cov)
print("Calculate l & v")
l, v = np.linalg.eig(cov)

mix = np.vstack((l,v))
mix = mix.T[np.lexsort(mix[::-1,:])].T[:,::-1]
v = np.delete(mix, 0, axis = 0)

plt.figure('Feature Map')
r, c = (4, 10)
for i in np.linspace(1, r * c, r * c).astype(np.int8):
    plt.subplot(r,c,i)
    plt.imshow(v[:, i-1].real.reshape(int(112 * scale), int(92 * scale)), cmap='gray')
    plt.axis('off')

v = v[:,0:100]

train_face = np.dot(train_face, v)
test_face = np.dot(test_face , v)

count = 0

for i in np.linspace(0, test_face.shape[0] - 1, test_face.shape[0]).astype(np.int64):
    sub = submean(train_face,test_face[i, :].reshape((1,test_face.shape[1])))
    dis = np.linalg.norm(sub, axis = 1)
    fig = np.argmin(dis)
    if train_face_number[fig] == test_face_number[i]:
        count = count + 1
correct_rate = count / test_face.shape[0]

print("Correct rate =", correct_rate * 100 , "%")
print("Finish...")
