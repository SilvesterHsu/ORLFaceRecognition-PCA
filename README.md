![principal components analysis](https://user-images.githubusercontent.com/20944641/30513670-bd46d20c-9b39-11e7-9ad8-405d8c175c30.png)
# Face recognition using PCA(Principal Components Analysis) with ORL database

[![Build Status](https://travis-ci.org/SilvesterHsu/ORLFaceRecognition-PCA.svg?branch=master)](https://travis-ci.org/SilvesterHsu/ORLFaceRecognition-PCA)	[![Scrutinizer Code Quality](https://scrutinizer-ci.com/g/SilvesterHsu/ORLFaceRecognition-PCA/badges/quality-score.png?b=master)](https://scrutinizer-ci.com/g/SilvesterHsu/ORLFaceRecognition-PCA/?branch=master)	[![Libraries.io for releases](https://img.shields.io/badge/release-v0.1.5-orange.svg)]()

This is a face recognition illustration using PCA via python.  
  
* [Introduction](https://github.com/SilvesterHsu/ORLFaceRecognition-PCA#introduction)  
* [Installation](https://github.com/SilvesterHsu/ORLFaceRecognition-PCA#installation)  
* [Run Program](https://github.com/SilvesterHsu/ORLFaceRecognition-PCA#run-program)
* [To Do](https://github.com/SilvesterHsu/ORLFaceRecognition-PCA#todo)  

## Introduction
This program is mainly used for face recognition. Face recognition can be broadly divided into two parts: **data processing** and **recognition**.  

### 1. data processing
We choose [The ORL Database of Faces](http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html) as database. After a series of basic image preprocessing, we map each image column quantitatively. After that, we piled the vectors horizontally into a matrix, and use [*Principal Components Analysis* ](http://psycnet.apa.org/record/1934-00645-001)to reduce the dimension of matrix. Finally, the column vector of each image is projected in a point in the high dimensional space.  
![image](https://user-images.githubusercontent.com/20944641/30518453-02e300c0-9bb1-11e7-8a3b-cc5996ef5c0a.png) 

### 2. recognition
We use [*Euclidean Distance*](https://en.wikipedia.org/wiki/Euclidean_distance) to measure the distance between two coordinate points in high dimensional space (each coordinate point represents a face image). The smaller the metric, the closer the two coordinate points are in the high-dimensional space (not near the strict sense, but near the Euclidean space), the more the face image represented by the two coordinate points similar. Here you can set the distance threshold to determine the distance at which the two coordinate points represented the same face.  
![image](https://user-images.githubusercontent.com/20944641/30518885-4459e9f0-9bbc-11e7-988d-fb78164b13c0.png)  
  
## Installation
The program uses **Mac OS 10.10** as a test platform, **Ubuntu 16.04** can work as well. Other operating systems have yet to be tested. Program uses Python3 as the programming language, so you also need to install Python3 environment, and will also need to use `pip3` to install some necessary packages.
### Environment
Operating Systems: **Mac OS 10.10**  
IDE: [**PyCharm CE**](https://www.jetbrains.com/pycharm/download/)

### Step 1: Install Python3
I only list the steps for installing Python3 on Mac, and the installation steps for the rest of the operating systems are available for download and documentation from [Python3](https://www.python.org/).  
First we need to install Xcode command line tool. It is kind of necessary environment we need to install python. So open a terminal, and run the command as below.  
```
xcode-select--install
```  
It is easy install python through [Homebrew](https://brew.sh/index_zh-tw.html). The function of Homebrew is much liable to that of command `apt` under **Ubuntu**. The means you install packages are also similar. For instance, we use `brew install <packages>` to install new packages.   
```
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```  
Now, well preparing, we can install Python via **HomeBrew**:  
```
brew install python3
```  
It won't take a long time. If don't success, try [Proxy Note](https://github.com/SilvesterHsu/ORLFaceRecognition-PCA#proxy-note). After installing python, you shall use `python3` to check whether **Python3** is installed. Besides, you can use `python3 -V` to check your python version.  
![image](https://user-images.githubusercontent.com/20944641/30519531-38a3458e-9bcb-11e7-9fdc-333a2b8b4f9e.png)  
#### Proxy Note
In addition, it is worth noting that the network maybe slow due to the GFW while using **HomeBrew**. You can, therefore, use [shadowsocks](https://github.com/shadowsocks/shadowsocks-iOS) in terminal to proxy your network.  
```
export ALL_PROXY=socks5://127.0.0.1:1080
```  

### Step 2: Install Packages

The necessary packages are already in the file `requirements.txt`. So you can use `pip3` to install them all in an easy way.  
```
cd PCA_ORL_Python
pip3 install -r requirements.txt
```  
#### Proxy Note
Proxy can also be used to speed up the process of `pip3 install`. Be aware, pip doesn't support `socks5`, so you can only use http proxy.  
```
export ALL_PROXY=http://server:port
```
## Run Program
To run this code, you can run both in a IDE or terminal.
```
python3 PCA.py --image-scale 0.4 --train-per-person 3 --print-feature-face True --principal-rate 0.3
```  
![image](https://user-images.githubusercontent.com/20944641/30792527-8d0cc870-a1ed-11e7-9e29-1bdd259c73c0.png)

## TODO 
* Offer mathematical formula documents



