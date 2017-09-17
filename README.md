![principal components analysis](https://user-images.githubusercontent.com/20944641/30513670-bd46d20c-9b39-11e7-9ad8-405d8c175c30.png)
# Face recognition using PCA(Principal Components Analysis) with ORL database

[![Build Status](https://travis-ci.org/SilvesterHsu/PCA_ORL_Python.svg?branch=master)](https://travis-ci.org/SilvesterHsu/PCA_ORL_Python)	[![Scrutinizer Code Quality](https://scrutinizer-ci.com/g/SilvesterHsu/PCA_ORL_Python/badges/quality-score.png?b=master)](https://scrutinizer-ci.com/g/SilvesterHsu/PCA_ORL_Python/?branch=master)	[![Libraries.io for releases](https://img.shields.io/badge/release-v0.0.1-orange.svg)]()

This is a face recognition illustration using PCA via python.  
  

## Introduction
This program is mainly used for face recognition. Face recognition can be broadly divided into two parts: **data processing** and **recognition**.  

### 1. data processing
We choose [The ORL Database of Faces](http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html) as database. After a series of basic image preprocessing, we map each image column quantitatively. After that, we piled the vectors horizontally into a matrix, using *Principal Components Analysis* to reduce the dimension of matrix. Finally, the column vector of each image is projected in a point in the high dimensional space.  
![image](https://user-images.githubusercontent.com/20944641/30518453-02e300c0-9bb1-11e7-8a3b-cc5996ef5c0a.png) 

### 2. recognition
We use *Euclidean Metric* to measure the distance between two coordinate points in high dimensional space (each coordinate point represents a face image). The smaller the metric, the closer the two coordinate points are in the high-dimensional space (not near the strict sense, but near the Euclidean space), the more the face image represented by the two coordinate points similar. Here you can set the distance threshold to determine the distance at which the two coordinate points represented the same face.  
![image](https://user-images.githubusercontent.com/20944641/30518831-f8b97160-9bba-11e7-903f-87104209c302.png)  

***
## Installation
The program uses **Mac OS 10.10** as a test platform, **Ubuntu 16.04** can work as well. Other operating systems have yet to be tested. Program uses Python3 as the programming language, so you also need to install Python3 environment, and will also need to use `pip3` to install some necessary packages.
### Environment
Operating Systems: **Mac OS 10.10**  
IDE: **PyCharm CE**

### Step 1: Install Python3
I only listed the steps for installing Python3 on Mac, and the installation steps for the rest of the operating systems are available for download and documentation from [Python3](https://www.python.org/).  
First we need to install Xcode command line tool. Open a terminal, run the commands:  
```
xcode-select--install
```  
It is easy install python through [Homebrew](https://brew.sh/index_zh-tw.html).  
```
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```  
The network maybe slow due to the GFW, you can use shadowsocks in terminal to speed up.  
```
export ALL_PROXY=socks5://127.0.0.1:1080
```  
Now, we are ready to install Python:  
```
brew install python3
```  
You can use `python3 -V` to check your python version.  

### Step 2: Install Packages

The necessary packages are already in the file *requirements.txt*. So you can use `pip3` to install them easily.  
```
pip install -r requirements.txt
```  
***
## TODO



