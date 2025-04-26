# DeBERTaMF

### Summary
本科毕业设计的部分代码，主要尝试对2016年的ConvMF进行了复刻和改进，原论文地址如下：
- Convolutional Matrix Factorization for Document Context-Aware Recommendation (*RecSys 2016*)
  - <a href="http://dm.postech.ac.kr/~cartopy" target="_blank">_**Donghyun Kim**_</a>, Chanyoung Park, Jinoh Oh, Seungyong Lee, Hwanjo Yu
- Deep Hybrid Recommender Systems via Exploiting Document Context and Statistics of Items (*Information Sciences (SCI)*)
   - <a href="http://dm.postech.ac.kr/~cartopy" target="_blank">_**Donghyun Kim**_</a>, Chanyoung Park, Jinoh Oh, Hwanjo Yu

在经典的ConvMF基础上，我们主要针对语言特征提取部分进行了改进。但是由于原论文使用的python版本较低，有很多包已经不再支持(比如用于架构CNN的graph类)，且DeBERTa等模型要求python>=3.6，因此我们在python= 3.8.19的环境下重新复刻了ConMF，并尝试使用DeBERTa和ALBERT提取物品文本信息的语言特征。这个Github repository中包含的是使用DeBERTa的代码部分。

### How to Run



### Configuration
You can evaluate our model with different settings in terms of the size of dimension, the value of hyperparameter, the number of convolutional kernal, and etc. Below is a description of all the configurable parameters and their defaults:
