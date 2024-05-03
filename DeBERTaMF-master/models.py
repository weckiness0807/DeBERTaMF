'''
Created on Dec 8, 2015

@author: donghyun
'''

import os
import time

from util import eval_RMSE
import math
import numpy as np
from text_analysis.models import DeBERTa_Model


def DeBERTaMF(res_dir, train_user, train_item, valid_user, test_user,
           R, DeBERTa_X, init_W=None, give_item_weight=True,
           max_iter=50, lambda_u=10, lambda_v=100, dimension=50,
           ):
    # explicit setting
    a = 1
    b = 0

    batch_size = 100
    num_user = R.shape[0]
    num_item = R.shape[1]
    PREV_LOSS = 1e-50
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    f1 = open(res_dir + '/state.log', 'w')

    Train_R_I = train_user[1]
    Train_R_J = train_item[1]
    Test_R = test_user[1]
    Valid_R = valid_user[1]

    if give_item_weight is True:
        item_weight = np.array([math.sqrt(len(i))                    # 计算每个物品被评分的次数，取平方根作为其初始权重
                                for i in Train_R_J], dtype=float)  
        item_weight = (float(num_item) / item_weight.sum()) * item_weight  # 将初始权重进行归一化
    else:
        item_weight = np.ones(num_item, dtype=float)

    pre_val_eval = 1e10

    print('-----try to create DeBERTa_Model-----')
    deberta_model = DeBERTa_Model(output_dimesion=dimension)
    print('-------try to extract_features-------')
    theta = deberta_model.extract_features(DeBERTa_X, batch_size)      #从文本信息中提取的物品特征
    np.random.seed(133)
    U = np.random.uniform(size=(num_user, dimension))   #(num_user, dimension)，用户的嵌入向量
    V = theta                                           #(num_item, dimension)，物品的嵌入向量

    endure_count = 5
    count = 0       #早停策略，防止过拟合
    for iteration in range(max_iter):
        loss = 0
        tic = time.time()       #获取当前的时间戳
        print ("%d iteration\t(patience: %d)" % (iteration, count))

        VV = b * (V.T.dot(V)) + lambda_u * np.eye(dimension)  # V.T.dot(V)：V转置与V点积；np.eye：生成对角阵
        sub_loss = np.zeros(num_user)

        for i in range(num_user):
            idx_item = train_user[0][i]     # 获取用户i评分过的物品索引
            V_i = V[idx_item]               # 物品的隐向量
            R_i = Train_R_I[i]              # train_user[1][i], 用户i物品的评分
            A = VV + (a - b) * (V_i.T.dot(V_i))                     # 更新用户隐向量的线性方程组的系数矩阵
            B = (a * V_i * (np.tile(R_i, (dimension, 1)).T)).sum(0) 

            U[i] = np.linalg.solve(A, B)        # A * U[i] = B，更新用户i的特征向量

            sub_loss[i] = -0.5 * lambda_u * np.dot(U[i], U[i])  # 用L2范数计算用户i的损失

        loss = loss + np.sum(sub_loss)

        sub_loss = np.zeros(num_item)
        UU = b * (U.T.dot(U))
        for j in range(num_item):
            idx_user = train_item[0][j]     # 获取评分过物品j的用户的索引
            U_j = U[idx_user]               # 获取用户的隐向量
            R_j = Train_R_J[j]              # 获取用户对第J个物品的评分

            tmp_A = UU + (a - b) * (U_j.T.dot(U_j))
            A = tmp_A + lambda_v * item_weight[j] * np.eye(dimension)  # 更新物品隐向量的系数矩阵
            B = (a * U_j * (np.tile(R_j, (dimension, 1)).T)
                 ).sum(0) + lambda_v * item_weight[j] * theta[j]
            V[j] = np.linalg.solve(A, B)

            sub_loss[j] = -0.5 * np.square(R_j * a).sum()
            sub_loss[j] = sub_loss[j] + a * np.sum((U_j.dot(V[j])) * R_j)
            sub_loss[j] = sub_loss[j] - 0.5 * np.dot(V[j].dot(tmp_A), V[j])

        loss = loss + np.sum(sub_loss)
        seed = np.random.randint(100000)
        print('-----try to train DeBERTa_Model-----')
        history = deberta_model.train( inputs= DeBERTa_X, labels=V, item_weight=item_weight) # 训练deberta，更新deberta的权重参数
        print('-------try to extract_features-------')
        theta = deberta_model.extract_features(DeBERTa_X, batch_size)          # 获取deberta的投影层的输出，更新物品的隐向量
        print('-------try to get loss-------')
        DeBERTa_loss = history                 # 获取deberta训练的最后一轮的损失

        loss = loss - 0.5 * lambda_v * DeBERTa_loss * num_item

        tr_eval = eval_RMSE(Train_R_I, U, V, train_user[0])     # 计算训练集上的均方根误差
        val_eval = eval_RMSE(Valid_R, U, V, valid_user[0])
        te_eval = eval_RMSE(Test_R, U, V, test_user[0])

        toc = time.time()
        elapsed = toc - tic      #这一轮迭代的时长

        converge = abs((loss - PREV_LOSS) / PREV_LOSS)      #计算损失的相对变化，检查模型是否收敛

        if (val_eval < pre_val_eval):
            deberta_model.save_model(res_dir + '/DeBERTa_weights.hdf5')
            np.savetxt(res_dir + '/U.dat', U)
            np.savetxt(res_dir + '/V.dat', V)
            np.savetxt(res_dir + '/theta.dat', theta)
        else:
            count = count + 1

        pre_val_eval = val_eval

        print ("Loss: %.5f Elpased: %.4fs Converge: %.6f Tr: %.5f Val: %.5f Te: %.5f" % (
            loss, elapsed, converge, tr_eval, val_eval, te_eval))
        f1.write("Loss: %.5f Elpased: %.4fs Converge: %.6f Tr: %.5f Val: %.5f Te: %.5f\n" % (
            loss, elapsed, converge, tr_eval, val_eval, te_eval))

        if (count == endure_count):
            break

        PREV_LOSS = loss

    f1.close()
