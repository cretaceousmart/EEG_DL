import pandas as pd
import os
from skimage.transform import resize
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

def train_svm_model(X_train, y_train, X_test, y_test):
    print("Training SVM model...")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    # 定义 SVM 模型，使用 RBF 核
    model = SVC(kernel='rbf', C=1.0, gamma='scale')

    # 训练模型
    model.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = model.predict(X_test)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    # 计算召回率
    recall = recall_score(y_test, y_pred)
    print(f"Recall: {recall}")

    # 计算精确率
    precision = precision_score(y_test, y_pred)
    print(f"Precision: {precision}")

    # 计算 F1 分数
    f1 = f1_score(y_test, y_pred)
    print(f"F1 Score: {f1}")

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:\n{conf_matrix}")

    return model

# 假设 X_train, X_test, y_train, y_test 已经被定义
# model = train_svm_model(X_train, y_train, X_test, y_test)
