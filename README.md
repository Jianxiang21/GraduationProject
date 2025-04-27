# 综合论文训练 / Graduation Project

## 📌 项目简介 / Project Overview

本项目旨在基于深度学习方法构建电力系统优化问题的预测模型，具体以 IEEE 118-bus 系统为研究对象。模型输入为负载分布，输出为最优发电方案及其对应的拉格朗日乘子，旨在通过数据驱动方式快速近似传统优化器的求解结果。

This project aims to develop a deep learning-based predictor for power system optimization solutions, focusing on the IEEE 118-bus system. The model takes the bus load vector as input and outputs the optimal power dispatch along with the corresponding Lagrange multipliers, offering a fast approximation to traditional optimization solvers.

---

## ✅ 当前进展 / Current Progress

- 构建并标准化训练数据集与验证集（验证集完全未见过）；
- 使用残差网络（ResNet）完成模型设计与训练；
- 模型预测结果包括发电机出力与多类约束的拉格朗日乘子；
- 项目结构清晰，代码模块化，便于扩展与评估;
- 发现模型对线性目标函数的表现较差，尝试将最优解单独拿出训练，最优解预测表现非常良好。
- 用验证集评估模型泛化性能；
- 构建primal dual的求解范式，求解LP和QP的算例，检测当前模型和求解模式的有效性

- Constructed standardized training and validation datasets (validation set is strictly unseen);
- Built and trained a ResNet-style model for prediction;
- Model outputs include optimal generator outputs and Lagrange multipliers for all constraints;
- Modular project structure for easy evaluation and further development；
- Observed poor performance on linear objective functions, but excellent results when training the model to predict the optimal solution separately.
- Evaluated model generalization using the validation set;
- Established a primal-dual solving paradigm for LP and QP problems, validating the effectiveness of the current model and solving approach.

---

## 🔧 下一步计划 / Next Steps

- 尝试数据增强，生成更多场景的数据
- 尝试计算其他优化问题算例

- Experiment with data augmentation to generate more diverse scenarios.
- Explore other optimization problem cases for model training and evaluation.
---
