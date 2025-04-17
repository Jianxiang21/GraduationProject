# 综合论文训练 / Graduation Project

## 📌 项目简介 / Project Overview

本项目旨在基于深度学习方法构建电力系统优化问题的预测模型，具体以 IEEE 118-bus 系统为研究对象。模型输入为负载分布，输出为最优发电方案及其对应的拉格朗日乘子，旨在通过数据驱动方式快速近似传统优化器的求解结果。

This project aims to develop a deep learning-based predictor for power system optimization solutions, focusing on the IEEE 118-bus system. The model takes the bus load vector as input and outputs the optimal power dispatch along with the corresponding Lagrange multipliers, offering a fast approximation to traditional optimization solvers.

---

## ✅ 当前进展 / Current Progress

- 构建并标准化训练数据集与验证集（验证集完全未见过）；
- 使用残差网络（ResNet）完成模型设计与训练；
- 模型预测结果包括发电机出力与多类约束的拉格朗日乘子；
- 项目结构清晰，代码模块化，便于扩展与评估。

- Constructed standardized training and validation datasets (validation set is strictly unseen);
- Built and trained a ResNet-style model for prediction;
- Model outputs include optimal generator outputs and Lagrange multipliers for all constraints;
- Modular project structure for easy evaluation and further development.

---

## 🔧 下一步计划 / Next Steps

- 使用验证集评估模型泛化性能；
- 可视化真实值与预测值的差异；
- 尝试改进模型结构或训练方法，提高准确性与鲁棒性。

- Evaluate model generalization using the validation set;
- Visualize prediction vs ground truth;
- Experiment with improved architectures or training strategies for better performance and robustness.

---
