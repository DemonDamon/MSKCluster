# 基站坐标异常值检测系统

基于时间序列坐标数据的基站定位异常检测算法实现，针对高度不平衡数据（异常率约0.17%）进行优化。

## 项目结构

```
base_station_anomaly_detection/
├── README.md                    # 项目说明
├── requirements.txt             # 依赖包
├── config/                      # 配置文件
│   └── config.yaml             # 算法参数配置
├── src/                        # 源代码
│   ├── __init__.py
│   ├── data/                   # 数据处理模块
│   │   ├── __init__.py
│   │   ├── preprocessor.py     # 数据预处理
│   │   └── generator.py        # 数据生成器
│   ├── models/                 # 算法模型
│   │   ├── __init__.py
│   │   ├── base_detector.py    # 基础检测器
│   │   ├── statistical_detector.py  # 统计方法检测器
│   │   ├── trajectory_detector.py   # 轨迹分析检测器
│   │   ├── clustering_detector.py   # 聚类检测器
│   │   └── ensemble_detector.py     # 集成检测器
│   ├── evaluation/             # 评估模块
│   │   ├── __init__.py
│   │   ├── metrics.py          # 评估指标
│   │   └── badcase_analyzer.py # Badcase分析
│   ├── visualization/          # 可视化模块
│   │   ├── __init__.py
│   │   ├── plotter.py          # 绘图工具
│   │   └── dashboard.py        # 可视化面板
│   └── utils/                  # 工具模块
│       ├── __init__.py
│       ├── geo_utils.py        # 地理计算工具
│       └── time_utils.py       # 时间处理工具
├── examples/                   # 示例代码
│   ├── train_model.py          # 训练示例
│   ├── inference.py            # 推理示例
│   └── evaluation.py           # 评估示例
├── tests/                      # 测试代码
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_models.py
│   └── test_evaluation.py
└── data/                       # 数据目录
    ├── raw/                    # 原始数据
    ├── processed/              # 处理后数据
    └── synthetic/              # 合成数据
```

## 核心功能

1. **多种异常检测算法**
   - 基于统计的速度/加速度阈值检测
   - 轨迹平滑与残差分析
   - 密度聚类异常检测
   - 集成检测器

2. **针对不平衡数据优化**
   - 动态阈值调整（白天/夜间）
   - 代价敏感学习
   - F2-Score优化

3. **完整的评估体系**
   - 多维度评估指标
   - Badcase深度分析
   - 可视化分析工具

4. **实用工具**
   - 数据预处理
   - 特征工程
   - 模型训练和推理
   - 结果可视化

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 生成示例数据
python examples/generate_data.py

# 训练模型
python examples/train_model.py

# 进行推理
python examples/inference.py

# 评估结果
python examples/evaluation.py
```