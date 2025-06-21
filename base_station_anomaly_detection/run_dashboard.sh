#!/bin/bash

# 基站坐标异常检测系统 - 启动脚本

echo "🚀 启动基站坐标异常检测系统..."
echo "=================================="

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "❌ Python未安装，请先安装Python"
    exit 1
fi

# 检查Streamlit
if ! python -c "import streamlit" &> /dev/null; then
    echo "📦 安装依赖包..."
    pip install -r requirements.txt
fi

# 检查数据文件
if [ ! -f "data/synthetic/sample_trajectory_data.csv" ]; then
    echo "📊 生成示例数据..."
    python examples/generate_data.py
fi

# 检查模型文件
if [ ! -f "models/statistical_detector_simple.joblib" ]; then
    echo "🤖 训练简化模型..."
    python examples/train_simple.py
fi

echo ""
echo "✅ 准备完成！"
echo "🌐 启动Web界面..."
echo ""
echo "访问地址: http://localhost:8501"
echo "按 Ctrl+C 停止服务"
echo ""

# 启动Streamlit应用
streamlit run examples/dashboard.py --server.port 8501 --server.address 0.0.0.0