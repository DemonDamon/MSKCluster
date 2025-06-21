"""
基站坐标异常检测系统 - 可视化面板
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging

from src.models.statistical_detector import StatisticalAnomalyDetector
from src.data.generator import TrajectoryGenerator
from src.utils.geo_utils import calculate_distance

# 配置页面
st.set_page_config(
    page_title="基站坐标异常检测系统",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@st.cache_data
def load_sample_data():
    """加载示例数据"""
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'synthetic', 'sample_trajectory_data.csv')
    
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df.head(1000)  # 只加载前1000条数据用于演示
    else:
        return None


@st.cache_resource
def load_model():
    """加载训练好的模型"""
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'statistical_detector_simple.joblib')
    
    if os.path.exists(model_path):
        detector = StatisticalAnomalyDetector()
        detector.load_model(model_path)
        return detector
    else:
        return None


def generate_test_data(num_points=100, anomaly_rate=0.05):
    """生成测试数据"""
    generator = TrajectoryGenerator()
    
    # 生成正常轨迹
    start_time = datetime.now()
    trajectory = generator.generate_normal_trajectory(
        start_time, duration_hours=num_points/12, user_type='commuter'
    )
    
    # 注入异常
    trajectory = generator.inject_anomalies(trajectory)
    
    return trajectory.head(num_points)


def preprocess_data(df):
    """预处理数据"""
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # 计算基础特征
    df['prev_lat'] = df['latitude'].shift(1)
    df['prev_lon'] = df['longitude'].shift(1)
    df['time_diff'] = df['timestamp'].diff().dt.total_seconds().fillna(0)
    
    # 计算距离和速度
    df['distance'] = df.apply(
        lambda row: calculate_distance(
            row['prev_lat'], row['prev_lon'],
            row['latitude'], row['longitude']
        ) if not pd.isna(row['prev_lat']) else 0,
        axis=1
    )
    
    df['speed'] = df['distance'] / df['time_diff'].replace(0, np.nan)
    df['speed'] = df['speed'].fillna(0)
    df['speed_kmh'] = df['speed'] * 3.6
    
    # 计算加速度
    df['prev_speed'] = df['speed'].shift(1)
    df['acceleration'] = (df['speed'] - df['prev_speed']) / df['time_diff'].replace(0, np.nan)
    df['acceleration'] = df['acceleration'].fillna(0)
    
    # 时间特征
    df['hour'] = df['timestamp'].dt.hour
    df['is_daytime'] = (df['hour'] >= 7) & (df['hour'] < 22)
    
    return df


def run_detection(detector, df):
    """运行异常检测"""
    if detector is None:
        st.error("模型未加载，请先训练模型")
        return df
    
    # 准备特征
    feature_cols = ['distance', 'speed_kmh', 'acceleration', 'hour', 'is_daytime', 'latitude', 'longitude']
    X = df[['timestamp'] + feature_cols]
    
    # 预测
    y_pred = detector.predict(X)
    y_scores = detector.predict_proba(X)
    
    # 添加预测结果
    df['predicted'] = y_pred
    df['anomaly_score'] = y_scores
    
    return df


def main():
    """主函数"""
    st.title("📡 基站坐标异常检测系统")
    st.markdown("---")
    
    # 侧边栏
    st.sidebar.title("系统控制")
    
    # 数据源选择
    data_source = st.sidebar.selectbox(
        "选择数据源",
        ["示例数据", "生成测试数据", "上传数据"]
    )
    
    # 加载数据
    df = None
    
    if data_source == "示例数据":
        df = load_sample_data()
        if df is None:
            st.sidebar.error("示例数据不存在，请先运行 generate_data.py")
    
    elif data_source == "生成测试数据":
        num_points = st.sidebar.slider("数据点数量", 50, 500, 100)
        anomaly_rate = st.sidebar.slider("异常率", 0.01, 0.1, 0.05)
        
        if st.sidebar.button("生成数据"):
            with st.spinner("生成测试数据..."):
                df = generate_test_data(num_points, anomaly_rate)
                st.sidebar.success("数据生成完成！")
    
    elif data_source == "上传数据":
        uploaded_file = st.sidebar.file_uploader(
            "上传CSV文件",
            type=['csv'],
            help="文件应包含 timestamp, latitude, longitude 列"
        )
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    if df is not None:
        # 数据预处理
        with st.spinner("预处理数据..."):
            df = preprocess_data(df)
        
        # 加载模型
        detector = load_model()
        
        # 运行检测
        if st.sidebar.button("运行异常检测"):
            with st.spinner("运行异常检测..."):
                df = run_detection(detector, df)
                st.sidebar.success("检测完成！")
        
        # 显示结果
        display_results(df)
    
    else:
        st.info("请选择数据源并加载数据")


def display_results(df):
    """显示检测结果"""
    
    # 基本统计
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("总数据点", len(df))
    
    with col2:
        if 'label' in df.columns:
            true_anomalies = df['label'].sum()
            st.metric("真实异常点", true_anomalies)
        else:
            st.metric("真实异常点", "未知")
    
    with col3:
        if 'predicted' in df.columns:
            predicted_anomalies = df['predicted'].sum()
            st.metric("预测异常点", predicted_anomalies)
        else:
            st.metric("预测异常点", "未检测")
    
    with col4:
        if 'anomaly_score' in df.columns:
            avg_score = df['anomaly_score'].mean()
            st.metric("平均异常分数", f"{avg_score:.3f}")
        else:
            st.metric("平均异常分数", "未计算")
    
    # 标签页
    tab1, tab2, tab3, tab4 = st.tabs(["轨迹地图", "时间序列", "数据表格", "统计分析"])
    
    with tab1:
        display_trajectory_map(df)
    
    with tab2:
        display_time_series(df)
    
    with tab3:
        display_data_table(df)
    
    with tab4:
        display_statistics(df)


def display_trajectory_map(df):
    """显示轨迹地图"""
    st.subheader("轨迹地图")
    
    # 创建地图
    fig = go.Figure()
    
    # 正常点
    normal_points = df[df.get('label', 0) == 0]
    if len(normal_points) > 0:
        fig.add_trace(go.Scattermapbox(
            lat=normal_points['latitude'],
            lon=normal_points['longitude'],
            mode='markers',
            marker=dict(size=8, color='blue'),
            name='正常点',
            text=normal_points.index
        ))
    
    # 真实异常点
    if 'label' in df.columns:
        true_anomalies = df[df['label'] == 1]
        if len(true_anomalies) > 0:
            fig.add_trace(go.Scattermapbox(
                lat=true_anomalies['latitude'],
                lon=true_anomalies['longitude'],
                mode='markers',
                marker=dict(size=15, color='red', symbol='x'),
                name='真实异常点',
                text=true_anomalies.index
            ))
    
    # 预测异常点
    if 'predicted' in df.columns:
        predicted_anomalies = df[df['predicted'] == 1]
        if len(predicted_anomalies) > 0:
            fig.add_trace(go.Scattermapbox(
                lat=predicted_anomalies['latitude'],
                lon=predicted_anomalies['longitude'],
                mode='markers',
                marker=dict(size=12, color='orange'),
                name='预测异常点',
                text=predicted_anomalies.index
            ))
    
    # 轨迹线
    fig.add_trace(go.Scattermapbox(
        lat=df['latitude'],
        lon=df['longitude'],
        mode='lines',
        line=dict(width=2, color='gray'),
        name='轨迹',
        showlegend=False
    ))
    
    # 设置地图
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()
    
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=10
        ),
        height=600,
        margin=dict(l=0, r=0, t=0, b=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)


def display_time_series(df):
    """显示时间序列"""
    st.subheader("时间序列分析")
    
    # 异常分数时间序列
    if 'anomaly_score' in df.columns:
        fig_score = px.line(df, x='timestamp', y='anomaly_score', 
                           title='异常分数时间序列')
        
        # 添加真实异常点
        if 'label' in df.columns:
            true_anomalies = df[df['label'] == 1]
            if len(true_anomalies) > 0:
                fig_score.add_scatter(
                    x=true_anomalies['timestamp'],
                    y=true_anomalies['anomaly_score'],
                    mode='markers',
                    marker=dict(size=10, color='red', symbol='x'),
                    name='真实异常点'
                )
        
        # 添加阈值线
        fig_score.add_hline(y=0.5, line_dash="dash", line_color="red", 
                           annotation_text="检测阈值")
        
        st.plotly_chart(fig_score, use_container_width=True)
    
    # 速度时间序列
    if 'speed_kmh' in df.columns:
        fig_speed = px.line(df, x='timestamp', y='speed_kmh', 
                           title='速度时间序列')
        
        # 添加真实异常点
        if 'label' in df.columns:
            true_anomalies = df[df['label'] == 1]
            if len(true_anomalies) > 0:
                fig_speed.add_scatter(
                    x=true_anomalies['timestamp'],
                    y=true_anomalies['speed_kmh'],
                    mode='markers',
                    marker=dict(size=10, color='red', symbol='x'),
                    name='真实异常点'
                )
        
        st.plotly_chart(fig_speed, use_container_width=True)


def display_data_table(df):
    """显示数据表格"""
    st.subheader("数据表格")
    
    # 过滤选项
    col1, col2 = st.columns(2)
    
    with col1:
        show_anomalies_only = st.checkbox("只显示异常点")
    
    with col2:
        if 'predicted' in df.columns:
            show_predicted_only = st.checkbox("只显示预测异常点")
        else:
            show_predicted_only = False
    
    # 过滤数据
    display_df = df.copy()
    
    if show_anomalies_only and 'label' in df.columns:
        display_df = display_df[display_df['label'] == 1]
    
    if show_predicted_only and 'predicted' in df.columns:
        display_df = display_df[display_df['predicted'] == 1]
    
    # 选择显示的列
    display_cols = ['timestamp', 'latitude', 'longitude']
    
    if 'speed_kmh' in df.columns:
        display_cols.append('speed_kmh')
    if 'distance' in df.columns:
        display_cols.append('distance')
    if 'label' in df.columns:
        display_cols.append('label')
    if 'predicted' in df.columns:
        display_cols.append('predicted')
    if 'anomaly_score' in df.columns:
        display_cols.append('anomaly_score')
    
    st.dataframe(display_df[display_cols], use_container_width=True)


def display_statistics(df):
    """显示统计分析"""
    st.subheader("统计分析")
    
    # 性能指标
    if 'label' in df.columns and 'predicted' in df.columns:
        st.subheader("检测性能")
        
        tp = ((df['label'] == 1) & (df['predicted'] == 1)).sum()
        fp = ((df['label'] == 0) & (df['predicted'] == 1)).sum()
        tn = ((df['label'] == 0) & (df['predicted'] == 0)).sum()
        fn = ((df['label'] == 1) & (df['predicted'] == 0)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("精确率", f"{precision:.3f}")
        with col2:
            st.metric("召回率", f"{recall:.3f}")
        with col3:
            st.metric("F1分数", f"{f1:.3f}")
        
        # 混淆矩阵
        st.subheader("混淆矩阵")
        confusion_data = pd.DataFrame({
            '预测正常': [tn, fn],
            '预测异常': [fp, tp]
        }, index=['实际正常', '实际异常'])
        
        st.dataframe(confusion_data)
    
    # 数据分布
    st.subheader("数据分布")
    
    if 'speed_kmh' in df.columns:
        fig_hist = px.histogram(df, x='speed_kmh', title='速度分布')
        st.plotly_chart(fig_hist, use_container_width=True)
    
    if 'anomaly_score' in df.columns:
        fig_score_hist = px.histogram(df, x='anomaly_score', title='异常分数分布')
        st.plotly_chart(fig_score_hist, use_container_width=True)


if __name__ == "__main__":
    main()