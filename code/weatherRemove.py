"""
气象归一化分析系统
功能：分离自然因素和人为因素影响，生成分析报告
"""

# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates  # 新增关键导入
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import partial_dependence
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.base import clone
from tqdm import tqdm


class WeatherNormalizer:
    """气象归一化分析核心类"""

    def __init__(self, config):
        """
        初始化分析器
        :param config: 配置字典，新增：
            random_forest_params - 随机森林参数字典（可选）
        """
        # 设置随机森林默认参数
        self.default_rf_params = {
            "n_estimators": 100,
            "max_depth": 10,
            "n_jobs": -1,
            "random_state": 42,
            "max_features": "sqrt",
        }

        # 合并用户自定义参数
        self.rf_params = {
            **self.default_rf_params,
            **config.get("random_forest_params", {}),
        }

        # 原始配置项
        self.config = config
        self.model = None
        os.makedirs(config["output_dir"], exist_ok=True)

    def _preprocess_data(self, df):
        """数据预处理管道"""
        # 时间特征工程
        df["date"] = pd.to_datetime(df[self.config["time_col"]])
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["day"] = df["date"].dt.day
        df["hour"] = df["date"].dt.hour
        df["weekday"] = df["date"].dt.weekday
        df["dayofyear"] = df["date"].dt.dayofyear

        # 数据过滤
        features = [
            c
            for c in df.columns
            if c not in self.config["exclude_cols"] + [self.config["target_col"]]
        ]

        return df.dropna(), features

    def train_model(self, df):
        """训练气象归一化模型（更新参数传递方式）"""
        cleaned_df, features = self._preprocess_data(df)

        # 准备数据
        X = cleaned_df[features]
        y = cleaned_df[self.config["target_col"]]

        # 训练随机森林模型（使用配置参数）
        self.model = RandomForestRegressor(**self.rf_params)
        self.model.fit(X, y)

        # 保存模型（新增参数元数据）
        model_meta = {
            "model_params": self.rf_params,
            "features": features,
            "target": self.config["target_col"],
        }
        joblib.dump(
            {"model": self.model, "meta": model_meta},
            os.path.join(self.config["output_dir"], "normalization_model.pkl"),
        )

    def analyze_results(self, df):
        """生成分析结果"""
        cleaned_df, features = self._preprocess_data(df)
        X = cleaned_df[features]

        # 生成预测值
        cleaned_df["human_effect"] = self.model.predict(X)
        cleaned_df["natural_effect"] = (
                cleaned_df[self.config["target_col"]] - cleaned_df["human_effect"]
        )

        # 保存结果
        result_path = os.path.join(self.config["output_dir"], "analysis_results.csv")
        cleaned_df.to_csv(result_path, index=False)

        # 生成模型评估
        self._generate_model_report(cleaned_df)

        return cleaned_df

    def _generate_model_report(self, df):
        """生成模型评估报告"""
        metrics = {
            "R2": r2_score(df[self.config["target_col"]], df["human_effect"]),
            "RMSE": np.sqrt(
                mean_squared_error(df[self.config["target_col"]], df["human_effect"])
            ),
        }
        pd.DataFrame([metrics]).to_csv(
            os.path.join(self.config["output_dir"], "model_metrics.csv"), index=False
        )

    def visualize_analysis(self, df, months_per_image=4):
        """支持配置单图显示月份数量的可视化分析"""
        # 日期预处理
        first_col_name = df.columns[0]
        if not np.issubdtype(df["date"].dtype, np.datetime64):
            df["date"] = pd.to_datetime(df["date"])

        # 颜色配置
        colors = {
            "human_effect": "#4B9AC9",  # 蓝
            "natural_effect": "#4BC96B",  # 绿
            "target_line": "#FF6B6B"  # 红
        }

        stations = df[first_col_name].unique()
        pbar = tqdm(total=len(stations), desc="Processing Stations/Province",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}")

        def plot_grouped_months(ax, df_group):
            """月份组绘图核心逻辑（含自动柱宽调整）"""
            # 日期数值转换与间隔计算（参考网页3的数值转换方法）
            dates = mdates.date2num(df_group["date"])

            # 自动计算柱宽（核心改造点）
            if len(dates) > 1:
                # 计算最小日期间隔（处理跨月数据）
                day_interval = np.diff(dates).min()  # 获取最小时间间隔
                auto_width = day_interval * 0.8  # 保留20%间距（参考网页6的间距建议）
            else:
                auto_width = 0.8  # 单数据点默认宽度

            # 堆叠柱状图（根据网页1的width参数调整方案）
            # 在绘制柱状图时添加label参数[1,6](@ref)
            bars = ax.bar(dates, df_group["human_effect"],
                          width=auto_width, color=colors["human_effect"],
                          label='Human Effect')  # 新增label

            ax.bar(dates, df_group["natural_effect"],
                   width=auto_width, bottom=df_group["human_effect"],
                   color=colors["natural_effect"],
                   label='Natural Effect')  # 新增label

            # 在绘制折线图时添加label参数[1](@ref)
            line = ax.plot(dates, df_group[self.config["target_col"]],
                           color=colors["target_line"], marker='o', markersize=4,
                           linestyle='--', linewidth=1.5,
                           label='Target Line')  # 新增label

            # 在函数末尾添加图例配置[6,8](@ref)
            ax.legend(loc='upper right',
                      frameon=True,
                      fontsize=8,
                      ncol=1,
                      title='Legend',
                      title_fontsize=9,
                      borderpad=0.8,
                      handlelength=1.5)

            # 坐标轴格式化
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

            # 数据标签
            for bar, y1, y2 in zip(bars, df_group["human_effect"], df_group["natural_effect"]):
                height = y1 + y2
                ax.text(bar.get_x() + bar.get_width() / 2, height + 0.1,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=6)

        # 主处理逻辑
        for station in stations:
            station_df = df[df[first_col_name] == station].sort_values('date')
            all_months = sorted(station_df["month"].unique())

            # 月份分组处理
            month_groups = [all_months[i:i + months_per_image]
                            for i in range(0, len(all_months), months_per_image)]

            for group_idx, month_group in enumerate(month_groups):
                # 动态计算子图布局（参考网页6/7的subplots方案）
                n_months = len(month_group)
                # 向下取整
                n_rows = int(np.floor(np.sqrt(n_months)))
                n_cols = int(np.ceil(n_months / n_rows))

                fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 10, n_rows * 6))
                axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

                # 绘制每个月份的图表
                for i, month in enumerate(month_group):
                    df_month = station_df[station_df["month"] == month]
                    plot_grouped_months(axes[i], df_month)
                    axes[i].set_title(f"Month: {month}", fontsize=10)

                # 隐藏空子图（参考网页8的布局优化）
                for j in range(i + 1, len(axes)):
                    axes[j].set_visible(False)

                # 整体布局调整（参考网页8的subplots_adjust）
                plt.subplots_adjust(wspace=0.3, hspace=0.4)
                plt.suptitle(
                    f"Station(Province) {station} - Group {group_idx + 1} ({min(month_group)}-{max(month_group)})",
                    y=0.98, fontsize=12)

                # 保存输出
                output_path = os.path.join(
                    self.config["output_dir"],
                    f"Station(Province)_{station}_Group{group_idx + 1}.png"
                )
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close()

            pbar.update(1)
        pbar.close()

    def partial_dependence_analysis(self, df):
        """部分依赖分析"""
        cleaned_df, features = self._preprocess_data(df)

        # 生成部分依赖图
        for feature in self.config["feature_cols"]:
            if feature in features:
                self._plot_partial_dependence(feature, cleaned_df)

    def _plot_partial_dependence(self, feature, df):
        """绘制单个特征的部分依赖图"""
        predictions = partial_dependence(
            self.model,
            df[self.config["feature_cols"]],
            features=[feature],
            grid_resolution=20,
        )
        print(predictions)
        pdp = predictions["average"]
        axes = predictions["grid_values"]
        # pdp = predictions[0]
        # axes = predictions[1]

        plt.figure(figsize=(10, 6))
        plt.plot(axes[0], pdp[0])
        plt.xlabel(feature)
        plt.ylabel("Partial Dependence")
        plt.title(f"{feature} Partial Dependence")
        plt.savefig(os.path.join(self.config["output_dir"], f"pdp_{feature}.png"))
        plt.close()


def main(input_path, config=None, province_mode=False):
    """主流程控制器"""
    # 配置参数（可根据实际数据调整）
    if config is None:
        config = {
            "target_col": "PM25",
            "time_col": "Date",
            "feature_cols": [
                "Mean_Pressure",
                "Mean_Humidity",
                "Mean_Temperature",
                "Mean_Wind_Speed",
                "Max_Wind_Direction",
                "dayofyear",
            ],
            "exclude_cols": ["Station_Code", "Station_Name", "v1", "Date", "date"],
            "output_dir": "../result",
            # 新增随机森林配置项（可调整以下参数）
            "random_forest_params": {
                "n_estimators": 200,  # 决策树数量（默认：100）
                "max_depth": 15,  # 最大深度（默认：10）
                # "min_samples_split": 20,  # 节点分裂最小样本（默认：2）
                # "max_features": "log2",  # 特征选择方式（默认："sqrt"）
                "n_jobs": -1,  # 并行计算（默认：-1）
                "random_state": 42,  # 随机种子（默认：42）
            }
        }

    # 加载数据
    raw_data = pd.read_csv(input_path)

    # 若不为省分析模式则把Station_Code放在第一列
    if not province_mode:
        # 明确指定列顺序
        cols = ["Station_Code", "Station_Name"] + [col for col in raw_data.columns if
                                                   col not in {"Station_Code", "Station_Name"}]
        raw_data = raw_data[cols]

    # 取前100行

    # raw_data = raw_data[:1000]

    # 初始化分析器
    analyzer = WeatherNormalizer(config)

    # 训练模型
    analyzer.train_model(raw_data)

    # 生成结果
    results = analyzer.analyze_results(raw_data)

    # 可视化分析
    analyzer.visualize_analysis(results)
    analyzer.partial_dependence_analysis(results)

    print("分析完成！结果保存在:", config["output_dir"])


if __name__ == "__main__":
    # 示例数据路径
    sample_data = {
        "Key": [1.0] * 5,
        "Name": ["万寿西宫"] * 5,
        "Date": ["2018-01-01", "2018-01-02", "2018-01-03", "2018-01-04", "2018-01-05"],
        "apm": [39.5625, 24.291667, 9.458333, 13.782609, 40.3],
        "wdm": [3.0, 3.0, 2.0, 3.0, 4.0],
        "wsm": [11.0, 20.0, 21.0, 22.0, 15.0],
        "tm": [-50.0, -45.0, -56.0, -54.0, -73.0],
        "map": [10192.0, 10261.0, 10310.0, 10256.0, 10185.0],
        "v1": ["PM25"] * 5,
    }

    # 生成测试数据
    pd.DataFrame(sample_data).to_csv("sample_input.csv", index=False)

    # 执行分析
    main(r"..\result\Matched_Data\2019\PM25\matched_data_with_PM25.csv")
