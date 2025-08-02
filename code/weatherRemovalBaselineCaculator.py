import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
import os
import math
from tqdm import tqdm
from scipy.stats import ks_2samp


class EnhancedBaseLineCalculator:
    COLORS = {"human": "#1f77b4", "natural": "#ff7f0e"}

    def __init__(self, year: str, pollutant: str, target_dir: str, r2_threshold: float = 0.95, min_valid_count: int = 5):
        self.year = year
        self.pollutant = pollutant
        self.target_dir = target_dir
        self.r2_threshold = r2_threshold      # R²阈值，默认0.95
        self.min_valid_count = min_valid_count  # 若有效数据点不足，则不做截断
        self._create_dirs()

    def _create_dirs(self):
        os.makedirs(f"{self.target_dir}/infl_draw_1", exist_ok=True)
        os.makedirs(f"{self.target_dir}/infl_draw_2", exist_ok=True)

    def load_data(self, csv_path: str):
        raw_df = pd.read_csv(csv_path, parse_dates=["Date"])
        # 取前10000行数据
        # raw_df = raw_df.head(10000)
        self.stations = {}
        first_col_name = raw_df.columns[0]

        # 按站点重组数据
        grouped = raw_df.groupby(first_col_name)
        for station, group in tqdm(grouped, desc="Loading stations/province"):
            sorted_group = group.sort_values("Date")
            self.stations[station] = {
                "human": sorted_group["human_effect"].values,
                "natural": sorted_group["natural_effect"].values,
                "name": sorted_group[first_col_name].iloc[0],
            }

    def _clean_data(self, data: np.ndarray) -> np.ndarray:
        """数据清洗：去除NaN和异常值"""
        cleaned = data[~np.isnan(data)]
        if len(cleaned) < 10:
            return np.array([])

        # 3sigma去极值
        mean, std = cleaned.mean(), cleaned.std()
        return cleaned[(cleaned >= mean - 3 * std) & (cleaned <= mean + 3 * std)]

    def _calc_r2_sequence(self, sorted_data: np.ndarray) -> list:
        """计算R²累积序列"""
        r2_seq = []
        for i in range(2, len(sorted_data)):
            x = np.arange(i)
            slope, _, r_value, _, _ = st.linregress(x, sorted_data[:i])
            r2_seq.append(r_value**2)
        return r2_seq

    def __youxiao(self, data_list: list):
        """
        优化后的R²序列有效性清洗：
        从后往前删除低于设定阈值的数据，但如果满足阈值的数据点数量少于min_valid_count，则保留原序列
        """
        threshold = self.r2_threshold
        valid_count = sum(1 for x in data_list if x >= threshold and not math.isnan(x))
        if valid_count < self.min_valid_count:
            # 有效数据不足，直接返回原序列（拷贝避免修改原列表）
            return data_list.copy()
        
        trimmed_list = data_list.copy()
        for i in range(len(trimmed_list) - 1, -1, -1):
            if trimmed_list[i] < threshold or math.isnan(trimmed_list[i]):
                trimmed_list.pop(i)
            else:
                break
        return trimmed_list

    def __get_inflection(self, data_list: list):
        """四分位后的最高点为拐点"""
        return max(data_list[len(data_list) // 6 :])

    def __inflection_2_exist(self, latter_data: list):
        latter_r2 = self.__youxiao(self._calc_r2_sequence(latter_data))
        if len(latter_r2) < 5:
            return False

        flection_2 = self.__get_inflection(latter_r2)
        tmp_index2 = latter_r2.index(flection_2)

        if tmp_index2 <= (len(latter_r2) // 6) + 2 or tmp_index2 >= len(latter_r2) - 3:
            return False
        return tmp_index2


    def _find_inflections(self, r2_seq: list, sorted_data: list) -> dict:
        """拐点查找逻辑"""
        # 有效性清洗
        cleaned_r2 = self.__youxiao(r2_seq.copy())
        len_youxiao = len(cleaned_r2)
        if not cleaned_r2:
            return {"inflection1": 0, "inflection2": 0}

        # 查找第一拐点
        flection1_val = self.__get_inflection(cleaned_r2)
        inf1 = cleaned_r2.index(flection1_val)

        # 查找第二拐点
        latter_data = sorted_data[inf1 + 1 :len_youxiao]
        inf2_relative = self.__inflection_2_exist(latter_data)

        if inf2_relative is not False:
            inf2 = inf1 + 1 + inf2_relative
            # 全局位置校验
            # if inf2 > len(sorted_data) * 0.8:  # 后20%不认作有效拐点
            #     inf2 = inf1
        else:
            inf2 = inf1

        return {"inflection1": inf1, "inflection2": inf2}
    
    # 基线选择器，根据拐点选择基线
    def _compare_distributions(self, sorted_data: np.ndarray, inf1: int, inf2: int) -> str:
        """比较分布相似性，返回选择的拐点标记"""
        # 数据分段
        pre_A = sorted_data[:inf1+1]         # 拐点1之前的区域
        mid_AB = sorted_data[inf1+1:inf2+1]  # 两个拐点之间的区域
        post_B = sorted_data[inf2+1:]        # 拐点2之后的区域
        
        # 有效性检查
        if len(mid_AB) < 10 or len(pre_A) < 10 or len(post_B) < 10:
            return "inf1"  # 数据不足时默认选第一个拐点
        
        # 执行K-S检验比较分布
        
        stat_pre, _ = ks_2samp(mid_AB, pre_A)
        stat_post, _ = ks_2samp(mid_AB, post_B)
        
        # 选择更相似的区间
        return "inf1" if stat_pre < stat_post else "inf2"


    # 修改 _plot_scatter 方法增加容错
    def _plot_scatter(self, station: str, data: dict):
        """绘制双指标拐点散点图（支持两个拐点）"""
        plt.figure(figsize=(10, 6))
        
        for metric in ['human', 'natural']:
            if data[metric].get('error'):
                continue  # 跳过有错误的数据
                
            r2_seq = data[metric].get('r2_seq', [])
            if not r2_seq:  # 空数据保护
                continue
                
            color = self.COLORS[metric]
            plt.plot(r2_seq, color=color, label=f'{metric.capitalize()} Effect', alpha=0.7)
            
            # 绘制第一个拐点
            inf1 = data[metric].get('inf1', -1)
            if inf1 >= 0 and inf1 < len(r2_seq):
                plt.scatter(inf1, r2_seq[inf1], 
                        color=color, marker='o' if metric=='human' else 's', 
                        edgecolor='k', label=f'{metric.capitalize()} Inflection 1')
            
            # 绘制第二个拐点
            inf2 = data[metric].get('inf2', -1)
            if inf2 >= 0 and inf2 < len(r2_seq) and inf2 != inf1:  # 确保拐点2有效且不与拐点1重合
                plt.scatter(inf2, r2_seq[inf2], 
                        color=color, marker='^' if metric=='human' else 'D',  # 使用不同标记区分拐点
                        edgecolor='k', label=f'{metric.capitalize()} Inflection 2')
        
        plt.title(f"{station} - {self.year} R² Sequence")
        plt.xlabel("Sample Index")
        plt.ylabel("R² Value")
        plt.legend()
        plt.savefig(f"{self.target_dir}/infl_draw_1/{station}-{self.year}-inflection1.png", dpi=150)
        plt.close()

    def _plot_histograms(self, station: str, data: dict):
        """增强累计频率图（添加累计曲线），根据拐点进行分段绘图
        若无拐点2则仅以拐点1分割，有则以两个拐点分割
        """
        # 判断是否存在第二拐点
        # 这里假定若inf2与inf1相等或为NaN，则认为不存在第二拐点
        if "error" in data["human"] or data["human"]["inf2"] == data["human"]["inf1"] or np.isnan(data["human"]["inf2"]):
            segments = [
                ("left", 0, data["human"]["inf1"]),
                ("right", data["human"]["inf1"], None)
            ]
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        else:
            segments = [
                ("left", 0, data["human"]["inf1"]),
                ("middle", data["human"]["inf1"], data["human"]["inf2"]),
                ("right", data["human"]["inf2"], None)
            ]
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        n_bins = 15

        for ax, (seg_name, start, end) in zip(axes, segments):
            # 处理human数据
            h_seg = data["human"]["sorted"][start:end]
            n, bins, _ = ax.hist(h_seg, bins=n_bins, color=self.COLORS["human"], alpha=0.5, density=True)

            # 添加累计频率曲线
            cumulative = np.cumsum(n) / np.sum(n)
            ax2 = ax.twinx()
            ax2.plot(bins[:-1], cumulative, "o-", color=self.COLORS["human"], markersize=4)
            ax2.set_ylim(0, 1)

            # 处理natural数据（按比例映射至相同分布区间）
            n_total = len(data["natural"]["sorted"])
            total_h = len(data["human"]["sorted"])
            n_start = int(start * (n_total / total_h))
            n_end = int(end * (n_total / total_h)) if end is not None else None
            n_seg = data["natural"]["sorted"][n_start:n_end]

            n_nat, bins_nat, _ = ax.hist(n_seg, bins=n_bins, color=self.COLORS["natural"], alpha=0.5, density=True)

            # 绘制natural数据累计频率曲线
            cumulative_nat = np.cumsum(n_nat) / np.sum(n_nat)
            ax2.plot(bins_nat[:-1], cumulative_nat, "s-", color=self.COLORS["natural"], markersize=4)

            ax.set_title(f"{seg_name.capitalize()} Segment")
            ax.set_xlabel("Effect Value")
            ax.set_ylabel("Density")
            ax2.set_ylabel("Cumulative Freq")

        plt.suptitle(f"{station} - {self.year} Distribution Comparison")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        # 添加label
        plt.figlegend(labels=["Human Effect", "Natural Effect"], loc="lower center", ncol=2)
        plt.savefig(f"{self.target_dir}/infl_draw_2/{station}-{self.year}-distribution.png", dpi=150)
        plt.close()


    def process_station(self, station: str) -> dict:
        """处理单个站点（修复键缺失问题）"""
        result = {'human': {'error': None}, 'natural': {'error': None}}  # 初始化默认结构
        
        for metric in ['human', 'natural']:
            # 数据清洗
            cleaned = self._clean_data(self.stations[station][metric])
            if len(cleaned) < 30:  # 数据量不足时跳过
                result[metric]['error'] = 'Insufficient data'
                # 初始化必要字段避免KeyError
                result[metric].update({
                    'sorted': np.array([]),
                    'r2_seq': [],
                    'inf1': -1,
                    'inf2': -1
                })
                continue
                
            # 排序数据
            sorted_data = np.sort(cleaned)
            
            # 计算R²序列
            r2_seq = self._calc_r2_sequence(sorted_data)
            
            # 查找拐点
            try:
                infs = self._find_inflections(r2_seq, sorted_data)
            except Exception as e:
                print(f"拐点计算错误: {str(e)}")
                infs = {'inflection1': 0, 'inflection2': 0}
            
            result[metric].update({
                'sorted': sorted_data,
                'r2_seq': r2_seq,
                'inf1': infs['inflection1'],
                'inf2': infs['inflection2']
            })
        
        return result

    # 修改generate_reports方法中的记录基线部分
    def generate_reports(self):
        """调整基线输出格式"""
        baseline_df = []

        for station in tqdm(self.stations, desc="Processing stations"):
            station_data = self.process_station(station)
            if station_data["human"]['error'] is not None or station_data["natural"]['error'] is not None:
                print(f"{station} Error!{station_data['human']['error']}")
                continue

            # 生成图表（保持不变）
            self._plot_scatter(station, station_data)
            self._plot_histograms(station, station_data)

            # 记录基线
            for metric in ["human", "natural"]:
                if station_data[metric]['error'] is not None:
                    continue

                data = station_data[metric]
                # 获取排序后的数据
                sorted_data = data["sorted"]
                
                # 新增分布相似性判断
                final_choice = self._compare_distributions(
                    sorted_data, 
                    data["inf1"], 
                    data["inf2"]
                )
                
                # 根据选择计算最终基线
                if final_choice == "inf1":
                    final_baseline = np.mean(sorted_data[:data["inf1"]+1])
                    final_inf = data["inf1"]
                else:
                    final_baseline = np.mean(sorted_data[:data["inf2"]+1])
                    final_inf = data["inf2"]

                baseline_df.append({
                    "Station/Province": station,
                    "Metric": metric,
                    "拐点1位置": data["inf1"],
                    "拐点2位置": data["inf2"],
                    "最终选择拐点": final_choice,
                    "最终基线值": final_baseline,
                    "分布相似性判断": "更接近前段" if final_choice == "inf1" else "更接近后段"
                })

        # 保存为程序2格式（修改列名）
        pd.DataFrame(baseline_df).to_csv(
            f"{self.target_dir}/{self.year}_分省(站点)基线.csv",
            index=False,
            columns=[
                "Station/Province", 
                "Metric", 
                "拐点1位置",
                "拐点2位置",
                "最终选择拐点",
                "最终基线值",
                "分布相似性判断"
            ],
        )


# 使用示例
if __name__ == "__main__":
    processor = EnhancedBaseLineCalculator(
        year="2019", pollutant="O3", target_dir="../result/Baseline_Data_With_WeatherRemove/2019/O3/"
    )
    processor.load_data("../result/WeatherRemove_Data/2019/O3/analysis_results.csv")
    processor.generate_reports()
