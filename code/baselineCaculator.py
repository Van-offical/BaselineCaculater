# -*- coding: utf-8 -*-
"""
-----基线计算器封装-----
1.导入分省逐日数据表格
2.去离散数据（并保存）
3.排序
4.R2序列计算（并保存）
5.R2序列绘制散点图（并保存）

-----累计频率图封装-----

-----带标记散点图封装-----

-----输出文件夹结构-----
-对于一种污染物(此为前置文件夹路径 pre_dir_path/)
  -年份
    -inflection_1_draw
    -inflection_2_draw
    -2018年去离散数据污染物表格.csv
    -2018分省基线.csv
    -2018年分省R2序列.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
import os
import math
from scipy.stats import ks_2samp
import chardet
from datetime import datetime


class BaseLineCalculator:
    """
    读入一个列标签为逐日PM2.5/O3数据、行标签为省/站点名的表格
    输出去离散数据表格、R2序列表格、
    """

    def __init__(
        self,
        year: str,
        pollutant: str,
        target_dir_pre: str,
        r2_threshold: float = 0.95,
        min_valid: int = 10,
    ):
        self.year = year
        self.pollutant = pollutant
        self.target_dir_pre = target_dir_pre
        self.r2_threshold = r2_threshold  # 可调阈值
        self.min_valid = (
            min_valid  # 如果累计序列中 ≥r2_threshold 的数据点少于此值，则不做截断
        )
        if not os.path.isdir(self.target_dir_pre) or not os.path.isdir(
            self.target_dir_pre + "/infl_draw_1"
        ):
            # 若结果路径不存在则提前创建文件夹
            os.makedirs(self.target_dir_pre + "/infl_draw_1")
            os.makedirs(self.target_dir_pre + "/infl_draw_2")

    def detect_encoding(self, file_path):
        with open(file_path, "rb") as f:
            result = chardet.detect(f.read())
        return result["encoding"]

    def auto_convert_csv(self, input_path):
        encoding = self.detect_encoding(input_path)
        # 读取CSV（兼容不同分隔符）
        df = pd.read_csv(input_path, index_col=0, parse_dates=False, encoding=encoding)

        # 判断当前结构：尝试解析列名是否为日期
        date_cols = []
        for col in df.columns:
            try:
                # 兼容多种日期格式（如YYYY-MM-dd、YYYY/MM/dd等）
                parsed_date = datetime.strptime(col, "%Y-%m-%d")
                date_cols.append(col)
            except ValueError:
                try:
                    parsed_date = datetime.strptime(col, "%Y/%m/%d")
                    date_cols.append(col)
                except:
                    pass

        # 若日期在列名中：直接处理格式
        if len(date_cols) > 0:
            # 转换列名为统一日期格式
            df.columns = [
                datetime.strptime(
                    col, "%Y-%m-%d" if "-" in col else "%Y/%m/%d"
                ).strftime("%Y-%m-%d")
                for col in df.columns
            ]
        else:
            # 若日期在行索引中：转换索引并转置
            df.index = pd.to_datetime(df.index).strftime("%Y-%m-%d")
            df = df.T
        return df

    def init_raw_data_from_csv(self, csv_file: str):
        """
        从csv文件读入一组（全年的）分站点逐日数据,本程序允许csv和excel两种原始数据读入
        """
        self.raw_data = self.auto_convert_csv(csv_file)

        # 数据格式转换
        # self.raw_data = 
        # 取前10000行数据
        # self.raw_data = self.raw_data.iloc[1033:1040,:]
        return

    def init_raw_data_from_excel(self, excel_file: str, excel_sheet: str):
        """
        从excel文件读入一组（全年的）分站点逐日数据,本程序允许csv和excel两种原始数据读入
        """
        self.raw_data = pd.read_excel(excel_file, excel_sheet, header=0)

        return

    def check_self_raw_data(self):
        """
        本程序提供原始数据检查功能
        """
        print(self.raw_data)

    def __one_sigma_to_nan(self, raw_data: pd.DataFrame):
        """
        私有计算组件，把列表中超出 1 sigma 的值,赋值为 np.nan
        """
        for i in range(raw_data.shape[0]):  # 对每一行分别用1sigma原则去离散值
            # 特殊情况：前1列是标签，从第2列开始
            Ser1 = raw_data.iloc[i, 1:]
            rule = (Ser1.mean() - 3 * Ser1.std() > Ser1) | (
                Ser1.mean() + 3 * Ser1.std() < Ser1
            )
            index = np.arange(Ser1.shape[0])[rule]
            cnt = 0
            for r in index:
                cnt = cnt + 1
                raw_data.iloc[i, r + 1] = np.nan
            print("第 %d 行去除了 %d 个离散值" % (i, cnt))
        return raw_data

    def save_de_discrete_data(self):
        """
        将原始数据剔除离散值后，保存在csv中
        """
        de_discrete_data_file = "%s/%s无离散值数据.csv" % (
            self.target_dir_pre,
            self.year,
        )
        self.de_discrete_data = self.__one_sigma_to_nan(self.raw_data)
        self.de_discrete_data.to_csv(de_discrete_data_file)
        return

    def __r2_res(self, list_to_cpt: list):
        """
        私有计算组件：计算一组数据的r2
        """
        lenth = len(list_to_cpt)
        x = list(range(1, lenth + 1))
        y = list_to_cpt
        # 线性拟合，可以返回斜率，截距，r 值，p 值，标准误差
        result = st.linregress(x, y)
        r_value = result.rvalue

        return r_value**2

    def __r2_accumulate(self, full_data: list):
        """
        私有计算组件：计算一组数据的逐步累积r2序列
        """
        full_data_lenth = len(full_data)
        r2_accum_list = []
        for i in range(1, full_data_lenth):
            temp_list = full_data[0 : i + 1]
            temp_r2_res = self.__r2_res(temp_list)
            r2_accum_list.append(temp_r2_res)
        return r2_accum_list

    def __youxiao(self, data_list: list):
        """
        优化后的私有计算组件：
        如果累计 R2 序列中大于等于设定阈值的有效数据点不足，则保留原序列，否则
        从后往前剔除低于阈值的数据（拷贝方式，不直接修改原始列表）。
        """
        threshold = self.r2_threshold
        # 统计累计序列中达到阈值的数据点数量
        valid_count = sum(1 for x in data_list if x >= threshold)
        if valid_count < self.min_valid:
            # 数据不足，不进行剔除
            return data_list.copy()

        # 否则，从后往前删除低于阈值或为 NaN 的数据，返回新列表
        trimmed_list = data_list.copy()
        for i in range(len(trimmed_list) - 1, -1, -1):
            if trimmed_list[i] < threshold or math.isnan(trimmed_list[i]):
                trimmed_list.pop(i)
            else:
                # 遇到第一个大于等于阈值的值后停止删除（保证后段连续有效）
                break
        return trimmed_list

    def __get_inflection(self, data_list: list):
        """
        私有计算组件：四分位后的最高点为一组数据的拐点
        """
        inflection1 = max(data_list[len(data_list) // 6 :])
        return inflection1

    def __inflection_2_exist(self, latter_data: list):
        """
        私有计算组件：后段重新计算r2序列，重找拐点
        """
        latter_r2_list = self.__youxiao(self.__r2_accumulate(latter_data))
        if len(latter_r2_list) < 10:
            return False
        flection_2 = max(latter_r2_list[len(latter_r2_list) // 6 :])
        tmp_index2 = latter_r2_list.index(flection_2)
        if (
            tmp_index2 <= (len(latter_r2_list) // 6) + 2
            or tmp_index2 >= len(latter_r2_list) - 3
        ):
            return False
        else:
            return tmp_index2

    def __inflection2_draw(self, all_data, index1, index2, filename_prefix):
        """
        私有计算组件：被两个拐点分隔的三段数据，分别做累计频率图
        """
        _left_file = filename_prefix + "左侧累计频率图.png"
        _mid_file = filename_prefix + "中间累计频率图.png"
        _right_file = filename_prefix + "右侧累计频率图.png"

        drawer1 = Histplot(
            all_data[: index1 + 1], num_bins=15, fsize=(10, 6.18), fig_file=_left_file
        )
        drawer1.draw_plot()
        drawer2 = Histplot(
            all_data[index1 + 1 : index2 + 1],
            num_bins=15,
            fsize=(10, 6.18),
            fig_file=_mid_file,
        )
        drawer2.draw_plot()
        drawer3 = Histplot(
            all_data[index2 + 1 :], num_bins=15, fsize=(10, 6.18), fig_file=_right_file
        )
        drawer3.draw_plot()

    # 选择拐点
    def __compare_distributions(self, sorted_data: list, inf1: int, inf2: int) -> str:
        """比较分布相似性，返回选择的拐点标记"""
        # 数据分段
        pre_A = sorted_data[: inf1 + 1]  # 拐点1之前的区域
        mid_AB = sorted_data[inf1 + 1 : inf2 + 1]  # 两个拐点之间的区域
        post_B = sorted_data[inf2 + 1 :]  # 拐点2之后的区域

        # 有效性检查（至少10个数据点）
        if len(mid_AB) < 10 or len(pre_A) < 10 or len(post_B) < 10:
            return "inf1"  # 数据不足时默认选第一个拐点

        # 执行K-S检验比较分布
        stat_pre, _ = ks_2samp(mid_AB, pre_A)
        stat_post, _ = ks_2samp(mid_AB, post_B)

        # 选择更相似的区间
        return "inf1" if stat_pre < stat_post else "inf2"

    def __r2_cnt_draw(self):
        """
        私有计算组件：主体处理框架：
        1.读入无离散值数据DF
        2.准备存放位置参数
        2.按行（省）遍历，去空并排序为单行raw_data
          2.1计算总体r2_list，得最大值索引index1，计算inflection_2_exist[index1+1:]
            2.1.1.若不存在：计算baseline[0:index1]，绘制散点图并保存
            2.1.2.若存在：计算[index1+1:]的r2_list，得最大值索引index2，绘制三段累计频率图，计算2个baseline
        """
        raw_data = self.de_discrete_data
        t_csv = "%s/%s年分省基线.csv" % (self.target_dir_pre, self.year)
        with open(t_csv, "w") as f:
            f.write("")
            f.close()
        # 修改结果记录部分（约第195行开始）
        headers = pd.DataFrame(
            [
                "provinces",
                "拐点1以下均值",
                "拐点2以下均值",
                "最终基线值",
                "选择拐点",
                "分布判断",
            ]
        ).T
        headers.to_csv(t_csv, mode="a",header=False)

        for index, row in raw_data.iterrows():
            station_name = str(index)
            row_data = row[1:].tolist()
            row_data = [x for x in row_data if math.isnan(x) == False]
            row_data = sorted(row_data)
            print("%s共%d个有效数据：" % (station_name, len(row_data)), end=" ")
            if len(row_data) < 10:
                print("数据量不足，跳过")
                continue
            else:
                print("开始计算")
            row_r2_raw = self.__r2_accumulate(row_data)
            row_r2_list = self.__youxiao(row_r2_raw)
            len_youxiao = len(row_r2_list)

            flection_1 = self.__get_inflection(row_r2_list)
            index1 = row_r2_list.index(flection_1)
            print("拐点1位置: " + str(index1), end="; ")
            # 此处是匹配第一种拐点2求法的参数带入后段r2_list，
            # 第二种拐点2求法应当带入后端原始数据并计算新r2序列
            index2 = self.__inflection_2_exist(row_data[index1 + 1 : len_youxiao])

            baseline1 = np.mean(row_data[0 : index1 + 1])
            baseline2 = np.nan

            if not index2:
                print("只有一个拐点")
                scatter_file = "%s/infl_draw_1/%s-%s-inflection1.png" % (
                    self.target_dir_pre,
                    station_name,
                    self.year,
                )
                drawer = Scatterplot(row_r2_list, scatter_file)
                drawer.draw_scatter(index_1=index1)
            else:
                index2 = index2 + index1 + 2
                print("拐点2位置: " + str(index2) + ";")
                hisplot_file = "%s/infl_draw_2/%s年%s地区-" % (
                    self.target_dir_pre,
                    self.year,
                    station_name,
                )
                self.__inflection2_draw(row_data, index1, index2, hisplot_file)

                scatter_file = "%s/infl_draw_1/%s-%s-inflection1.png" % (
                    self.target_dir_pre,
                    station_name,
                    self.year,
                )
                drawer2 = Scatterplot(row_r2_list, scatter_file)
                drawer2.draw_scatter(index1, index2)

                baseline2 = np.mean(row_data[0 : index2 + 1])

            # 新增分布判断逻辑
            if index2:  # 当存在有效拐点2时
                final_choice = self.__compare_distributions(row_data, index1, index2)
                if final_choice == "inf1":
                    final_baseline = baseline1
                    dist_judge = "更接近前段"
                else:
                    final_baseline = baseline2
                    dist_judge = "更接近后段"
            else:
                final_choice = "inf1"
                final_baseline = baseline1
                dist_judge = "单拐点"

            # 修改记录字段
            flection_record = pd.DataFrame(
                [
                    station_name,
                    baseline1,
                    baseline2,
                    final_baseline,
                    final_choice,
                    dist_judge,
                ]
            ).T

            flection_record.to_csv(t_csv, mode="a", header=False, index=False)

    def full_process(self):
        self.save_de_discrete_data()
        self.__r2_cnt_draw()
        return "Calculate Over!"


class Histplot:
    def __init__(
        self,
        x_data,
        num_bins,
        linetype="o-",
        legends=None,
        xlabel=None,
        ylabels=None,
        fsize=(10, 10),
        df_line=1,
        fig_file=None,
    ):
        """
        :param x_data: 数据x列表
        :param num_bins: x的分组数
        :param linetype: 累计频率曲线的样式，默认为红色实心点
        :param legends: 图例名，默认为 "线性拟合结果", "实测值"
        :param xlabel:x坐标轴标题名，默认为 "数据x"
        :param ylabels:双y坐标轴标题名，默认为 "计数", "累计频率"
        :param df_line:是否显示累计频率曲线
        Frequency，Cumulative Frequency，PM2.5 concentration
        """
        if legends is None:
            legends = ["Frequency", "Cumulative Frequency"]
        if xlabel is None:
            xlabel = "PM25 concentration"
        if ylabels is None:
            ylabels = ["Frequency", "Cumulative Frequency"]
        self.x_data = x_data
        self.num_bins = num_bins
        self.linetype = linetype
        self.fsize = fsize
        self.legends = legends
        self.xlabel = xlabel
        self.ylabels = ylabels
        self.df_line = df_line
        if fig_file is None:
            self.fig_file = "保存位置有问题.png"
        else:
            self.fig_file = fig_file

    def change_legend(self, new_legends):
        # 将图例名称改为new_legends
        self.legends = new_legends

    def change_ylabel(self, new_labels):
        # 将双y轴坐标轴标题改为new_labels
        self.ylabels = new_labels

    def change_xlabel(self, new_label):
        # 将x轴坐标轴标题改为new_label
        self.xlabel = new_label

    def change_linetype(self, new_linetype):
        # 将累计频率线的格式改为new_lintype
        self.linetype = new_linetype

    def draw_plot(self):
        fs = self.fsize  # 画布大小
        # 利用seaborn库对字体大小进行统一设置，为fgsize[1]的0.12倍，即画布纵向大小为1000时，font_scale=1.2
        sns.set_style("ticks")
        sns.set_context("talk", font_scale=fs[1] * 0.15)
        plt.rc("font", family="Times New Roman")

        # 设置画布
        fig, ax = plt.subplots(figsize=fs)

        # ax:绘制频率直方图
        n, bins, patches = ax.hist(
            self.x_data, self.num_bins, rwidth=0.9, color="#bbb", label=self.legends[0]
        )
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabels[0])

        # ax2:绘制累计频率曲线
        if self.df_line:
            leiji_n = [sum(n[:i]) / len(self.x_data) for i in range(len(n) + 1)]
            ax2 = ax.twinx()
            ax2.plot(
                bins,
                leiji_n,
                self.linetype,
                ms=fs[0] * 0.5,
                color="#666",
                markeredgecolor="k",
                label=self.legends[1],
            )
            ax2.set_ylim(0, 1)
            ax2.set_ylabel(self.ylabels[1])

        fig.tight_layout()
        plt.savefig(self.fig_file, dpi=200)
        plt.close()


class Scatterplot:
    def __init__(self, x_data, fig_file=None):
        self.data = x_data
        if fig_file is None:
            self.fig_file = "保存位置有问题.png"
        else:
            self.fig_file = fig_file

    def draw_scatter(self, index_1=0, index_2=0):
        # plt.rc('font',family='Times New Roman')

        y = self.data
        x = np.arange(0, len(y))

        if not index_2:
            index_1_x = index_1
            index_1_y = y[index_1_x] + 0.005

            fig, ax = plt.subplots(figsize=(8, 5))

            ax.set_xlabel("Sample Index")
            ax.set_ylabel("Coefficient of Determination")

            ax.scatter(x, y, marker=".", s=2, color="black")

            ax2 = ax
            ax2.scatter(index_1_x, index_1_y, marker="v")

            fig.tight_layout()
            plt.savefig(self.fig_file, dpi=200)
            plt.close()
        else:
            index_1_x = index_1
            index_1_y = y[index_1_x] + 0.005
            index_2_x = index_2
            index_2_y = y[index_2_x] + 0.005

            fig, ax = plt.subplots(figsize=(8, 5))

            ax.set_xlabel("Sample Index")
            ax.set_ylabel("Coefficient of Determination")

            ax.scatter(x, y, marker=".", s=2, color="black")

            ax2 = ax
            ax2.scatter(index_1_x, index_1_y, marker="v")
            ax3 = ax
            ax3.scatter(index_2_x, index_2_y, marker="v")

            fig.tight_layout()
            plt.savefig(self.fig_file, dpi=200)
            plt.close()


if __name__ == "__main__":
    # for i in range(2018,2024):

    #     year = str(i)
    #     pollutant = "PM25"
    #     res_dir = "result/%s/%s"%(pollutant, year)
    #     # result/PM25/2018

    #     source_excel = "data/原始数据/PM25/provinces/%s分省日均.xlsx"%(year)
    #     source_sheet = "%s年PM2.5分省平均"%(year)

    #     my_calculator = BaseLineCalculator(year, pollutant, res_dir)
    #     my_calculator.init_raw_data_from_excel(source_excel, source_sheet)
    #     my_calculator.full_process()

    year = "2019"
    pollutant = "PM25"
    res_dir = "./result/Baseline_Data/%s/%s" % (year, pollutant)
    if not os.path.isdir(res_dir):
        # 若结果路径不存在则提前创建文件夹
        os.makedirs(res_dir)
    # result/PM25/2018
    # 2018年PM2.5分省平均
    source_csv = r"2018_province_daily_pm25.csv"
    my_calculator = BaseLineCalculator(year, pollutant, res_dir)
    # my_calculator.init_raw_data_from_excel(source_excel, source_sheet)
    my_calculator.init_raw_data_from_csv(source_csv)
    sigal = my_calculator.full_process()
