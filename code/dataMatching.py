import os
import re
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree


# ---------------------------
# 辅助函数：归一化和模糊匹配
# ---------------------------
def normalize(text):
    """将文本转换为小写并去除空格、括号、连字符等字符"""
    return re.sub(r"[\s\(\)（）\-]", "", text).lower()


def fuzzy_match_column(df, candidates):
    """
    在 DataFrame 的列名中查找包含候选字符串之一的列名
    :param df: 输入 DataFrame
    :param candidates: 候选字符串列表
    :return: 匹配到的列名；若未找到则返回 None
    """
    for col in df.columns:
        norm_col = normalize(col)
        for candidate in candidates:
            if normalize(candidate) in norm_col:
                return col
    return None


# ---------------------------
# 数据标准化函数
# ---------------------------
def standardize_merge_keys(df):
    """
    标准化气象数据中的合并键列为固定的中文名称
    标准键包括：省份、站名、区站号、纬度、经度、年、月、日
    :param df: 输入 DataFrame
    :return: 标准化后的 DataFrame
    """
    merge_keys_map = {
        "省份": ["省份", "province"],
        "站名": ["站名", "stationname", "station name"],
        "区站号": ["区站号", "stationnumber", "station no", "stationid"],
        "纬度": ["纬度", "latitude"],
        "经度": ["经度", "longitude"],
        "年": ["年", "year"],
        "月": ["月", "month"],
        "日": ["日", "day"],
    }
    for standard, candidates in merge_keys_map.items():
        actual = fuzzy_match_column(df, candidates)
        if actual is not None and actual != standard:
            df.rename(columns={actual: standard}, inplace=True)
    return df


def standardize_meteorological_variables(df, var_type):
    """
    对气象数据中的变量列进行模糊匹配并重命名为统一名称
    :param df: 输入 DataFrame
    :param var_type: 变量类型，可取值 "prs", "rhu", "tem", "win"
                     分别对应气压、湿度、气温、风速数据
    :return: 标准化后的 DataFrame
    """
    if var_type == "prs":
        col = fuzzy_match_column(df, ["平均本站气压", "平均气压"])
        if col is not None:
            df.rename(columns={col: "Pressure_raw"}, inplace=True)
    elif var_type == "rhu":
        col = fuzzy_match_column(df, ["平均相对湿度", "平均湿度"])
        if col is not None:
            df.rename(columns={col: "Humidity_raw"}, inplace=True)
    elif var_type == "tem":
        col = fuzzy_match_column(df, ["平均气温"])
        if col is not None:
            df.rename(columns={col: "Temperature_raw"}, inplace=True)
    elif var_type == "win":
        col = fuzzy_match_column(df, ["平均风速"])
        if col is not None:
            df.rename(columns={col: "WindSpeed_raw"}, inplace=True)
        col2 = fuzzy_match_column(df, ["极大风速风向", "极大风速的风向"])
        if col2 is not None:
            df.rename(columns={col2: "WindDirection_raw"}, inplace=True)
    return df


def standardize_station_data(df):
    """
    对站点数据中的关键字段进行模糊匹配，并重命名为英文名称
    重命名规则：
      - 监测点编码 -> Station_Code
      - 监测点名称 -> Station_Name
      - 经度 -> Longitude
      - 纬度 -> Latitude
    :param df: 输入站点 DataFrame
    :return: 标准化后的 DataFrame
    """
    mapping = {
        "Station_Code": ["监测点编码", "stationcode", "station code"],
        "Station_Name": ["监测点名称", "stationname", "station name"],
        "Longitude": ["经度", "longitude"],
        "Latitude": ["纬度", "latitude"],
    }
    for std, candidates in mapping.items():
        actual = fuzzy_match_column(df, candidates)
        if actual is not None and actual != std:
            df.rename(columns={actual: std}, inplace=True)
    return df


# ---------------------------
# 数据加载与清洗函数
# ---------------------------
def load_station_data(station_data_path):
    """
    读取站点数据，打印预览及缺失值统计信息，并标准化字段名称
    :param station_data_path: 站点数据文件路径
    :return: 站点数据 DataFrame
    """
    station_data = pd.read_csv(station_data_path)
    station_data = standardize_station_data(station_data)
    print("站点数据预览:")
    print(station_data.head())

    # 统计缺失值
    missings = station_data.isna().sum()
    print("缺失值统计:")
    print(missings.head())
    missing_values = missings[missings > 0]
    if not missing_values.empty:
        print("存在缺失值的字段:")
        print(missing_values)
    return station_data


def load_and_clean_meteorological_data(file_path, value_col_candidates, drop_last=True):
    """
    读取单个气象数据文件，删除最后一行（如果存在），替换异常值32766，
    并采用线性插值补全缺失值
    :param file_path: 数据文件路径
    :param value_col_candidates: 候选数值列名称列表，例如 ["平均本站气压", "平均气压"]
    :param drop_last: 是否删除最后一行，默认 True
    :return: 清洗后的 DataFrame
    """
    df = pd.read_csv(file_path)
    if drop_last:
        df = df.drop(df.index[-1])
    df = df.replace(32766, np.nan)
    df = df.interpolate(method="linear")
    # 如果传入为字符串，则转换为列表
    if isinstance(value_col_candidates, str):
        value_col_candidates = [value_col_candidates]
    actual_col = fuzzy_match_column(df, value_col_candidates)
    if actual_col is None:
        raise ValueError(f"在文件 {file_path} 中未找到匹配 {value_col_candidates} 的列")
    # 转换为数值类型
    df[actual_col] = pd.to_numeric(df[actual_col], errors="coerce")
    # 将匹配到的列重命名为候选列表中的第一个
    df.rename(columns={actual_col: value_col_candidates[0]}, inplace=True)
    return df


# ---------------------------
# 数据合并与转换函数
# ---------------------------
def merge_meteorological_data(data_prs, data_rhu, data_tem, data_win):
    """
    合并气压、湿度、气温和风速数据，进行单位转换并构造日期列
    合并键为：["省份", "站名", "区站号", "纬度", "经度", "年", "月", "日"]
    最终输出的字段会重命名为英文
    :param data_prs: 气压数据 DataFrame
    :param data_rhu: 湿度数据 DataFrame
    :param data_tem: 气温数据 DataFrame
    :param data_win: 风速数据 DataFrame
    :return: 合并后的 DataFrame
    """
    # 对每个数据集先标准化合并键
    for df in [data_prs, data_rhu, data_tem, data_win]:
        standardize_merge_keys(df)
    # 标准化各数据的变量列
    data_prs = standardize_meteorological_variables(data_prs, "prs")
    data_rhu = standardize_meteorological_variables(data_rhu, "rhu")
    data_tem = standardize_meteorological_variables(data_tem, "tem")
    data_win = standardize_meteorological_variables(data_win, "win")

    # 定义合并键（中文）
    merge_keys = ["省份", "站名", "区站号", "纬度", "经度", "年", "月", "日"]

    # 只保留合并键及必要的变量列，避免因其他重复列导致冲突
    data_prs = data_prs[merge_keys + ["Pressure_raw"]]
    data_rhu = data_rhu[merge_keys + ["Humidity_raw"]]
    data_tem = data_tem[merge_keys + ["Temperature_raw"]]
    data_win = data_win[merge_keys + ["WindSpeed_raw", "WindDirection_raw"]]

    # 依次合并各数据集
    df_merge = pd.merge(data_prs, data_rhu, on=merge_keys, how="inner")
    df_merge = pd.merge(df_merge, data_tem, on=merge_keys, how="inner")
    df_merge = pd.merge(df_merge, data_win, on=merge_keys, how="inner")

    # 单位转换
    df_merge["Mean_Pressure"] = df_merge["Pressure_raw"] / 10.0
    df_merge["Mean_Humidity"] = df_merge["Humidity_raw"]
    df_merge["Mean_Temperature"] = df_merge["Temperature_raw"] / 10.0
    df_merge["Mean_Wind_Speed"] = df_merge["WindSpeed_raw"] / 10.0
    df_merge["Max_Wind_Direction"] = df_merge["WindDirection_raw"]

    # 构造日期列，格式为 yyyy-MM-dd
    df_merge["Date"] = df_merge.apply(
        lambda row: f"{int(row['年']):04d}-{int(row['月']):02d}-{int(row['日']):02d}",
        axis=1,
        # 构造日期列，格式为 yyyy-MM-dd
        result_type='reduce'
    )

    # 将纬度和经度从度分格式转换为十进制度
    df_merge["纬度"] = df_merge["纬度"].apply(dms_to_decimal)
    df_merge["经度"] = df_merge["经度"].apply(dms_to_decimal)

    # 将合并键中的中文列重命名为英文
    df_merge.rename(
        columns={
            "省份": "Province",
            "站名": "Station_Name",
            "区站号": "Station_Number",
            "纬度": "Latitude",
            "经度": "Longitude",
        },
        inplace=True,
    )

    # 选择最终输出的列（英文版）
    final_cols = [
        "Province",
        "Station_Name",
        "Station_Number",
        "Latitude",
        "Longitude",
        "Date",
        "Mean_Pressure",
        "Mean_Humidity",
        "Mean_Temperature",
        "Mean_Wind_Speed",
        "Max_Wind_Direction",
    ]
    final_df = df_merge[final_cols].copy()

    return final_df


def dms_to_decimal(dms):
    """
    将度分格式转换为十进制度
    如果数值大于 140，则认为是度分格式；否则认为已经是十进制度
    :param dms: 输入的经纬度数值
    :return: 十进制度数值
    """
    try:
        dms = float(dms)
    except:
        return dms
    if dms > 140:
        degrees = int(dms // 100)
        minutes = dms % 100
        return round(degrees + minutes / 60, 4)
    else:
        return dms


# ---------------------------
# 数据匹配函数
# ---------------------------
def match_stations(df, stations):
    """
    将气象站数据与监测站列表根据经纬度最小距离进行匹配
    对于监测站中的每个有效站点，找到 df 中最近的坐标（可能对应多条时间序列数据）
    输出新的 DataFrame，包含 'Station_Code', 'Station_Name', 'Station_Number', 'Date',
    'Mean_Pressure' 等字段
    :param df: 合并后的气象数据 DataFrame
    :param stations: 站点数据 DataFrame，要求包含 'Station_Code', 'Station_Name', 'Longitude', 'Latitude'
    :return: 匹配后的 DataFrame
    """
    # 过滤掉缺少经纬度的站点
    valid_stations = stations.dropna(subset=["Longitude", "Latitude"]).reset_index(
        drop=True
    )
    missing_stations = stations[
        stations[["Longitude", "Latitude"]].isnull().any(axis=1)
    ]
    if not missing_stations.empty:
        print("以下站点缺少经纬度信息，将跳过:")
        print(missing_stations)

    # 从 df 中提取唯一的坐标集合，用于构造 kd-tree
    unique_coords = (
        df[["Latitude", "Longitude"]].drop_duplicates().reset_index(drop=True)
    )
    tree = cKDTree(unique_coords[["Latitude", "Longitude"]].values)

    # 对每个站点查询在 df 中最近的坐标
    station_coords = valid_stations[["Latitude", "Longitude"]].values
    distances, indices = tree.query(station_coords)

    result_frames = []
    # 检查unique_coords是否为空
    if unique_coords.empty:
        print("警告: 没有找到有效的坐标数据，无法进行站点匹配。")
        return pd.DataFrame()
        
    for i, unique_idx in enumerate(indices):
        # 确保索引在有效范围内
        if 0 <= unique_idx < len(unique_coords):
            station_info = valid_stations.iloc[i]
            coord = unique_coords.iloc[unique_idx]
            
            # 选择 df 中与该坐标相同的所有行（即该气象站的完整时间序列数据）
            matched_rows = df[
                (df["Latitude"] == coord["Latitude"])
                & (df["Longitude"] == coord["Longitude"])
            ].copy()
            if not matched_rows.empty:
                matched_rows["Station_Code"] = station_info["Station_Code"]
                matched_rows["Station_Name"] = station_info["Station_Name"]
                result_frames.append(matched_rows)
        else:
            print(f"警告: 站点索引 {i} 的坐标索引 {unique_idx} 超出范围，将跳过该站点。")
            # 调整输出字段顺序
            matched_rows = matched_rows[
                [
                    # 'Station_Code', 'Station_Name', 'Station Number', 'Date',
                    # 'Mean Pressure (hPa)', 'Mean Humidity', 'Mean Temperature (℃)',
                    # 'Mean Wind Speed (m/s)', 'Max Wind Direction'
                    "Station_Code",
                    "Station_Name",
                    "Province",
                    "Date",
                    "Mean_Pressure",
                    "Mean_Humidity",
                    "Mean_Temperature",
                    "Mean_Wind_Speed",
                    "Max_Wind_Direction",
                ]
            ]
            result_frames.append(matched_rows)

    if result_frames:
        result_df = pd.concat(result_frames, ignore_index=True)
    else:
        result_df = pd.DataFrame(
            columns=[
                "Station_Code",
                "Station_Name",
                "Province",
                "Date",
                "Mean_Pressure",
                "Mean_Humidity",
                "Mean_Temperature",
                "Mean_Wind_Speed",
                "Max_Wind_Direction",
            ]
        )
    return result_df


# ---------------------------
# O3 数据合并函数
# ---------------------------
def merge_value_data(matched_df, value_path, value_name="value1"):
    """
    读取 o3.csv 文件，将其从宽格式转换为长格式，
    并根据 'Station_Code' 和 'Date'（格式 yyyy-MM-dd）与 matched_df 进行合并，
    添加数值列（例如 o3）及额外列 v1，值为 value_name
    :param matched_df: 已匹配的气象数据 DataFrame
    :param value_path: o3 数据文件路径
    :param value_name: 数值列名称，默认 'value1'
    :return: 合并后的 DataFrame
    """
    # 读取 o3.csv，假设第一列为站点编码
    value_df = pd.read_csv(value_path, index_col=0)
    # 将索引重置为列，并重命名为 'Station_Code'
    value_df = value_df.reset_index()
    # 确保索引列被重命名为 'Station_Code'
    if 'index' in value_df.columns:
        value_df = value_df.rename(columns={'index': 'Station_Code'})
    else:
        # 如果索引有名称，使用该名称重命名
        index_name = value_df.columns[0]
        value_df = value_df.rename(columns={index_name: 'Station_Code'})
    # 使用 melt 将宽格式转换为长格式
    value_long = value_df.melt(
        id_vars="Station_Code", var_name="Date", value_name=value_name
    )

    # 自动推断日期格式并转换
    value_long["Date"] = pd.to_datetime(value_long["Date"], infer_datetime_format=True)
    matched_df["Date"] = pd.to_datetime(matched_df["Date"], infer_datetime_format=True)

    # 根据 'Station_Code' 和 'Date' 合并数据，采用左连接保留 matched_df 的所有记录
    merged_df = pd.merge(
        matched_df, value_long, on=["Station_Code", "Date"], how="left"
    )

    # 添加额外列 v1，值为 value_name
    merged_df["v1"] = value_name

    return merged_df


# ---------------------------
# 主数据处理流程函数
# ---------------------------
def process_data(
    station_data_path,
    data_prs_path,
    data_rhu_path,
    data_tem_path,
    data_win_path,
    value_path,
    value_name="value1",
    output_dir="../data/MeteorologicalData/2019/",
):
    """
    完整的数据处理流程：
      1. 加载站点数据并标准化字段
      2. 加载并清洗气象数据（气压、湿度、气温、风速）
      3. 合并气象数据并转换为英文表头
      4. 根据经纬度匹配站点
      5. 根据 'Station_Code' 和 'Date' 合并 O3 数据（或其他数值数据）
      6. 保存最终结果到 CSV 文件（英文表头）
    :return: 最终合并后的各阶段 DataFrame
    """
    # 读取站点数据并标准化字段
    station_data = load_station_data(station_data_path)

    # 读取并清洗气象数据，传入候选字段以实现模糊匹配
    data_prs = load_and_clean_meteorological_data(
        data_prs_path, ["平均本站气压", "平均气压"]
    )
    data_rhu = load_and_clean_meteorological_data(
        data_rhu_path, ["平均相对湿度", "平均湿度"]
    )
    data_tem = load_and_clean_meteorological_data(data_tem_path, ["平均气温"])
    data_win = load_and_clean_meteorological_data(data_win_path, ["平均风速"])

    # 合并气象数据并转换为英文表头
    final_df = merge_meteorological_data(data_prs, data_rhu, data_tem, data_win)
    print("合并后的气象数据预览:")
    print(final_df.head())

    # 根据 'Station_Code', 'Station_Name', 'Longitude', 'Latitude' 等字段匹配站点
    matched_df = match_stations(final_df, station_data)
    print("匹配后的站点预览:")
    print(matched_df.head())

    # 检查matched_df是否为空
    if matched_df.empty:
        print("警告: 没有找到匹配的站点数据，无法继续合并污染物数据。")
        return []
        
    # 根据 'Station_Code' 和 'Date' 合并 O3（或其他数值）数据
    merged_df = merge_value_data(matched_df, value_path, value_name)
    print("合并 污染物 数据后的预览:")
    print(merged_df.head())

    # 线性填充value缺失值
    merged_df = merged_df.interpolate(method='linear', limit_direction='both')

    # 去除完全重复的行
    merged_df = merged_df.drop_duplicates()

    # 按照省份分割输出数据文件
    province_groups = merged_df.groupby("Province")
    for province, group in province_groups:
        # 创建输出目录（如果不存在）并保存最终结果到 CSV 文件
        os.makedirs(os.path.join(output_dir, province) , exist_ok=True)
        output_path = os.path.join(output_dir, province, "matched_data_with_" + value_name + ".csv")
        # merged_df[].to_csv(output_path, index=False)
        group.to_csv(output_path, index=False)
        print(f"最终结果已保存到 {output_path}")
    # 返回省份名列表
    return list(province_groups.groups.keys())


# ---------------------------
# 主程序入口
# ---------------------------
if __name__ == "__main__":
    # 文件路径配置（根据需要修改）
    station_data_path = "../data/StationList/StationList.csv"
    data_prs_path = "../data/MeteorologicalData/2019/PRS/2019PRS.csv"
    data_rhu_path = "../data/MeteorologicalData/2019/RHU/2019RHU.csv"
    data_tem_path = "../data/MeteorologicalData/2019/TEM/2019TEM.csv"
    data_win_path = "../data/MeteorologicalData/2019/WIN/2019WIN.csv"
    value_path = "../data/MeteorologicalData/2019/O3/2019-O3-AVE.csv"  # O3 数据文件路径

    output_dir = "../data/"

    # 调用数据处理流程，value_name 参数例如传入 'o3'
    process_data(
        station_data_path,
        data_prs_path,
        data_rhu_path,
        data_tem_path,
        data_win_path,
        value_path=value_path,
        value_name="o3",
        output_dir=output_dir,
    )
