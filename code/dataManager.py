import os
import re
import glob
import pandas as pd
from tkinter import Tk, filedialog, messagebox

def display_available_data():
    """
    扫描 ../data/MeteorologicalData 文件夹，
    打印当前各年份下各子文件夹中的数据文件信息。
    """
    base_dir = os.path.join("..", "data", "MeteorologicalData")
    if not os.path.isdir(base_dir):
        print("数据文件夹不存在。")
        return
    print("当前数据文件信息：")
    for year in os.listdir(base_dir):
        year_path = os.path.join(base_dir, year)
        if os.path.isdir(year_path):
            print(f"年份 {year}:")
            for folder in os.listdir(year_path):
                folder_path = os.path.join(year_path, folder)
                if os.path.isdir(folder_path):
                    files = os.listdir(folder_path)
                    print(f"  {folder}: {files}")

def check_meteo_file_format(file_path: str, file_type: str, config: dict) -> bool:
    """
    检查气象数据文件格式是否符合规定：
    - 对于 PRS、RHU、TEM、WIN 类型，
      必选字段要求：
         站名、区站号、【纬度 或 纬度（度、分）】、【经度 或 经度（度、分）】、
         年、月、日；
      对于 PRS：任选字段要求存在【平均本站气压 或 平均气压】；
      对于 RHU：任选字段要求存在【平均相对湿度 或 平均湿度】；
      对于 TEM 和 WIN：直接要求存在相应字段（WIN 数据还要求存在“平均风速”和“极大风速风向”）。
    """
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except Exception as e:
        messagebox.showerror("文件读取错误", f"无法读取文件: {file_path}\n错误信息: {e}")
        return False
    
    columns = df.columns.tolist()
    required = config["meteorological_data"]["required_columns"].get(file_type)
    if not required:
        messagebox.showerror("文件类型错误", f"未知的文件类型: {file_type}")
        return False
    
    # 检查必选字段
    for field in required["mandatory"]:
        if isinstance(field, list):
            # 若 field 为候选列表，则至少要有其中一个存在
            if not any(f in columns for f in field):
                messagebox.showerror("文件格式错误", f"{file_type} 文件缺少必选字段之一: {field}")
                return False
        else:
            if field not in columns:
                messagebox.showerror("文件格式错误", f"{file_type} 文件缺少必选字段: {field}")
                return False

    # 检查任选字段（若存在），至少要有一个
    if "optional" in required:
        if not any(opt in columns for opt in required["optional"]):
            messagebox.showerror("文件格式错误", f"{file_type} 文件至少需要存在任选字段之一: {required['optional']}")
            return False
    return True

import re
import pandas as pd
from tkinter import messagebox
from datetime import datetime

import re
import pandas as pd
from tkinter import messagebox
from datetime import datetime

def check_pollutant_file_format(file_path: str) -> bool:
    """
    检查污染物数据文件格式：
    - 第一列（可选）检查为 '区站号'
    - 其余列标题可能包含括号及单位信息，需去除括号内容后验证格式是否为 yyyy-MM-dd 或 yyyy/MM/dd，
      同时支持月份和日期为1或2位数字；如果格式为 yyyy/MM/dd（或带 / 分隔符），则转换为 yyyy-MM-dd 再保存文件。
    """
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except Exception as e:
        messagebox.showerror("文件读取错误", f"无法读取文件: {file_path}\n错误信息: {e}")
        return False

    columns = df.columns.tolist()
    if not columns:
        messagebox.showerror("文件格式错误", "文件没有列标题")
        return False

    # 可根据需要启用第一列的检查
    # if columns[0] != "区站号":
    #     messagebox.showerror("文件格式错误", "污染物数据文件第一列必须为 '区站号'")
    #     return False

    # 修改正则表达式，允许 - 或 / 作为日期分隔符，并允许月份和日期为1或2位
    date_pattern = re.compile(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}$")

    def clean_column(col):
        # 去除括号及其内容（支持英文括号和中文括号）
        col = re.sub(r"\([^)]*\)", "", col)
        col = re.sub(r"（[^）]*）", "", col)
        return col.strip()

    new_columns = [columns[0]]  # 第一列保持不变
    changed = False
    for col in columns[1:]:
        cleaned = clean_column(col)
        if not date_pattern.fullmatch(cleaned):
            messagebox.showerror("文件格式错误",
                f"污染物数据文件列标题 '{col}' 处理后 '{cleaned}' 不是有效的日期格式 (yyyy-MM-dd 或 yyyy/MM/dd)")
            return False
        # 根据分隔符判断日期格式
        try:
            if "/" in cleaned:
                dt = datetime.strptime(cleaned, "%Y/%m/%d")
            elif "-" in cleaned:
                dt = datetime.strptime(cleaned, "%Y-%m-%d")
            else:
                messagebox.showerror("文件格式错误", f"无法识别日期分隔符: {cleaned}")
                return False
            standardized = dt.strftime("%Y-%m-%d")
            if standardized != cleaned:
                changed = True
            new_columns.append(standardized)
        except Exception as e:
            messagebox.showerror("日期转换错误", f"无法转换日期 '{cleaned}': {e}")
            return False

    # 更新 DataFrame 的列标题
    df.columns = new_columns
    # 如果有转换，则保存文件
    if changed:
        try:
            df.to_csv(file_path, index=False, encoding='utf-8')
            print(f"文件 {file_path} 日期格式已更新为 yyyy-MM-dd")
        except Exception as e:
            messagebox.showerror("保存错误", f"保存更新后的文件失败: {e}")
            return False
    return True

# 以下是添加数据和选择数据的其他函数保持不变（见前面示例代码）
def clear_existing_data(target_dir: str):
    """
    清空目标目录下所有旧数据文件（若存在），确保每个年份的数据仅存一份。
    """
    if os.path.isdir(target_dir):
        files = os.listdir(target_dir)
        for f in files:
            file_path = os.path.join(target_dir, f)
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"删除旧文件 {file_path} 失败: {e}")

def interactive_add_data(config: dict):
    """
    交互式添加数据：
    1. 用户首先输入数据所属年份。
    2. 对于气象数据（PRS、RHU、TEM、WIN），逐个询问是否添加或更新；
       如果已有数据，则选择更新（删除旧文件）；不存在则询问是否添加。
    3. 然后允许用户添加/更新污染物数据（可添加多个），输入污染物名称后执行同样操作。
    复制文件完成后，会重新检查文件格式，如果检查失败则删除复制的文件并给出提示。
    """
    from tkinter import Tk, filedialog, messagebox
    root = Tk()
    root.withdraw()
    
    year = input("请输入数据所属年份 (例如 2019): ").strip()
    base_dir = os.path.join("..", "data", "MeteorologicalData", year)
    os.makedirs(base_dir, exist_ok=True)
    
    meteo_types = ["PRS", "RHU", "TEM", "WIN"]
    for mtype in meteo_types:
        target_dir = os.path.join(base_dir, mtype)
        if os.path.isdir(target_dir) and os.listdir(target_dir):
            update = input(f"{year} 年的 {mtype} 数据已存在，是否更新? (y/n): ").strip().lower()
            if update != 'y':
                continue
        else:
            add = input(f"{year} 年的 {mtype} 数据不存在，是否添加? (y/n): ").strip().lower()
            if add != 'y':
                continue
        clear_existing_data(target_dir)
        os.makedirs(target_dir, exist_ok=True)
        new_file = filedialog.askopenfilename(title=f"选择 {year} 年 {mtype} 数据文件")
        if new_file:
            target_path = os.path.join(target_dir, os.path.basename(new_file))
            try:
                with open(new_file, 'rb') as fr, open(target_path, 'wb') as fw:
                    fw.write(fr.read())
                # 复制后进行检查
                from dataManager import check_meteo_file_format  # 确保该函数已定义
                if not check_meteo_file_format(target_path, mtype, config):
                    messagebox.showerror("文件检查失败", f"复制后的 {mtype} 数据格式不符合要求，已删除文件")
                    os.remove(target_path)
                else:
                    print(f"{mtype} 数据已更新/添加到 {target_dir}，格式检查通过")
            except Exception as e:
                print(f"添加 {mtype} 数据失败: {e}")
        else:
            print(f"未选择 {mtype} 数据文件")
    
    while True:
        pol_choice = input("是否添加/更新某个污染物数据? (y/n): ").strip().lower()
        if pol_choice != 'y':
            break
        pollutant = input("请输入污染物名称 (例如 O3 或 PM25): ").strip().upper()
        target_dir = os.path.join(base_dir, pollutant)
        if os.path.isdir(target_dir) and os.listdir(target_dir):
            update = input(f"{year} 年的 {pollutant} 数据已存在，是否更新? (y/n): ").strip().lower()
            if update != 'y':
                continue
        else:
            add = input(f"{year} 年的 {pollutant} 数据不存在，是否添加? (y/n): ").strip().lower()
            if add != 'y':
                continue
        clear_existing_data(target_dir)
        os.makedirs(target_dir, exist_ok=True)
        new_file = filedialog.askopenfilename(title=f"选择 {year} 年 {pollutant} 数据文件")
        if new_file:
            target_path = os.path.join(target_dir, os.path.basename(new_file))
            try:
                with open(new_file, 'rb') as fr, open(target_path, 'wb') as fw:
                    fw.write(fr.read())
                # 对污染物数据进行复制后检查
                from dataManager import check_pollutant_file_format
                if not check_pollutant_file_format(target_path):
                    messagebox.showerror("文件检查失败", f"复制后的 {pollutant} 数据格式不符合要求，已删除文件")
                    os.remove(target_path)
                else:
                    print(f"{pollutant} 数据已更新/添加到 {target_dir}，格式检查通过")
            except Exception as e:
                print(f"添加 {pollutant} 数据失败: {e}")
        else:
            print("未选择文件")
    root.destroy()


def select_calculation_data():
    """
    扫描 ../data/MeteorologicalData 下所有年份，
    检查每个年份是否存在完整的气象数据（PRS, RHU, TEM, WIN）且至少存在一个污染物文件夹，
    列出所有可供计算的组合，供用户选择。
    返回 (year, pollutant)
    """
    base_dir = os.path.join("..", "data", "MeteorologicalData")
    available = []
    meteo_types = ["PRS", "RHU", "TEM", "WIN"]
    for year in os.listdir(base_dir):
        year_path = os.path.join(base_dir, year)
        if not os.path.isdir(year_path):
            continue
        has_all_meteo = all(os.path.isdir(os.path.join(year_path, mt)) and os.listdir(os.path.join(year_path, mt))
                            for mt in meteo_types)
        if not has_all_meteo:
            continue
        for folder in os.listdir(year_path):
            if folder in meteo_types:
                continue
            pol_dir = os.path.join(year_path, folder)
            if os.path.isdir(pol_dir) and os.listdir(pol_dir):
                available.append((year, folder))
    if not available:
        print("没有可供计算的数据组合！")
        return None, None
    print("可供计算的数据组合：")
    for idx, (y, pol) in enumerate(available, 1):
        print(f"{idx}. {y} 年, 污染物: {pol}")
    choice = input("请选择组合编号: ").strip()
    try:
        choice = int(choice)
        selected = available[choice-1]
        return selected
    except Exception as e:
        print("选择错误，退出")
        return None, None
