import os
import pandas as pd
from tqdm import tqdm  # 进度条库（可选）

def xlsx_to_csv(input_folder, output_folder=None, encoding='utf-8'):
    """
    将指定文件夹内的xlsx文件批量转换为csv文件
    参数:
        input_folder: 包含xlsx文件的输入文件夹路径
        output_folder: 输出csv的文件夹路径（默认同输入文件夹）
        encoding: 输出文件的编码格式
    """
    # 处理输出路径
    output_folder = output_folder or input_folder
    os.makedirs(output_folder, exist_ok=True)
    
    # 获取所有xlsx文件
    xlsx_files = [f for f in os.listdir(input_folder) 
                 if f.lower().endswith('.xlsx')]
    
    if not xlsx_files:
        print(f"在 {input_folder} 中未找到xlsx文件")
        return
    
    # 转换统计
    success = 0
    failed = []
    
    # 带进度条的转换（需要安装tqdm库）
    for filename in tqdm(xlsx_files, desc="转换进度"):
    # 普通循环（不需要额外库）
    # for filename in xlsx_files:
        try:
            # 构造完整路径
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(
                output_folder, 
                f"{os.path.splitext(filename)[0]}.csv"
            )
            
            # 读取Excel文件
            df = pd.read_excel(input_path, engine='openpyxl')
            
            # 保存CSV
            df.to_csv(output_path, index=False, encoding=encoding)
            success += 1
            
        except Exception as e:
            failed.append((filename, str(e)))
    
    # 打印结果
    print(f"\n转换完成: {success}/{len(xlsx_files)} 个文件成功")
    if failed:
        print("\n失败文件:")
        for f, err in failed:
            print(f"- {f}: {err}")

if __name__ == "__main__":
    # 使用示例
    # 遍历文件夹为data下的所有子文件夹 调用xlsx_to_csv函数
    print(os.getcwd())
    input_dir = '../data'
    output_dir = '../data'
    print('开始转换')
    # 遍历data文件夹下的所有子文件夹 打印路径
    for dirs in os.listdir(input_dir):
        if os.path.isdir(os.path.join(input_dir, dirs)):
            print(os.path.join(input_dir, dirs))
            xlsx_to_csv(os.path.join(input_dir, dirs), os.path.join(output_dir, dirs))

    
    # xlsx_to_csv(
    #     input_folder=input_dir,
    #     output_folder=output_dir,
    #     encoding='utf-8-sig'  # 推荐用于中文兼容性
    # )