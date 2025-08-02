import pandas as pd

# 1. 读取原始 CSV 文件
input_file = '../result/analysis_results.csv'  # 请将此处修改为实际文件路径
df = pd.read_csv(input_file)

# 按 Station_Code 和 Date 分组
groups = df.groupby(["Station_Code", "Date"])

# 用于记录完全重复的组中需要删除的行索引
indices_to_drop = []

for name, group in groups:
    if len(group) > 1:
        # 去除完全重复的行
        group_dedup = group.drop_duplicates()
        if len(group_dedup) == 1:
            # 如果去重后只有一行，说明组内完全相同，
            # 则保留第一条，记录其它重复行的索引以便删除
            indices_to_drop.extend(group.index[1:].tolist())
        else:
            # 如果组内存在差异，打印该组数据供检查
            print(f"Group {name} contains non-identical duplicates:")
            print(group)
            
# 删除完全重复的冗余记录
df = df.drop(indices_to_drop)

# print("清理后的 DataFrame：")
# print(df_cleaned)


# 2. 根据 Station_Code 和 Date 对 human_effect 进行透视
human_df = df.pivot(index='Station_Code', columns='Date', values='human_effect')
# 按日期排序（由于日期格式为 'YYYY-MM-DD'，直接排序即可）
human_df = human_df.sort_index(axis=1)
# 将 Station_Code 恢复为普通列
human_df.reset_index(inplace=True)

# 3. 保存 human.csv
human_df.to_csv('human.csv', index=False)

# 4. 同理，对 natural_effect 进行透视
nature_df = df.pivot(index='Station_Code', columns='Date', values='natural_effect')
nature_df = nature_df.sort_index(axis=1)
nature_df.reset_index(inplace=True)

# 5. 保存 nature.csv
nature_df.to_csv('nature.csv', index=False)

print("转换完成：生成 human.csv 和 nature.csv")
