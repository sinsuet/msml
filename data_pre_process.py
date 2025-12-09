import pandas as pd
import os

def preprocess_data():
    """
    读取 .tab 格式的原始数据，进行预处理并保存为 CSV。
    """
    # 修改输入文件名以匹配您下载的 .tab 文件
    input_file = "data/090113_TotWatDat_cor_merge_Price.tab"
    output_file = "data/data_ferraroprice.csv"

    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"警告: 未找到输入文件 '{input_file}'")
        print("请确保您已将 .tab 文件放入 data 文件夹中。")
        return

    try:
        # 使用 pandas 读取 .tab 文件
        # sep='\t' 指定分隔符为制表符
        original_data = pd.read_csv(input_file, sep='\t')
        
        # 对应 R 代码逻辑：
        # 计算 Y (2007年6月到9月用水量之和)
        # 注意：这里假设 .tab 文件中的列名与原 .dta 文件一致
        original_data['Y'] = (original_data['jun07_3x'] + 
                              original_data['jul07_3x'] + 
                              original_data['aug07_3x'] + 
                              original_data['sep07_3x'])
        
        # 对应 R 代码：mutate(D = treatment)
        # 原始数据中 'treatment' 列即为 D
        original_data['D'] = original_data['treatment']

        # 选择需要的列
        cols_to_keep = [
            'Y', 'D',
            'jun06', 'jul06', 'aug06', 'sep06', 'oct06', 'nov06', 
            'dec06', 'jan07', 'feb07', 'mar07', 'apr07_3x', 'may07_3x'
        ]
        
        # 筛选列
        df_selected = original_data[cols_to_keep]

        # 保存为 CSV
        if not os.path.exists('data'):
            os.makedirs('data')
            
        df_selected.to_csv(output_file, index=False)
        print(f"数据预处理完成，已保存至 {output_file}")

    except Exception as e:
        print(f"处理数据时出错: {e}")
        print("提示：请检查 .tab 文件的列名是否与代码中引用的一致。")

if __name__ == "__main__":
    preprocess_data()