import pandas as pd
import numpy as np

def load_multi_asset_data(file_path):
    # 1. 基础读取：针对 YYYYMMDD 格式的整数日期进行解析
    df = pd.read_feather(file_path)
    df['date'] = pd.to_datetime(df['time'], format='%Y%m%d')
    df = df.sort_values(['date', 'asset'])

    # 2. 清洗 Volume 和 Ret
    # 新数据 volume 是浮点数，但为了兼容性保留 convert_volume
    def convert_volume(v):
        if not isinstance(v, str): return v
        v = v.upper()
        if 'B' in v: return float(v.replace('B', '')) * 1e9
        elif 'M' in v: return float(v.replace('M', '')) * 1e6
        elif 'K' in v: return float(v.replace('K', '')) * 1e3
        return float(v)

    df['volume'] = df['volume'].apply(convert_volume)
    
    # 收益率处理：新数据 changes 已经是数值小数
    # 统一命名为 rise_fall_norm 以适配 Env 中的调用
    df['rise_fall_norm'] = df['ret'].astype(float)


    # 3. 按资产分别计算特征（防止资产间数据干扰）
    assets = df['asset'].unique()
    processed_dfs = []
    
    for asset in assets:
        asset_df = df[df['asset'] == asset].copy().sort_values('date')
        
        asset_df['close_feat'] = asset_df['close'] / asset_df['close'].rolling(20).mean()
        asset_df['open_feat']  = asset_df['open'] / asset_df['close'] 
        asset_df['high_feat']  = asset_df['high'] / asset_df['close']
        asset_df['low_feat']   = asset_df['low'] / asset_df['close']
        
        # 成交量特征：已有的逻辑
        asset_df['vol_feat']   = asset_df['volume'] / (asset_df['volume'].rolling(20).mean() + 1e-9)
        
        # 收益率特征：直接使用 changes
        asset_df['ret_feat']   = asset_df['ret'].astype(float)
        asset_df['main_ret_slp_feat']   = asset_df['main_ret_slp'].astype(float)
        
        asset_df['capvol0_feat']   = asset_df['capvol0'].astype(float)
        asset_df['tr_feat']   = asset_df['tr'].astype(float)
        
        
        
        processed_dfs.append(asset_df)

    # 4. 合并并处理缺失值（由于使用了 rolling，前 20 行会产生空值）
    combined_df = pd.concat(processed_dfs).dropna()
    
    # 确保返回的数据按日期和资产名称排序，这对 Env 中的 Observation 对齐至关重要
    combined_df = combined_df.sort_values(['date', 'asset']).reset_index(drop=True)
    
    return combined_df
