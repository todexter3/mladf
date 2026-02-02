import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from envs.trading_envs import TimingEnv
from agents.agent import PPOAgent
from data_loader.data_loader import load_multi_asset_data

class Exp_PPO:
    def __init__(self, args):
        self.args = args
        self.data = load_multi_asset_data(args.data_path)
        
        # --- 修改 1: 按日期切分数据，确保不打断资产对齐 ---
        unique_dates = sorted(self.data['date'].unique())
        split_date = unique_dates[int(len(unique_dates) * 0.8)]
        
        train_data = self.data[self.data['date'] < split_date].copy()
        test_data = self.data[self.data['date'] >= split_date].copy()
        
        self.train_env = TimingEnv(train_data, args)
        self.test_env = TimingEnv(test_data, args)
        
        sample_obs = self.train_env.reset()
        # agent 内部会自动根据 state_dim 初始化 ActorCritic
        self.agent = PPOAgent(len(sample_obs), args)

    def train(self):
        for ep in range(self.args.n_epochs):
            state = self.train_env.reset()
            ep_reward = 0
            done = False
            while not done:
                # action 此时是长度为 n_assets 的向量
                action, logp, val = self.agent.select_action(state)
                next_state, reward, done, info = self.train_env.step(action)
                
                # 修改 2: 确保存储的是 numpy 格式的动作向量
                self.agent.store_transition((state, action, reward, done, logp, val))
                state = next_state
                ep_reward += reward
                
                if len(self.agent.buffer) >= self.args.buffer_size:
                    self.agent.update()
            
            print(f"Epoch: {ep} | Train Reward: {ep_reward:.4f}")
            torch.save(self.agent.net.state_dict(), os.path.join(self.args.checkpoints, 'latest.pth'))

    def test(self):
        self.agent.net.load_state_dict(torch.load(os.path.join(self.args.checkpoints, 'latest.pth')))
        state = self.test_env.reset()
        done = False
        
        results = []
        # 新增：用于存储每个资产每一步的原始收益率
        all_asset_returns = []

        while not done:
            action, _, _ = self.agent.select_action(state, deterministic=True)
            
            # 为了获取单资产收益，我们需要在 step 之前记录日期
            # 或者从 env 的 next_rets 中提取。为了不破坏 Env 封装，我们在 info 里加一下
            next_state, reward, done, info = self.test_env.step(action)
            
            # 我们需要在 step 逻辑里把 next_rets 传出来，
            # 如果你没有修改 Env，建议在 TimingEnv.step 的 info 里加入: 'raw_rets': next_rets
            results.append(info)
            state = next_state
        
        res_df = pd.DataFrame(results)

        # --- 新增：处理单个资产的固定权重收益 (Buy & Hold) ---
        # 假设每个 info 里都有 'raw_rets' (需要在 Env.step 的 info 字典里加上这一行)
        # 如果不想改 Env，可以通过 res_df['pos'] 和 pnl 逆推，但建议直接改 Env 
        # 这里假设你已经在 Env.step 的 info 中添加了 info['raw_rets'] = next_rets
        raw_rets_matrix = np.stack([res['raw_rets'] for res in results]) # (T, n_assets)
        pos_matrix = np.stack([res['pos'] for res in results])

        n_assets = self.test_env.n_assets
        fixed_weight = 1.0 / n_assets

        # 计算各项指标
        res_df['excess_ret'] = res_df['agent_ret'] - res_df['bench_ret']
        res_df['cum_agent'] = (1 + res_df['agent_ret']).cumprod()
        res_df['cum_bench'] = (1 + res_df['bench_ret']).cumprod()
        res_df['cum_excess'] = res_df['excess_ret'].cumsum()
        
        # --- 可视化部分 ---
        plt.figure(figsize=(14, 25))

        # 1. 累计收益对比图 (模型 vs 基准 vs 各个独立资产)
        plt.subplot(6, 1, 1)
        # 绘制模型和基准
        plt.plot(res_df['cum_agent'], label='RL Strategy (Dynamic)', color='red', linewidth=3)
        plt.plot(res_df['cum_bench'], label='Benchmark (Equal Weight)', color='blue', linestyle='--', linewidth=2)
        
        # 绘制单个资产的收益曲线 (假设满仓该资产，不调仓)
        
        plt.title('Strategy vs Individual Assets Performance')
        plt.ylabel('Cumulative Return')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.grid(True)

        plt.subplot(6, 1, 2)
        colors = plt.cm.get_cmap('tab10', n_assets)
        for i in range(n_assets):
            asset_name = self.test_env.assets[i]
            
            # 1. 计算该资产在固定权重(1/N)下的累计收益
            # 收益 = 1 + (单日收益 * 1/N)，然后累乘
            cum_fixed = (1 + raw_rets_matrix[:, i] * fixed_weight).cumprod()
            
            # 2. 计算该资产在模型动态权重下的累计收益
            # 收益 = 1 + (单日收益 * 模型权重)，然后累乘
            cum_dynamic = (1 + raw_rets_matrix[:, i] * pos_matrix[:, i]).cumprod()
            
            # 绘制固定权重线 (虚线)
            plt.plot(cum_fixed, linestyle='--', color=colors(i), alpha=0.4, 
                     label=f'{asset_name} (Fixed 1/{n_assets})')
            
            # 绘制动态权重线 (实线)
            plt.plot(cum_dynamic, linestyle='-', color=colors(i), linewidth=1.5,
                     label=f'{asset_name} (RL Dynamic)')
            
        plt.title('Figure 2: Individual Asset Contribution (Fixed vs Dynamic)', fontsize=14)
        plt.ylabel('Cumulative Return Contribution')
        # 将图例放在右侧，避免遮挡
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
        plt.grid(True, alpha=0.3)

        # 2. Alpha 曲线 (保持不变)
        plt.subplot(6, 1, 3)
        plt.plot(res_df['cum_excess'], label='Cumulative Excess Return', color='black')
        plt.fill_between(res_df.index, res_df['cum_excess'], 0, alpha=0.2, color='gray')
        plt.title('Alpha (Strategy - Benchmark)')
        plt.legend(); plt.grid(True)

        # 3. 资产仓位堆叠图 (保持不变)
        plt.subplot(6, 1, 4)
        pos_array = np.stack(res_df['pos'].values)
        cash_array = 1.0 - np.sum(pos_array, axis=1)
        labels = [f'Asset {a}' for a in self.test_env.assets] + ['CASH']
        stack_data = np.column_stack([pos_array, cash_array])
        plt.stackplot(res_df.index, stack_data.T, labels=labels, alpha=0.8)
        plt.title('Portfolio Composition (Allocation)')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.grid(True, alpha=0.3)

        # 4. 现金流水平图 (保持不变)
        plt.subplot(6, 1, 5)
        plt.plot(res_df.index, cash_array, label='Cash Weight (Idle)', color='green')
        plt.title('Risk Exposure (1 - Total Asset Weight)')
        plt.ylim(-0.05, 1.05); plt.legend(); plt.grid(True)

        # 5. 单日收益分布对比 (新增：查看模型是否规避了极端回撤)
        plt.subplot(6, 1, 6)
        plt.hist(res_df['agent_ret'], bins=50, alpha=0.5, label='Strategy Ret Distribution', color='red')
        plt.hist(res_df['bench_ret'], bins=50, alpha=0.5, label='Benchmark Ret Distribution', color='blue')
        plt.title('Return Distribution Comparison')
        plt.legend(); plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.args.res_path, 'detailed_analysis.png'))
        plt.show()
    '''
    def test(self):
        # ... 加载模型逻辑保持不变 ...
        self.agent.net.load_state_dict(torch.load(os.path.join(self.args.checkpoints, 'latest.pth')))
        state = self.test_env.reset()
        done = False
        
        results = []
        while not done:
            action, _, _ = self.agent.select_action(state, deterministic=True)
            next_state, reward, done, info = self.test_env.step(action)
            # info 里面现在包含了 'pos' (数组) 和 'agent_ret' 等
            results.append(info)
            state = next_state
        
        res_df = pd.DataFrame(results)
        # 计算各项指标
        res_df['excess_ret'] = res_df['agent_ret'] - res_df['bench_ret']
        res_df['cum_agent'] = (1 + res_df['agent_ret']).cumprod()
        res_df['cum_bench'] = (1 + res_df['bench_ret']).cumprod()
        res_df['cum_excess'] = res_df['excess_ret'].cumsum()
        
        res_df.to_csv(os.path.join(self.args.res_path, 'test_results.csv'))
        
        # --- 修改 3: 可视化部分 ---
        # --- 修改 3: 可视化部分 ---
        plt.figure(figsize=(12, 22)) # 稍微增加高度以容纳更多子图

        # 1. 累计收益图
        plt.subplot(4, 1, 1)
        plt.plot(res_df['cum_agent'], label='Multi-Asset Strategy', color='red')
        plt.plot(res_df['cum_bench'], label='Benchmark (Equal Weight)', color='blue', linestyle='--')
        plt.title('Cumulative Returns')
        plt.legend(); plt.grid(True)

        # 2. Alpha 曲线
        plt.subplot(4, 1, 2)
        plt.plot(res_df['cum_excess'], label='Cumulative Excess Return', color='black')
        plt.fill_between(res_df.index, res_df['cum_excess'], 0, alpha=0.2, color='gray')
        plt.title('Alpha Curve')
        plt.legend(); plt.grid(True)

        # 3. 资产仓位堆叠图 (Area Chart)
        plt.subplot(4, 1, 3)
        pos_array = np.stack(res_df['pos'].values)  # shape: (T, n_assets)
        cash_array = 1.0 - np.sum(pos_array, axis=1) # 计算现金部分
        
        # 准备堆叠数据：资产1, 资产2, ..., 资产N, 现金
        labels = [f'Asset {a}' for a in self.test_env.assets] + ['CASH']
        stack_data = np.column_stack([pos_array, cash_array])
        
        # 使用 stackplot 绘制堆叠面积图，能直观看出权重分配
        plt.stackplot(res_df.index, stack_data.T, labels=labels, alpha=0.8)
        plt.title('Portfolio Composition Over Time (Stacked)')
        plt.ylabel('Weight')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1)) # 将图例放到外面防止挡住曲线
        plt.grid(True, alpha=0.3)

        # 4. 现金流水平图 (单独观察现金仓位变化)
        plt.subplot(4, 1, 4)
        plt.plot(res_df.index, cash_array, label='Cash Weight (Idle Funds)', color='green', linewidth=2)
        plt.fill_between(res_df.index, cash_array, 0, color='green', alpha=0.1)
        plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.3) # 参考线
        plt.title('Cash Position (Market Avoidance)')
        plt.ylim(-0.05, 1.05)
        plt.legend(); plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.args.res_path, 'performance_analysis.png'))
        plt.close()
        
    '''