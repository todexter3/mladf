import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from envs.trading_envs import TimingEnv
from agents.agent import PPOAgent
from data_loader.data_loader import load_multi_asset_data
import logging

class Exp_PPO:
    def __init__(self, args):
        self.args = args
        self.data = load_multi_asset_data(args.data_path)

        self.all_assets = sorted(self.data['asset'].unique()) 
        self.args.n_assets = len(self.all_assets)
        
        # --- 修改 1: 按日期切分数据，确保不打断资产对齐 ---
        unique_dates = sorted(self.data['date'].unique())

        n = len(unique_dates)
        train_end = unique_dates[int(n * 0.65)]
        eval_end  = unique_dates[int(n * 0.8)]

        train_data = self.data[self.data['date'] < train_end].copy()
        eval_data  = self.data[(self.data['date'] >= train_end) &(self.data['date'] < eval_end)].copy()
        test_data  = self.data[self.data['date'] >= eval_end].copy()

        '''
        self.train_env = TimingEnv(train_data, args)
        self.test_env = TimingEnv(test_data, args)
        self.eval_env = TimingEnv(eval_data, args)
        '''
        self.train_env = TimingEnv(train_data, args, all_assets=self.all_assets)
        self.eval_env  = TimingEnv(eval_data, args, all_assets=self.all_assets)
        self.test_env  = TimingEnv(test_data, args, all_assets=self.all_assets)
        
        sample_obs = self.train_env.reset()
        # agent 内部会自动根据 state_dim 初始化 ActorCritic
        self.agent = PPOAgent(len(sample_obs), args)


        self.train_rewards = []
        self.train_losses = []
        self.train_value_losses = []
        self.train_entropies = []

    def train(self):

        s_train = self.train_env.reset()
        s_eval  = self.eval_env.reset()
        s_test  = self.test_env.reset()

        print("Train state dim:", s_train.shape)
        print("Eval  state dim:", s_eval.shape)
        print("Test  state dim:", s_test.shape)

        best_eval_reward = -np.inf

        for ep in range(self.args.n_epochs):
            state = self.train_env.reset()
            ep_reward = 0
            done = False
            while not done:
                # action 此时是长度为 n_assets 的向量
                action, logp, val = self.agent.select_action(state)
                next_state, reward, done, info = self.train_env.step(action)
                
                
                self.agent.store_transition((state, action, reward, done, logp, val))
                state = next_state
                ep_reward += reward  
                
                if len(self.agent.buffer) >= self.args.buffer_size:
                    loss_info=self.agent.update()

                    if loss_info is not None:
                        self.train_losses.append(loss_info.get('loss', np.nan))
                        self.train_value_losses.append(loss_info.get('value_loss', np.nan))
                        self.train_entropies.append(loss_info.get('entropy', np.nan))

            self.train_rewards.append(ep_reward)

            eval_reward = self.evaluate_policy(self.eval_env)

            logging.info(
                f"[Epoch {ep:03d}] "
                f"TrainReward={ep_reward:.4f} | "
                f"EvalReward={eval_reward:.4f} | "
                f"Loss={self.train_losses[-1] if self.train_losses else 'NA'}"
            )

            torch.save(
                {
                    "model": self.agent.net.state_dict(),
                    "optimizer": self.agent.optimizer.state_dict(),
                    "epoch": ep,
                    "eval_reward": eval_reward,
                },
                os.path.join(self.args.checkpoints, "latest.pth")
            )

            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward

                torch.save(
                    {
                        "model": self.agent.net.state_dict(),
                        "optimizer": self.agent.optimizer.state_dict(),
                        "epoch": ep,
                        "eval_reward": eval_reward,
                    },
                    os.path.join(self.args.checkpoints, "best.pth")
                )

                logging.info(
                    f"New best model saved! EvalReward={best_eval_reward:.4f}"
                )

        self.plot_training_curves()

    def test(self):
        ckpt = torch.load(os.path.join(self.args.checkpoints, "best.pth"),map_location=self.args.device)
        self.agent.net.load_state_dict(ckpt["model"])
        
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
        raw_rets_matrix = np.stack([res['raw_rets'] for res in results])  # (T, n_assets)
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

    def plot_training_curves(self):
        save_path = os.path.join(self.args.res_path, 'training_curves.png')

        fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)

        axes[0].plot(self.train_rewards, label='Episode Reward')
        axes[0].set_title('Training Reward')
        axes[0].legend(); axes[0].grid(True)

        if len(self.train_losses) > 0:
            axes[1].plot(self.train_losses, label='Policy Loss')
            axes[1].set_title('Policy Loss')
            axes[1].legend(); axes[1].grid(True)

        if len(self.train_value_losses) > 0:
            axes[2].plot(self.train_value_losses, label='Value Loss')
            axes[2].set_title('Value Loss')
            axes[2].legend(); axes[2].grid(True)

        if len(self.train_entropies) > 0:
            axes[3].plot(self.train_entropies, label='Entropy')
            axes[3].set_title('Policy Entropy')
            axes[3].legend(); axes[3].grid(True)

        axes[-1].set_xlabel('Update Step / Epoch')

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        logging.info(f"Training curves saved to {save_path}")

    def evaluate_policy(self, env, n_episodes=1):
        """
        Run policy in eval mode, without gradient or buffer update.
        Return average episode reward.
        """
        self.agent.net.eval()

        total_reward = 0.0

        with torch.no_grad():
            for _ in range(n_episodes):
                state = env.reset()
                done = False
                ep_reward = 0.0

                while not done:
                    action, _, _ = self.agent.select_action(
                        state, deterministic=True
                    )
                    state, reward, done, info = env.step(action)
                    ep_reward += reward

                total_reward += ep_reward

        self.agent.net.train()
        return total_reward / n_episodes
