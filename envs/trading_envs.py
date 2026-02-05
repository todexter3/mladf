import numpy as np
import collections


class TimingEnv:
    def __init__(self, df, args):
        """
        df: 必须包含列
            - date
            - asset
            - ma5_ratio
            - vol_ratio
            - rise_fall_norm   (作为 next return)
        """
        self.df = df.sort_values(['date', 'asset']).reset_index(drop=True)
        self.args = args

        # ===== 基本维度 =====
        self.assets = sorted(self.df['asset'].unique())
        self.n_assets = len(self.assets)
        self.asset2idx = {a: i for i, a in enumerate(self.assets)}

        self.unique_dates = sorted(self.df['date'].unique())
        self.max_step = len(self.unique_dates) - 1
        self.current_step = 0

        self.window_size = args.window_size
        self.feat_dim = 3  # ma5_ratio, vol_ratio, rise_fall_norm

        # ===== 历史收益（可选）=====
        self.ret_history = {
            asset: collections.deque(maxlen=self.window_size)
            for asset in self.assets
        }

        # ===== 仓位 =====
        self.pos = np.zeros(self.n_assets, dtype=np.float32)
        self.prev_pos = np.zeros(self.n_assets, dtype=np.float32)

        # ===== state_dim（给 PPO 用）=====
        self.state_dim = self.n_assets * self.feat_dim + self.n_assets+1
        self.action_dim = self.n_assets + 1

    # ------------------------------------------------------------------
    def reset(self):
        self.current_step = 0
        self.pos[:] = 0.0
        self.prev_pos[:] = 0.0

        for asset in self.assets:
            self.ret_history[asset].clear()

        return self._get_obs()

    # ------------------------------------------------------------------
    def _get_obs(self):
        """
        obs = [每个资产的 3 个特征, 当前仓位]
        维度恒定: n_assets * 3 + n_assets
        """
        curr_date = self.unique_dates[self.current_step]
        day_data = self.df[self.df['date'] == curr_date]

        feats = np.zeros(self.n_assets * self.feat_dim, dtype=np.float32)

        for _, row in day_data.iterrows():
            idx = self.asset2idx[row['asset']]
            feats[idx * self.feat_dim:(idx + 1) * self.feat_dim] = [
                row['ma5_ratio'],
                row['vol_ratio'],
                row['rise_fall_norm']
            ]

        market_ret = np.mean([row['rise_fall_norm'] for _, row in day_data.iterrows()])

        obs = np.concatenate([feats, self.pos, [market_ret]], axis=0)

        return obs

    def step(self, actions):

        asset_weights = actions[:self.n_assets]
        risk_budget  = np.clip(actions[self.n_assets], 0.0, 1.0)

        asset_weights = risk_budget * asset_weights
        cash_weight   = 1.0 - risk_budget



        # ===== 下一期收益 =====
        next_date = self.unique_dates[self.current_step + 1]
        next_day_data = self.df[self.df['date'] == next_date]
        next_rets = np.zeros(self.n_assets, dtype=np.float32)

        for _, row in next_day_data.iterrows():
            idx = self.asset2idx[row['asset']]
            next_rets[idx] = row['rise_fall_norm']

        # ===== PnL =====
        pnl = np.sum(asset_weights * next_rets)
        port_ret = pnl


        bench_ret = np.mean(next_rets)

   

        # ===== 风险调整 =====
        vol = np.std(next_rets) + 1e-6
        excess_ret = port_ret - bench_ret
        risk_adj_reward = excess_ret / vol



        # ===== 换手成本（anneal）=====
        warmup_steps = 200
        effective_commission = self.args.commission * min(1.0, self.current_step / warmup_steps)
        turnover_cost = effective_commission * np.sum(np.abs(asset_weights - self.pos))

        
        # ===== cash penalty =====
        cash_penalty = 0.0
        cash_threshold = 0.5
        cash_penalty_coef = 0.1

        if cash_weight > cash_threshold:
            cash_penalty = cash_penalty_coef * (cash_weight - cash_threshold)

         
        # 每一步随机挑一个资产
        
        reward = risk_adj_reward -turnover_cost- cash_penalty

        # ===== 状态更新 =====
        self.prev_pos = self.pos.copy()
        self.pos = asset_weights.copy()
        self.current_step += 1

        done = self.current_step >= self.max_step - 1

        info = {
            'agent_ret': port_ret,
            'bench_ret': bench_ret,
            'raw_rets': next_rets.copy(),
            'cash_weight': cash_weight,
            'pos': asset_weights.copy()
        }

        return self._get_obs(), reward, done, info


       
