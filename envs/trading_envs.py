import numpy as np
import collections


class TimingEnv:
    def __init__(self, df, args, all_assets=None):
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
        self.assets = all_assets if all_assets is not None else sorted(self.df['asset'].unique())
        self.n_assets = len(self.assets)
        self.asset2idx = {a: i for i, a in enumerate(self.assets)}

        self.unique_dates = sorted(self.df['date'].unique())
        self.max_step = len(self.unique_dates) - 1
        self.current_step = 0

        self.window_size = args.window_size
        self.feature_dim = args.feature_dim  # ma5_ratio, vol_ratio, rise_fall_norm

        # ===== 历史收益（可选）=====
        self.ret_history = {
            asset: collections.deque(maxlen=self.window_size)
            for asset in self.assets
        }

        # ===== 仓位 =====
        self.pos = np.zeros(self.n_assets, dtype=np.float32)
        self.prev_pos = np.zeros(self.n_assets, dtype=np.float32)

        # ===== state_dim（给 PPO 用）=====
        self.state_dim = self.n_assets * self.feature_dim + self.n_assets+ self.n_assets +1
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

        feats = np.zeros(self.n_assets * self.feature_dim, dtype=np.float32)
        masks = np.ones(self.n_assets, dtype=np.float32)

        for _, row in day_data.iterrows():
            idx = self.asset2idx[row['asset']]
            masks[idx] = 0.0
            feats[idx * self.feature_dim : (idx + 1) * self.feature_dim] = [
                row['close_feat'],
                row['open_feat'],
                row['high_feat'],
                row['low_feat'],
                row['vol_feat'],
                row['ret_feat'],
                row['main_ret_slp_feat'],
                row['tr_feat'],
                row['capvol0_feat']
            ]

        market_ret = np.mean([row['rise_fall_norm'] for _, row in day_data.iterrows()])

        obs = np.concatenate([feats, self.pos, masks, [market_ret]], axis=0)

        return obs

    def step(self, actions):

        asset_weights = actions[:self.n_assets]
        cash_weight   = actions[self.n_assets]

        '''
        import random
        risk_budget =random.uniform(0,1)

        #risk_budget  = np.clip(actions[self.n_assets], 0.0, 1.0)

        asset_weights = risk_budget * asset_weights
        
        cash_weight   = 1.0 - risk_budget
        '''

        # ===== 下一期收益 =====
        next_date = self.unique_dates[self.current_step + 1]
        next_day_data = self.df[self.df['date'] == next_date]

        next_rets = np.zeros(self.n_assets, dtype=np.float32)

        for _, row in next_day_data.iterrows():
            idx = self.asset2idx[row['asset']]
            next_rets[idx] = row['rise_fall_norm']

        # ===== PnL =====
        linear_rets = np.exp(next_rets) - 1.0
        gross_pnl = np.sum(self.pos * linear_rets)

        port_ret = gross_pnl
        


        bench_ret = np.mean(next_rets)

        turnover = np.sum(np.abs(asset_weights - self.pos))
        commission_cost = turnover * self.args.commission

        reward = gross_pnl - commission_cost

   

        



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


       
