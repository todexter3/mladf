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

    # ------------------------------------------------------------------
    def step(self, actions):
        """
        actions: np.ndarray, shape = (n_assets,)
        PPOAgent 已保证这一点
        """
        # ===== 1. 动作约束 =====
        asset_logits = actions[:self.n_assets]
        risk_budget = actions[self.n_assets]   # ∈ [0,1]，这是“总风险敞口”

        # 资产内部做 softmax（只在资产内部）
        asset_weights_raw = asset_logits / (np.sum(asset_logits) + 1e-9)

        asset_weights = risk_budget * asset_weights_raw

        cash_weight = 1.0 - risk_budget


        

        # ===== 2. 取下一期收益（严格按 asset 对齐）=====
        next_date = self.unique_dates[self.current_step + 1]
        next_day_data = self.df[self.df['date'] == next_date]
        next_rets = np.zeros(self.n_assets, dtype=np.float32)



        for _, row in next_day_data.iterrows():
            idx = self.asset2idx[row['asset']]
            next_rets[idx] = row['rise_fall_norm']

        # ===== 3. 计算 PnL & 成本 =====
        rf = 0.0  # 或者用 short-term rate

        pnl = np.sum(asset_weights * next_rets)
        cash_ret = cash_weight * rf

        port_ret = pnl + cash_ret

        # benchmark：全风险资产等权
        bench_ret = np.mean(next_rets)

        reward = (
            port_ret
            - bench_ret                     # 核心：超额收益
            - self.args.commission * np.sum(np.abs(asset_weights - self.pos))
)

        



        # ===== 4. 更新状态 =====
        self.prev_pos = self.pos.copy()
        self.pos = asset_weights.copy()
        self.current_step += 1

        done = self.current_step >= self.max_step - 1

        info = {
            'agent_ret': pnl,
            'bench_ret': bench_ret,
            'pos': asset_weights.copy(),
            'total_weight': float(np.sum(asset_weights)),
            'raw_rets': next_rets.copy()
        }

        # ===== 5. 防御性断言（强烈建议保留）=====
        assert self.pos.shape == next_rets.shape, \
            f"pos {self.pos.shape}, rets {next_rets.shape}"

        return self._get_obs(), reward, done, info