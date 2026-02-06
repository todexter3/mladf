import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta, Dirichlet


class ActorCritic(nn.Module):
    def __init__(self, args, state_dim, n_assets=3, hidden_dim=128, n_layers=2):
        super().__init__()

        self.args = args
        self.n_assets = n_assets
        self.feature_dim = args.feature_dim
        self.action_dim = n_assets + 1  # assets + risk_budget

        # ================= Feature extractor =================
        layers = [nn.Linear(state_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()] #
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()] #
        self.feature = nn.Sequential(*layers)

        # ================= Actor heads =================
        # —— 资产权重：Dirichlet concentration 参数
        self.asset_conc_head = nn.Linear(hidden_dim, self.action_dim)

        # —— risk budget：Beta 分布
        #self.alpha_risk = nn.Linear(hidden_dim, 1)
        #self.beta_risk  = nn.Linear(hidden_dim, 1)

        # ================= Critic =================
        self.value_head = nn.Linear(hidden_dim, 1)

        # ---------- 初始化 ----------
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

        # risk budget 初始化为 mean ≈ 0.5
        # Beta(a,b) 的均值 = a / (a+b)
        #nn.init.constant_(self.alpha_risk.bias, 0.5)
        #nn.init.constant_(self.beta_risk.bias,  0.5)

    # -------------------------------------------------------
    def forward(self, state):
        z = self.feature(state)

        # Dirichlet concentration（必须 > 0）
        asset_conc = F.softplus(self.asset_conc_head(z)) + 0.1  # (B, n_assets)

        # risk budget Beta
        #alpha_r = F.softplus(self.alpha_risk(z)) + 0.1
        #beta_r  = F.softplus(self.beta_risk(z))  + 0.1

        value = self.value_head(z)

        return asset_conc, value

    # -------------------------------------------------------
    def act(self, state, deterministic=False):
        mask_start = self.n_assets * self.feature_dim + self.n_assets
        mask = state[:, mask_start : mask_start + self.n_assets]

        asset_conc, value = self.forward(state)

        effective_conc = asset_conc * (1 - mask) + 1e-6 * mask

        dist_asset = Dirichlet(effective_conc)

        
        #dist_risk  = Beta(alpha_r, beta_r)
        

        if deterministic:
            asset_w = effective_conc / effective_conc.sum(dim=-1, keepdim=True)
            #risk_w  = alpha_r / (alpha_r + beta_r)
        else:
            asset_w = dist_asset.sample()
            #risk_w  = dist_risk.sample()

        #action = torch.cat([asset_w, risk_w], dim=-1)
        action=asset_w

        log_prob = (
            dist_asset.log_prob(asset_w)
        )

        return action, log_prob, value

    # -------------------------------------------------------
    def evaluate(self, state, action):
        asset_conc, value = self.forward(state)

        #asset_action = action[:, :self.n_assets]
        #risk_action  = action[:, self.n_assets:]

        dist_asset = Dirichlet(asset_conc)
        #dist_risk  = Beta(alpha_r, beta_r)

        log_prob = (
            dist_asset.log_prob(action)
    
        )

        entropy = (
            dist_asset.entropy()
        )

        return log_prob, entropy, value.squeeze(-1)
