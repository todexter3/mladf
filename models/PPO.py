import torch
import torch.nn as nn
from torch.distributions import Beta
import torch.nn.functional as F


class ActorCritic(nn.Module):
    def __init__(self,args, state_dim,n_assets=3, hidden_dim=128, n_layers=2):
        super().__init__()

        self.args=args
        self.n_assets = n_assets
        self.action_dim = n_assets + 1

        layers = []
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        self.feature = nn.Sequential(*layers)

        # Actor heads & Critic head
        self.alpha_asset = nn.Linear(hidden_dim, self.n_assets)
        self.beta_asset  = nn.Linear(hidden_dim, self.n_assets)

        self.alpha_risk = nn.Linear(hidden_dim, 1)
        self.beta_risk  = nn.Linear(hidden_dim, 1)

        self.value_head = nn.Linear(hidden_dim, 1)


        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

        nn.init.constant_(self.alpha_risk.bias, 0.0)
        nn.init.constant_(self.beta_risk.bias,  2.0)

    def forward(self, state):
        z = self.feature(state)

        alpha_a = F.softplus(self.alpha_asset(z)) + 1.0
        beta_a  = F.softplus(self.beta_asset(z))  + 1.0

        alpha_r = F.softplus(self.alpha_risk(z)) + 1.0
        beta_r  = F.softplus(self.beta_risk(z))  + 1.0

        value = self.value_head(z)

        return (alpha_a, beta_a), (alpha_r, beta_r), value

    def act(self, state, deterministic=False):
        (alpha_a, beta_a), (alpha_r, beta_r), value = self.forward(state)

        dist_asset = Beta(alpha_a, beta_a)
        dist_risk  = Beta(alpha_r, beta_r)

        if deterministic:
            asset_raw = alpha_a / (alpha_a + beta_a)
            risk_raw  = alpha_r / (alpha_r + beta_r)
        else:
            asset_raw = dist_asset.sample()
            risk_raw  = dist_risk.sample()


        action = torch.cat([asset_raw, risk_raw], dim=-1)

        log_prob = (
            dist_asset.log_prob(asset_raw).sum(dim=-1)
            + dist_risk.log_prob(risk_raw).sum(dim=-1)
        )

        return action, log_prob, value


    def evaluate(self, state, action):

        (alpha_a, beta_a), (alpha_r, beta_r), value = self.forward(state)

        asset_action = action[:, :self.n_assets]
        risk_action  = action[:, self.n_assets:]

        dist_asset = Beta(alpha_a, beta_a)
        dist_risk  = Beta(alpha_r, beta_r)

        log_prob = (
            dist_asset.log_prob(asset_action).sum(dim=-1)
            + dist_risk.log_prob(risk_action).sum(dim=-1)
        )

        entropy = (
            dist_asset.entropy().sum(dim=-1)
            + dist_risk.entropy().sum(dim=-1)
        )

        return log_prob, entropy, value.squeeze(-1)