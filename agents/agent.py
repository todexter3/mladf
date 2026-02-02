import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.PPO import ActorCritic


class PPOAgent:
    def __init__(self, state_dim, args):
        self.args = args
        self.device = args.device

        self.net = ActorCritic(
            args=self.args,
            state_dim=state_dim,
            n_assets=args.n_assets,
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers
        ).to(self.device)

        self.optimizer = optim.Adam(self.net.parameters(), lr=args.lr)
        self.buffer = []

    # Action selection
    def select_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_raw, logp, value = self.net.act(state, deterministic)

        # --- 修复错误 3: 处理多资产动作向量 ---
        # 因为现在是多资产，action_raw 是一个形状为 [1, n_assets] 的张量
        # 不能用 .item()，必须转为 numpy 数组并去掉 batch 维度
        action = action_raw.squeeze(0).cpu().numpy()

        return (
            action,         # 现在返回的是 numpy 数组，例如 [0.3, 0.4, 0.1]
            logp.item(),    # logp 已经是 sum(-1) 后的标量，可以用 .item()
            value.item()
        )


    def store_transition(self, transition):
        self.buffer.append(transition)


    def update(self):
        states = torch.as_tensor(
            np.stack([t[0] for t in self.buffer]),
            dtype=torch.float32,
            device=self.device
        )

        actions = torch.as_tensor(
            np.stack([t[1] for t in self.buffer]), # 使用 np.stack 堆叠数组
            dtype=torch.float32,
            device=self.device
        )

        rewards = torch.as_tensor(
            [t[2] for t in self.buffer],
            dtype=torch.float32,
            device=self.device
        )

        dones = torch.as_tensor(
            [1.0 if bool(t[3]) else 0.0 for t in self.buffer],
            dtype=torch.float32,
            device=self.device
        )

        old_logps = torch.as_tensor(
            [t[4] for t in self.buffer],
            dtype=torch.float32,
            device=self.device
        )

        old_values = torch.as_tensor(
            [t[5] for t in self.buffer],
            dtype=torch.float32,
            device=self.device
        )

        # GAE
        T = rewards.size(0)
        advantages = torch.zeros(T, device=self.device)
        returns = torch.zeros(T, device=self.device)
        gae = 0.0

        for t in reversed(range(T)):
            # 这里 next_value 需要处理最后一步
            if t == T - 1:
                next_value = 0.0
            else:
                next_value = old_values[t + 1]

            delta = rewards[t] + self.args.gamma * next_value * (1.0 - dones[t]) - old_values[t]
            gae = delta + self.args.gamma * self.args.gae_lambda * (1.0 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = gae + old_values[t]

        # normalize advantage
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Optimization
        for _ in range(self.args.n_epochs):
            logps, entropy, values = self.net.evaluate(states, actions)
            values = values.squeeze(-1)

            ratio = torch.exp(logps - old_logps)
            surr1 = ratio * advantages
            surr2 = torch.clamp(
                ratio,
                1 - self.args.clip_eps,
                1 + self.args.clip_eps
            ) * advantages

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * (returns - values).pow(2).mean()
            entropy_loss = entropy.mean()

            loss = (
                policy_loss
                + self.args.vf_coef * value_loss
                - self.args.ent_coef * entropy_loss
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
            self.optimizer.step()

        self.buffer.clear()