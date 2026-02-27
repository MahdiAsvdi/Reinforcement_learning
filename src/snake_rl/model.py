import copy
import os

import torch
import torch.nn as nn
import torch.optim as optim


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        if isinstance(hidden_size, int):
            hidden_sizes = [hidden_size, hidden_size]
        else:
            hidden_sizes = list(hidden_size)

        layers = []
        last_size = input_size
        for layer_size in hidden_sizes:
            layers.append(nn.Linear(last_size, layer_size))
            layers.append(nn.ReLU())
            last_size = layer_size

        layers.append(nn.Linear(last_size, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def save(self, file_name="model.pth", folder_path="./model"):
        os.makedirs(folder_path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(folder_path, file_name))

    def load(self, file_name="model.pth", folder_path="./model", map_location=None):
        path = os.path.join(folder_path, file_name)
        state_dict = torch.load(path, map_location=map_location)
        self.load_state_dict(state_dict)
        self.eval()


class QTrainer:
    def __init__(
        self,
        model,
        lr,
        gamma,
        tau=0.01,
        grad_clip=5.0,
        weight_decay=1e-5,
        device=None,
    ):
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.grad_clip = grad_clip

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.online_model = model.to(self.device)
        self.target_model = copy.deepcopy(model).to(self.device)
        self.target_model.eval()

        # Backward compatibility with previous code paths.
        self.model = self.online_model

        self.optimizer = optim.AdamW(
            self.online_model.parameters(), lr=self.lr, weight_decay=weight_decay
        )
        self.criterion = nn.SmoothL1Loss()

    def _soft_update_target(self):
        with torch.no_grad():
            for target_param, online_param in zip(
                self.target_model.parameters(), self.online_model.parameters()
            ):
                target_param.data.mul_(1.0 - self.tau)
                target_param.data.add_(online_param.data, alpha=self.tau)

    def train_step(self, state, action, reward, next_state, done):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        next_state = torch.as_tensor(next_state, dtype=torch.float32, device=self.device)
        action = torch.as_tensor(action, dtype=torch.long, device=self.device)
        reward = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        done = torch.as_tensor(done, dtype=torch.bool, device=self.device)

        # Single sample -> batch size 1
        if state.ndim == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = done.unsqueeze(0)

        # One-hot actions -> indices
        if action.ndim > 1:
            action_idx = action.argmax(dim=1, keepdim=True)
        else:
            action_idx = action.view(-1, 1)

        q_pred_all = self.online_model(state)
        q_pred = q_pred_all.gather(1, action_idx).squeeze(1)

        with torch.no_grad():
            # Double DQN: select action by online net, evaluate by target net.
            next_action_idx = self.online_model(next_state).argmax(dim=1, keepdim=True)
            next_q = self.target_model(next_state).gather(1, next_action_idx).squeeze(1)
            q_target = reward + (~done).float() * self.gamma * next_q

        self.optimizer.zero_grad(set_to_none=True)
        loss = self.criterion(q_pred, q_target)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_model.parameters(), self.grad_clip)
        self.optimizer.step()

        self._soft_update_target()
        return float(loss.item())


