"""Small, explicitly action-conditioned multi-camera latent dynamics."""
import torch
from torch import nn


class TriViewDynamics(nn.Module):
    def __init__(self, feature_dim=768, action_dim=14, views=3, hidden=256):
        super().__init__()
        self.config = dict(feature_dim=feature_dim, action_dim=action_dim,
                           views=views, hidden=hidden)
        self.visual = nn.Sequential(nn.Linear(views * feature_dim, hidden), nn.GELU())
        self.actions = nn.GRU(action_dim * 2 + 1, hidden // 2, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + hidden // 2, hidden), nn.GELU(),
                                  nn.Linear(hidden, views * feature_dim))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, initial, action, time):
        if initial.ndim != 3 or action.ndim != 3 or time.shape != action.shape[:2] + (1,):
            raise ValueError('Expected initial[B,V,D], actions[B,T,A], time[B,T,1]')
        if initial.shape[1:] != (self.config['views'], self.config['feature_dim']):
            raise ValueError('Feature shape differs from the training configuration')
        if action.shape[-1] != self.config['action_dim'] or action.shape[0] != initial.shape[0]:
            raise ValueError('Action convention/shape differs from training')
        prefix, _ = self.actions(torch.cat((action, action - action[:, :1], time), dim=-1))
        visual = self.visual(initial.flatten(1))[:, None].expand(-1, action.shape[1], -1)
        delta = self.head(torch.cat((visual, prefix), dim=-1))
        # Enforce exact initial observation and continuous time-dependent residual.
        delta = delta.reshape(initial.shape[0], action.shape[1], *initial.shape[1:])
        return initial[:, None] + delta * time[..., None]
