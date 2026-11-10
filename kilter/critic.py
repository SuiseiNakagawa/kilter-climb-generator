"""Load the trained graph model and grade any list of climbs with it."""
from __future__ import annotations

import numpy as np
import torch

from .data import PROJECT_DIR, Climb
from .graph import ClimbGNN, build_graph


class GradeCritic:
    def __init__(self, path=PROJECT_DIR / "models" / "gnn.pt"):
        ckpt = torch.load(path, weights_only=False)
        self.placement_index = ckpt["placement_index"]
        self.mean, self.std = ckpt["mean"], ckpt["std"]
        self.model = ClimbGNN(len(self.placement_index), dim=ckpt["args"]["dim"], layers=ckpt["args"]["layers"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    @torch.no_grad()
    def predict(self, climbs: list[Climb], batch_size: int = 1024) -> np.ndarray:
        """Numeric difficulty for each climb (use grades.to_v_number for V-grades)."""
        from .graph import collate
        graphs = [build_graph(c, self.placement_index) for c in climbs]
        out = [self.model(collate(graphs[i:i + batch_size])) * self.std + self.mean
               for i in range(0, len(graphs), batch_size)]
        return torch.cat(out).numpy()
