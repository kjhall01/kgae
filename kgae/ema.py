import copy
import torch

class EMA:
    def __init__(self, model, decay=0.995):
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_avg = (
                    self.decay * self.shadow[name]
                    + (1.0 - self.decay) * param.detach()
                )
                self.shadow[name] = new_avg.clone()

    def apply_shadow(self, model):
        """Load EMA weights into model (for eval)."""
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.detach().clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        """Restore original weights after eval."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}
