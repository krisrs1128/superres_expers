#!/usr/bin/env python
import torch

def save_checkpoint(model, optimizer, path):
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }

    torch.save(state, path)
