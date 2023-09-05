import torch
import torch.nn as nn

def gradident_penalty(crit,real,fake,device="cpu"):
    BATCH_SIZE,C,H,W,L = real.shape
    epsilon = torch.rand(BATCH_SIZE, 1, 1, 1, 1).repeat(1,C,H,W,L).to(device)
    mixed_images = real * epsilon + fake * (1 - epsilon)
    mixed_scores = crit(mixed_images)

    # take the gradident of scores with respect to images
    gradient = torch.autograd.grad(
        inputs=mixed_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores), 
        create_graph=True,
        retain_graph=True,
    )[0]

    gradient = gradient.view(len(gradient), -1)
    gradient_norm = gradient.norm(2, dim=1)
    penalty = torch.mean( (gradient_norm-1)**2 )
    return penalty
