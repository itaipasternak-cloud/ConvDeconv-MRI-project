"""
Standalone, importable fit() function for multicoil ConvDecoder fitting.
"""

import copy
import os
import numpy as np
import torch


def fit(net, img_noisy_var, num_channels, net_input, apply_f, mask,
        mask1d=None, scaling_factor=None, num_iter=5000, LR=0.01,
        checkpoint_dir=None, checkpoint_every=1000, dtype=torch.cuda.FloatTensor):

    net_input = net_input.type(dtype)
    p = [x for x in net.parameters()]
    mse_wrt_noisy = np.zeros(num_iter)

    print("optimize with adam", LR)
    optimizer = torch.optim.Adam(p, lr=LR)
    mse = torch.nn.MSELoss()

    best_net = copy.deepcopy(net)
    best_mse = 1000000.0

    for i in range(num_iter):
        def closure():
            optimizer.zero_grad()
            out = net(net_input.type(dtype))
            loss = mse(apply_f(out, mask), img_noisy_var)
            loss.backward()
            mse_wrt_noisy[i] = loss.data.cpu().numpy()
            if i % 10 == 0:
                print("Iteration %05d    Train loss %f " % (i, loss.data), "\r", end="")
            return loss

        loss = optimizer.step(closure)

        if best_mse > 1.005 * loss.data:
            best_mse = loss.data
            best_net = copy.deepcopy(net)

        if checkpoint_dir is not None and (i % checkpoint_every == 0 or i == num_iter - 1):
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save({
                "iteration": i,
                "model_state_dict": best_net.state_dict(),
                "net_input": net_input,
                "mask": mask,
                "mask1d": mask1d,
                "scaling_factor": scaling_factor,
            }, os.path.join(checkpoint_dir, "checkpoint.pt"))

    net = best_net
    return mse_wrt_noisy, net
