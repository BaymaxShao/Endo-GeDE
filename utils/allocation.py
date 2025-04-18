import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from models import gvemode


def exponential_scaling(values, target_sum, exponent):
    values = np.array(values)
    scaled_values = np.power(values, exponent)
    scaled_integers = np.round((scaled_values / scaled_values.sum()) * target_sum).astype(int)

    while scaled_integers.sum() != target_sum:
        difference = target_sum - scaled_integers.sum()
        if difference > 0:
            scaled_integers[np.argmin(scaled_values - scaled_integers)] += 1
        else:
            scaled_integers[np.argmax(scaled_values - scaled_integers)] -= 1

    return scaled_integers


def fix_finger(w, bins=100, pl_fitting=True, EVALS_THRESH=1e-4, filter_zeros=False):
    eigs = torch.square(torch.linalg.svdvals(w).flatten())
    eigs, _ = torch.sort(eigs, descending=False)

    if filter_zeros:
        nz_eigs = eigs[eigs > EVALS_THRESH]
        N = len(nz_eigs)
    else:
        # print(f"{name} Skip Filter Zero")
        nz_eigs = eigs
        N = len(nz_eigs)

    log_nz_eigs = torch.log(nz_eigs)
    alphas = torch.zeros(N - 1)
    Ds = torch.ones(N - 1)
    if pl_fitting:
        hist_nz_eigs = torch.log10(nz_eigs)
        min_e, max_e = hist_nz_eigs.min(), hist_nz_eigs.max()
        counts = torch.histc(hist_nz_eigs, bins, min=min_e, max=max_e)
        boundaries = torch.linspace(min_e, max_e, bins + 1)
        h = counts, boundaries
        ih = torch.argmax(h[0])
        xmin2 = 10 ** h[1][ih]
        xmin_min = torch.log10(0.95 * xmin2)
        xmin_max = 1.5 * xmin2

    for i, xmin in enumerate(nz_eigs[:-1]):
        if pl_fitting == True:
            if xmin < xmin_min:
                continue
            if xmin > xmin_max:
                break

        n = float(N - i)
        #seq = torch.arange(n).cuda(nz_eigs.device)
        alpha = 1 + n / (torch.sum(log_nz_eigs[i:]) - n * log_nz_eigs[i])
        alphas[i] = alpha
        if alpha > 1:
            seq = torch.arange(n, device=nz_eigs.device)
            Ds[i] = torch.max(torch.abs(
                1 - (nz_eigs[i:] / xmin) ** (-alpha + 1) - seq / n
            ))

    min_D_index = torch.argmin(Ds)
    final_alpha = alphas[min_D_index]

    return final_alpha


def find_layers(module, layers=[nn.Linear], name=''):

    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def calculate_expert(model):
    all_layer_alpha = []
    layers = model.encoder.blocks

    for i, layer in enumerate(layers):

        subset = find_layers(layer)
        print(f"Processing layer {i + 1}--subset--{subset}")
        layer_final_alpha = []
        for name in subset:
            # if name == 'attn.proj':
            #     continue
            layer_final_alpha.append(fix_finger(subset[name].weight.data.float()))
        all_layer_alpha.append(torch.stack(layer_final_alpha).mean().item())
        print(f"alpha value of layer {i+1} ---{torch.stack(layer_final_alpha).mean().item()} ")

    torch.cuda.empty_cache()
    return all_layer_alpha

def allocate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=25)
    parser.add_argument('--beta', type=float, default=2.5)
    parser.add_argument('--target_sum', type=int, default=55)

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    model = gvemode.gv_emode(
                backbone_size="base", r=4, peft_type=None,
                image_shape=(224, 280), pretrained_path='pretrained_model',
                residual_block_indexes=[])
    model.eval()

    distribution = calculate_expert(model)

    print("Distribution:", distribution)
    quantized_vector = exponential_scaling(distribution, args.target_sum, args.beta)
    print("Total expert number:", sum(quantized_vector))
    print("expert allocation: ", ','.join(map(str, quantized_vector)))

    topkk = [2 if n > 1 else 1 for n in quantized_vector]
    topkk = ','.join(map(str, topkk))
    print("top_k: ", topkk)

    return quantized_vector, topkk
