import argparse

import torch

from utils import get_network_npmc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', type=bool, default=False)
    parser.add_argument('-save-path',type=str, default='exported_model.pt',help='model path to be saved')

    args = parser.parse_args()

    model = get_network_npmc(args)
    model.eval()
    graph = torch.fx.Tracer().trace(model)
    traced_model = torch.fx.GraphModule(model, graph)
    torch.save(traced_model, args.save_path)