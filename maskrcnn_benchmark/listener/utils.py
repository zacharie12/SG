import torch

def format_scores(scores, true_index, device):
    true_score = scores[true_index]
    score_length = scores.size(0)
    true_tensor = torch.full((score_length,), true_score.item()).to(device)
    binary = torch.ones(score_length).to(device)

    return true_tensor, scores, binary
        

def collate_sgs(sgs, device):
    nodes, pair_idx, edges = sgs

    output = []
    for node, pair, edge in zip(nodes, pair_idx, edges):
        tensor_pair = pair.t().contiguous()
        node = node.to(device)
        tensor_pair = tensor_pair.to(device)
        edge = edge.to(device)
        output.append((node, tensor_pair, edge))

    return output

