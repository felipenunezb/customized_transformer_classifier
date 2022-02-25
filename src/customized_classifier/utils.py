
import torch.nn as nn
import torch

#Get top predictions, 
def get_top_preds(model, outputs, topk_=3):

    results = {}
    smax = nn.Softmax(dim=-1)

    #Products
    out_prod = smax(outputs.logits)
    top_prods = torch.topk(out_prod, topk_)

    results['label_01'] = {}
    vals, ixs = top_prods
    for ix, val in zip(ixs.tolist()[0], vals.tolist()[0]):
        results['label_01'][model.config.id2label.get(ix,'Unknown')] = val

    #Issues
    out_issue = smax(outputs.logits_v2)
    top_issues = torch.topk(out_issue, topk_)

    results['label_02'] = {}
    vals, ixs = top_issues
    for ix, val in zip(ixs.tolist()[0], vals.tolist()[0]):
        results['label_02'][model.config.id2label_v2.get(str(ix),'Unknown')] = val

    return results
    