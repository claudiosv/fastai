from .imports import *
from .torch_imports import *



# There are 2 versions of each metrics function, depending on the type of the prediction tensor:
# *    torch preds/log_preds
# *_np numpy preds/log_preds
#

def mrr_non_interactive(preds, targs):
    summ = .0
    total = to_np(preds).shape[0]
    for pred, targ in zip(to_np(preds), to_np(targs)):
        rank = find_sorted_array_position(pred, pred[targ])
        summ += float(1) / rank
    return summ / total
def mrr(preds, targs):
    if isinstance(targs, Variable): targs = targs.data
    pred_values = preds.gather(1, targs.view(-1, 1))
    guessed_positions = find_sorted_array_position_tensor(preds, pred_values)
    return torch.mean(torch.reciprocal(guessed_positions.float()))

def MRR(gen_preds_targs, n_batches):
    batch_count = 0
    count_all=0
    summ = .0
    tqdm1 = tqdm(gen_preds_targs, leave=False, total=n_batches)
    for pred, targ, input in tqdm1:
        for i_in_batch, (pr, t) in enumerate(zip(pred, targ)):
            rank = find_sorted_array_position(pr, pr[t])
            summ += float(1) / rank
            count_all +=1
        batch_count += 1
        temporary_result = {"mrr": f"{summ/count_all:.3}"}
        tqdm1.set_postfix(**temporary_result)
    return temporary_result

def find_sorted_array_position(np_array, value):
    count = 1
    for elm in np_array:
        if elm > value:
            count += 1
    return count


def find_sorted_array_position_tensor(tensor, values_tensor):
    dim0, dim1 = tensor.shape
    return torch.sum(tensor > values_tensor.resize_((dim0, 1)).expand(dim0, dim1), 1).add(1)

def top_k(gen_preds_targs, n_batches, ks, cat=None):
    cat = cat if cat else len(cat)
    sorted_ks = sorted([k * -1 for k in ks])
    min_k = sorted_ks[0]
    batch_count = 0
    count_true = [0] * len(ks)
    count_all=0
    examples = []
    tqdm1 = tqdm(gen_preds_targs, leave=False, total=n_batches)
    for pred, targ, input in tqdm1:
        bs = input[0].shape[1]
        bptt = input[0].shape[0]
        top_k_ind = np.argpartition(pred, sorted_ks, 1)[:,min_k:]
        for i_in_batch, (pr, t) in enumerate(zip(top_k_ind, targ)):
            for ind, k in enumerate(sorted_ks):
                if t in pr[k:]:
                    count_true[ind] += 1
                else:
                    if ind == (len(ks) - cat) and len(examples) < 100 and i_in_batch // bs > int(0.75 * bptt):
                        examples.append((input[0][:, i_in_batch % bs], i_in_batch // bs, pr[k:], t))
                    break

            count_all +=1
        batch_count += 1
        temporary_result = {f"top{str(-k)}": f"{float(c)/count_all:.3}" for k, c in zip(sorted_ks, count_true)}
        tqdm1.set_postfix(**temporary_result)
    return temporary_result, examples

def accuracy_np(preds, targs):
    preds = np.argmax(preds, 1)
    return (preds==targs).mean()

def accuracy(preds, targs):
    preds = torch.max(preds, dim=1)[1]
    return (preds == (targs.data if isinstance(targs, Variable) else targs)).float().mean()

def accuracy_np(preds, targs):
    preds = np.argmax(preds, 1)
    return (preds==targs).mean()

def accuracy_thresh(thresh):
    return lambda preds,targs: accuracy_multi(preds, targs, thresh)

def accuracy_multi(preds, targs, thresh):
    return ((preds>thresh).float()==targs).float().mean()

def accuracy_multi_np(preds, targs, thresh):
    return ((preds>thresh)==targs).mean()

def recall(log_preds, targs, thresh=0.5, epsilon=1e-8):
    preds = torch.exp(log_preds)
    pred_pos = torch.max(preds > thresh, dim=1)[1]
    tpos = torch.mul((targs.byte() == pred_pos.byte()), targs.byte())
    return tpos.sum()/(targs.sum() + epsilon)

def recall_np(preds, targs, thresh=0.5, epsilon=1e-8):
    pred_pos = preds > thresh
    tpos = torch.mul((targs.byte() == pred_pos), targs.byte())
    return tpos.sum()/(targs.sum() + epsilon)

def precision(log_preds, targs, thresh=0.5, epsilon=1e-8):
    preds = torch.exp(log_preds)
    pred_pos = torch.max(preds > thresh, dim=1)[1]
    tpos = torch.mul((targs.byte() == pred_pos.byte()), targs.byte())
    return tpos.sum()/(pred_pos.sum() + epsilon)

def precision_np(preds, targs, thresh=0.5, epsilon=1e-8):
    pred_pos = preds > thresh
    tpos = torch.mul((targs.byte() == pred_pos), targs.byte())
    return tpos.sum()/(pred_pos.sum() + epsilon)

def fbeta(log_preds, targs, beta, thresh=0.5, epsilon=1e-8):
    """Calculates the F-beta score (the weighted harmonic mean of precision and recall).
    This is the micro averaged version where the true positives, false negatives and
    false positives are calculated globally (as opposed to on a per label basis).

    beta == 1 places equal weight on precision and recall, b < 1 emphasizes precision and
    beta > 1 favors recall.
    """
    assert beta > 0, 'beta needs to be greater than 0'
    beta2 = beta ** 2
    rec = recall(log_preds, targs, thresh)
    prec = precision(log_preds, targs, thresh)
    return (1 + beta2) * prec * rec / (beta2 * prec + rec + epsilon)

def fbeta_np(preds, targs, beta, thresh=0.5, epsilon=1e-8):
    """ see fbeta """
    assert beta > 0, 'beta needs to be greater than 0'
    beta2 = beta ** 2
    rec = recall_np(preds, targs, thresh)
    prec = precision_np(preds, targs, thresh)
    return (1 + beta2) * prec * rec / (beta2 * prec + rec + epsilon)

def f1(log_preds, targs, thresh=0.5): return fbeta(log_preds, targs, 1, thresh)
def f1_np(preds, targs, thresh=0.5): return fbeta_np(preds, targs, 1, thresh)

def f2(preds, targs): return fbeta(preds, targs, 2)
