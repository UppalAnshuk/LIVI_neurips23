import numpy as np
from sklearn.metrics import roc_auc_score, mean_squared_error
import torch
from bnn_src.models import LeNet
import scipy.stats as st
import torch.nn.functional as F
import torch.distributions as dist
from bnn_src.utils import mfvipredict, avbpredict, mnfpredict

def get_auroc(py_in, py_out):
    py_in, py_out = py_in.cpu().numpy(), py_out.cpu().numpy()
    labels = np.zeros(len(py_in) + len(py_out), dtype='int32')
    labels[:len(py_in)] = 1
    examples = np.concatenate([py_in.max(1), py_out.max(1)])
    return roc_auc_score(labels, examples)


def get_fpr95(py_in, py_out):
    py_in, py_out = py_in.cpu().numpy(), py_out.cpu().numpy()
    conf_in, conf_out = py_in.max(1), py_out.max(1)
    tpr = 95
    perc = np.percentile(conf_in, 100 - tpr)
    fp = np.sum(conf_out >= perc)
    fpr = np.sum(conf_out >= perc) / len(conf_out)
    return fpr.item(), perc.item()


def get_brier_score(probs, targets):
    targets = F.one_hot(targets, num_classes=probs.shape[1])
    return torch.mean(torch.sum((probs - targets) ** 2, axis=1)).item()


def get_calib(pys, y_true, M=100):
    #TODO implement ACE according to Measuring Calibration in Deep Learning
    pys, y_true = pys.cpu().numpy(), y_true.cpu().numpy() 
    # Put the confidence into M bins
    population, bins = np.histogram(pys, M, range=(0, 1))
    labels = pys.argmax(1)
    labels_all = np.flip(pys.argsort(axis=-1),axis=-1)
    confs = np.max(pys, axis=1)
    probabilities = np.sort(pys, axis=1)
    conf_idxs = np.digitize(confs, bins)
    
    # Accuracy and avg. confidence per bin
    accs_bin = []
    confs_bin = []
    nitems_bin = []
    print(f'confidence indices, meaning in which bin does the most probable prediction lie {conf_idxs}')
    #breakpoint()
    for i in range(M):
        labels_i = labels[conf_idxs == i]
        y_true_i = y_true[conf_idxs == i]
        confs_i = confs[conf_idxs == i]

        acc = np.nan_to_num(np.mean(labels_i == y_true_i), 0)
        conf = np.nan_to_num(np.mean(confs_i), 0)

        accs_bin.append(acc)
        confs_bin.append(conf)
        nitems_bin.append(len(labels_i))
    
    SCE = 0
    for k in range(pys.shape[-1]):
        kth_prob_bins = np.digitize(probabilities[:,k],bins)
        acc_bin_sce = []
        confs_bin_sce = []
        nitems_bin_sce = []
        for i in range(M):
            labels_k_i = labels_all[:,k][kth_prob_bins == i]
            y_true_k_i = y_true[kth_prob_bins == i]
            confs_k_i = probabilities[:,k][kth_prob_bins == i]
            
            acc_k = np.nan_to_num(np.mean(labels_k_i == y_true_k_i), 0)
            conf_k = np.nan_to_num(np.mean(confs_k_i), 0)
            
            acc_bin_sce.append(acc_k)
            confs_bin_sce.append(conf_k)
            nitems_bin_sce.append(len(labels_k_i))
            
        acc_bin_sce, confs_bin_sce = np.array(acc_bin_sce), np.array(confs_bin_sce)
        nitems_bin_sce = np.array(nitems_bin_sce)
        SCE += np.average(np.abs(confs_bin_sce - acc_bin_sce), weights=nitems_bin_sce / nitems_bin_sce.sum())
        
            
    SCE = SCE/(k+1) #implemented here as per eqn 3 in Measuring Calibration in Deep Learning Nixon et al. 2019
        
    
    accs_bin, confs_bin = np.array(accs_bin), np.array(confs_bin)
    nitems_bin = np.array(nitems_bin)

    ECE = np.average(np.abs(confs_bin - accs_bin), weights=nitems_bin / nitems_bin.sum())
    MCE = (np.max(np.abs(accs_bin - confs_bin)),np.argmax(np.abs(accs_bin - confs_bin)))
    

    return ECE, MCE, SCE


def get_calib_regression(pred_means, pred_stds, y_true, return_hist=False, M=10):
    '''
    Kuleshov et al. ICML 2018, eq. 9
    * pred_means, pred_stds, y_true must be np.array's
    * Set return_hist to True to also return the "histogram"---useful for visualization (see paper)
    '''
    T = len(y_true)
    ps = np.linspace(0, 1, M)
    cdf_vals = [st.norm(m, s).cdf(y_t) for m, s, y_t in zip(pred_means, pred_stds, y_true)]
    p_hats = np.array([len(np.where(cdf_vals <= p)[0]) / T for p in ps])
    cal = T * mean_squared_error(ps, p_hats)  # Sum-squared-error

    return (cal, ps, p_hats) if return_hist else cal


def get_sharpness(pred_stds):
    '''
    Kuleshov et al. ICML 2018, eq. 10
    pred_means be np.array
    '''
    return np.mean(pred_stds ** 2)


def empirical_cdf(model_dict: dict, ood_dataloader, n_samples: int = 0):
    """

    :param model_dict: has to list the models and the type in str
    :param ood_dataloader: torch dataloader with the ood dataset
    :param n_samples: for model that can use multiple samples
    :return:
    """
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    #x, y = next(iter(ood_dataloader))
    #x, y = x.to(device), y.to(device)
    emp_pred = list()
    for k, v in model_dict.items():
        if k == 'mfvi':
            model = v.to(device)
            y_prob,_ = mfvipredict(v, ood_dataloader, n_samples=n_samples)
            calc_entropy_dist = dist.Categorical(probs=torch.from_numpy(y_prob))
            emp_pred.append(calc_entropy_dist.entropy())
        elif k == 'mnf':
            model = v.to(device)
            y_prob,_ = mnfpredict(v, ood_dataloader, n_samples=n_samples)
            calc_entropy_dist = dist.Categorical(probs=y_prob)
            emp_pred.append(calc_entropy_dist.entropy())
        elif k == 'avb':
            model = v
            y_prob = avbpredict(v, ood_dataloader, n_samples=n_samples)
            calc_entropy_dist = dist.Categorical(probs=y_prob)
            emp_pred.append(calc_entropy_dist.entropy())
        else:
            for i, data in enumerate(ood_dataloader):
                x, y = data
                x, y = x.to(device), y.to(device)

                if k == 'de':
                    _py = 0
                    assert isinstance(v[0], LeNet), "for deep ensembles provide a list of loaded models"
                    for model in v:
                        model.to(device)
                        _py += 1 / len(v) * torch.softmax(model(x), -1).detach()
                    #emp_pred.append(-1 * _py * torch.log(_py))
                    calc_entropy_dist = dist.Categorical(probs=_py)
                    emp_pred.append(calc_entropy_dist.entropy())
                elif k == 'imp_vi':
                    _py = 0
                    v.bnn.to(device)
                    base_sample, params, weights_to_layers = v.bnn.sample_params(batch_size=n_samples, device=device)
                    noisy_params = params + 0.05*torch.randn_like(params)
                    for s in range(n_samples):
                        weights_to_layers(noisy_params[s, :])
                        f_s = v.bnn(x)  # The second return is KL
                        # if self.bnn.named_children()
                        _py += torch.softmax(f_s, 1).detach()
                    _py /= n_samples
                    #emp_pred.append(-1 * _py * torch.log(_py))
                    calc_entropy_dist = dist.Categorical(probs=_py)
                    emp_pred.append(calc_entropy_dist.entropy())
                elif k == 'lap':
                    model = v
                    y_prob = model(
                        x, pred_type='glm', link_approx='mc', n_samples=n_samples)
                    #emp_pred.append(-1 * y_prob.detach() * torch.log(y_prob.detach()))
                    calc_entropy_dist = dist.Categorical(probs=y_prob)
                    emp_pred.append(calc_entropy_dist.entropy())
                else:
                    #map model
                    model = v.to(device)
                    y_prob = torch.softmax(model(x),1)
                    #emp_pred.append(-1 * y_prob.detach() * torch.log(y_prob.detach())+ 1e-7)
                    calc_entropy_dist = dist.Categorical(probs=_py)
                    emp_pred.append(calc_entropy_dist.entropy())

    return emp_pred


