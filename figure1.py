import pickle

import numpy as np
import torch

from torch import Tensor
from typing import Union
import torch.nn.functional as F

import matplotlib.pyplot as plt

from IPython import embed


def sum_sinusoid(
    start: float,
    num_data_points: int,
    end: float,
    eps=1e-6,
    astensor: bool = True,
    func_return: Union[bool, Tensor, np.ndarray] = False,
):

    if isinstance(func_return, Tensor) or isinstance(func_return, np.ndarray):
        y = (
            func_return
            + 0.8 * np.sin(2 * np.pi * (func_return + eps))
            + 0.8 * np.sin(4 * np.pi * (func_return + eps))
            + eps
        )
        return y
    elif func_return:
        x = np.linspace(start=start, stop=end, num=num_data_points, dtype=np.float32)
        x_test = np.linspace(
            start=start, stop=end, num=int(num_data_points * 0.2)
        ).astype(
            np.float32
        )  # 20 percent of the train data
        y = (
            x
            + 0.8 * np.sin(2 * np.pi * (x + eps))
            + 0.8 * np.sin(4 * np.pi * (x + eps))
            + eps
        )
        return y

    else:
        x = np.linspace(start=start, stop=end, num=num_data_points, dtype=np.float32)
        x_test = np.linspace(
            start=start, stop=end, num=int(num_data_points * 0.2)
        ).astype(
            np.float32
        )  # 20 percent of the train data
        y = (
            x
            + 0.8 * np.sin(2 * np.pi * (x + eps))
            + 0.8 * np.sin(4 * np.pi * (x + eps))
            + eps
            + 0.3 * np.random.randn(*x.shape)
        )
        y = y.astype(np.float32, copy=False)
        y_test = (
            x_test
            + 0.8 * np.sin(2 * np.pi * (x_test + eps))
            + 0.8 * np.sin(4 * np.pi * (x_test + eps))
            + eps
        )
        y_test = y_test.astype(np.float32, copy=False)

        if astensor:
            y = tr.from_numpy(y)
            x = tr.from_numpy(x)
            y_test = tr.from_numpy(y_test)
            x_test = tr.from_numpy(x_test)
        return (
            y[:, np.newaxis],
            x[:, np.newaxis],
            y_test[:, np.newaxis],
            x_test[:, np.newaxis],
        )


if __name__ == "__main__":

    # with open("data/output/plot_toy_posterior_submission.pkl", "rb") as f:
    with open("demo_plots/neurips23/neurips23/plot_toy_posterior_submission_gap_icml.pkl", "rb") as f:
        data = pickle.load(f)

    with open("avb_toy_posterior.pkl", "rb") as f_avb:
        avb = pickle.load(f_avb)
        avb['toy_avb'] = (avb['toy_avb'][0].cpu().numpy(),avb['toy_avb'][1],avb['toy_avb'][2]) 
        data["toy_avb"] = avb["toy_avb"]
        
    with open('plot_hmc_posterior_gapNIPS.pkl','rb') as f_hmc:
        hmc = pickle.load(f_hmc)
        data['hmc_bnn'] = hmc['hmc_bnn']
    x_train, y_train = data["train_data"]
    x_test, y_test = data["test_data"]

    # Model evaluations:
    x = data["plot_x"]

    # x_std = x / data["data_normalization_factor"]
    # true_func = sum_sinusoid(x.min(), 200, x.max(), func_return=x)

    tf_x = np.linspace(-2, 2, 200)  # / data["data_normalization_factor"]
    tf_x_std = tf_x / data["data_normalization_factor"]
    true_func = sum_sinusoid(tf_x.min(), 200, tf_x.max(), func_return=tf_x)

    models_to_plot = {
        "Mean-field VI": "mfvi_bnn",
        #"LIVI acc-jac": "acc_jac_no_init",
        "AVB": "toy_avb",
        "LIVI": "acc_jac_no_init",
        #"LIVI diff-lb": "diff_lb_no_init",
        "HMC": "hmc_bnn",
        #"DE": "deep_ensembles_5_seeds",
    }

    # model_names = [
    #    "MFVI",
    #    "LIVI acc-jac",
    #    "LIVI diff-lb",
    #    "HMC",
    #    "DE",
    # ]

    cmap = plt.cm.Blues

    fig, ax = plt.subplots(
        #2, 2,
        1,
        len(models_to_plot),
        figsize=(12, 3) if len(models_to_plot) == 5 else (8, 2.3), #(6.5, 2.3), #(6.5, 4), #(6.5, 2.5),
        #layout="constrained",
        sharey=True,
        sharex=True,
        squeeze=False,
    )

    for m, (model_name, model) in enumerate(models_to_plot.items()):

        if model == "deep_ensembles_5_seeds":
            avg_pred, mixed_noise_interval, seeds, model_conf = data[model]
            conf_lower = model_conf[0].squeeze()
            conf_upper = model_conf[1].squeeze()
            noise_lower, noise_upper = mixed_noise_interval  # .squeeze()
        else:
            avg_pred, conf_interval, mixed_noise_interval = data[model]
            conf_lower, conf_upper = conf_interval.squeeze()
            if model == "hmc_bnn":
                data_noise = np.sqrt(1 / mixed_noise_interval["prec"].numpy().mean())
                noise_lower = conf_lower - 2 * data_noise
                noise_upper = conf_upper + 2 * data_noise
            else:
                noise_lower, noise_upper = mixed_noise_interval.squeeze()

        #col = m % 2
        #row = int(m >= 2)
        col = m
        row = 0
        #print(row, col)

        #embed()
        # stop
        # print()
        # print()
        # Data

        full_uncertainty = ax[row, col].fill_between(
            x,
            noise_lower,
            noise_upper,
            color=cmap(0.25),
            edgecolor="none",
            label="Epistemic + aleatoric",
            # alpha=0.3
        )
        func_uncertainty = ax[row, col].fill_between(
            x,
            conf_lower,
            conf_upper,
            color=cmap(0.4),
            edgecolor="none",
            label="Epistemic uncertainty",
            # alpha=0.3
        )
        mean_func, = ax[row, col].plot(
            x,
            avg_pred,
            color=cmap(0.7),
            label="Mean prediction"
            # alpha=0.7

        )

        ax[row, col].plot(tf_x_std, true_func, c=plt.cm.binary(0.6), ls="--")
        # ax[m].scatter(x_test, y_test, c=plt.cm.binary(0.8), ) #label="Test data")
        ax[row, col].scatter(
            x_train,
            y_train,
            color=plt.cm.binary(0.8),
            s=5,
        )  # label="Training data")

        ax[row, col].set_xlim(left=-1.5, right=1.5)
        ax[row, col].set_ylim(bottom=-5, top=5)

        ax[row, col].set_title(model_name)

    #fig.legend(bbox_to_anchor=(0.15, 0.02, 0.5, 0.1), ncol=3,
    fig.legend(bbox_to_anchor=(0.5, 0.0), ncol=3,
               loc="lower center",
               handles=[mean_func, func_uncertainty, full_uncertainty]
               )
    plt.tight_layout()
    #plt.subplots_adjust(left=0.07, right= 0.995, top=0.93, bottom=0.15)
    plt.subplots_adjust(left=0.05, right= 0.995, top=0.90, bottom=0.25)
    #plt.show()
    plt.savefig("fig1.pdf")
    #plt.savefig("fig1.png")