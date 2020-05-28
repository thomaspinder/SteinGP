import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import ndarray
from gpflow.models import GPModel


def plot_boundary(m: GPModel, X: ndarray, y: ndarray, ax=None):
    x_grid = np.linspace(min(X[:, 0]), max(X[:, 0]), 40)
    y_grid = np.linspace(min(X[:, 1]), max(X[:, 1]), 40)
    xx, yy = np.meshgrid(x_grid, y_grid)
    Xplot = np.vstack((xx.flatten(), yy.flatten())).T
    mask = y[:, 0] == 1

    p, _ = m.predict_y(Xplot)  # here we only care about the mean
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))


#     plt.figure(figsize=(7, 7))
    ax.plot(X[mask, 0], X[mask, 1], "oC0", mew=0, alpha=0.5, label="1")
    ax.plot(X[np.logical_not(mask), 0],
            X[np.logical_not(mask), 1],
            "oC1",
            mew=0,
            alpha=0.5,
            label="0")

    _ = ax.contour(
        xx,
        yy,
        p.numpy().reshape(*xx.shape),
        [0.5],  # plot the p=0.5 contour line only
        colors="k",
        linewidths=1.8,
        zorder=100,
    )
    ax.legend(loc='best')
    ax.axis("off")


def make_predictive_plot(ax,
                         dataset,
                         mu: ndarray,
                         sigma: ndarray,
                         lik='gaussian',
                         plt_type="testing"):
    X, Y, Xte, Yte = dataset
    test_type = [ax.scatter if lik == 'bernoulli' else ax.plot][0]
    test_label = ["Testing points" if plt_type == "testing" else plt_type][0]
    test_type(Xte, Yte.flatten(), label=test_label, color="green", alpha=0.5)
    ax.plot(Xte, mu, label="Predictive mean", color="blue")
    ax.fill_between(Xte[:, 0],
                    mu[:, 0].numpy() - 1.96 * sigma[:, 0].numpy(),
                    mu[:, 0].numpy() + 1.96 * sigma[:, 0].numpy(),
                    alpha=0.2,
                    label="Predictive_uncertainty",
                    color="blue")
    ax.plot(X, Y, 'o', color="black", markersize=5, label="Training points")
    handles, labels = ax.get_legend_handles_labels()
    labels, ids = np.unique(labels, return_index=True)
    handles = [handles[i] for i in ids]
    ax.legend(handles, labels, loc='upper left')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    return ax


def progress_plot(ax, progress, model_name: str):
    ax.plot(progress, label=model_name, linewidth=2)
    ax.set_xlabel("Optimisation iteration")
    ax.set_ylabel("Marginal log-likelihood")
    ax.legend(loc="lower right")


def make_gpr_plot(model,
                  particles,
                  Xfull,
                  Yfull,
                  X,
                  Y,
                  mu,
                  sigma,
                  logf,
                  gif=False):
    n_iter = len(logf)
    # adam_mll = pd.read_csv("quick_svgd/adam_1particle_comparison.csv")
    with plt.style.context("seaborn-notebook"):
        fig = plt.figure(figsize=(18, 8))
        layout = (2, 2)
        predict_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        mll_ax = plt.subplot2grid(layout, (1, 0))
        particle_ax = plt.subplot2grid(layout, (1, 1))

        mll_ax.plot(logf, label="SteinGP", linewidth=2)
        mll_ax.set_xlabel("Optimisation iteration")
        mll_ax.set_ylabel("Marginal log-likelihood")
        mll_ax.legend(loc="lower right")
        if gif:
            mll_ax.set_ylim(-70, -20)

        predict_ax.plot(Xfull,
                        Yfull.flatten(),
                        label="Latent function",
                        color="green",
                        alpha=0.5)
        predict_ax.plot(Xfull, mu, label="Predictive mean", color="blue")
        predict_ax.fill_between(Xfull[:, 0],
                                mu[:, 0].numpy() - 1.96 * sigma[:, 0].numpy(),
                                mu[:, 0].numpy() + 1.96 * sigma[:, 0].numpy(),
                                alpha=0.2,
                                label="Predictive_uncertainty",
                                color="blue")
        predict_ax.plot(X,
                        Y,
                        'o',
                        color="black",
                        markersize=5,
                        label="Training points")
        handles, labels = predict_ax.get_legend_handles_labels()
        labels, ids = np.unique(labels, return_index=True)
        handles = [handles[i] for i in ids]
        predict_ax.legend(handles, labels, loc='upper left')
        predict_ax.set_xlabel("X")
        predict_ax.set_ylabel("Y")
        if gif:
            predict_ax.set_ylim(-1.75, 2.25)

        cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for p, lab, col, pa in zip(particles,
                                   ['Lengthscale', 'Variance', 'Obs. noise'],
                                   cols[:particles.shape[0]],
                                   model.trainable_parameters):
            particle_ax.axhline(pa.transform(np.mean(p)).numpy(),
                                alpha=0.7,
                                color=col)
            particle_ax.text(particles.shape[1],
                             pa.transform(np.mean(p)).numpy(),
                             "{} mean".format(lab))
            particle_ax.plot(pa.transform(p).numpy(),
                             'o',
                             label=lab,
                             color=col)
        handles, labels = particle_ax.get_legend_handles_labels()
        labels, ids = np.unique(labels, return_index=True)
        handles = [handles[i] for i in ids]
        particle_ax.legend(handles, labels, loc='best')
        particle_ax.set_xlabel("Particle index")
        particle_ax.set_ylabel("Particle value")
        particle_ax.set_title("Final SVGD particles")
        particle_ax.set_xticks(np.arange(particles.shape[1] + 1))
        if gif:
            particle_ax.set_ylim(-0.2, 0.8)

        plt.tight_layout()
        plt.figtext(
            0.5,
            0.96,
            "Recovering a realisation from a GP with lengthscale=0.2, variance=0.3 and obs. noise=0.2",
            wrap=True,
            horizontalalignment='center',
            fontsize=12)
        if gif:
            plt.savefig("quick_svgd/gif/{}_signal_recovery.png".format(
                int((n_iter - 1) / 10)))
        else:
            plt.savefig("plots/regression.png")
        plt.close(fig)


def plot_K(K, dK, iteration, filename):
    with plt.style.context("seaborn-notebook"):
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
        im1 = ax[0].imshow(K)
        # ax[0].spines['top'].set_visible(False)
        # ax[0].spines['bottom'].set_visible(False)
        # ax[0].spines['right'].set_visible(False)
        # ax[0].spines['left'].set_visible(False)
        # ax[0].tick_params(left=False, right=False, top=False, bottom=False)
        # # Turn off tick labels
        # ax[0].set_yticklabels([])
        # ax[0].set_xticklabels([])
        ax[0].set_title("Kernel matrix")
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im1, cax=cax, orientation='vertical')

        im2 = ax[1].imshow(dK)
        # ax[1].spines['top'].set_visible(False)
        # ax[1].spines['bottom'].set_visible(False)
        # ax[1].spines['right'].set_visible(False)
        # ax[1].spines['left'].set_visible(False)
        # ax[1].tick_params(left=False, right=False, top=False, bottom=False)
        # # Turn off tick labels
        # ax[1].set_yticklabels([])
        # ax[1].set_xticklabels([])
        ax[1].set_title("Kernel derivative")
        # at = AnchoredText("Iteration: {}".format(iteration),
        #                   prop=dict(size=15), frameon=True,
        #                   loc='lower right',
        #                   )
        # at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        # ax[1].add_artist(at)
        fig.suptitle('Iteration: {}'.format(iteration), fontsize=16)
        divider = make_axes_locatable(ax[1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im2, cax=cax, orientation='vertical')
        # plt.tight_layout()
        plt.savefig("quick_svgd/kernels/{}".format(filename))
    plt.close()


def make_sgpr_plot(model,
                   particles,
                   Xfull,
                   Yfull,
                   X,
                   Y,
                   mu,
                   sigma,
                   logf,
                   gif=False):
    n_iter = len(logf)
    with plt.style.context("seaborn-notebook"):
        fig = plt.figure(figsize=(18, 8))
        layout = (2, 2)
        predict_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        mll_ax = plt.subplot2grid(layout, (1, 0))
        particle_ax = plt.subplot2grid(layout, (1, 1))

        mll_ax.plot(logf, label="steingp", linewidth=2)
        mll_ax.set_xlabel("Optimisation iteration")
        mll_ax.set_ylabel("Marginal log-likelihood")
        mll_ax.legend(loc="lower right")
        if gif:
            mll_ax.set_ylim(-70, -20)

        predict_ax.plot(Xfull,
                        Yfull.flatten(),
                        label="Latent function",
                        color="green",
                        alpha=0.5)
        predict_ax.plot(Xfull, mu, label="Predictive mean", color="blue")
        predict_ax.fill_between(Xfull[:, 0],
                                mu[:, 0].numpy() - 1.96 * sigma[:, 0].numpy(),
                                mu[:, 0].numpy() + 1.96 * sigma[:, 0].numpy(),
                                alpha=0.2,
                                label="Predictive_uncertainty",
                                color="blue")
        predict_ax.plot(X,
                        Y,
                        'o',
                        color="black",
                        markersize=5,
                        label="Training points")
        handles, labels = predict_ax.get_legend_handles_labels()
        labels, ids = np.unique(labels, return_index=True)
        handles = [handles[i] for i in ids]
        predict_ax.legend(handles, labels, loc='upper left')
        predict_ax.set_xlabel("X")
        predict_ax.set_ylabel("Y")
        if gif:
            predict_ax.set_ylim(-1.75, 2.25)

        cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for p, lab, col, pa in zip(particles,
                                   ['Lengthscale', 'Variance', 'Obs. noise'],
                                   cols[:particles.shape[0]],
                                   model.trainable_parameters):
            particle_ax.axhline(pa.transform(np.mean(p)).numpy(),
                                alpha=0.7,
                                color=col)
            particle_ax.text(particles.shape[1],
                             pa.transform(np.mean(p)).numpy(),
                             "{} mean".format(lab))
            particle_ax.plot(pa.transform(p).numpy(),
                             'o',
                             label=lab,
                             color=col)
        handles, labels = particle_ax.get_legend_handles_labels()
        labels, ids = np.unique(labels, return_index=True)
        handles = [handles[i] for i in ids]
        particle_ax.legend(handles, labels, loc='best')
        particle_ax.set_xlabel("Particle index")
        particle_ax.set_ylabel("Particle value")
        particle_ax.set_title("Final SVGD particles")
        particle_ax.set_xticks(np.arange(particles.shape[1] + 1))
        if gif:
            particle_ax.set_ylim(-0.2, 0.8)

        plt.tight_layout()
        plt.figtext(
            0.5,
            0.96,
            "Recovering a realisation from a GP with lengthscale=0.2, variance=0.3 and obs. noise=0.2",
            wrap=True,
            horizontalalignment='center',
            fontsize=12)
        plt.show()
        # if gif:
        #     plt.savefig("quick_svgd/gif/{}_signal_recovery.png".format(
        #         int((n_iter - 1) / 10)))
        # else:
        #     plt.savefig("quick_svgd/sgpr_output.png")
        plt.close(fig)


def make_breathe_plot(model,
                      particles,
                      Xfull,
                      Yfull,
                      X,
                      Y,
                      Xte,
                      Yte,
                      mu,
                      sigma,
                      logf,
                      gif=False):
    n_iter = len(logf)
    adam_mll = pd.read_csv("quick_svgd/adam_1particle_comparison.csv")
    with plt.style.context("seaborn-notebook"):
        fig = plt.figure(figsize=(18, 8))
        layout = (2, 2)
        predict_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        mll_ax = plt.subplot2grid(layout, (1, 0))
        particle_ax = plt.subplot2grid(layout, (1, 1))

        mll_ax.plot(logf, label="steingp", linewidth=2)
        # mll_ax.plot(adam_mll, label="Adam Opt.", linewidth=2)
        mll_ax.set_xlabel("Optimisation iteration")
        mll_ax.set_ylabel("Marginal log-likelihood")
        mll_ax.legend(loc="lower right")
        if gif:
            mll_ax.set_ylim(-70, -20)

        predict_ax.plot(Xfull,
                        Yfull.flatten(),
                        label="True data_old",
                        color="green",
                        alpha=0.5)
        predict_ax.plot(Xte, mu, label="Predictive mean", color="blue")
        predict_ax.fill_between(Xte[:, 0],
                                mu[:, 0] - 1.96 * sigma[:, 0],
                                mu[:, 0] + 1.96 * sigma[:, 0],
                                alpha=0.2,
                                label="Predictive_uncertainty",
                                color="blue")
        # TODO: Fix inducing point plot
        # predict_ax.plot(model.inducing_variable.Z.numpy(),
        #                 'o',
        #                 color="black",
        #                 markersize=6,
        #                 label="Inducing points")
        handles, labels = predict_ax.get_legend_handles_labels()
        labels, ids = np.unique(labels, return_index=True)
        handles = [handles[i] for i in ids]
        predict_ax.legend(handles, labels, loc='upper left')
        predict_ax.set_xlabel("X")
        predict_ax.set_ylabel("Y")
        if gif:
            predict_ax.set_ylim(-1.75, 2.25)

        cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for p, lab, col, pa in zip(particles,
                                   ['Lengthscale', 'Variance', 'Obs. noise'],
                                   cols[:particles.shape[0]],
                                   model.trainable_parameters):
            particle_ax.axhline(pa.transform(np.mean(p)).numpy(),
                                alpha=0.7,
                                color=col)
            particle_ax.text(particles.shape[1],
                             pa.transform(np.mean(p)).numpy(),
                             "{} mean".format(lab))
            particle_ax.plot(pa.transform(p).numpy(),
                             'o',
                             label=lab,
                             color=col)
        handles, labels = particle_ax.get_legend_handles_labels()
        labels, ids = np.unique(labels, return_index=True)
        handles = [handles[i] for i in ids]
        particle_ax.legend(handles, labels, loc='best')
        particle_ax.set_xlabel("Particle index")
        particle_ax.set_ylabel("Particle value")
        particle_ax.set_title("Final SVGD particles")
        particle_ax.set_xticks(np.arange(particles.shape[1] + 1))
        if gif:
            particle_ax.set_ylim(-0.2, 0.8)

        plt.tight_layout()
        plt.figtext(0.5,
                    0.94,
                    "Predictions of the Whitecross AQ station",
                    wrap=True,
                    horizontalalignment='center',
                    fontsize=12)
        if gif:
            plt.savefig("quick_svgd/gif/{}_signal_recovery.png".format(
                int((n_iter - 1) / 10)))
        else:
            plt.savefig("quick_svgd/breathe_output.png")
        plt.close(fig)


def complement(l, universe=None):
    """
    Return the complement of a list of integers, as compared to
    a given "universe" set. If no universe is specified,
    consider the universe to be all integers between
    the minimum and maximum values of the given list.
    """
    if universe is not None:
        universe = set(universe)
    else:
        universe = set(range(min(l), max(l) + 1))
    return sorted(universe - set(l))


def make_gpmc_plot(model,
                   particles,
                   Xfull,
                   Yfull,
                   X,
                   Y,
                   mu,
                   sigma,
                   logf,
                   gif=False):
    n_iter = len(logf)
    with plt.style.context("seaborn-notebook"):
        fig = plt.figure(figsize=(18, 8))
        layout = (2, 2)
        predict_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        mll_ax = plt.subplot2grid(layout, (1, 0))
        particle_ax = plt.subplot2grid(layout, (1, 1))

        mll_ax.plot(logf, label="steingp", linewidth=2)
        mll_ax.set_xlabel("Optimisation iteration")
        mll_ax.set_ylabel("Marginal log-likelihood")
        mll_ax.legend(loc="lower right")
        if gif:
            mll_ax.set_ylim(-70, -20)

        predict_ax.plot(Xfull,
                        Yfull.flatten(),
                        label="Latent function",
                        color="green",
                        alpha=0.5)
        predict_ax.plot(Xfull, mu, label="Predictive mean", color="blue")
        predict_ax.fill_between(Xfull[:, 0],
                                mu[:, 0].numpy() - 1.96 * sigma[:, 0].numpy(),
                                mu[:, 0].numpy() + 1.96 * sigma[:, 0].numpy(),
                                alpha=0.2,
                                label="Predictive_uncertainty",
                                color="blue")
        predict_ax.plot(X,
                        Y,
                        'o',
                        color="black",
                        markersize=5,
                        label="Training points")
        handles, labels = predict_ax.get_legend_handles_labels()
        labels, ids = np.unique(labels, return_index=True)
        handles = [handles[i] for i in ids]
        predict_ax.legend(handles, labels, loc='upper left')
        predict_ax.set_xlabel("X")
        predict_ax.set_ylabel("Y")
        if gif:
            predict_ax.set_ylim(-1.75, 2.25)

        cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for idx, (p, lab, col, pa) in enumerate(
                zip(particles, [
                    '', 'Matern Lengthscale', 'Matern Variance',
                    'Bias Variance', "obs_noise"
                ], cols[:particles.shape[0]], model.trainable_parameters)):
            if idx != 0:
                particle_ax.axhline(pa.transform(np.mean(p)).numpy(),
                                    alpha=0.7,
                                    color=col)
                particle_ax.text(particles.shape[1] + 0.1,
                                 pa.transform(np.mean(p)).numpy(),
                                 "{} mean".format(lab))
                particle_ax.plot(pa.transform(p).numpy(),
                                 'o',
                                 label=lab,
                                 color=col,
                                 markersize=5,
                                 alpha=0.8)
        handles, labels = particle_ax.get_legend_handles_labels()
        labels, ids = np.unique(labels, return_index=True)
        handles = [handles[i] for i in ids]
        particle_ax.legend(handles, labels, loc='best')
        particle_ax.set_xlabel("Particle index")
        particle_ax.set_ylabel("Particle value")
        particle_ax.set_title("Final SVGD particles")
        particle_ax.set_xticks(np.arange(particles.shape[1] + 1))
        if gif:
            particle_ax.set_ylim(-0.2, 0.8)

        plt.tight_layout()
        plt.figtext(0.5,
                    0.96,
                    "SVGD to fit exponential data_old",
                    wrap=True,
                    horizontalalignment='center',
                    fontsize=12)
        if gif:
            plt.savefig("quick_svgd/gif/{}_signal_recovery.png".format(
                int((n_iter - 1) / 10)))
        else:
            plt.savefig("quick_svgd/exponential_nparticles_gaussian.png")
        plt.close(fig)


def make_bern_plot(model,
                   particles,
                   Xfull,
                   Yfull,
                   X,
                   Y,
                   mu,
                   sigma,
                   logf,
                   gif=False):
    n_iter = len(logf)
    samples = model.predict_f_samples(Xfull, 10).numpy().squeeze().T
    with plt.style.context("seaborn-notebook"):
        fig = plt.figure(figsize=(18, 8))
        layout = (2, 2)
        predict_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        mll_ax = plt.subplot2grid(layout, (1, 0))
        particle_ax = plt.subplot2grid(layout, (1, 1))

        mll_ax.plot(logf, label="steingp", linewidth=2)
        mll_ax.set_xlabel("Optimisation iteration")
        mll_ax.set_ylabel("Marginal log-likelihood")
        mll_ax.legend(loc="lower right")
        if gif:
            mll_ax.set_ylim(-70, -20)
        predict_ax.plot(Xfull, mu, label="Predictive mean", color="blue")
        predict_ax.fill_between(Xfull[:, 0],
                                mu[:, 0].numpy() - 1.96 * sigma[:, 0].numpy(),
                                mu[:, 0].numpy() + 1.96 * sigma[:, 0].numpy(),
                                alpha=0.2,
                                label="Predictive_uncertainty",
                                color="blue")
        predict_ax.scatter(X,
                           Y,
                           marker='o',
                           color="red",
                           label="Training points",
                           alpha=0.7)
        predict_ax.scatter(Xfull,
                           Yfull.flatten(),
                           marker="x",
                           label="Original dataset",
                           color="green",
                           alpha=0.7)
        handles, labels = predict_ax.get_legend_handles_labels()
        labels, ids = np.unique(labels, return_index=True)
        handles = [handles[i] for i in ids]
        predict_ax.legend(handles, labels, loc='upper left')
        predict_ax.set_xlabel("X")
        predict_ax.set_ylabel("Y")
        if gif:
            predict_ax.set_ylim(-1.75, 2.25)

        cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for idx, (p, lab, col, pa) in enumerate(
                zip(particles, [
                    '', 'Matern Lengthscale', 'Matern Variance',
                    'Bias Variance', "obs_noise"
                ], cols[:particles.shape[0]], model.trainable_parameters)):
            if idx != 0:
                particle_ax.axhline(pa.transform(np.mean(p)).numpy(),
                                    alpha=0.7,
                                    color=col)
                particle_ax.text(particles.shape[1] + 0.1,
                                 pa.transform(np.mean(p)).numpy(),
                                 "{} mean".format(lab))
                particle_ax.plot(pa.transform(p).numpy(),
                                 'o',
                                 label=lab,
                                 color=col,
                                 markersize=5,
                                 alpha=0.8)
        handles, labels = particle_ax.get_legend_handles_labels()
        labels, ids = np.unique(labels, return_index=True)
        handles = [handles[i] for i in ids]
        particle_ax.legend(handles, labels, loc='best')
        particle_ax.set_xlabel("Particle index")
        particle_ax.set_ylabel("Particle value")
        particle_ax.set_title("Final SVGD particles")
        particle_ax.set_xticks(np.arange(particles.shape[1] + 1))
        if gif:
            particle_ax.set_ylim(-0.2, 0.8)

        plt.tight_layout()
        plt.figtext(0.5,
                    0.96,
                    "SVGD to fit exponential data_old",
                    wrap=True,
                    horizontalalignment='center',
                    fontsize=12)
        if gif:
            plt.savefig("quick_svgd/gif/{}_signal_recovery.png".format(
                int((n_iter - 1) / 10)))
        else:
            plt.savefig("plots/toy_data/bernoulli_nparticles.png")
        plt.close(fig)
