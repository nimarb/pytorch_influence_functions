from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
import os

def get_mnist_data():
    train_data = datasets.MNIST('data', train=True, download=True, transform=transforms.Compose([
                       transforms.ToTensor(),
                   ]))
    x_train, y_train = train_data.data, train_data.targets

    test_data = datasets.MNIST('data', train=False, download=True, transform=transforms.Compose([
                       transforms.ToTensor(),
                   ]))

    # set train data's last 5000 data as validation data, as in the paper
    x_train, y_train = x_train[:-5000], y_train[:-5000]

    x_test, y_test = test_data.data, test_data.targets
    x_train, x_test = x_train.reshape([-1, 784]) / 255.0, x_test.reshape([-1, 784]) / 255.0  # divided by 255, improve convergence

    return (x_train, y_train), (x_test, y_test)


def visualize_result(actual_loss_diff, estimated_loss_diff, save_path=None):
    from matplotlib.ticker import MaxNLocator, FuncFormatter

    r2_s = r2_score(actual_loss_diff, estimated_loss_diff)

    max_abs = np.max([np.abs(actual_loss_diff), np.abs(estimated_loss_diff)])
    min_, max_ = -max_abs * 1.1, max_abs * 1.1
    plt.rcParams.update({'font.size': 15})
    tick_label_size = 8

    fig, ax = plt.subplots()

    ax.scatter(actual_loss_diff, estimated_loss_diff, zorder=2, s=10)
    ax.set_title('Linear(approx)')
    ax.set_xlabel('Actual diff in loss')
    ax.set_ylabel('Predicted diff in loss')
    range_ = [min_, max_]
    ax.plot(range_, range_, 'k-', alpha=0.2, zorder=1)
    text = 'MAE = {:.03}\nR2 score = {:.03}'.format(mean_absolute_error(actual_loss_diff, estimated_loss_diff),
                                                    r2_s)
    ax.text(max_abs, -max_abs, text, verticalalignment='bottom', horizontalalignment='right')
    ax.set_xlim(min_, max_)
    ax.set_ylim(min_, max_)

    # Using scientific notation for xticks and yticks
    ax.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))

    # Adjusting the x and y ticks to be symmetric
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, symmetric=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, symmetric=True))
    ax.xaxis.set_major_formatter(FuncFormatter('{:.0e}'.format))
    ax.yaxis.set_major_formatter(FuncFormatter('{:.0e}'.format))
    # smaller the tick size
    ax.xaxis.set_tick_params(labelsize=tick_label_size)
    ax.yaxis.set_tick_params(labelsize=tick_label_size)
    # make plt to be a square
    plt.gca().set_aspect('equal', adjustable='box')
    if save_path is not None:
        plt.savefig(os.path.join(save_path, "result.png"))
    else:
        plt.show()

    return r2_s
