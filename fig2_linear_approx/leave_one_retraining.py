"""
reproduce the fig2 middle plot in the paper, remove one training sample and retrain the logistic regression model on MINIST 10 classes
2023-10-29
"""

import torch
from sklearn import linear_model
import numpy as np
from tqdm import tqdm
import pickle
from utils import get_mnist_data, visualize_result
from model import LogisticRegression as LR

from influence_functions_toolkits.influence_functions import (
    calc_influence_single,
)

# HYPARAMS
EPOCH = 10
BATCH_SIZE = 100
CLASS_A, CLASS_B = 1, 7
TEST_INDEX = 5
WEIGHT_DECAY = 0.01  # same as original paper
OUTPUT_DIR = '../results'
SAMPLE_NUM = 100
RECURSION_DEPTH = 1000
R = 10
SEED = 17

# set seed
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


class DataSet:
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        out_data = self.data[idx]
        out_label = self.targets[idx]

        return out_data, out_label


def get_accuracy(model, test_loader):
    """
    test whether the weight transferred from sklearn model to pytorch model is correct
    """
    correct = 0
    total = 0

    with torch.no_grad():
        for data in tqdm(test_loader):
            images, labels = data

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the model on the test images: %d %%' % (100 * correct / total))
    return correct / total


def leave_one_out():
    (x_train, y_train), (x_test, y_test) = get_mnist_data()
    # print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    train_sample_num = len(x_train)
    print("len(x_train):", len(x_train))

    train_data = DataSet(x_train, y_train)
    test_data = DataSet(x_test, y_test)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    # prepare sklearn model to train w as used in original paper code
    C = 1.0 / (train_sample_num * WEIGHT_DECAY)
    sklearn_model = linear_model.LogisticRegression(C=C, solver='lbfgs', tol=1e-8, fit_intercept=False,
                                                    multi_class='multinomial', warm_start=True)

    # prepare pytorch model to compute influence function
    torch_model = LR(weight_decay=WEIGHT_DECAY, is_multi=True)

    # train
    sklearn_model.fit(x_train, y_train.ravel())
    print('LBFGS training took %s iter.' % sklearn_model.n_iter_)

    # assign W into pytorch model
    w_opt = sklearn_model.coef_
    with torch.no_grad():
        torch_model.w = torch.nn.Parameter(
            torch.tensor(w_opt, dtype=torch.float)  # torch.Size([10, 784])
        )
    get_accuracy(torch_model, test_loader)

    # calculate original loss
    x_test_input = torch.FloatTensor(x_test[TEST_INDEX: TEST_INDEX + 1])
    y_test_input = torch.LongTensor(y_test[TEST_INDEX: TEST_INDEX + 1])

    test_data = DataSet(x_test[TEST_INDEX: TEST_INDEX + 1], y_test[TEST_INDEX: TEST_INDEX + 1])
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)


    test_loss_ori = torch_model.loss(torch_model(x_test_input), y_test_input, train=False).detach().cpu().numpy()

    print('Original loss       :{}'.format(test_loss_ori))

    loss_diff_approx, _, _, _, = calc_influence_single(torch_model, train_loader, test_loader, test_id_num=0,
                                                       recursion_depth=RECURSION_DEPTH, r=R, damp=0, scale=25)
    loss_diff_approx = - torch.FloatTensor(loss_diff_approx).cpu().numpy()

    # get high and low loss diff indice, checking stability
    sorted_indice = np.argsort(loss_diff_approx)
    sample_indice = np.concatenate([sorted_indice[-int(SAMPLE_NUM / 2):], sorted_indice[:int(SAMPLE_NUM / 2)]])

    # calculate true loss diff
    loss_diff_true = np.zeros(SAMPLE_NUM)
    for i, index in zip(range(SAMPLE_NUM), sample_indice):
        print('[{}/{}]'.format(i + 1, SAMPLE_NUM))

        # get minus one dataset
        x_train_minus_one = np.delete(x_train, index, axis=0)
        y_train_minus_one = np.delete(y_train, index, axis=0)

        # retrain
        C = 1.0 / ((train_sample_num - 1) * WEIGHT_DECAY)
        sklearn_model_minus_one = linear_model.LogisticRegression(C=C, fit_intercept=False, tol=1e-8, solver='lbfgs')
        sklearn_model_minus_one.fit(x_train_minus_one, y_train_minus_one.ravel())
        print('LBFGS training took {} iter.'.format(sklearn_model_minus_one.n_iter_))

        # assign w on tensorflow model
        w_retrain = sklearn_model_minus_one.coef_
        with torch.no_grad():
            torch_model.w = torch.nn.Parameter(
                torch.tensor(w_retrain, dtype=torch.float)
            )

        # get retrain loss
        test_loss_retrain = torch_model.loss(torch_model(x_test_input), y_test_input,
                                             train=False).detach().cpu().numpy()

        # get true loss diff
        loss_diff_true[i] = test_loss_retrain - test_loss_ori

        print('Original loss       :{}'.format(test_loss_ori))
        print('Retrain loss        :{}'.format(test_loss_retrain))
        print('True loss diff      :{}'.format(loss_diff_true[i]))
        print('Estimated loss diff :{}'.format(loss_diff_approx[index]))

    pickle.dump(loss_diff_true, open('loss_diff_true.pkl', 'wb'))
    pickle.dump(loss_diff_approx[sample_indice], open('loss_diff_approx.pkl', 'wb'))
    r2_score = visualize_result(loss_diff_true, loss_diff_approx[sample_indice], OUTPUT_DIR)


if __name__ == "__main__":
    leave_one_out()
    loss_diff_true = pickle.load(open('loss_diff_true.pkl', 'rb'))
    loss_diff_approx = pickle.load(open('loss_diff_approx.pkl', 'rb'))
    visualize_result(loss_diff_true, loss_diff_approx, OUTPUT_DIR)
