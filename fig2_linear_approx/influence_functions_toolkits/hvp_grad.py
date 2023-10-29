#! /usr/bin/env python3
import torch
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector
from torch.autograd import grad
from torch.autograd.functional import vhp
from torch.utils.data import DataLoader
from tqdm import tqdm

from fig2_linear_approx.influence_functions_toolkits.utils import (
    conjugate_gradient,
    load_weights,
    make_functional,
    tensor_to_tuple,
)


def s_test_cg(x_test, y_test, model, train_loader, damp, gpu=-1, verbose=True, loss_func="cross_entropy"):

    if gpu > 0:
        x_test, y_test = x_test.cuda(), y_test.cuda()

    v_flat = parameters_to_vector(grad_z(x_test, y_test, model, gpu, loss_func=loss_func))

    def hvp_fn(x):

        x_tensor = torch.tensor(x, requires_grad=False)
        if gpu > 0:
            x_tensor = x_tensor.cuda()

        params, names = make_functional(model)
        # Make params regular Tensors instead of nn.Parameter
        params = tuple(p.detach().requires_grad_() for p in params)
        flat_params = parameters_to_vector(params)

        hvp = torch.zeros_like(flat_params)

        for x_train, y_train in train_loader:

            if gpu > 0:
                x_train, y_train = x_train.cuda(), y_train.cuda()

            def f(flat_params_):
                split_params = tensor_to_tuple(flat_params_, params)
                load_weights(model, names, split_params)
                out = model(x_train)
                loss = calc_loss(out, y_train)
                return loss

            batch_hvp = vhp(f, flat_params, x_tensor, strict=True)[1]

            hvp += batch_hvp / float(len(train_loader))

        with torch.no_grad():
            load_weights(model, names, params, as_params=True)
            damped_hvp = hvp + damp * v_flat

        return damped_hvp.cpu().numpy()

    def print_function_value(_, f_linear, f_quadratic):
        print(
            f"Conjugate function value: {f_linear + f_quadratic}, lin: {f_linear}, quad: {f_quadratic}"
        )

    debug_callback = print_function_value if verbose else None

    result = conjugate_gradient(
        hvp_fn,
        v_flat.cpu().numpy(),
        debug_callback=debug_callback,
        avextol=1e-8,
        maxiter=100,
    )

    result = torch.tensor(result)
    if gpu > 0:
        result = result.cuda()

    return result


def s_test(x_test, y_test, model, i, samples_loader, gpu=-1, damp=0.01, scale=25.0, loss_func="cross_entropy"):
    """s_test can be precomputed for each test point of interest, and then
    multiplied with grad_z to get the desired value for each training point.
    Here, stochastic estimation is used to calculate s_test. s_test is the
    Inverse Hessian Vector Product.

    Arguments:
        x_test: torch tensor, test data points, such as test images
        y_test: torch tensor, contains all test data labels
        model: torch NN, model used to evaluate the dataset
        i: the sample number
        samples_loader: torch DataLoader, can load the training dataset
        gpu: int, GPU id to use if >=0 and -1 means use CPU
        damp: float, dampening factor
        scale: float, scaling factor

    Returns:
        h_estimate: list of torch tensors, s_test"""

    v = grad_z(x_test, y_test, model, gpu, loss_func=loss_func)
    h_estimate = v

    params, names = make_functional(model)
    # Make params regular Tensors instead of nn.Parameter
    params = tuple(p.detach().requires_grad_() for p in params)

    # TODO: Dynamically set the recursion depth so that iterations stop once h_estimate stabilises
    progress_bar = tqdm(samples_loader, desc=f"IHVP sample {i}")
    for i, (x_train, y_train) in enumerate(progress_bar):

        if gpu > 0:
            x_train, y_train = x_train.cuda(), y_train.cuda()

        def f(*new_params):
            load_weights(model, names, new_params)
            out = model(x_train)
            loss = calc_loss(out, y_train, loss_func=loss_func)
            return loss

        hv = vhp(f, params, tuple(h_estimate), strict=True)[1]

        # Recursively calculate h_estimate
        with torch.no_grad():
            h_estimate = [
                _v + (1 - damp) * _h_e - _hv / scale
                for _v, _h_e, _hv in zip(v, h_estimate, hv)
            ]

            if i % 100 == 0:
                norm = sum([h_.norm() for h_ in h_estimate])
                progress_bar.set_postfix({"est_norm": norm.item()})

    with torch.no_grad():
        load_weights(model, names, params, as_params=True)

    return h_estimate


def calc_loss(logits, labels, loss_func="cross_entropy"):
    """Calculates the loss

    Arguments:
        logits: torch tensor, input with size (minibatch, nr_of_classes)
        labels: torch tensor, target expected by loss of size (0 to nr_of_classes-1)
        loss_func: str, specify loss function name

    Returns:
        loss: scalar, the loss"""
    
    if loss_func == "cross_entropy":
        if logits.shape[-1] == 1:
            loss = F.binary_cross_entropy_with_logits(logits, labels.type(torch.float))
        else:
            criterion = torch.nn.CrossEntropyLoss()
            loss = criterion(logits, labels)
    elif loss_func == "mean":
        loss = torch.mean(logits)
    else:
        raise ValueError("{} is not a valid value for loss_func".format(loss_func))

    return loss


def grad_z(x, y, model, gpu=-1, loss_func="cross_entropy"):
    """Calculates the gradient z. One grad_z should be computed for each
    training sample.

    Arguments:
        x: torch tensor, training data points
            e.g. an image sample (batch_size, 3, 256, 256)
        y: torch tensor, training data labels
        model: torch NN, model used to evaluate the dataset
        gpu: int, device id to use for GPU, -1 for CPU

    Returns:
        grad_z: list of torch tensor, containing the gradients
            from model parameters to loss"""
    model.eval()

    # initialize
    if gpu > 0:
        print(gpu)
        x, y = x.cuda(), y.cuda()

    prediction = model(x)

    loss = calc_loss(prediction, y, loss_func=loss_func)

    # Compute sum of gradients from model parameters to loss
    return grad(loss, model.parameters())


def s_test_sample(
    model,
    x_test,
    y_test,
    train_loader,
    gpu=-1,
    damp=0.01,
    scale=25,
    recursion_depth=5000,
    r=1,
    loss_func="cross_entropy",
):
    """Calculates s_test for a single test image taking into account the whole
    training dataset. s_test = invHessian * nabla(Loss(test_img, model params))

    Arguments:
        model: pytorch model, for which s_test should be calculated
        x_test: test image
        y_test: test image label
        train_loader: pytorch dataloader, which can load the train data
        gpu: int, device id to use for GPU, -1 for CPU (default)
        damp: float, influence function damping factor |
        scale: float, influence calculation scaling factor (to keep hessian <= I) | in the paper code use 25
        recursion_depth: int, number of recursions to perform during s_test
            calculation, increases accuracy. r*recursion_depth should equal the
            training dataset size.
        r: int, number of iterations of which to take the avg.
            of the h_estimate calculation; r*recursion_depth should equal the
            training dataset size.
        loss_func: cross_entropy

    Returns:
        s_test_vec: torch tensor, contains s_test for a single test image
    """

    """
    initialize inverse_hvp as a list of tensors with zeros, which should be first s_test as described in the paper
    H_0^(-1)v = v
    """
    inverse_hvp = [
        torch.zeros_like(params, dtype=torch.float) for params in model.parameters()
    ]

    for i in range(r): # repeat r times to get average

        hessian_loader = DataLoader(
            train_loader.dataset,
            sampler=torch.utils.data.RandomSampler(
                train_loader.dataset, True, num_samples=recursion_depth # as mentioned in paper, use "enought" samples
            ),
            batch_size=1,
            # num_workers=4,
        )

        cur_estimate = s_test(
            x_test, y_test, model, i, hessian_loader, gpu=gpu, damp=damp, scale=scale, loss_func=loss_func,
        )

        with torch.no_grad():
            inverse_hvp = [
                old + (cur / scale) for old, cur in zip(inverse_hvp, cur_estimate) # update inverse_hvp by adding new cur_estimate
            ]

    with torch.no_grad():
        inverse_hvp = [component / r for component in inverse_hvp]

    return inverse_hvp
