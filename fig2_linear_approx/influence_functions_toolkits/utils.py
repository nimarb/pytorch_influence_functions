import sys
import json
import logging
from pathlib import Path
from datetime import datetime as dt
from typing import Sequence

import numpy as np
import torch
from scipy.optimize import fmin_ncg


def save_json(
    json_obj,
    json_path,
    append_if_exists=False,
    overwrite_if_exists=False,
    unique_fn_if_exists=True,
):
    """Saves a json file

    Arguments:
        json_obj: json, json object
        json_path: Path, path including the file name where the json object
            should be saved to
        append_if_exists: bool, append to the existing json file with the same
            name if it exists (keep the json structure intact)
        overwrite_if_exists: bool, xor with append, overwrites any existing
            target file
        unique_fn_if_exsists: bool, appends the current date and time to the
            file name if the target file exists already.
    """
    if isinstance(json_path, str):
        json_path = Path(json_path)

    if overwrite_if_exists:
        append_if_exists = False
        unique_fn_if_exists = False

    if unique_fn_if_exists:
        overwrite_if_exists = False
        append_if_exists = False
        if json_path.exists():
            time = dt.now().strftime("%Y-%m-%d-%H-%M-%S")
            json_path = (
                json_path.parents[0] / f"{str(json_path.stem)}_{time}"
                f"{str(json_path.suffix)}"
            )

    if overwrite_if_exists:
        append_if_exists = False
        with open(json_path, "w+") as fout:
            json.dump(json_obj, fout, indent=2)
        return

    if append_if_exists:
        if json_path.exists():
            with open(json_path, "r") as fin:
                read_file = json.load(fin)
            read_file.update(json_obj)
            with open(json_path, "w+") as fout:
                json.dump(read_file, fout, indent=2)
            return

    with open(json_path, "w+") as fout:
        json.dump(json_obj, fout, indent=2)


def display_progress(text, current_step, last_step, enabled=True, fix_zero_start=True):
    """Draws a progress indicator on the screen with the text preceeding the
    progress

    Arguments:
        test: str, text displayed to describe the task being executed
        current_step: int, current step of the iteration
        last_step: int, last possible step of the iteration
        enabled: bool, if false this function will not execute. This is
            for running silently without stdout output.
        fix_zero_start: bool, if true adds 1 to each current step so that the
            display starts at 1 instead of 0, which it would for most loops
            otherwise.
    """
    if not enabled:
        return

    # Fix display for most loops which start with 0, otherwise looks weird
    if fix_zero_start:
        current_step = current_step + 1

    term_line_len = 80
    final_chars = [":", ";", " ", ".", ","]
    if text[-1:] not in final_chars:
        text = text + " "
    if len(text) < term_line_len:
        bar_len = term_line_len - (
            len(text) + len(str(current_step)) + len(str(last_step)) + len("  / ")
        )
    else:
        bar_len = 30
    filled_len = int(round(bar_len * current_step / float(last_step)))
    bar = "=" * filled_len + "." * (bar_len - filled_len)

    bar = f"{text}[{bar:s}] {current_step:d} / {last_step:d}"
    if current_step < last_step - 1:
        # Erase to end of line and print
        sys.stdout.write("\033[K" + bar + "\r")
    else:
        sys.stdout.write(bar + "\n")

    sys.stdout.flush()


def init_logging(filename=None):
    """Initialises log/stdout output

    Arguments:
        filename: str, a filename can be set to output the log information to
            a file instead of stdout"""
    log_lvl = logging.INFO
    log_format = "%(asctime)s: %(message)s"
    if filename:
        logging.basicConfig(
            handlers=[logging.FileHandler(filename), logging.StreamHandler(sys.stdout)],
            level=log_lvl,
            format=log_format,
        )
    else:
        logging.basicConfig(stream=sys.stdout, level=log_lvl, format=log_format)


def get_default_config():
    """Returns a default config file"""
    config = {
        "outdir": "outdir",
        "seed": 42,
        "gpu": 0,
        "dataset": "CIFAR10",
        "num_classes": 10,
        "test_sample_num": 1,
        "test_start_index": 0,
        "recursion_depth": 1,
        "r_averaging": 1,
        "scale": None,
        "damp": None,
        "calc_method": "img_wise",
        "log_filename": None,
    }

    return config


def conjugate_gradient(ax_fn, b, debug_callback=None, avextol=None, maxiter=None):
    """Computes the solution to Ax - b = 0 by minimizing the conjugate objective
    f(x) = x^T A x / 2 - b^T x. This does not require evaluating the matrix A
    explicitly, only the matrix vector product Ax.

    From https://github.com/kohpangwei/group-influence-release/blob/master/influence/conjugate.py.

    Args:
      ax_fn: A function that return Ax given x.
      b: The vector b.
      debug_callback: An optional debugging function that reports the current optimization function. Takes two
          parameters: the current solution and a helper function that evaluates the quadratic and linear parts of the
          conjugate objective separately. (Default value = None)
      avextol:  (Default value = None)
      maxiter:  (Default value = None)

    Returns:
      The conjugate optimization solution.

    """

    cg_callback = None
    if debug_callback:
        cg_callback = lambda x: debug_callback(
            x, -np.dot(b, x), 0.5 * np.dot(x, ax_fn(x))
        )

    result = fmin_ncg(
        f=lambda x: 0.5 * np.dot(x, ax_fn(x)) - np.dot(b, x),
        x0=np.zeros_like(b),
        fprime=lambda x: ax_fn(x) - b,
        fhess_p=lambda x, p: ax_fn(p),
        callback=cg_callback,
        avextol=avextol,
        maxiter=maxiter,
    )

    return result


def del_attr(obj, names):
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])


def set_attr(obj, names, val):
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)


def make_functional(model):
    orig_params = tuple(model.parameters())
    # Remove all the parameters in the model
    names = []

    for name, p in list(model.named_parameters()):
        del_attr(model, name.split("."))
        names.append(name)

    return orig_params, names


def load_weights(model, names, params, as_params=False):
    for name, p in zip(names, params):
        if not as_params:
            set_attr(model, name.split("."), p)
        else:
            set_attr(model, name.split("."), torch.nn.Parameter(p))


def tensor_to_tuple(vec, parameters):
    r"""Convert one vector to the parameters

    Adapted from
    https://pytorch.org/docs/master/generated/torch.nn.utils.vector_to_parameters.html#torch.nn.utils.vector_to_parameters

    Arguments:
        vec (Tensor): a single vector represents the parameters of a model.
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.
    """
    if not isinstance(vec, torch.Tensor):
        raise TypeError('expected torch.Tensor, but got: {}'
                        .format(torch.typename(vec)))

    # Pointer for slicing the vector for each parameter
    pointer = 0

    split_tensors = []
    for param in parameters:

        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        split_tensors.append(vec[pointer:pointer + num_param].view_as(param))

        # Increment the pointer
        pointer += num_param

    return tuple(split_tensors)


def parameters_to_vector(parameters):
    r"""Convert parameters to one vector

    Arguments:
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.

    Returns:
        The parameters represented by a single vector
    """
    # Flag for the device where the parameter is located

    vec = []
    for param in parameters:
        vec.append(param.view(-1))

    return torch.cat(vec)
