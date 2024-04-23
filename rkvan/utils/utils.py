import os
import torch
import inspect
import re
import numpy as np

from datetime import datetime
from loguru import logger
import random

import pdb

def init_experiment(args, runner_name=None, exp_id=None, resume_path=None):
    if resume_path is not None:
        log_dir     = resume_path
        args.resume = True

    else:
        # Get filepath of calling script
        if runner_name is None:
            runner_name = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))).split(".")[-2:]

        root_dir = os.path.join(args.exp_root, *runner_name)
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)

        # Either generate a unique experiment ID, or use one which is passed
        if exp_id is None:

            if args.exp_name is None:
                raise ValueError("Need to specify the experiment name")
            # Unique identifier for experiment
            now = '{}_({:02d}.{:02d}.{}_|_'.format(args.exp_name, datetime.now().day, datetime.now().month, datetime.now().year) + \
                datetime.now().strftime("%S.%f")[:-3] + ')'

            log_dir = os.path.join(root_dir, 'log', now)
            while os.path.exists(log_dir):
                now = '({:02d}.{:02d}.{}_|_'.format(datetime.now().day, datetime.now().month, datetime.now().year) + \
                    datetime.now().strftime("%S.%f")[:-3] + ')'

                log_dir = os.path.join(root_dir, 'log', now)

        else:

            log_dir = os.path.join(root_dir, 'log', f'{exp_id}')

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    logger.add(os.path.join(log_dir, 'log.txt'))
    args.logger = logger
    args.log_dir = log_dir

    # Instantiate directory to save models to
    model_root_dir = os.path.join(args.log_dir, 'checkpoints')
    if not os.path.exists(model_root_dir):
        os.mkdir(model_root_dir)

    args.model_dir = model_root_dir
    args.model_path = os.path.join(args.model_dir, 'model.pt')
    args.best_model_path = os.path.join(args.model_dir, 'model_best.pt')

    print(f'Experiment saved to: {args.log_dir}')

    hparam_dict = {}

    for k, v in vars(args).items():
        if isinstance(v, (int, float, str, bool, torch.Tensor)):
            hparam_dict[k] = v

    return args


# model structure relevant
def print_model_para_name(model):
    for name, param in model.named_parameters():
        print(name)


def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def freeze_layer(model, layers_name_freeze=None, layer_name_keep=None):
    # backbone.conv1 backbone.bn1 backbone.layer1 .......
    if layers_name_freeze != None:
        for name, param in model.named_parameters():
            for fl in layers_name_freeze:
                if fl in name:
                    param.requires_grad = False
    if layer_name_keep != None:
        for name, param in model.named_parameters():
            for fl in layer_name_keep:
                if fl not in name:
                    param.requires_grad = False
    return model


def set_seed(seed):
    if seed == 0:
        print(' random seed')
        torch.backends.cudnn.benchmark = True
    else:
        print('manual seed:', seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def set_gpu(args):
    gpu_list = [int(x) for x in args.gpu.split(',')]
    print('use gpu:', gpu_list)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return gpu_list.__len__()


def load_trained_paras(path: str, models: list, keys:list, map_location="cpu", logger=None, sub_level=None):
    if logger:
        logger.info(f"Load pretrained model [{model.__class__.__name__}] from {path}")
    if os.path.exists(path):
        # From local
        state_dict = torch.load(path, map_location)
    # elif path.startswith("http"):
    #     # From url
    #     state_dict = load_state_dict_from_url(path, map_location=map_location, check_hash=False)
    else:
        raise Exception(f"Cannot find {path} when load pretrained")
    
    model_trained = []
    for i in range(len(models)):
        model, key = models[i], keys[i]
        model = load_pretrained_dict(model, state_dict, key)
        model_trained.append(model)

    return model_trained


def _auto_drop_invalid(model: torch.nn.Module, state_dict: dict, logger=None):
    """ Strip unmatched parameters in state_dict, e.g. shape not matched, type not matched.

    Args:
        model (torch.nn.Module):
        state_dict (dict):
        logger (logging.Logger, None):

    Returns:
        A new state dict.
    """
    ret_dict = state_dict.copy()
    invalid_msgs = []
    for key, value in model.state_dict().items():
        if key in state_dict:
            # Check shape
            new_value = state_dict[key]
            if value.shape != new_value.shape:
                invalid_msgs.append(f"{key}: invalid shape, dst {value.shape} vs. src {new_value.shape}")
                ret_dict.pop(key)
            elif value.dtype != new_value.dtype:
                invalid_msgs.append(f"{key}: invalid dtype, dst {value.dtype} vs. src {new_value.dtype}")
                ret_dict.pop(key)
    if len(invalid_msgs) > 0:
        warning_msg = "ignore keys from source: \n" + "\n".join(invalid_msgs)
        if logger:
            logger.warning(warning_msg)
        else:
            import warnings
            warnings.warn(warning_msg)
    return ret_dict


def load_pretrained_dict(model: torch.nn.Module, state_dict: dict, key:str, logger=None, sub_level=None):
    """ Load parameters to model with
    1. Sub name by revise_keys For DataParallelModel or DistributeParallelModel.
    2. Load 'state_dict' again if possible by key 'state_dict' or 'model_state'.
    3. Take sub level keys from source, e.g. load 'backbone' part from a classifier into a backbone model.
    4. Auto remove invalid parameters from source.
    5. Log or warning if unexpected key exists or key misses.

    Args:
        model (torch.nn.Module):
        state_dict (dict): dict of parameters
        logger (logging.Logger, None):
        sub_level (str, optional): If not None, parameters with key startswith sub_level will remove the prefix
            to fit actual model keys. This action happens if user want to load sub module parameters
            into a sub module model.
    """
    revise_keys = [(r'^module\.', '')]
    state_dict = state_dict[key]
    for p, r in revise_keys:
        state_dict = {re.sub(p, r, k): v for k, v in state_dict.items()}

    if sub_level:
        sub_level = sub_level if sub_level.endswith(".") else (sub_level + ".")
        sub_level_len = len(sub_level)
        state_dict = {key[sub_level_len:]: value
                      for key, value in state_dict.items()
                      if key.startswith(sub_level)}

    state_dict = _auto_drop_invalid(model, state_dict, logger=logger)

    load_status = model.load_state_dict(state_dict, strict=False)
    unexpected_keys = load_status.unexpected_keys
    missing_keys = load_status.missing_keys
    err_msgs = []
    if unexpected_keys:
        err_msgs.append('unexpected key in source '
                        f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msgs.append('missing key in source '
                        f'state_dict: {", ".join(missing_keys)}\n')
    err_msgs = '\n'.join(err_msgs)

    if len(err_msgs) > 0:
        if logger:
            logger.warning(err_msgs)
        else:
            import warnings
            warnings.warn(err_msgs)
    return model