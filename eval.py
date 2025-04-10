
from ctypes.util import test
from re import A
import yaml
from torch.utils.tensorboard import SummaryWriter
import torch
import os
import shutil
import sys
import csv
from pathlib import Path
import time

import argparse
import pickle
from models.resnet_simclr import ResNetSimCLR
# from clinical_ts.cpc import CPCModel
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from clinical_ts.simclr_dataset_wrapper import SimCLRDataSetWrapper
from clinical_ts.eval_utils_cafa import eval_scores, eval_scores_bootstrap
from clinical_ts.timeseries_utils import aggregate_predictions
import pdb
from copy import deepcopy
from os.path import join, isdir
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def parse_args():
    parser = argparse.ArgumentParser("Finetuning tests")
    parser.add_argument("--model_file")
    parser.add_argument("--log_dir", help='path to logging folder', default='evaluation_logs/')
    # parser.add_argument("--model_folder", help='path to pre-trained model logging folder')
    parser.add_argument("--method", help='choose a model from swav, simclr or byol', default='swav')
    parser.add_argument('--data_path', dest="data_path", type=str, help='path to dataset used dataset')
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--l_epochs", type=int, default=0, help="number of head-only epochs (these are performed first)")
    parser.add_argument("--f_epochs", type=int, default=0, help="number of finetuning epochs (these are perfomed after head-only training")
    parser.add_argument("--discriminative_lr", default=False, action="store_true")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--linear_evaluation", default=False, action="store_true", help="use linear evaluation")
    parser.add_argument("--load_finetuned", action="store_true", default=False)
    parser.add_argument("--eval_only", action="store_true", default=False, help="only evaluate mode")
    parser.add_argument("--verbose", action="store_true", default=False)
    
    parser.add_argument("--hidden", default=False, action="store_true")
    parser.add_argument("--lr_schedule", default="{}")
    parser.add_argument("--use_pretrained", default=False, action="store_true")
    parser.add_argument("--test_noised", default=False, action="store_true", help="validate also on a noisy dataset")
    parser.add_argument("--noise_level", default=1, type=int, help="level of noise induced to the second validations set")
    parser.add_argument("--folds", default=8, type=int, help="number of folds used in finetuning (between 1-8)")
    parser.add_argument("--tag", default="")
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--cpc", action="store_true", default=False)
    parser.add_argument("--normalize", action="store_true", default=False, help="normalize dataset with ptbxl mean and std")
    parser.add_argument("--bn_head", action="store_true", default=False)
    parser.add_argument("--ps_head", type=float, default=0.0)
    parser.add_argument("--conv_encoder", action="store_true", default=False)
    parser.add_argument("--base_model", default="xresnet1d50")
    parser.add_argument("--widen", default=1, type=int, help="use wide xresnet1d50")
    args = parser.parse_args()
    return args


def get_new_state_dict(init_state_dict, lightning_state_dict, method="simclr"):
    from collections import OrderedDict
    # lightning_state_dict = lightning_state_dict["state_dict"]
    new_state_dict = OrderedDict()

    if method == "simclr":
        for key in init_state_dict:
            if "features" in key:
                l_key = key.replace("features", "encoder.features")
            if l_key in lightning_state_dict.keys():
                new_state_dict[key] = lightning_state_dict[l_key]
    elif method == "swav":
        for key in init_state_dict:
            if "features" in key:
                l_key = key.replace("features", "model.features")
            if l_key in lightning_state_dict.keys():
                new_state_dict[key] = lightning_state_dict[l_key]
    elif method == "byol":
        for key in init_state_dict:
            l_key = "online_network.encoder." + key
            if l_key in lightning_state_dict.keys():
                new_state_dict[key] = lightning_state_dict[l_key]
    else:
        raise("method unknown")
    new_state_dict["l1.weight"] = init_state_dict["l1.weight"]
    new_state_dict["l1.bias"] = init_state_dict["l1.bias"]
    if "l2.weight" in init_state_dict.keys():
        new_state_dict["l2.weight"] = init_state_dict["l2.weight"]
        new_state_dict["l2.bias"] = init_state_dict["l2.bias"]

    assert(len(init_state_dict) == len(new_state_dict))
    return new_state_dict


def adjust(model, num_classes, hidden=False):
    in_features = model.l1.in_features
    last_layer = torch.nn.modules.linear.Linear(
        in_features, num_classes).to(device)
    if hidden:
        model.l1 = torch.nn.modules.linear.Linear(
            in_features, in_features).to(device)
        model.l2 = last_layer
    else:
        model.l1 = last_layer

    def def_forward(self):
        def new_forward(x):
            h = self.features(x)
            h = h.squeeze()

            x = self.l1(h)
            if hidden:
                x = F.relu(x)
                x = self.l2(x)
            return x
        return new_forward

    model.forward = def_forward(model)


def configure_optimizer(model, batch_size, head_only=False, discriminative_lr=False, base_model="xresnet1d", optimizer="adam"):
    loss_fn = F.binary_cross_entropy_with_logits
    if base_model == "xresnet1d":
        wd = 1e-3
        # linear evaluation
        if head_only:
            # linearly scale learning rate with batchsize
            # lr = (5e-3*(batch_size/512))
            lr = 5e-3
            optimizer = torch.optim.AdamW(
                model.l1.parameters(), lr=lr, weight_decay=wd)
        # fine-tuning
        else:
            # use higher learning rate than backbone to preserve learned features 
            lr = 5e-3
            if not discriminative_lr:
                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=lr, weight_decay=wd)
            else:
                param_dict = dict(model.named_parameters())
                keys = param_dict.keys()
                weight_layer_nrs = set()
                for key in keys:
                    if "features" in key:
                        # parameter names have the form features.x
                        weight_layer_nrs.add(key[9])
                weight_layer_nrs = sorted(weight_layer_nrs, reverse=True)
                features_groups = []
                while len(weight_layer_nrs) > 0:
                    if len(weight_layer_nrs) > 1:
                        features_groups.append(list(filter(
                            lambda x: "features." + weight_layer_nrs[0] in x or "features." + weight_layer_nrs[1] in x,  keys)))
                        del weight_layer_nrs[:2]
                    else:
                        features_groups.append(
                            list(filter(lambda x: "features." + weight_layer_nrs[0] in x,  keys)))
                        del weight_layer_nrs[0]
                # filter linear layers
                linears = list(filter(lambda x: "l" in x, keys))
                groups = [linears] + features_groups
                optimizer_param_list = []
                tmp_lr = lr

                for layers in groups:
                    layer_params = [param_dict[param_name]
                                    for param_name in layers]
                    optimizer_param_list.append(
                        {"params": layer_params, "lr": tmp_lr})
                    tmp_lr /= 4
                optimizer = torch.optim.AdamW(
                    optimizer_param_list, lr=lr, weight_decay=wd)
        print("lr", lr)
        print("wd", wd)
        print("batch size", batch_size)

    else:
        raise("model unknown")
    return loss_fn, optimizer


def load_model(num_classes, use_pretrained, hidden=False, location="./checkpoints/moco_baselinewonder200.ckpt", method="swav", base_model="xresnet1d50", out_dim=16, widen=1):
    if use_pretrained:
        print("load pre-trained model from " + location)
        if "xresnet1d" in base_model:
            model = ResNetSimCLR(base_model, out_dim, hidden=hidden, widen=widen).to(device)
            model_state_dict = torch.load(location, weights_only=True, map_location=device)
            if "state_dict" in model_state_dict.keys():
                model_state_dict = model_state_dict["state_dict"]
            if "l1.weight" in model_state_dict.keys():  # load already fine-tuned model
                model_classes = model_state_dict["l1.weight"].shape[0]
                if model_classes != num_classes:
                    raise Exception("Loaded model has different output dim ({}) than needed ({})".format(
                        model_classes, num_classes))
                adjust(model, num_classes, hidden=hidden)
                if not hidden and "l2.weight" in model_state_dict:
                    del model_state_dict["l2.weight"]
                    del model_state_dict["l2.bias"]
                model.load_state_dict(model_state_dict)
            else:  # load pretrained model
                base_dict = model.state_dict()
                model_state_dict = get_new_state_dict(
                    base_dict, model_state_dict, method=method)
                model.load_state_dict(model_state_dict)
                adjust(model, num_classes, hidden=hidden)
        else:
            raise Exception("model unknown")
    else:
        if "xresnet1d" in base_model:
            model = ResNetSimCLR(base_model, out_dim, hidden=hidden, widen=widen).to(device)
            adjust(model, num_classes, hidden=hidden)
        else:
            raise Exception("model unknown")

    return model

# val_preds, macro_auc, macro_f1, macro_agg_auc, macro_agg_f1 = evaluate(model, valid_loader)
def evaluate(model, dataloader, verbose, test=False): 
    preds, targs = eval_model(model, dataloader, test, verbose=verbose)
    scores = eval_scores(targs, preds, parallel=True)
    
    preds_agg, targs_agg = aggregate_predictions(preds, targs)
    scores_agg = eval_scores(targs_agg, preds_agg, parallel=True)
    
    macro_auc = scores["label_AUC"]["macro"]
    macro_auc_agg = scores_agg["label_AUC"]["macro"]
    
    macro_f1 = scores['f1']['macro']
    macro_f1_agg = scores_agg["f1"]["macro"]
    
    return preds, macro_auc, macro_f1, macro_auc_agg, macro_f1_agg, targs


def set_train_eval(model, linear_evaluation):
    if linear_evaluation:
        model.features.eval()
    else:
        model.train()


def train_model(model, train_loader, valid_loader, test_loader, epochs, loss_fn, optimizer, head_only=True, linear_evaluation=False, percentage=1, lr_schedule=None, save_model_at=None, writer=None, global_step=0, verbose=False):
    if head_only:
        if linear_evaluation:
            print("linear evaluation for {} epochs".format(epochs))
        else:
            print("head-only for {} epochs".format(epochs))
    else:
        print("fine tuning for {} epochs".format(epochs))

    # freeze parameters if head_only
    if head_only:
        for key, param in model.named_parameters():
            if "l1." not in key and "head." not in key:
                param.requires_grad = False
        print("copying state dict before training for sanity check after training")
    else:
        for param in model.parameters():
            param.requires_grad = True

    data_type = model.features[0][0].weight.type()
    set_train_eval(model, linear_evaluation)
    state_dict_pre = deepcopy(model.state_dict())
    
    loss_per_epoch = []
    macro_agg_f1_per_epoch = []
    max_batches = len(train_loader)
    break_point = int(percentage*max_batches)
    
    best_macro_auc = 0      # best validation macro AUC
    best_macro_f1_agg = 0  # aggeragete over crops?? 
    best_val_epoch = 0      # epoch of best macro AUC
    best_val_preds = None   # save predictions of best validation epoch?
    test_macro_auc = 0      # test macro AUC
    test_macro_auc_agg = 0  # test aggregate AUC over?

    for epoch in range(epochs):
        if type(lr_schedule) == dict:
            if epoch in lr_schedule.keys():
                for param_group in optimizer.param_groups:
                    param_group['lr'] /= lr_schedule[epoch]

        # print epoch on single line instead of tqdm
        if not verbose:
            print(f"Train epoch ({epoch+1}/{epochs})", end='\r')
        # train
        total_loss_one_epoch = 0
        for batch_idx, samples in enumerate(tqdm(train_loader, desc=f'Trainloader Epoch({epoch+1}/{epochs})', disable=not verbose)):
            if batch_idx == break_point:
                print("break at batch nr.", batch_idx)
                break
            # print(len(samples))
            # print(samples[0].shape)
            # print(samples[1].shape)
            # print('\n'*4)
            data = samples[0].to(device).type(data_type)
            labels = samples[1].to(device).type(data_type)
            optimizer.zero_grad()
            preds = model(data)
            loss = loss_fn(preds, labels)
            loss.backward()
            optimizer.step()
            
            # log train loss at each step
            if writer is not None:
                prefix = 'linear/' if linear_evaluation else 'finetune/'
                writer.add_scalar('Train Loss (Step)', loss.item(), global_step)
                global_step += 1

            total_loss_one_epoch += loss.item()
            # if(batch_idx % 100 == 0):
            #     print(epoch, batch_idx, loss.item())
        loss_per_epoch.append(total_loss_one_epoch)

        # validation
        val_preds, macro_auc, macro_f1, macro_auc_agg, macro_f1_agg, val_ytrue = evaluate(model, valid_loader, verbose=verbose)
        macro_agg_f1_per_epoch.append(macro_f1_agg)
        
        if writer is not None:
            writer.add_scalar(prefix + "Train Loss (Epoch)", total_loss_one_epoch, epoch)
            writer.add_scalar(prefix + "Val Macro AUC", macro_auc, epoch)
            writer.add_scalar(prefix + "Val Macro F1", macro_f1, epoch)

        if verbose:
            print(f'Validaiton epoch {epoch+1}, train loss {total_loss_one_epoch}, macro f1 {macro_f1}')

        # save best model based on highest macro f1 on validation set 
        if macro_f1_agg > best_macro_f1_agg:
            torch.save(model.state_dict(), save_model_at)
            best_macro_f1_agg = macro_f1_agg
            best_macro_f1 = macro_f1
            best_epoch = epoch
            best_val_preds = val_preds
        

    # Testing: load best validation model and evaluate on test set
    model.load_state_dict(torch.load(save_model_at, weights_only=True))
    print('Finished training')
    print(f'Loaded best validation model from epoch={best_epoch}')
    test_preds, test_macro_auc, test_macro_f1, test_macro_auc_agg, test_macro_f1_agg, test_ytrue = evaluate(model, test_loader, test=True, verbose=verbose)
    
    if writer is not None:
        writer.add_scalar(prefix + "Test Macro AUC", test_macro_auc, epoch)
        writer.add_scalar(prefix + "Test Macro F1", test_macro_f1, epoch)

    set_train_eval(model, linear_evaluation)

    if epochs > 0:
        sanity_check(model, state_dict_pre, linear_evaluation, head_only)
    return loss_per_epoch, macro_agg_f1_per_epoch, best_macro_f1, best_macro_f1_agg, test_macro_f1, test_macro_f1_agg, best_epoch, best_val_preds, test_preds, val_ytrue, test_ytrue, global_step


def sanity_check(model, state_dict_pre, linear_evaluation, head_only):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    print("=> loading state dict for sanity check")
    state_dict = model.state_dict()
    if linear_evaluation:
        for k in list(state_dict.keys()):
            # only ignore fc layer
            if 'fc.' in k or 'head.' in k or 'l1.' in k:
                continue

            equals = (state_dict[k].cpu() == state_dict_pre[k].cpu()).all()
            if (linear_evaluation != equals):
                raise Exception(
                    '=> failed sanity check in {}'.format("linear_evaluation"))
    elif head_only:
        for k in list(state_dict.keys()):
            # only ignore fc layer
            if 'fc.' in k or 'head.' in k:
                continue

            equals = (state_dict[k].cpu() == state_dict_pre[k].cpu()).all()
            if (equals and "running_mean" in k):
                raise Exception(
                    '=> failed sanity check in {}'.format("head-only"))
    # else:
    #     for k in list(state_dict.keys()):
    #         equals=(state_dict[k].cpu() == state_dict_pre[k].cpu()).all()
    #         if equals:
    #             pdb.set_trace()
    #             raise Exception('=> failed sanity check in {}'.format("fine_tuning"))

    print("=> sanity check passed.")


def eval_model(model, valid_loader, test=False, verbose=False):
    data_type = model.features[0][0].weight.type()
    model.eval()
    preds = []
    targs = []
    if not test:
        desc="Valloader"
    else:
        desc="Testloader"
    with torch.no_grad():
        for batch_idx, samples in enumerate(tqdm(valid_loader, desc=desc, disable=not verbose)):
            data = samples[0].to(device).type(data_type)
            preds_tmp = torch.sigmoid(model(data))
            targs.append(samples[1])
            preds.append(preds_tmp.cpu())
        preds = torch.cat(preds).numpy()
        targs = torch.cat(targs).numpy()
    return preds, targs


def get_dataset(batch_size, num_workers, data_path, signal_fs, train_records, validation_records, test_records, apply_noise=False, t_params=None, test=False, normalize=False):
    # when test=True, the wrapper returns train, test instead of train, validation
    if apply_noise:
        transformations = ["BaselineWander", "PowerlineNoise", "EMNoise", "BaselineShift"]
        if normalize:
            transformations.append("Normalize")
        dataset = SimCLRDataSetWrapper(batch_size, num_workers, data_path, signal_fs, train_records, validation_records, test_records,
                                            mode="linear_evaluation", transformations=transformations, t_params=t_params, test=test)
    else:
        if normalize:
            # always use PTB-XL stats
            transformations = ["Normalize"]
            dataset = SimCLRDataSetWrapper(batch_size, num_workers, data_path, signal_fs, train_records, validation_records, test_records,
                                                mode="linear_evaluation", transformations=transformations, test=test)
        else:
            # dataset = SimCLRDataSetWrapper(batch_size,num_workers,None,"(12, 250)",None,data_path,[data_path],None,None,
            #                                mode="linear_evaluation", percentage=percentage, folds=folds, test=test, ptb_xl_label="label_all")
            dataset = SimCLRDataSetWrapper(batch_size, num_workers, data_path, signal_fs, train_records, validation_records, test_records,
                                                mode="linear_evaluation", transformations=None, test=test)
 
    train_loader, valid_loader = dataset.get_data_loaders()
    return dataset, train_loader, valid_loader

if __name__ == "__main__":
    args = parse_args()
    
    # load config from checkpoint folder or fall back to default config
    run_folder = Path(args.model_file).parent.parent

    config_path = run_folder / "config.txt"
    if config_path.is_file():
        config_file = config_path
    else:
        config_file = "bolts_config.yaml"

    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('Using config ' + str(config_file))
    print(config)
    # get settings from config file
    data_path = config["dataset"]["data_path"]
    signal_fs = config["dataset"]["signal_fs"]
    train_records = config["dataset"]["train_records"]
    validation_records = config["dataset"]["val_records"]
    test_records = config["dataset"]["test_records"]
       
    # load train, validation and test datasets
    _, train_loader, valid_loader = get_dataset(args.batch_size, args.num_workers, data_path, signal_fs=signal_fs, 
                                                train_records=train_records, validation_records=validation_records, test_records=test_records,
                                                test=False, normalize=args.normalize)
    _, _, test_loader = get_dataset(args.batch_size, args.num_workers, data_path, signal_fs=signal_fs, 
                                                train_records=train_records, validation_records=validation_records, test_records=test_records,
                                                test=True, normalize=args.normalize)
    print(f"Loaded datasets - Train: {len(train_loader.dataset)}, Valid: {len(valid_loader.dataset)}, Test: {len(test_loader.dataset)}")

    # create prefix tag for saved model and results
    tag = args.tag if args.use_pretrained else "ran_" + args.tag
    tag = "eval_" + tag if args.eval_only else tag

    if args.test_noised:
        t_params_by_level = {
            1: {"bw_cmax": 0.05, "em_cmax": 0.25, "pl_cmax": 0.1, "bs_cmax": 0.5},
            2: {"bw_cmax": 0.1, "em_cmax": 0.5, "pl_cmax": 0.2, "bs_cmax": 1},
            3: {"bw_cmax": 0.1, "em_cmax": 1, "pl_cmax": 0.2, "bs_cmax": 2},
            4: {"bw_cmax": 0.2, "em_cmax": 1, "pl_cmax": 0.4, "bs_cmax": 2},
            5: {"bw_cmax": 0.2, "em_cmax": 1.5, "pl_cmax": 0.4, "bs_cmax": 2.5},
            6: {"bw_cmax": 0.3, "em_cmax": 2, "pl_cmax": 0.5, "bs_cmax": 3},
        }
        if args.noise_level not in t_params_by_level.keys():
            raise("noise level does not exist")
        t_params = t_params_by_level[args.noise_level]
        dataset, _, noise_valid_loader = get_dataset(
            args.batch_size, args.num_workers, args.data_path, apply_noise=True, t_params=t_params, test=args.test)
    else:
        noise_valid_loader = None

    losses, macros, predss, result_macros, result_macros_agg, test_macros, test_macros_agg, noised_macros, noised_macros_agg, true_labels = [
    ], [], [], [], [], [], [], [], [], []

    val_preds_fin, test_preds_fin = [], []
    m1, m2, l1, l2 = [], [], [], []
    ckpt_epoch_lin, ckpt_epoch_fin = None, None
    global_step = 0
    
    # define the logging naming convention
    linear_str = "_linear" if args.l_epochs > 0 else ""
    finetune_str = "_finetuned" if args.f_epochs > 0 else ""
    
    _, mid, run_suffix = run_folder.name.partition(args.method)
    run_name = mid + run_suffix
    run_name += linear_str
    run_name += finetune_str
    name = time.strftime("%d-%m-%Y-%H-%M") + '_' + run_name

    eval_log_folder = os.path.join(args.log_dir, name)
    print('Logging current run in:', eval_log_folder)

    # init tensorboard
    tb_writer = SummaryWriter(log_dir=eval_log_folder)
    
    # set checkpoint and results locations
    save_model_at = os.path.join(eval_log_folder, "checkpoints")
    if args.f_epochs == 0:  # only head-only training (linear evaluation)
        # os.path.dirname(args.model_file) is checkpoints folder
        results_filename = os.path.join(eval_log_folder, "result_linear.pkl")
    else:
        results_filename = os.path.join(eval_log_folder, "result_finetuned.pkl")

    # load model state dict and configure the optimizer
    model = load_model(
        num_classes=3, use_pretrained=args.use_pretrained or args.load_finetuned, hidden=args.hidden,
        location=args.model_file, method=args.method
    )
    loss_fn, optimizer = configure_optimizer(model, args.batch_size, head_only=True, discriminative_lr=args.discriminative_lr)

    # train and evaluate the model
    if not args.eval_only:
        if not isdir(save_model_at):
            os.makedirs(save_model_at)
        
        # first train only the head for l_epochs
        if args.l_epochs != 0:
            print(f"\n======================== Linear evaluation for {args.l_epochs} epochs")
            # loss_per_epoch, macro_agg_f1_per_epoch, best_macro_f1, best_macro_f1_agg, test_macro_f1, test_macro_f1_agg, best_epoch, best_val_preds, test_preds
            l1, m1, bm, bm_agg, tm, tm_agg, ckpt_epoch_lin, val_preds_lin, test_preds_lin, val_ytrue_lin, test_ytrue_lin, global_step = train_model(model, train_loader, valid_loader, test_loader, args.l_epochs, loss_fn,
                                                                                                optimizer, head_only=True, linear_evaluation=args.linear_evaluation, lr_schedule=args.lr_schedule, 
                                                                                                save_model_at=os.path.join(save_model_at, "best_val_linear.pt"), writer=tb_writer, verbose=args.verbose)
            # print best f1 macro
            print('Linear evaluation results:')
            if bm != 0:
                print(f"Best Validation macro F1 at epoch={ckpt_epoch_lin} for head-only training: {bm_agg}")
            if tm != 0:
                print(f"Test macro F1 for head-only training: {tm_agg}")
        # NOTE: linear_evaluation argument? what if head only without linear evaluation?
        
        # fine-tuning epochs
        if args.f_epochs != 0:
            print(f"\n======================== Fine-tuning for {args.f_epochs} epochs")
            # load the linear trained head if both f_epochs and l_epochs are set
            if args.l_epochs != 0:
                model = load_model(num_classes=3, use_pretrained=args.use_pretrained or args.load_finetuned, hidden=args.hidden,
                                   location=join(save_model_at, "best_val_linear.pt"), method=args.method)
                print("Loaded linear eval model from", join(save_model_at, "best_val_linear.pt"))
            # optimizer for fine-tuning
            loss_fn, optimizer = configure_optimizer(model, args.batch_size, head_only=False, discriminative_lr=args.discriminative_lr)
            # fine-tune model
            l2, m2, bm, bm_agg, tm, tm_agg, ckpt_epoch_fin, val_preds_fin, test_preds_fin, val_ytrue_fin, test_ytrue_fin,  _ = train_model(model, train_loader, valid_loader, test_loader, args.f_epochs, loss_fn,
                                                                                                optimizer, head_only=False, linear_evaluation=False, lr_schedule=args.lr_schedule, 
                                                                                                save_model_at=os.path.join(save_model_at, "best_val_finetuned.pt"), writer=tb_writer, global_step=global_step, verbose=args.verbose)
                            
            print('Fine-tuning results:')
            if bm != 0:
                print(f"Best Validation macro F1 at epoch={ckpt_epoch_fin} after fine-tuning: {bm_agg}")
            if tm != 0:
                print(f"Test macro F1 after fine-tuning: {tm_agg}")
            
        losses.append(l1+l2)
        macros.append(m1+m2)
        test_macros.append(tm)
        test_macros_agg.append(tm_agg)
        result_macros.append(bm)
        result_macros_agg.append(bm_agg)

    else:
        test_preds, eval_macro_auc, eval_macro_f1, eval_macro_auc_agg, eval_macro_f1_agg, test_ytrue = evaluate(model, test_loader)
        result_macros.append(eval_macro_f1)
        result_macros_agg.append(eval_macro_f1_agg)
        if args.verbose:
            print("macro:", eval_macro_f1)
    
    if args.l_epochs != 0:
        predss.append((val_preds_lin, test_preds_lin))
        true_labels.append((val_ytrue_lin, test_ytrue_lin))
    
    if args.f_epochs != 0:
        predss.append((val_preds_fin, test_preds_fin))
        true_labels.append((val_ytrue_fin, test_ytrue_fin))

    if noise_valid_loader is not None:
        _, _, noise_macro_f1, _, noise_macrof1__agg = evaluate(model, noise_valid_loader)
        noised_macros.append(noise_macro_f1)
        noised_macros_agg.append(noise_macrof1__agg)

    # close tensorboard writer
    tb_writer.close()
    
    res = {"filename": results_filename, "epochs": args.l_epochs+args.f_epochs, "model_location": save_model_at, 'run_name': run_name,
           "losses": losses, "macros": macros, "predss": predss, "true_labels":true_labels, "result_macros": result_macros, "test_macros": test_macros, 
           "noised_macros": noised_macros, "noised_macros_agg": noised_macros_agg, "ckpt_epoch_lin": ckpt_epoch_lin, "ckpt_epoch_fin": ckpt_epoch_fin,
           "discriminative_lr": args.discriminative_lr, "hidden": args.hidden, "lr_schedule": args.lr_schedule,
           "use_pretrained": args.use_pretrained, "linear_evaluation": args.linear_evaluation, "loaded_finetuned": args.load_finetuned,
           "eval_only": args.eval_only, "noise_level": args.noise_level, "test_noised": args.test_noised, "normalized": args.normalize}
    pickle.dump(res, open(results_filename, "wb"))
    print("dumped results to", results_filename)
    if args.verbose:
        print(res)
    print("Done!")
    
    
# example usage:
# python eval.py  --model_file "experiment_logs/07-04-2025-21-52_swav_DTW_w=3_r=5/checkpoints/best_pretrained_swav_epoch=33-val_loss=4.4183 copy.ckpt" \
#     --method "swav" --use_pretrained --batch_size 128 --l_epochs 15 --linear_evaluation
#     --logdir evaluation_logs/

# python eval.py --model_folder "./experiment_logs/07-04-2025-21-52_swav_DTW_w=3_r=5" --model_file "experiment_logs/07-04-2025-21-52_swav_DTW_w=3_r=5/checkpoints/best_pretrained_swav_epoch=33-val_loss=4.4183 copy.ckpt" --method "swav" --use_pretrained --batch_size 128 --l_epochs 15 --linear_evaluation