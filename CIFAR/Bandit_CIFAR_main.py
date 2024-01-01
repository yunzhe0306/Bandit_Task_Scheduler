import argparse
import copy
import os
import pdb
import random
import sys
from collections import OrderedDict
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

from anil import ANIL
from data_generator_new import CIFAR_100, CIFAR_Noisy_Task, CIFAR_Error_Analysis

sys.path.append(os.path.abspath(".."))
from Bandit_Adaptive_Sampler_pkg.Bandit_Adaptive_Sampler_New_dataset import Bandit_Sampler
from Bandit_Adaptive_Sampler_pkg.Bandit_Config_New_dataset import get_bandit_sampler_config
from Bandit_Adaptive_Sampler_pkg.Ensemble_Model import AveragedModelWeights


parser = argparse.ArgumentParser(description='graph transfer')
parser.add_argument('--datasource', default='cifar100', type=str,
                    help='sinusoid or omniglot or cifar100 or mixture or metadataset or metadataset_leave_one_out or multiscale')
parser.add_argument('--select_data', default=-1, type=int, help='-1,0,1,2,3')
parser.add_argument('--test_dataset', default=-1, type=int,
                    help='which dataset to be test: 0: bird, 1: texture, 2: aircraft, 3: fungi, -1 is test all')
parser.add_argument('--num_test_task', default=600, type=int, help='number of test tasks.')
parser.add_argument('--test_epoch', default=-1, type=int, help='test epoch, only work when test start')

## Training options

parser.add_argument('--update_lr', default=0.01, type=float, help='inner learning rate')
parser.add_argument('--meta_lr', default=0.001, type=float, help='the base learning rate of the generator')
parser.add_argument('--num_updates', default=5, type=int, help='num_updates in anil')
parser.add_argument('--num_updates_test', default=10, type=int, help='num_updates in anil')

parser.add_argument('--num_filters', default=32, type=int,
                    help='number of filters for conv nets -- 32 for cifar100, 64 for omiglot.')
parser.add_argument('--weight_decay', default=0.0, type=float, help='weight decay')

## Logging, saving, and testing options
parser.add_argument('--log', default=1, type=int, help='if false, do not log summaries, for debugging code.')
parser.add_argument('--logdir', default='./logs', type=str,
                    help='directory for summaries and checkpoints.')
parser.add_argument('--datadir', default='../data/', type=str, help='directory for datasets.')

parser.add_argument('--resume', default=0, type=int, help='resume training if there is a model available')

parser.add_argument('--test_set', default=1, type=int,
                    help='Set to true to test on the the test set, False for the validation set.')
parser.add_argument('--trail', default=0, type=int, help='trail for each layer')

parser.add_argument('--trace', default=0, help='whether to trace the learning curve')

parser.add_argument('--replace', default=0, type=int, help='whether to allow sampling the same task, 1 for allow')
parser.add_argument('--limit_data', default=False, type=int, help='whether to use limited data to do the experiments')
parser.add_argument('--out_learner', default='anil', type=str)
parser.add_argument('--tb_dir', default='./tensorboard', type=str)

parser.add_argument('--mix', default=False, type=int)
parser.add_argument("--grad_type", default='norm_cos', type=str)
parser.add_argument("--lambda0", default=1.0, type=float)
parser.add_argument("--limit_classes", default=16, type=int)
parser.add_argument("--debug", action="store_true", default=False)
parser.add_argument("--finetune", action='store_true', default=False)
parser.add_argument("--temperature", type=float, default=0.1)
parser.add_argument("--scheduler_lr", type=float, default=0.001)
parser.add_argument("--t_decay", type=int, default=True)
parser.add_argument("--warmup", type=int, default=0)
parser.add_argument("--pretrain_iter", type=int, default=20000)
parser.add_argument("--seed", default=1)


# ==================================================================================================

parser.add_argument('--num_classes', default=5, type=int,
                    help='number of classes used in classification (e.g. 5-way classification).')

#
parser.add_argument('--buffer_size', type=int, default=10)
parser.add_argument('--meta_batch_size', default=2, type=int, help='number of tasks sampled per meta-update')

#
parser.add_argument('--update_batch_size', default=5, type=int,
                    help='number of examples used for inner gradient update (K for K-shot learning).')
parser.add_argument('--update_batch_size_eval', default=5, type=int,
                    help='number of examples used for inner gradient test (K for K-shot learning).')

#
parser.add_argument('--metatrain_iterations', default=20000, type=int,
                    help='number of metatraining iterations.')  # 15k for omniglot, 50k for sinusoid

# ==================================================================================================
parser.add_argument('--ensemble_model_num', default=5, type=int)

parser.add_argument('--noisy_task_ratio', default=0.5, type=float, help='noisy task portion')
parser.add_argument('--noise', default=0.5, type=float, help='noise ratio')
parser.add_argument('--train', default=1, type=int, help='True to train, False to test.')
parser.add_argument("--device", default='cuda:0')
parser.add_argument('--sampling_method', default='Bandit')

parser.add_argument('--copy_noisy_task_flag', default=False, type=bool)
#
parser.add_argument('--skewed_task_distribution_flag', default=False, type=bool)

# ==================================================================================================

args = parser.parse_args()
print(args)

assert torch.cuda.is_available()
torch.backends.cudnn.benchmark = True


def set_seed(seed):
    """Sets seed"""
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


set_seed(args.seed)

exp_string = 'data_' + str(args.datasource) + '.mbs_' + str(
    args.meta_batch_size) + '.metalr' + str(args.meta_lr) + '.innerlr' + str(args.update_lr)
if args.sampling_method is not None:
    exp_string += '.Sample-{}'.format(args.sampling_method)
if args.noise > 0:
    exp_string += f'_noise_{args.noise}'
if args.sampling_method == 'Bandit':
    exp_string += str(args.device).replace(':', '-')
exp_string += str(args.metatrain_iterations)

if not os.path.exists(args.logdir + '/' + exp_string + '/'):
    os.makedirs(args.logdir + '/' + exp_string + '/')


now = datetime.now()
dt_string = now.strftime("%d-%m-%Y_%H-%M-%S") + '_{}_{}_{}_{}_{}'.format(str(args.noise), str(args.noisy_task_ratio), str(args.copy_noisy_task_flag), str(args.num_classes), str(args.skewed_task_distribution_flag))
folder_str = './logs/Bandit-{}'.format(dt_string)
os.mkdir(folder_str)

# Logger
# Recording console output
class Logger(object):
    def __init__(self, stdout):
        self.terminal = sys.stdout
        self.log = open('{}/Bandit_log_{}_'
                        .format(folder_str, dt_string) + str(args.sampling_method) + ".log", "w")
        self.out = stdout
        print("date and time =", dt_string)

    def write(self, message):
        self.log.write(message)
        self.log.flush()
        self.terminal.write(message)

    def flush(self):
        pass


sys.stdout = Logger(sys.stdout)
print(args)
print(exp_string)


def weight_norm(task_weight):
    sum_weight = torch.sum(task_weight)
    assert sum_weight > 0
    if sum_weight == 0:
        sum_weight = 1.0
    task_weight = task_weight / sum_weight

    return task_weight


def get_inner_loop_parameter_dict(params):
    """
    Returns a dictionary with the parameters to use for inner loop updates.
    :param params: A dictionary of the network's parameters.
    :return: A dictionary of the parameters to use for the inner loop optimization process.
    """
    param_dict = dict()
    indexes = []
    for i, (name, param) in enumerate(params):
        if param.requires_grad:
            if "1.weight" in name or '1.bias' in name:
                continue
            param_dict[name] = param.to(device=args.device)
            indexes.append(i)

    return param_dict, indexes


def update_moving_avg(mv, reward, count):
    return mv + (reward.item() - mv) / (count + 1)


def train(args, anil, optimiser, bandit_config_parser, run_i):
    Print_Iter = 100
    Save_Iter = 500
    print_loss, print_acc, print_loss_support = 0.0, 0.0, 0.0

    train_steps = [1]
    train_losses = []
    train_probs = []
    train_clean_probs = []
    train_noisy_probs = []
    train_clean_losses = []
    train_noisy_losses = []
    train_accs = []

    # ensemble_model
    ensemble_model = AveragedModelWeights(model=anil, device=args.device, model_num=args.ensemble_model_num)

    if args.datasource.lower() == 'cifar100':
        if args.finetune:
            if args.sampling_method == 'Bandit':
                name = 'Bandit'
            elif args.sampling_method is None:
                name = 'ANIL'
            else:
                name = args.sampling_method
            found = False
            for file in os.listdir("./models/"):
                if file.startswith(f"{name}_{args.noise}_model_{args.update_batch_size}shot"):
                    found = True
                    break
            assert found
            anil.load_state_dict(torch.load(f"./models/{file}"))
            print("Load model successfully")
            dataloader = CIFAR_100(args, 'val')

            args.sampling_method = None
            args.DAML = False
            args.FocalLoss = False

        else:
            if args.sampling_method == 'Bandit':
                # dataloader = CIFAR_MW(args, 'train')
                dataloader = CIFAR_Noisy_Task(args, mode='train')
            else:
                dataloader = CIFAR_100(args, 'train')

    # Error analysis data loader
    train_full_data_loader = CIFAR_Error_Analysis(args=args, mode='train', noisy_tasks=dataloader.noisy_tasks)
    test_full_data_loader = CIFAR_Error_Analysis(args=args, mode='test')

    if not os.path.exists(args.logdir + '/' + exp_string + '/'):
        os.makedirs(args.logdir + '/' + exp_string + '/')

    if args.sampling_method == 'ATS':
        # these are for the calculation of the dimensions
        names_weights_copy, indexes = get_inner_loop_parameter_dict(anil.learner.named_parameters())
        scheduler = Scheduler(len(names_weights_copy), args.buffer_size, grad_indexes=indexes).to(args.device)

        if args.resume != 0 and args.train:
            scheduler_file = '{0}/{2}/scheduler{1}'.format(args.logdir, args.test_epoch, exp_string)
            print(scheduler_file)
            scheduler.load_state_dict(torch.load(scheduler_file))
        scheduler_optimizer = torch.optim.Adam(scheduler.parameters(), lr=args.scheduler_lr)

    elif args.sampling_method == 'Bandit':

        # bandit_config_parser = get_bandit_sampler_config()
        #
        meta_model_param_num = 0
        for name, param in anil.learner.named_parameters():
            meta_model_param_num += param.numel()
        print("--- Meta model param size: ", meta_model_param_num)
        print("--- Pooling meta size: ", int(meta_model_param_num // bandit_config_parser.g_pool_step_meta) + 1)
        bandit_scheduler = Bandit_Sampler(input_dim=meta_model_param_num, hidden_size=bandit_config_parser.hidden_size,
                                          alpha=bandit_config_parser.alpha,
                                          g_pool_step_meta=bandit_config_parser.g_pool_step_meta,
                                          g_pool_step_bandit=bandit_config_parser.g_pool_step_bandit,
                                          buffer_size=bandit_config_parser.buffer_size,
                                          lr_rate=bandit_config_parser.lr_rate,
                                          artificial_exp_score=bandit_config_parser.artificial_exp_score,
                                          device=args.device)

    if args.t_decay:
        T = 1
    else:
        T = 0
    moving_avg_reward = 0
    lambda0 = args.lambda0

    if args.resume != 0 and args.train:
        T *= 0.999 ** args.test_epoch

    state_dict = copy.deepcopy(anil.state_dict())

    warmup = args.warmup

    ######
    if args.noise > 0:
        np.save(folder_str + '/noisy_tasks_Algo_{}_Run_{}.npy'
                .format(args.sampling_method, str(run_i)), dataloader.noisy_tasks)
    selected_classes_by_algo = []

    final_acc = None
    cumulative_time = 0.0
    for step, data in enumerate(dataloader):

        if step > args.metatrain_iterations:
            break

        task_losses = []
        task_losses_val = []
        task_acc = []
        task_accs_val = []

        if args.sampling_method is None:
            if args.datasource == 'cifar100' and not args.finetune:
                x_spt, y_spt, x_qry, y_qry, noisy_or_not, task_classes = data
            else:
                x_spt, y_spt, x_qry, y_qry, task_classes = data
                noisy_or_not = None
            x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).cuda(), y_spt.squeeze(0).cuda(), \
                                         x_qry.squeeze(0).cuda(), y_qry.squeeze(0).cuda()
            for meta_batch in range(args.meta_batch_size):
                meta_batch_result = anil(x_spt[meta_batch], y_spt[meta_batch], x_qry[meta_batch],
                                         y_qry[meta_batch])

                loss_val, acc_val = meta_batch_result

                task_losses.append(loss_val)
                task_acc.append(acc_val)

            meta_batch_loss_raw = torch.stack(task_losses)
            meta_batch_loss = meta_batch_loss_raw.mean()

            meta_batch_acc = torch.stack(task_acc).mean()

            optimiser.zero_grad()
            meta_batch_loss.backward()
            optimiser.step()

            train_steps.append(train_steps[-1] + args.meta_batch_size)
            train_losses.append(meta_batch_loss.item())
            train_accs.append(meta_batch_acc.item())

            if noisy_or_not is not None:
                clean_idx = torch.where(noisy_or_not == 0)
                noisy_idx = torch.where(noisy_or_not > 0)
                if len(clean_idx[0]) > 0:
                    train_clean_losses.append(meta_batch_loss_raw[clean_idx].mean().item())
                else:
                    train_clean_losses.append(-1)

                if len(noisy_idx[0]) > 0:
                    train_noisy_losses.append(meta_batch_loss_raw[noisy_idx].mean().item())
                else:
                    train_noisy_losses.append(-1)

        elif args.sampling_method == 'ATS':
            (x_spt, y_spt, x_qry, y_qry, x_spt_val, y_spt_val, x_qry_val, y_qry_val, noisy_or_not, task_classes) = data

            scheduler.tasks = [[x1, y1, x2, y2] for (x1, y1, x2, y2) in zip(x_spt, y_spt, x_qry, y_qry)]

            names_weights_copy, _ = get_inner_loop_parameter_dict(anil.learner.named_parameters())

            pt = int(step / (args.metatrain_iterations + 1) * 100)

            meta_batch_loss_raw, _, all_task_weight = scheduler.get_weight(anil, pt)
            torch.cuda.empty_cache()
            all_task_prob = torch.softmax(all_task_weight.reshape(-1), dim=-1)

            selected_tasks_idx = scheduler.sample_task(all_task_prob, args.meta_batch_size, replace=args.replace)

            selected_losses = scheduler.compute_loss(selected_tasks_idx, anil)

            meta_batch_loss = torch.mean(selected_losses)

            # Prepare for second time forward
            fast_weights = OrderedDict(anil.learner.named_parameters())
            gradients = torch.autograd.grad(meta_batch_loss, fast_weights.values(), create_graph=True)

            fast_weights = OrderedDict(
                (name, param - anil.args.update_lr * grad)
                for ((name, param), grad) in zip(fast_weights.items(), gradients)
            )

            # Second time forward, on validation dataset compute the gradient of scheduler to update it.
            x_spt_val, y_spt_val, x_qry_val, y_qry_val = x_spt_val.cuda(), y_spt_val.cuda(), \
                                                         x_qry_val.cuda(), y_qry_val.cuda()

            for meta_batch in range(args.meta_batch_size):
                x1, y1, x2, y2 = x_spt_val[meta_batch], y_spt_val[meta_batch], x_qry_val[meta_batch], y_qry_val[
                    meta_batch]
                x1, y1, x2, y2 = x1.squeeze(0).cuda(), y1.squeeze(0).cuda(), \
                                 x2.squeeze(0).cuda(), y2.squeeze(0).cuda()
                loss_train_val, acc_train_val = anil.forward_val(x1, y1, x2, y2, fast_weights)

                task_losses_val.append(loss_train_val.detach())
                task_accs_val.append(acc_train_val)

            loss = 0
            for i in selected_tasks_idx:
                loss = loss - scheduler.m.log_prob(i)
            reward = torch.stack(task_accs_val).mean()
            loss *= (reward - moving_avg_reward)

            moving_avg_reward = update_moving_avg(moving_avg_reward, reward, step)

            scheduler_optimizer.zero_grad()
            loss.backward()
            scheduler_optimizer.step()

            meta_batch_acc = 0
            task_losses = []
            task_acc = []

            # Third time forward, on training dataset, update the learner
            if step < warmup:
                optimiser.zero_grad()
                meta_batch_loss.backward()
                optimiser.step()
                anil.load_state_dict(state_dict)
            else:
                task_losses = []
                task_acc = []
                T *= 0.99
                meta_batch_loss_raw, _, all_task_weight = scheduler.get_weight(anil, pt, detach=True)
                all_task_prob = torch.softmax(all_task_weight.reshape(-1) / max(args.temperature, T), dim=-1)
                selected_tasks_idx = scheduler.sample_task(all_task_prob, args.meta_batch_size, replace=args.replace)
                selected_tasks_idx = torch.stack(selected_tasks_idx)
                selected_tasks_idx_unique, count = torch.unique(selected_tasks_idx, return_counts=True)

                for i, idx in enumerate(selected_tasks_idx_unique):
                    x1, y1, x2, y2 = scheduler.tasks[idx]
                    x1, y1, x2, y2 = x1.squeeze(0).cuda(), y1.squeeze(0).cuda(), \
                                     x2.squeeze(0).cuda(), y2.squeeze(0).cuda()
                    loss_train_train, acc_train_train = anil(x1, y1, x2, y2)

                    task_losses.extend([loss_train_train] * count[i])
                    task_acc.extend([acc_train_train] * count[i])

                meta_batch_loss = torch.stack(task_losses).mean()
                meta_batch_acc = torch.stack(task_acc).mean()

                optimiser.zero_grad()
                meta_batch_loss.backward()
                optimiser.step()

                train_steps.append(train_steps[-1] + 1)
                train_losses.append(meta_batch_loss_raw.data.tolist())
                train_probs.append(all_task_prob.data.tolist())
                # train_losses.append(meta_batch_loss.item())
                # train_accs.append(meta_batch_acc.item())
                clean_idx = torch.where(noisy_or_not == 0)
                noisy_idx = torch.where(noisy_or_not > 0)
                if len(clean_idx[0]) > 0:
                    train_clean_probs.append(all_task_prob[clean_idx].mean().item())
                    train_clean_losses.append(meta_batch_loss_raw[clean_idx].mean().item())
                else:
                    train_clean_probs.append(-1)
                    train_clean_losses.append(-1)

                if len(noisy_idx[0]) > 0:
                    train_noisy_probs.append(all_task_prob[noisy_idx].mean().item())
                    train_noisy_losses.append(meta_batch_loss_raw[noisy_idx].mean().item())
                else:
                    train_noisy_probs.append(-1)
                    train_noisy_losses.append(-1)

        # =============================================================================================
        elif args.sampling_method == 'Bandit':
            start_time = time.time()
            (x_spt, y_spt, x_qry, y_qry, x_spt_val, y_spt_val, x_qry_val, y_qry_val, noisy_or_not, task_classes) = data

            # Calclate task probs
            new_tasks = [[x1, y1, x2, y2] for (x1, y1, x2, y2) in zip(x_spt, y_spt, x_qry, y_qry)]
            new_tasks_num = len(new_tasks)
            reward_ests, explore_score_ests = \
                bandit_scheduler.get_reward_est_and_exp_score(meta_model=anil, tasks=new_tasks)
            
            #
            if bandit_config_parser.alpha > 0:
                raw_scores = (bandit_config_parser.alpha * reward_ests) + explore_score_ests
                overall_score = F.softmax(raw_scores, dim=0)
            else:
                raw_scores = reward_ests
                overall_score = F.softmax(reward_ests, dim=0)

            # Make recommendations - direct
            rec_indices = list(torch.topk(raw_scores.reshape(-1, ).flatten(), args.meta_batch_size).indices)
            rec_tasks = [new_tasks[r_i] for r_i in rec_indices]

            # Calculate meta-loss w.r.t. selected tasks
            meta_loss_selected, meta_batch_acc_selected = \
                bandit_scheduler.calculate_meta_loss_with_tasks(tasks=rec_tasks, meta_model=anil)
            meta_batch_loss = meta_loss_selected.mean()
            meta_batch_acc = meta_batch_acc_selected.mean()

            # ==========================================================================================
            # Derive the model parameters after inner-loop optimization with selected tasks
            meta_parameters = OrderedDict(anil.learner.named_parameters())

            # ------------------------------------------------------------------------------------
            for t_id in rec_indices:
                x1, y1, x2, y2 = new_tasks[t_id]
                x1, y1, x2, y2 = x1.squeeze(0).float().to(args.device), y1.squeeze(0).long().to(args.device), \
                                 x2.squeeze(0).float().to(args.device), y2.squeeze(0).long().to(args.device)
                loss_train_val, acc = anil.forward_val(x1, y1, x2, y2, meta_parameters)
                task_losses_val.append(loss_train_val)

            # ==========================================================================================
            # Update the meta-model using the selected tasks
            optimiser.zero_grad()
            meta_batch_loss.backward(retain_graph=False)
            optimiser.step()

            # ==========================================================================================
            # Calculate true rewards and exploration scores for updating the bandit sampler
            rec_indix_vec = torch.stack(rec_indices)
            selected_tasks_idx_np = torch.tensor(rec_indix_vec).cpu().numpy()
            selected_classes_by_algo.append(task_classes[selected_tasks_idx_np.reshape(-1, ), :])
            selected_overall_score = overall_score[rec_indix_vec, :]
            selected_reward_ests = reward_ests[rec_indix_vec, :]
            val_loss = torch.stack(task_losses_val).reshape(-1, 1)

            # True rewards
            true_rewards, raw_loss = bandit_scheduler.cal_true_reward_sampling(anil, new_tasks, rec_indices, new_tasks_num,
                                                                    selected_overall_score, validation_task_num=5)
            alpha_coef = bandit_scheduler.alpha

            # True exploration score
            true_exploration_scores = \
                alpha_coef * (true_rewards - selected_reward_ests) + (1 - alpha_coef) * val_loss

            # Update information and update bandit sampler
            bandit_scheduler.update_data(rewards=true_rewards, explore_scores=true_exploration_scores,
                                         selected_task_indices=rec_indices, candidate_task_num=new_tasks_num)

            # Train the bandit sampler per 10 / 100 iterations
            if step % bandit_config_parser.train_per_step == 0:
                bandit_scheduler.train_model(step=step)

            #
            cumulative_time += (time.time() - start_time)
            if step != 0 and step % 500 == 0:
                print("Step: {}, Time elapsed: {}".format(step, cumulative_time))

        else:
            print(f"Samplling method {args.sampling_method} not recognized.")
            raise NotImplementedError

        # Error analysis data ----------------
        if step != 0 and step % 500 == 0:
            run_error_analysis(args, anil, step, train_full_data_loader, run_i, mode='train')
            run_error_analysis(args, anil, test_epoch=step, error_analysis_data_loader=test_full_data_loader,
                               run_i=run_i, mode='test')

        if step != 0 and step % Print_Iter == 0:
            if args.sampling_method == 'ATS':
                print(
                    'iter: {}, loss_all: {}, acc: {}, probs:{}'.format(
                        step, print_loss, print_acc, all_task_prob.data.tolist()[:3]))

            else:
                print('iter: {}, loss_all: {}, acc: {}'.format(
                    step, print_loss, print_acc))

            sys.stdout.flush()

            print_loss, print_acc, SPL_task_num = 0.0, 0.0, 0
        else:
            print_loss += meta_batch_loss / Print_Iter
            print_acc += meta_batch_acc / Print_Iter

        if step != 0 and step % Save_Iter == 0:
            test_epoch, mean_acc = test(args, anil, step)
            # ensemble_model.update_parameters(anil, mean_loss=(1 - mean_acc))
            # en_model = ensemble_model.get_avg_model()
            # _, __ = test(args, en_model, step, ensemble_test_flag=True)

            final_acc = mean_acc
            torch.save(anil.state_dict(),
                       '{0}/{2}/model{1}'.format(args.logdir, step, exp_string))
            if args.sampling_method == 'scheduler' or args.sampling_method == 'ALFA':
                torch.save(scheduler.state_dict(), '{0}/{2}/scheduler{1}'.format(args.logdir, step, exp_string))
    #
    np.save(folder_str + '/chosen_classes_Algo_{}_Run_{}.npy'.format(args.sampling_method, str(run_i)),
            np.array(selected_classes_by_algo))
    run_error_analysis(args, anil, test_epoch=99999, error_analysis_data_loader=test_full_data_loader,
                       run_i=run_i, mode='test')

    #
    return final_acc


def test(args, anil, test_epoch, ensemble_test_flag=False):
    res_acc = []
    meta_batch_size = args.meta_batch_size
    args.meta_batch_size = 1

    # dataloader = CIFAR_MW(args, 'test')
    dataloader = CIFAR_Noisy_Task(args, mode='test')

    for step, (x_spt, y_spt, x_qry, y_qry, task_classes) in enumerate(dataloader):
        if step > 600:
            break
        x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(args.device), y_spt.squeeze(0).to(args.device), \
                                     x_qry.squeeze(0).to(args.device), y_qry.squeeze(0).to(args.device)
        acc_val = anil(x_spt, y_spt, x_qry, y_qry)[1]
        res_acc.append(acc_val.item())

    res_acc = np.array(res_acc)
    mean_acc = np.mean(res_acc)

    if ensemble_test_flag:
        print('Ensemble test_epoch is {}, acc is {}, ci95 is {}'.format(test_epoch, np.mean(res_acc),
                                                                        1.96 * np.std(res_acc) / np.sqrt(
                                                                            args.num_test_task * args.meta_batch_size)))
    else:
        print('test_epoch is {}, acc is {}, ci95 is {}'.format(test_epoch, np.mean(res_acc),
                                                               1.96 * np.std(res_acc) / np.sqrt(
                                                                   args.num_test_task * args.meta_batch_size)))
    args.meta_batch_size = meta_batch_size

    return test_epoch, mean_acc


def run_error_analysis(args, anil, test_epoch, error_analysis_data_loader, run_i, mode='train'):

    error_analysis_data_loader.target_class_counter = 0
    # 64 tasks for training, 20 tasks for testing
    if mode == 'train':
        class_num = error_analysis_data_loader.data.shape[0]
        task_acc_dict = {t_i: [] for t_i in range(class_num)}
        for step in range(128):
            data_step = error_analysis_data_loader.get_sampled_data(error_analysis_data_loader.data, setting='train')
            x_spt, y_spt, x_qry, y_qry, noisy_or_not, task_classes = data_step
            batch_task_num = x_spt.shape[0]
            x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(args.device), y_spt.squeeze(0).to(args.device), \
                                         x_qry.squeeze(0).to(args.device), y_qry.squeeze(0).to(args.device)
            for meta_batch in range(batch_task_num):
                loss_val, acc_val = anil(x_spt[meta_batch], y_spt[meta_batch], x_qry[meta_batch], y_qry[meta_batch],
                                         refined_acc=True)
                this_chosen_classes = task_classes[meta_batch, :]
                for class_i, this_class in enumerate(this_chosen_classes):
                    class_mean_val = \
                        acc_val[class_i * args.update_batch_size: (class_i+1) * args.update_batch_size].mean()
                    task_acc_dict[this_class].append(class_mean_val.item())
    else:
        test_loss_val_list = []
        class_num = 20
        task_acc_dict = {t_i: [] for t_i in range(20)}
        for step in range(128):
            data_step = error_analysis_data_loader.get_sampled_data(error_analysis_data_loader.data, setting='test')
            x_spt, y_spt, x_qry, y_qry, task_classes = data_step
            batch_task_num = x_spt.shape[0]
            x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(args.device), y_spt.squeeze(0).to(args.device), \
                                         x_qry.squeeze(0).to(args.device), y_qry.squeeze(0).to(args.device)
            for meta_batch in range(batch_task_num):
                loss_val, acc_val = anil(x_spt[meta_batch], y_spt[meta_batch], x_qry[meta_batch], y_qry[meta_batch],
                                         refined_acc=True)
                test_loss_val_list.append(loss_val.detach().cpu().item())
                this_chosen_classes = task_classes[meta_batch, :]
                for class_i, this_class in enumerate(this_chosen_classes):
                    class_mean_val = \
                        acc_val[class_i * args.update_batch_size: (class_i + 1) * args.update_batch_size].mean()
                    task_acc_dict[this_class].append(class_mean_val.item())
        #
        mean_val_loss = np.mean(np.array(test_loss_val_list))
        print("--- Iteration: {}, Testing Avg Loss: {}".format(str(test_epoch), str(mean_val_loss)))

    #
    task_dist_for_acc = np.zeros([class_num, ])
    for i in range(class_num):
        task_dist_for_acc[i] = np.mean(task_acc_dict[i])
    if mode == 'train':
        np.save(folder_str + '/class_acc_dist_step_{}_Algo_{}_Run_{}.npy'.format(test_epoch, args.sampling_method,
                                                                                 str(run_i)),
                task_dist_for_acc)
    else:
        np.save(folder_str + '/class_acc_dist_step_{}_{}_Algo_{}_Run_{}.npy'.format('TEST', test_epoch,
                                                                                    args.sampling_method,
                                                                                    str(run_i)),
                task_dist_for_acc)




def main():
    # for cifar100
    final_layer_size = 128
    stride = 1

    best_acc = -1
    best_config = None


    exp_settings_list = [
        {'shot': 5, 'noise': args.noise}
    ]


    bandit_config_parser = get_bandit_sampler_config()
    args.update_batch_size = exp_settings_list[0]['shot']
    args.update_batch_size_eval = exp_settings_list[0]['shot']
    args.noise = exp_settings_list[0]['noise']
    args.skew_pattern = 1
    print(args)
    
    # --------------------------------
    if args.out_learner == 'anil':
        anil = ANIL(args, final_layer_size=final_layer_size, stride=stride).to(args.device)
        if args.resume != 0 and args.train == 1:
            model_file = '{0}/{2}/model{1}'.format(args.logdir, args.test_epoch, exp_string)
            print(model_file)
            anil.load_state_dict(torch.load(model_file))

        meta_optimiser = torch.optim.Adam(list(anil.parameters()),
                                          lr=args.meta_lr, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError

    if args.train == 1:
        with torch.backends.cudnn.flags(enabled=False):
            run_encoding = "00"
            final_acc = train(args, anil, meta_optimiser, bandit_config_parser=bandit_config_parser, run_i=run_encoding)
            if final_acc > best_acc:
                best_config = copy.deepcopy(bandit_config_parser)
                best_acc = final_acc


if __name__ == '__main__':
    main()
