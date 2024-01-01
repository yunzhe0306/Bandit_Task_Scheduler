import argparse
import os
import sys
import time
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import random
from scipy.stats import pearsonr

sys.path.append(os.path.abspath(".."))
from Bandit_Adaptive_Sampler_pkg.Bandit_Adaptive_Sampler_Drug import Bandit_Sampler
from Bandit_Adaptive_Sampler_pkg.Bandit_Config_Drug import get_bandit_sampler_config
from anil import ANIL
from data import MetaLearningSystemDataLoader
from learner import FCNet


parser = argparse.ArgumentParser(description='graph transfer')
parser.add_argument('--datasource', default='drug', type=str,
                    help='drug')
parser.add_argument('--dim_w', default=1024, type=int, help='dimension of w')
parser.add_argument('--dim_y', default=1, type=int, help='dimension of w')
parser.add_argument('--dataset_name', default='assay', type=str,
                    help='dataset_name.')
parser.add_argument('--dataset_path', default='ci9b00375_si_002.txt', type=str,
                    help='dataset_path.')
parser.add_argument('--type_filename', default='ci9b00375_si_001.txt', type=str,
                    help='type_filename.')
parser.add_argument('--compound_filename', default='ci9b00375_si_003.txt', type=str,
                    help='Directory of data files.')

parser.add_argument('--fp_filename', default='compound_fp.npy', type=str,
                    help='fp_filename.')

parser.add_argument('--target_assay_list', default='591252', type=str,
                    help='target_assay_list')

parser.add_argument('--train_seed', default=0, type=int, help='train_seed')

parser.add_argument('--val_seed', default=0, type=int, help='val_seed')

parser.add_argument('--test_seed', default=0, type=int, help='test_seed')

parser.add_argument('--train_val_split', default=[0.9588, 0.0177736202, 0.023386342], type=list, help='train_val_split')
parser.add_argument('--num_evaluation_tasks', default=100, type=int, help='num_evaluation_tasks')
parser.add_argument('--drug_group', default=17, type=int, help='drug group')

parser.add_argument('--update_lr', default=0.01, type=float, help='inner learning rate')
parser.add_argument('--meta_lr', default=0.001, type=float, help='the base learning rate of the generator')
parser.add_argument('--num_updates', default=5, type=int, help='num_updates in anil')
parser.add_argument('--test_num_updates', default=5, type=int, help='num_updates in anil')
parser.add_argument('--weight_decay', default=0.0, type=float, help='weight decay')

parser.add_argument('--logdir', default='./logs', type=str,
                    help='directory for summaries and checkpoints.')
parser.add_argument('--data_dir', default='../data/drug', type=str, help='directory for datasets.')
parser.add_argument('--resume', default=0, type=int, help='resume training if there is a model available')
parser.add_argument('--test_epoch', default=-1, type=int, help='test epoch, only work when test start')
parser.add_argument('--trial', default=0, type=int, help='trial for each layer')
parser.add_argument('--debug', action='store_true', default=False, help='debug mode')

parser.add_argument("--gamma", default=0.25, type=float)
parser.add_argument("--seed", default=1, type=int)
parser.add_argument('--simple_loss', default=False, action='store_true')


# ======================
parser.add_argument('--buffer_size', type=int, default=10)
parser.add_argument('--meta_batch_size', default=2, type=int, help='number of tasks sampled per meta-update')

parser.add_argument("--noise", default=0.5, type=float)

parser.add_argument('--update_batch_size', default=5, type=int,
                    help='number of examples used for inner gradient update (K for K-shot learning).')
parser.add_argument('--update_batch_size_eval', default=5, type=int,
                    help='number of examples used for inner gradient test (K for K-shot learning).')

parser.add_argument('--train', default=1, type=int, help='True to train, False to test.')
parser.add_argument('--metatrain_iterations', default=10, type=int,
                    help='number of metatraining iterations.')  # 15k for omniglot, 50k for sinusoid
# Logging, saving, and testing options
parser.add_argument('--sampling_method', default='Bandit', help='specified sampling method')
parser.add_argument("--device", default='cuda:0')
# ======================



args = parser.parse_args()


def set_seed(seed):
    """Sets seed"""
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


set_seed(args.seed)

exp_string = 'data_' + str(args.datasource) + '.mbs_' + str(
    args.meta_batch_size) + '.metalr' + str(args.meta_lr) + '.innerlr' + str(args.update_lr)
if args.trial > 0:
    exp_string += '.trial{}'.format(args.trial)
if args.sampling_method is not None:
    exp_string += '.Sample-{}'.format(args.sampling_method)
if args.noise > 0:
    exp_string += f'_noise_{args.noise}'
if args.simple_loss:
    exp_string += '_simple_loss'
if args.sampling_method == 'Bandit':
    exp_string += str(args.device).replace(':', '-')
exp_string += str(args.metatrain_iterations)

exp_string += '.drug_group-{}'.format(args.drug_group)

if not os.path.exists(args.logdir + '/' + exp_string + '/'):
    os.makedirs(args.logdir + '/' + exp_string + '/')

args.data_dir = args.data_dir + '/' + args.datasource

# Logger
# Recording console output
class Logger(object):
    def __init__(self, stdout):
        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
        self.terminal = sys.stdout
        self.log = open('{}/text_logs/Main_log_{}_'.format(args.logdir, dt_string) + str(args.sampling_method) + ".log", "w")
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
        if "1.weight" in name or '1.bias' in name:
            continue
        if param.requires_grad:
            param_dict[name] = param.to(device=args.device)
            indexes.append(i)

    return param_dict, indexes


def update_moving_avg(mv, reward, count):
    return mv + (reward.item() - mv) / (count + 1)


def train(args, anil, optimiser, dataloader, bandit_config_parser):
    Print_Iter = 100
    Save_Iter = 500
    print_loss, print_acc, print_loss_support = 0.0, 0.0, 0.0

    train_steps = [1]
    train_losses = []
    train_probs = []

    data_each_epoch = 4100
    moving_avg_reward = 0

    if args.debug:
        print("Training start:")
        sys.stdout.flush()

    bandit_scheduler = None
    if args.sampling_method == 'Bandit':
        #
        meta_model_param_num = 0
        for name, param in anil.learner.named_parameters():
            meta_model_param_num += param.numel()
        bandit_scheduler = Bandit_Sampler(input_dim=meta_model_param_num, hidden_size=bandit_config_parser.hidden_size,
                                          alpha=bandit_config_parser.alpha,
                                          g_pool_step_meta=bandit_config_parser.g_pool_step_meta,
                                          g_pool_step_bandit=bandit_config_parser.g_pool_step_bandit,
                                          buffer_size=bandit_config_parser.buffer_size,
                                          lr_rate=bandit_config_parser.lr_rate,
                                          artificial_exp_score=bandit_config_parser.artificial_exp_score,
                                          device=args.device)

    count = 0
    time_init = time.time()

    for epoch in range(args.metatrain_iterations):

        if args.sampling_method == 'Bandit':
            train_data_all = dataloader.get_train_val_batches()
        else:
            train_data_all = dataloader.get_train_batches()

        print(" ----------- Finish reading dataset")

        for step, cur_data in enumerate(train_data_all):

            task_losses = []
            task_losses_val = []

            if args.sampling_method is None:
                support_set_x, support_set_y, support_set_z, support_set_assay, \
                target_set_x, target_set_y, target_set_z, target_set_assay, seed = cur_data

                for meta_batch in range(args.meta_batch_size):
                    x1, y1, x2, y2 = support_set_x[meta_batch].squeeze().float().to(args.device), support_set_y[
                        meta_batch].squeeze().float().to(args.device), \
                                     target_set_x[meta_batch].squeeze().float().to(args.device), target_set_y[
                                         meta_batch].squeeze().float().to(args.device)
                    loss_val = anil(x1, y1, x2, y2)
                    task_losses.append(loss_val)

                meta_batch_loss_raw = torch.stack(task_losses)
                meta_batch_loss = meta_batch_loss_raw.mean()

                optimiser.zero_grad()
                meta_batch_loss.backward()
                optimiser.step()

            # =====================================================================================
            elif args.sampling_method == 'Bandit':

                # Get data
                support_set_x, support_set_y, support_set_z, support_set_assay, \
                target_set_x, target_set_y, target_set_z, target_set_assay, seed, \
                support_set_x_val, support_set_y_val, support_set_z_val, support_set_assay_val, \
                target_set_x_val, target_set_y_val, target_set_z_val, target_set_assay_val, seed_val = cur_data

                # Calclate task probs
                new_tasks = [[x1, y1, x2, y2] for (x1, y1, x2, y2) in
                             zip(support_set_x, support_set_y, target_set_x, target_set_y)]
                new_tasks_num = len(new_tasks)
                # 
                reward_ests, explore_score_ests = \
                    bandit_scheduler.get_reward_est_and_exp_score(meta_model=anil, tasks=new_tasks)
                #
                if bandit_config_parser.alpha > 0:
                    raw_scores = (bandit_config_parser.alpha * reward_ests) + explore_score_ests
                    overall_score = F.softmax(raw_scores, dim=0)
                else:
                    raw_scores = reward_ests
                    overall_score = F.softmax(reward_ests, dim=0)

                # Make recommendations
                rec_indices = \
                    bandit_scheduler.sample_arms_given_probs(probs=overall_score.reshape(-1, ),
                                                             batch_num=args.meta_batch_size)
                rec_tasks = [new_tasks[r_i] for r_i in rec_indices]

                # Calculate meta-loss w.r.t. selected tasks
                # 
                meta_loss_selected = bandit_scheduler.calculate_meta_loss_with_tasks(tasks=rec_tasks, meta_model=anil)
                # 
                meta_batch_loss = meta_loss_selected.mean()

                # ==========================================================================================
                # Derive the model parameters after inner-loop optimization with selected tasks
                meta_parameters = OrderedDict(anil.learner.named_parameters())
                meta_gradients = torch.autograd.grad(meta_batch_loss, meta_parameters.values(), create_graph=True)

                meta_gradients_dict = OrderedDict(
                    (name, param - anil.args.update_lr * grad)
                    for ((name, param), grad) in zip(meta_parameters.items(), meta_gradients)
                )

                # ------------------------------------------------------------------------------------
                for t_id in rec_indices:
                    x1, y1, x2, y2 = new_tasks[t_id]
                    x1, y1, x2, y2 = x1.squeeze(0).float().to(args.device), y1.squeeze(0).float().to(args.device), \
                                     x2.squeeze(0).float().to(args.device), y2.squeeze(0).float().to(args.device)
                    loss_train_val = anil.forward_val(x1, y1, x2, y2, meta_gradients_dict)
                    task_losses_val.append(loss_train_val)

                # ==========================================================================================
                # Update the meta-model using the selected tasks
                optimiser.zero_grad()
                meta_batch_loss.backward(retain_graph=False)
                optimiser.step()

                # ==========================================================================================
                # Calculate true rewards and exploration scores for updating the bandit sampler
                rec_indix_vec = torch.stack(rec_indices)
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

            # =====================================================================================
            else:
                print(f"Sampling method {args.sampling_method} not recognized.")
                raise NotImplementedError

            if step != 0 and step % Print_Iter == 0:
                print('epoch: {}, iter: {}, mse: {}'.format(epoch, step, print_loss))
                print("- Time elapsed: ", time.time() - time_init)
                sys.stdout.flush()
                print_loss = 0.0
            else:
                print_loss += meta_batch_loss / Print_Iter

            if step != 0 and step % Save_Iter == 0:
                torch.save(anil.learner.state_dict(),
                           '{0}/{2}/model{1}'.format(args.logdir, step + epoch * data_each_epoch, exp_string))
                if args.sampling_method == 'Bandit':
                    torch.save(bandit_scheduler.state_dict(),
                               '{0}/{2}/schedulernet{1}_bandit'.format(args.logdir, step + epoch * data_each_epoch,
                                                                       exp_string))

            #
            count += 1

        # Validation per epoch -----------------------------------------------------
        anil.args.train = False
        t_acc = test(args, epoch, anil, dataloader)
        anil.args.train = True
        print('\n\n ---------- epoch is: {} mean validation acc. is: {} ---------- \n\n'.format(epoch, t_acc))


def test(args, epoch, anil, dataloader):
    res_acc = []

    valid_cnt = 0

    test_data_all = dataloader.get_test_batches()

    for step, cur_data in enumerate(test_data_all):
        support_set_x, support_set_y, support_set_z, support_set_assay, \
        target_set_x, target_set_y, target_set_z, target_set_assay, seed = cur_data

        mse_loss, pred_label, actual_label = anil(support_set_x[0].squeeze().to(args.device),
                                                  support_set_y[0].squeeze().to(args.device),
                                                  target_set_x[0].squeeze().to(args.device),
                                                  target_set_y[0].squeeze().to(args.device))

        r2 = np.square(pearsonr(actual_label.cpu().numpy(), pred_label.detach().cpu().numpy())[0])
        res_acc.append(r2)

        if r2 > 0.3:
            valid_cnt += 1

    res_acc = np.array(res_acc)
    median = np.median(res_acc, 0)
    mean = np.mean(res_acc, 0)

    print('Test_acc --- epoch is: {} mean is: {}, median is: {}, cnt>0.3 is: {}'.format(epoch, mean, median, valid_cnt))
    return mean


def main():
    bandit_config_parser = get_bandit_sampler_config()

    # ----------------------------------------
    exp_settings_list = [
        {'shot': 5, 'noise': 0.5}
    ]

    args.update_batch_size = exp_settings_list[0]['shot']
    args.update_batch_size_eval = exp_settings_list[0]['shot']
    args.noise = exp_settings_list[0]['noise']
    print(args)

    # -----------------------------------
    learner = FCNet(args=args, x_dim=args.dim_w, hid_dim=500).to(args.device)

    anil = ANIL(args, learner)

    if args.resume == 1 and args.train == 1:
        model_file = '{0}/{2}/model{1}'.format(args.logdir, args.test_epoch, exp_string)
        print("model_file:", model_file)
        learner.load_state_dict(torch.load(model_file))

    meta_optimiser = torch.optim.Adam(list(learner.parameters()),
                                      lr=args.meta_lr, weight_decay=args.weight_decay)

    dataloader = MetaLearningSystemDataLoader(args, target_assay=args.target_assay_list)
    mean = []
    if args.train == 1:
        with torch.backends.cudnn.flags(enabled=False):
            train(args, anil, meta_optimiser, dataloader, bandit_config_parser)
    else:
        args.meta_batch_size = 1
        epoch_num = args.metatrain_iterations
        print("--- Testing epoch num: ", epoch_num)
        if args.sampling_method == 'Bandit':
            save_mean_file = '{0}/{2}/model{1}'.format(args.logdir, 'MEAN_VAL_', exp_string)
            for epoch in range(500, 4000 * epoch_num, 100):
                model_file = '{0}/{2}/model{1}'.format(args.logdir, epoch, exp_string)
                if os.path.exists(model_file):
                    learner.load_state_dict(torch.load(model_file))
                    mean.append(test(args, epoch, anil, dataloader))
                else:
                    continue
            np.save(save_mean_file + '.npy', np.array(mean))
        else:
            save_mean_file = '{0}/{2}/model{1}'.format(args.logdir, 'MEAN_VAL_', exp_string)
            for epoch in range(500, 4000 * epoch_num, 100):
                model_file = '{0}/{2}/model{1}'.format(args.logdir, epoch, exp_string)
                if os.path.exists(model_file):
                    learner.load_state_dict(torch.load(model_file))
                    mean.append(test(args, epoch, anil, dataloader))
                else:
                    continue
            np.save(save_mean_file + '.npy', np.array(mean))


if __name__ == '__main__':
    main()
