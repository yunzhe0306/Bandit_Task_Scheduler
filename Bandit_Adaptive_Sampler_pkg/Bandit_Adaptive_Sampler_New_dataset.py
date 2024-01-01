from collections import OrderedDict
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.distributions import Categorical
import torch.optim as optim


class FC_Net(nn.Module):
    def __init__(self, input_dim, hidden_size, buffer_size=20000, lr_rate=0.0001, embedding_layer_hidden_size=-1, device=None):
        super(FC_Net, self).__init__()
        self.hidden_size = hidden_size
        self.embed_dim = input_dim
        self.lr_rate = lr_rate
        self.buffer_size = buffer_size
        self.device = device

        if embedding_layer_hidden_size <= 0:
            self.fc1 = nn.Linear(input_dim, hidden_size)
            self.act = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, 1)
        else:
            layer_list = [nn.Linear(input_dim, embedding_layer_hidden_size), nn.ReLU(), nn.Linear(embedding_layer_hidden_size, hidden_size)]
            self.fc1 = nn.Sequential(*layer_list)
            self.act = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, 1)

        #
        self.selected_contexts = []
        self.labels = []
        self.candidate_tasks = []

    def update_data(self, contexts, labels, this_tasks):
        contexts = contexts.detach()
        labels = labels.detach()
        for i, t_i in enumerate(this_tasks):
            self.selected_contexts.append(contexts[i, :].reshape(-1, ))
            self.labels.append(labels[i, :].reshape(-1, ))
        #
        if self.buffer_size > 0:
            self.selected_contexts = self.selected_contexts[-self.buffer_size:]
            self.labels = self.labels[-self.buffer_size:]

    def train_model(self):
        time_length = len(self.selected_contexts)
        optimizer = optim.Adam(self.parameters(), lr=self.lr_rate)
        index = np.arange(time_length)
        np.random.shuffle(index)

        cnt = 0
        tot_loss = 0
        while True:
            batch_loss = 0
            for idx in index:
                c = self.selected_contexts[idx].to(self.device)
                r = self.labels[idx].to(self.device)
                optimizer.zero_grad()
                loss = (self.forward(c) - r) ** 2
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
                tot_loss += loss.item()
                cnt += 1
                if cnt >= 1000:
                    return tot_loss / cnt
            if batch_loss / time_length <= 1e-3:
                return batch_loss / time_length

    def forward(self, x):
        out = self.act(self.fc1(x))
        out = self.fc2(out)

        return out


# ======================================================================================================
# ======================================================================================================
class Bandit_Sampler(nn.Module):
    def __init__(self, input_dim, hidden_size, alpha=0.5, g_pool_step_meta=-1, g_pool_step_bandit=-1, buffer_size=20000,
                 lr_rate=0.0001, artificial_exp_score=-1, embedding_layer_hidden_size=-1, device=None):
        super(Bandit_Sampler, self).__init__()
        self.g_pool_step_meta = g_pool_step_meta
        self.g_pool_step_bandit = g_pool_step_bandit
        self.alpha = alpha
        self.artificial_exp_score = artificial_exp_score
        self.device = device
        self.print_flag= False

        #
        if embedding_layer_hidden_size <= 0:
            if g_pool_step_meta > 0:
                meta_param_num = int((input_dim // g_pool_step_meta)) + 1
            else:
                meta_param_num = input_dim
            self.exploit_net_f_1 = FC_Net(meta_param_num, hidden_size, buffer_size=buffer_size, lr_rate=lr_rate,
                                        device=device).to(device)
        else:
            self.g_pool_step_meta = 1
            meta_param_num = input_dim
            self.exploit_net_f_1 = FC_Net(meta_param_num, hidden_size, buffer_size=buffer_size, lr_rate=lr_rate, 
                                            embedding_layer_hidden_size=embedding_layer_hidden_size, device=device).to(device)
        self.explore_net_f_2 = FC_Net(input_dim=2 * self.get_f_2_input_size(g_pool_step_bandit=g_pool_step_bandit),
                                      hidden_size=hidden_size, buffer_size=buffer_size,
                                      lr_rate=lr_rate, device=device).to(device)
        #
        self.batch_loss_func = nn.CrossEntropyLoss(reduce=False, reduction='none')

        # Arm contexts for the candidates in the current round
        self.gradient_support_vec, self.gradient_query_vec, self.gradient_all_vec = None, None, None
        self.g_vec_concat = None

    def get_f_2_input_size(self, g_pool_step_bandit=-1):
        user_total_param_count = sum(param.numel() for param in self.exploit_net_f_1.parameters())
        if g_pool_step_bandit > 0:
            user_total_param_count = int((user_total_param_count // g_pool_step_bandit)) + 1
            print("--- Pooling f_1 size: ", user_total_param_count)
        return user_total_param_count

    def update_data(self, rewards, explore_scores, selected_task_indices, candidate_task_num):
        f_1_inputs = self.gradient_all_vec[selected_task_indices, :]
        f_2_inputs = self.g_vec_concat[selected_task_indices, :]
        self.exploit_net_f_1.update_data(contexts=f_1_inputs, labels=rewards, this_tasks=selected_task_indices)
        self.explore_net_f_2.update_data(contexts=f_2_inputs, labels=explore_scores, this_tasks=selected_task_indices)

        # ---------------------------------
        # Add artificial exploration scores
        if self.artificial_exp_score > 0:
            all_tasks = torch.arange(candidate_task_num).to(self.device).reshape(1, -1)
            selected_task_indices = torch.stack(selected_task_indices).reshape(1, -1)
            combined = torch.cat((selected_task_indices, all_tasks), dim=1)
            uniques, counts = combined.unique(return_counts=True)
            other_task_indices = uniques[counts == 1].reshape(-1, )
            other_f_2_inputs = self.g_vec_concat[other_task_indices, :]
            arti_exp_scores = torch.ones((other_f_2_inputs.shape[0], 1)).to(self.device) * self.artificial_exp_score

            self.explore_net_f_2.update_data(contexts=other_f_2_inputs, labels=arti_exp_scores,
                                             this_tasks=other_task_indices)

    def train_model(self, step, print_per_step=100):
        l_1 = self.exploit_net_f_1.train_model()
        l_2 = self.explore_net_f_2.train_model()

        if step % print_per_step == 0:
            print("- Bandit Training loss: ", l_1, l_2)

    def calculate_meta_loss_with_tasks(self, tasks, meta_model):
        """Output the loss, given mate-model and selected tasks"""
        task_losses = []
        task_acc = []
        for this_task in tasks:
            x1, y1, x2, y2 = this_task
            x1, y1, x2, y2 = x1.squeeze(0).float().to(self.device), y1.squeeze(0).long().to(self.device), \
                             x2.squeeze(0).float().to(self.device), y2.squeeze(0).long().to(self.device)
            loss_val, acc_val = meta_model(x1, y1, x2, y2)
            task_losses.append(loss_val)
            task_acc.append(acc_val)
        return torch.stack(task_losses), torch.stack(task_acc)

    # ============================================================================
    def get_meta_params_with_tasks(self, meta_model, task_indices, tasks, indices_layer_wird_grad=None):
        init_meta_params = OrderedDict(meta_model.learner.named_parameters())
        param_all, param_support, param_query = [], [], []
        self.raw_outer_parameter_list, self.raw_outer_parameter_logits_list = [], []

        for this_task in tasks:
            support_x, support_y, query_x, query_y = this_task
            support_x, support_y, query_x, query_y = \
                support_x.squeeze(0).float().to(self.device), support_y.squeeze(0).long().to(self.device), \
                query_x.squeeze(0).float().to(self.device), query_y.squeeze(0).long().to(self.device)
            support_count, query_count = support_x.shape[0], query_x.shape[0]

            # Get inner loop weights -----------------------
            logits = meta_model.learner(support_x)
            loss_support = self.batch_loss_func(logits, support_y).mean()
            inner_gradients = torch.autograd.grad(loss_support, init_meta_params.values(), create_graph=True)
            #
            # One-step inner loop optim
            inner_loop_weight_list = []
            inner_loop_weight_vec = []
            for ((name, param), grad) in zip(init_meta_params.items(), inner_gradients):
                i_gradient = param - meta_model.args.update_lr * grad
                inner_loop_weight_list.append((name, i_gradient))
                inner_loop_weight_vec.append(i_gradient.detach().reshape(-1, ))
            #
            inner_loop_weights = OrderedDict(
                (name, param - meta_model.args.update_lr * grad)
                for ((name, param), grad) in zip(init_meta_params.items(), inner_gradients)
            )
            inner_loop_weights_logits = OrderedDict(
                {"weight": inner_loop_weights['logits.weight'], "bias": inner_loop_weights['logits.bias']})

            # Get outer loop weights ------------------------
            outer_logits = \
                meta_model.learner.functional_forward_val(query_x, inner_loop_weights, inner_loop_weights_logits,
                                                          is_training=True).squeeze()
            loss_query = self.batch_loss_func(outer_logits, query_y).mean()
            outer_gradients = torch.autograd.grad(loss_query, inner_loop_weights.values(), create_graph=True)

            # For validation use
            raw_outer_loop_weights = OrderedDict(
                (name, param - meta_model.args.update_lr * grad)
                for ((name, param), grad) in zip(inner_loop_weights.items(), outer_gradients)
            )
            raw_outer_loop_weights_logits = OrderedDict(
                {"weight": raw_outer_loop_weights['logits.weight'], "bias": raw_outer_loop_weights['logits.bias']})
            self.raw_outer_parameter_list.append(raw_outer_loop_weights)
            self.raw_outer_parameter_logits_list.append(raw_outer_loop_weights_logits)

            #
            # One-step inner loop optim
            outer_loop_weight_vec = []
            for ((name, param), grad) in zip(inner_loop_weights.items(), outer_gradients):
                o_gradient = param - meta_model.args.update_lr * grad
                outer_loop_weight_vec.append(o_gradient.detach().reshape(-1, ))

            #
            inner_loop_weight_vec = torch.cat(inner_loop_weight_vec).unsqueeze(0).unsqueeze(0)
            outer_loop_weight_vec = torch.cat(outer_loop_weight_vec).unsqueeze(0).unsqueeze(0)

            #
            if self.g_pool_step_meta > 0:
                inner_loop_weight_vec = F.avg_pool1d(inner_loop_weight_vec, kernel_size=self.g_pool_step_meta,
                                                     stride=self.g_pool_step_meta, ceil_mode=True).squeeze(
                    0).squeeze(0)
                outer_loop_weight_vec = F.avg_pool1d(outer_loop_weight_vec, kernel_size=self.g_pool_step_meta,
                                                     stride=self.g_pool_step_meta, ceil_mode=True).squeeze(
                    0).squeeze(0)

            #
            param_all.append(outer_loop_weight_vec)
            param_support.append(inner_loop_weight_vec)
            param_query.append(outer_loop_weight_vec)

        # Stack the contexts
        param_all_vec = torch.stack(param_all)
        param_support_vec = torch.stack(param_support)
        param_query_vec = torch.stack(param_query)

        return param_support_vec, param_query_vec, param_all_vec

    def get_reward_est_and_exp_score(self, meta_model, tasks, indices_layer_wird_grad=None):
        gradient_support_vec, gradient_query_vec, gradient_all_vec = \
            self.get_meta_params_with_tasks(meta_model, None, tasks, indices_layer_wird_grad)
        self.gradient_support_vec, self.gradient_query_vec, self.gradient_all_vec = \
            gradient_support_vec, gradient_query_vec, gradient_all_vec

        # output of f_1 ------------------------------------------------
        reward_ests = self.exploit_net_f_1(gradient_all_vec)

        # Get the gradient of f_1 ------------------------------------------------
        # Support set
        point_ests_s = self.exploit_net_f_1(gradient_support_vec)
        # Extract the Gradients
        g_list_s = []

        # Get the current parameters of the f_1 model, for calculating the gradients
        f_1_weights = OrderedDict(self.exploit_net_f_1.named_parameters())

        # Calculate gradients for support set
        for fx in point_ests_s:

            # Calculate the Gradients with autograd.grad()
            this_g_list = []
            grad_tuple = torch.autograd.grad(fx, f_1_weights.values(), create_graph=True)
            for grad in grad_tuple:
                this_g_list.append(grad.reshape(-1, ))
            g = torch.cat(this_g_list)
            g_list_s.append(g)

            #
            del grad_tuple

        g_vec_s = torch.stack(g_list_s, dim=0)
        if self.g_pool_step_bandit > 0:
            g_vec_s = F.avg_pool1d(g_vec_s.unsqueeze(0), kernel_size=self.g_pool_step_bandit,
                                   stride=self.g_pool_step_bandit, ceil_mode=True).squeeze(0)

        # Query set
        point_ests_q = self.exploit_net_f_1(gradient_query_vec)
        # Extract the Gradients
        g_list_q = []
        for fx in point_ests_q:

            # Calculate the Gradients with autograd.grad()
            this_g_list = []
            grad_tuple = torch.autograd.grad(fx, f_1_weights.values(), create_graph=True)
            for grad in grad_tuple:
                this_g_list.append(grad.reshape(-1, ))
            g = torch.cat(this_g_list)
            g_list_q.append(g)

            #
            del grad_tuple

        g_vec_q = torch.stack(g_list_q, dim=0)
        if self.g_pool_step_bandit > 0:
            g_vec_q = F.avg_pool1d(g_vec_q.unsqueeze(0), kernel_size=self.g_pool_step_bandit,
                                   stride=self.g_pool_step_bandit, ceil_mode=True).squeeze(0)

        # Input of the f_2
        g_vec_concat = torch.cat([g_vec_s, g_vec_q], dim=1)
        self.g_vec_concat = g_vec_concat

        # Output of f_2 ------------------------------------------------
        explore_score_ests = self.explore_net_f_2(g_vec_concat)

        return reward_ests.reshape(-1, 1), explore_score_ests.reshape(-1, 1)


    def cal_true_reward_sampling(self, meta_model, tasks, selected_indices, new_tasks_num, selected_overall_score,
                                 validation_task_num=3):
        val_loss_per_task = []
        for task_i, this_task in enumerate(tasks):
            if task_i not in selected_indices:
                continue

            val_taks_indices = torch.tensor(np.random.choice(len(tasks), validation_task_num, replace=False))
            val_loss_list = []
            for val_task_i in val_taks_indices:
                support_x, support_y, query_x, query_y = tasks[val_task_i]
                support_x, support_y, query_x, query_y = \
                    support_x.squeeze(0).float().to(self.device), support_y.squeeze(0).long().to(self.device), \
                    query_x.squeeze(0).float().to(self.device), query_y.squeeze(0).long().to(self.device)

                # Get inner loop weights -----------------------
                initial_loop_weights = self.raw_outer_parameter_list[task_i]
                initial_loop_weights_logits = self.raw_outer_parameter_logits_list[task_i]

                ####
                inner_logits = \
                    meta_model.learner.functional_forward_val(support_x, initial_loop_weights,
                                                              initial_loop_weights_logits,
                                                              is_training=True).squeeze()
                loss_support = self.batch_loss_func(inner_logits, support_y).mean()
                inner_gradients = torch.autograd.grad(loss_support, initial_loop_weights.values(), create_graph=True)

                # One-step inner loop optim
                inner_loop_weights = OrderedDict(
                    (name, param - meta_model.args.update_lr * grad)
                    for ((name, param), grad) in zip(initial_loop_weights.items(), inner_gradients)
                )
                inner_loop_weights_logits = OrderedDict(
                    {"weight": inner_loop_weights['logits.weight'], "bias": inner_loop_weights['logits.bias']})

                # Get outer loop weights ------------------------
                outer_logits = \
                    meta_model.learner.functional_forward_val(query_x, inner_loop_weights, inner_loop_weights_logits,
                                                              is_training=True).squeeze()
                loss_query = self.batch_loss_func(outer_logits, query_y).mean()
                val_loss_list.append(loss_query.reshape(-1, ))
            avg_val_loss = torch.cat(val_loss_list).mean()
            val_loss_per_task.append(avg_val_loss.reshape(-1, 1))
        val_loss = torch.cat(val_loss_per_task).reshape(-1, 1)
        #
        true_rewards = self.scale_transform(val_loss)

        return true_rewards, val_loss

    def scale_transform(self, source):
        # Transform the loss into fixed value range.
        return 10 * torch.exp(-0.0001 * torch.exp(5 * source))

