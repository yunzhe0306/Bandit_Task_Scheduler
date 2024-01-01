import copy

import numpy as np
import torch
from torch.nn import Module
from copy import deepcopy


class AveragedModel(Module):

    def __init__(self, model, device=None):
        super(AveragedModel, self).__init__()
        self.module = deepcopy(model)
        self.module.zero_grad(set_to_none=True)

        def avg_fn(averaged_model_parameter, model_parameter, num_averaged):
                return averaged_model_parameter + \
                       (model_parameter - averaged_model_parameter) / (num_averaged + 1)

        #
        self.module = self.module.to(device)
        self.avg_fn = avg_fn
        self.n_averaged = 0

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def avg_BN_running_stats(self, avg_model, this_model):
        for avg_m, new_m in zip(avg_model.modules(), this_model.modules()):
            if isinstance(avg_m, torch.nn.BatchNorm2d):
                # Avg stats
                old_mean_sum = avg_m.running_mean.data * self.n_averaged
                old_var_sum = avg_m.running_var.data * self.n_averaged
                avg_m.running_mean.copy_((old_mean_sum + new_m.running_mean.data) / (self.n_averaged + 1))
                avg_m.running_var.copy_((old_var_sum + new_m.running_var.data) / (self.n_averaged + 1))

    def disable_BN_running_stats(self, model):
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.running_mean.fill_(0.2)
                m.running_var.fill_(0.3)

    def update_parameters(self, model, mean_loss):
        #
        with torch.no_grad():
            for p_swa, p_model in zip(self.module.parameters(), model.parameters()):
                device = p_swa.device
                p_model_ = p_model.detach().to(device)
                if self.n_averaged == 0:
                    p_swa.data.copy_(p_model_.data)
                else:
                    p_swa.data.copy_(
                        self.avg_fn(p_swa.detach(), p_model_, self.n_averaged).data
                    )

        # self.avg_BN_running_stats(self.module, model)
        self.module.apply(self.disable_BN_running_stats)
        self.n_averaged += 1


# =================================================================================

class AveragedModelWeights(Module):
    def __init__(self, model, device=None, model_num=10):
        super(AveragedModelWeights, self).__init__()
        self.module_list = [deepcopy(model).to(device)]
        self.disable_BN_running_stats(self.module_list[0], mean_val=0, var_val=0)
        self.mean_loss_list = {0: [1]}
        self.n_averaged = 1
        self.model_num_per_seed = model_num
        self.this_seed_idx = 0

        def avg_fn(averaged_model_parameter, model_parameter, num_averaged):
            old_param = averaged_model_parameter * num_averaged
            new_param = (old_param + model_parameter) / (num_averaged + 1)
            return new_param

        #
        self.avg_fn = avg_fn

    def forward(self, *args, **kwargs):
        return self.get_avg_model()(*args, **kwargs)

    def avg_BN_running_stats(self, avg_model, this_model, num_averaged):
        for avg_m, new_m in zip(avg_model.modules(), this_model.modules()):
            if isinstance(avg_m, torch.nn.BatchNorm2d):
                # Avg stats
                old_mean_sum = avg_m.running_mean.data * num_averaged
                old_var_sum = avg_m.running_var.data * num_averaged
                avg_m.running_mean.copy_((old_mean_sum + new_m.running_mean.data) / (num_averaged + 1))
                avg_m.running_var.copy_((old_var_sum + new_m.running_var.data) / (num_averaged + 1))

    def get_avg_model(self):
        indices = self.get_indices_max_var()
        this_module = None
        with torch.no_grad():
            for i, m_i in enumerate(indices):
                if this_module is None:
                    this_module = copy.deepcopy(self.module_list[0])
                else:
                    model = self.module_list[m_i]
                    for p_swa, p_model in zip(this_module.parameters(), model.parameters()):
                        device = p_swa.device
                        p_model_ = p_model.detach().to(device)
                        p_swa.data.copy_(
                            self.avg_fn(p_swa.detach(), p_model_, i).data
                        )
                    #
                    self.avg_BN_running_stats(this_module, model, i)

        return this_module

    def disable_BN_running_stats(self, model, mean_val=0.1, var_val=0.1):
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.running_mean.fill_(mean_val)
                m.running_var.fill_(var_val)

    def update_parameters(self, model, mean_loss):
        #
        this_model = copy.deepcopy(model)
        self.module_list.append(this_model)
        if self.this_seed_idx not in self.mean_loss_list:
            self.mean_loss_list[self.this_seed_idx] = []
        self.mean_loss_list[self.this_seed_idx].append(mean_loss)

        self.n_averaged += 1

    def get_indices_max_var(self):

        max_indices = None
        count = 0
        for seed_idx in range(self.this_seed_idx + 1):
            mean_loss_arr = -1 * np.array(self.mean_loss_list[seed_idx])
            if self.model_num_per_seed >= len(self.mean_loss_list[seed_idx]):
                this_max_indices = np.arange(len(self.mean_loss_list[seed_idx]))
            else:
                this_max_indices = np.argpartition(mean_loss_arr, -self.model_num_per_seed)[-self.model_num_per_seed:]
            #
            if max_indices is None:
                max_indices = this_max_indices
            else:
                this_max_indices += count
                max_indices = np.concatenate([max_indices, this_max_indices])
            count += len(self.mean_loss_list[seed_idx])
        return max_indices


# =================================================================================
class AveragedModelPred(Module):

    def __init__(self, model, device=None, model_num=10):
        super(AveragedModelPred, self).__init__()
        self.module_list = [deepcopy(model).to(device)]
        self.mean_loss_list = [1]
        self.n_averaged = 1
        self.model_num = model_num

    def forward(self, *args, **kwargs):
        this_pred = None
        indices = self.get_indices_max_var()
        for i, m_i in enumerate(indices):

            if this_pred is None:
                this_pred = self.module_list[m_i](*args, **kwargs)
            else:
                this_pred += self.module_list[m_i](*args, **kwargs)

        return this_pred / self.model_num
        # return this_pred / torch.sum(mean_loss_arr)

    def get_indices_max_var(self):
        if self.model_num >= self.n_averaged:
            return range(self.n_averaged)
        else:
            mean_loss_arr = -1 * np.array(self.mean_loss_list)
            max_indices = np.argpartition(mean_loss_arr, -self.model_num)[-self.model_num:]
        return max_indices

    def avg_BN_running_stats(self, avg_model, this_model):
        for avg_m, new_m in zip(avg_model.modules(), this_model.modules()):
            if isinstance(avg_m, torch.nn.BatchNorm2d):
                # Use the latest running stats
                avg_m.running_mean.copy_(new_m.running_mean.data)
                avg_m.running_var.copy_(new_m.running_var.data)

    def update_parameters(self, model, mean_loss):
        #
        this_model = copy.deepcopy(model)
        self.avg_BN_running_stats(this_model, model)
        self.module_list.append(this_model)
        self.mean_loss_list.append(mean_loss)

        self.n_averaged += 1
