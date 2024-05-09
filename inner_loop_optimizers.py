import torch
import torch.nn as nn
import copy


class GradientDescentLearningRule(nn.Module):
    """Simple (stochastic) gradient descent learning rule.
    For a scalar error function `E(p[0], p_[1] ... )` of some set of
    potentially multidimensional parameters this attempts to find a local
    minimum of the loss function by applying updates to each parameter of the
    form
        p[i] := p[i] - learning_rate * dE/dp[i]
    With `learning_rate` a positive scaling parameter.
    The error function used in successive applications of these updates may be
    a stochastic estimator of the true error function (e.g. when the error with
    respect to only a subset of data-points is calculated) in which case this
    will correspond to a stochastic gradient descent learning rule.
    """

    def __init__(self, device, learning_rate=1e-3):
        """Creates a new learning rule object.
        Args:
            learning_rate: A postive scalar to scale gradient updates to the
                parameters by. This needs to be carefully set - if too large
                the learning dynamic will be unstable and may diverge, while
                if set too small learning will proceed very slowly.
        """
        super(GradientDescentLearningRule, self).__init__()
        assert learning_rate > 0., 'learning_rate should be positive.'
        self.learning_rate = torch.ones(1) * learning_rate
        self.learning_rate.to(device)

    def update_params(self, names_weights_dict, names_grads_wrt_params_dict, num_step, tau=0.9):
        """Applies a single gradient descent update to all parameters.
        All parameter updates are performed using in-place operations and so
        nothing is returned.
        Args:
            grads_wrt_params: A list of gradients of the scalar loss function
                with respect to each of the parameters passed to `initialise`
                previously, with this list expected to be in the same order.
        """
        updated_names_weights_dict = dict()
        for key in names_weights_dict.keys():
            updated_names_weights_dict[key] = names_weights_dict[key] - self.learning_rate * names_grads_wrt_params_dict[key]

        return updated_names_weights_dict


class LSLRGradientDescentLearningRule(nn.Module):
    """Simple (stochastic) gradient descent learning rule.
    For a scalar error function `E(p[0], p_[1] ... )` of some set of
    potentially multidimensional parameters this attempts to find a local
    minimum of the loss function by applying updates to each parameter of the
    form
        p[i] := p[i] - learning_rate * dE/dp[i]
    With `learning_rate` a positive scaling parameter.
    The error function used in successive applications of these updates may be
    a stochastic estimator of the true error function (e.g. when the error with
    respect to only a subset of data-points is calculated) in which case this
    will correspond to a stochastic gradient descent learning rule.
    """

    def __init__(self, device, total_num_inner_loop_steps, use_learnable_learning_rates, use_learnable_weight_decay,
                 LAPID, random_init, init_learning_rate=1e-3, init_weight_decay=5e-4, momentum=0.9, weight_decay=0.0001,
                 netserov=False, dampening=0):
        """Creates a new learning rule object.
        Args:
            init_learning_rate: A postive scalar to scale gradient updates to the
                parameters by. This needs to be carefully set - if too large
                the learning dynamic will be unstable and may diverge, while
                if set too small learning will proceed very slowly.
        """
        super(LSLRGradientDescentLearningRule, self).__init__()
        print(init_learning_rate)
        assert init_learning_rate > 0., 'learning_rate should be positive.'

        self.LAPID = LAPID
        self.random_init = random_init

        self.init_lr_val = init_learning_rate
        self.init_wd_val = init_weight_decay

        self.init_pspl_weight_decay = torch.ones(1)
        self.init_pspl_weight_decay.to(device)

        self.init_learning_rate = torch.ones(1) * init_learning_rate
        self.init_learning_rate.to(device)
        self.total_num_inner_loop_steps = total_num_inner_loop_steps
        self.use_learnable_learning_rates = use_learnable_learning_rates
        self.use_learnable_weight_decay = use_learnable_weight_decay
        self.init_weight_decay = torch.ones(1) * init_weight_decay
        self.init_bias_decay = torch.ones(1)

        # PID 优化器的参数
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.netserov = netserov
        self.dampening = dampening

    def initialise(self, names_weights_dict):
        self.names_learning_rates_dict = nn.ParameterDict()

        if self.LAPID:
            if self.random_init:
                self.names_beta_dict_per_param = nn.ParameterDict()

            # self.names_I_dict = nn.ParameterDict()
            # self.names_D_dict = nn.ParameterDict()
            self.buf = dict()
            self.g_buf = dict()

            for idx, (key, param) in enumerate(names_weights_dict.items()):
                if self.random_init:
                    pass
                    # per-param weight decay for random init
                    # self.names_I_dict[key.replace(".", "-")] = nn.Parameter(
                    #     data=torch.ones(param.shape) * self.init_weight_decay * self.init_learning_rate,
                    #     requires_grad=self.use_learnable_learning_rates)
                    #
                    # self.names_D_dict[key.replace(".", "-")] = nn.Parameter(
                    #     data=torch.ones(self.total_num_inner_loop_steps + 1),
                    #     requires_grad=self.use_learnable_learning_rates)
                else:
                    # 重点 ---------------------------
                    # per-step per-layer meta-learnable weight decay bias term (for more stable training and better performance by 2~3%)
                    # self.names_beta_dict[key.replace(".", "-")] = nn.Parameter(
                    #     data=torch.ones(
                    #         self.total_num_inner_loop_steps + 1) * self.init_weight_decay * self.init_learning_rate,
                    #     requires_grad=self.use_learnable_learning_rates)

                    # 每层网络 每个内环步骤 一个学习率
                    self.names_learning_rates_dict[key.replace(".", "-")] = nn.Parameter(
                        data=torch.ones(self.total_num_inner_loop_steps + 1) * self.init_learning_rate, requires_grad=self.use_learnable_learning_rates)

                    # PID 优化器, 记录 I_buf 和 D_buf 【不可学习参数】
                    # self.buf[key.replace(".", "-") + 'I_buffer'] = torch.zeros_like(param.data)
                    # self.buf[key.replace(".", "-") + 'D_buffer'] = torch.zeros_like(param.data)

                    # 生成的 PID 优化器系数 I D 【可学习参数】，内循环更新时每步一个学习率
                    # self.names_I_dict[key.replace(".", "-")] = nn.Parameter(data=torch.ones(
                    #     self.total_num_inner_loop_steps + 1) * self.init_learning_rate,
                    #                                                         requires_grad=self.use_learnable_learning_rates)
                    #
                    # self.names_D_dict[key.replace(".", "-")] = nn.Parameter(data=torch.ones(
                    #     self.total_num_inner_loop_steps + 1) * self.init_learning_rate,
                    #                                                         requires_grad=self.use_learnable_learning_rates)

                # per-step per-layer meta-learnable learning rate bias term (for more stable training and better performance by 2~3%)
                # self.names_alpha_dict[key.replace(".", "-")] = nn.Parameter(
                #     data=torch.ones(self.total_num_inner_loop_steps + 1) * self.init_learning_rate,
                #     requires_grad=self.use_learnable_learning_rates)

    def update_params(self, names_weights_dict, names_grads_wrt_params_dict,
                      #   generated_P_params,
                      generated_I_params, generated_D_params, num_step, tau=0.1):
        """Applies a single gradient descent update to all parameters.
        All parameter updates are performed using in-place operations and so
        nothing is returned.
        Args:
            grads_wrt_params: A list of gradients of the scalar loss function
                with respect to each of the parameters passed to `initialise`
                previously, with this list expected to be in the same order.
        """
        updated_names_weights_dict = dict()

        for key in names_grads_wrt_params_dict.keys():
            # beta = (1 - generated_beta * meta-learned per-step-per-layer bias term)
            # alpha = generated_alpha * meta-learned per-step-per-layer bias term)
            if self.LAPID:
                if self.random_init:
                    pass
                #     updated_names_weights_dict[key] = (1 - self.names_beta_dict_per_param[key.replace(".", "-")] *
                #                                        generated_beta_params[key] *
                #                                        self.names_beta_dict[key.replace(".", "-")][num_step]) * \
                #                                       names_weights_dict[key] - generated_alpha_params[key] * \
                #                                       self.names_alpha_dict[key.replace(".", "-")][num_step] * \
                #                                       names_grads_wrt_params_dict[key]
                else:
                    for key in names_grads_wrt_params_dict.keys():
                        # cur_layer_grad = copy.deepcopy(names_grads_wrt_params_dict[key].data)
                        # if self.weight_decay != 0:
                        #     cur_layer_grad = cur_layer_grad + self.weight_decay * names_weights_dict[key]

                        # if self.momentum != 0:
                        #     I_buf = self.buf[key.replace(".", "-") + 'I_buffer']
                        #     # I_buf * momentum + (1 - dampening) * grad
                        #     I_buf = I_buf * self.momentum + (1 - self.dampening) * cur_layer_grad
                        #     self.buf[key.replace(".", "-") + 'I_buffer'] = I_buf.data

                        #     # -------------------------------------------------
                        #     D_buf = self.buf[key.replace(".", "-") + 'D_buffer']
                        #     last_grad = copy.deepcopy(D_buf.data)
                        #     # D_buf * momentum + (1 - momentum) * (d_p - g_buf)
                        #     D_buf = D_buf * self.momentum + (1 - self.momentum) * (cur_layer_grad - last_grad)
                        #     self.buf[key.replace(".", "-") + 'D_buffer'] = D_buf.data
                        #     cur_layer_grad += generated_I_params[key] * I_buf + generated_D_params[key] * D_buf

                        # # updated_names_weights_dict[key] = generated_P_params[key] * names_weights_dict[key] - \
                        # #                                   self.names_learning_rates_dict[key.replace(".", "-")][
                        # #                                       num_step] * cur_layer_grad
                        # updated_names_weights_dict[key] = names_weights_dict[key] - self.names_learning_rates_dict[key.replace(".", "-")][
                        #     num_step] * cur_layer_grad

                        cur_layer_grad = copy.deepcopy(names_grads_wrt_params_dict[key].data)
                        
                        if self.weight_decay != 0:
                            cur_layer_grad = cur_layer_grad + self.weight_decay * names_weights_dict[key]

                        # =============================================================================================
                        if self.momentum != 0:
                            # ============================ KI  第一次赋值 ========================================
                            if (key.replace(".", "-") + 'I_buffer') not in self.buf:
                                I_buf = torch.zeros_like(names_grads_wrt_params_dict[key].data)
                                I_buf = I_buf * self.momentum + cur_layer_grad  # # 第一次直接赋值等于梯度，d_p
                                self.buf[key.replace(".", "-") + 'I_buffer'] = I_buf.data
                            else:
                                I_buf = self.buf[key.replace(".", "-") + 'I_buffer']
                                # I_buf * momentum + (1 - dampening) * grad
                                I_buf = I_buf * self.momentum + (1 - self.dampening) * cur_layer_grad
                                self.buf[key.replace(".", "-") + 'I_buffer'] = I_buf.data
                                # I_buf += mul_(self.momentum).add_(1 - self.dampening, cur_layer_grad)

                            # ============================ KD  第一次赋值 ========================================
                            if (key.replace(".", "-") + 'D_buffer') not in self.buf:
                                self.g_buf[key.replace(".", "-") + 'g_buf'] = torch.zeros_like(names_grads_wrt_params_dict[key].data)
                                self.g_buf[key.replace(".", "-") + 'g_buf'] = cur_layer_grad

                                D_buf = torch.zeros_like(names_grads_wrt_params_dict[key].data)
                                D_buf = D_buf * self.momentum + (cur_layer_grad - self.g_buf[key.replace(".", "-") + 'g_buf'])
                                self.buf[key.replace(".", "-") + 'D_buffer'] = D_buf.data
                            else:
                                # -------------------------------------------------
                                D_buf = self.buf[key.replace( ".", "-") + 'D_buffer']
                                g_buf = self.g_buf[key.replace(".", "-") + 'g_buf']
                                # D_buf * momentum + (1 - momentum) * (d_p - g_buf)
                                # D_buf.mul_(self.momentum).add_(1 - self.momentum, cur_layer_grad - last_grad)
                                D_buf = D_buf * self.momentum + (1 - self.momentum) * (cur_layer_grad - g_buf)
                                self.buf[key.replace( ".", "-") + 'D_buffer'] = D_buf.data
                                # 对于下次来说，这次的梯度就是上一次的梯度(历史梯度)
                                self.g_buf[key.replace(".", "-") + 'g_buf'] = cur_layer_grad.clone()

                            cur_layer_grad += generated_I_params[key] * I_buf + generated_D_params[key] * D_buf

                        updated_names_weights_dict[key] = names_weights_dict[key] - self.names_learning_rates_dict[key.replace(".", "-")][num_step] * cur_layer_grad
            else:
                updated_names_weights_dict[key] = names_weights_dict[key] - self.init_lr_val * names_grads_wrt_params_dict[key]

        return updated_names_weights_dict
