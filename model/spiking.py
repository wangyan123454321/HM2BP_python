import torch
import torch.nn as nn
from torch.nn import functional as ff
from torch.autograd import Function as gf
import math
from . import functions as fun

class mm_linear(gf):
    @staticmethod
    @torch.jit.script
    def forward_(input, weight, bias, weight_lateral, threshold, tau_m, tau_s, t_ref, use_bias, use_weight_lateral):
        """
        mmbp MLP的前向传播
        @params:
            ctx: 上下文
            input: torch.tensor 输入张量，形状为[time_steps(T), batch_size(B), input_shape(I)]
            weight: torch.tensor 权重，形状为[output_shape(O), input_shape(I)]
            bias: torch.tensor 偏置，形状为[output_shape(O)]
            weight_lateral: torch.tensor 对输出的权重，形状为[output_shape(O)]
            threshold: float 权重
            tau_m: float 参数τm
            tau_s: float 参数τs
            t_ref: float 不应期长度
        @return:
            output: torch.tensor 输出张量，形状为[time_steps(T), batch_size(B), output_shape(O)]
        """
        T = input.shape[0]
        output = []
        ep = torch.tensor(0.)
        u = torch.tensor(0.)
        o = torch.tensor(0.)
        v_t_ref = torch.tensor(0.)
        times_pre = torch.tensor(0.)
        times_post = torch.tensor(0.)

        # 遍历所有的time_step
        for t in range(T):
            # 输入{O_j^{k-1}(t)}
            i = input[t] # [input_shape]
            # 利用torch自带的linear函数，通过与权重相乘（WI+b，对应元素就是\sum_{j}{w_{ij}*O_j^{k-1}(t)}+b_{i}）计算突触后电位

            if use_bias:
                i_response = ff.linear(i, weight, bias) # [output_shape, input_shape] * [input_shape, 1] + [output_shape] = [output_shape]
            else:
                i_response = ff.linear(i, weight) # [output_shape, input_shape] * [input_shape, 1] = [output_shape]
            # 如果没有初始化，初始化一些值（初始值全为0）
            if t == 0:
                ep = torch.zeros_like(i_response, dtype = torch.float) # [output_shape]
                u = torch.zeros_like(i_response, dtype = torch.float) # [output_shape]
                o = torch.zeros_like(i_response, dtype = torch.float) # [output_shape]
                v_t_ref = torch.zeros_like(i_response, dtype = torch.float) # [output_shape]
                times_pre = torch.zeros_like(i, dtype = torch.float) - 1. # [input_shape]
                times_post = torch.zeros_like(i_response, dtype = torch.float) - 1. # [output_shape]
            # 如果具有侧抑制，加入侧抑制的效果
            if use_weight_lateral:
                i_response = i_response + o * weight_lateral # [output_shape] + [output_shape] * [output_shape] = [output_shape]
            # 逐元素计算电位值{U_i^{k}(t)}
            # ep -= ep / TAU_S; ep += response;
            ep = ep * (1. - 1. / tau_s) + i_response
            # v -= v / TAU_M; v += ep / TAU_S;
            u = (u * (1. - 1. / tau_m) + ep / tau_s)
            # if(t_ref > 0) v = 0;
            u = u * (1. - v_t_ref.gt(0).float())
            # 逐元素计算脉冲是否发射{O_i^{k}(t)}
            # curOutput[o_idx + t * outputSize] = v > threshold ?  true : false;
            o = u.gt(threshold).float()
            # 更新不应期
            # if(t_ref > 0) t_ref--; t_ref = v > threshold ? T_REFRAC : t_ref;
            v_t_ref = torch.clamp(t_ref * o + (v_t_ref - 1.), min = 0., max = t_ref)
            output.append(o)
            times_pre = torch.max((i * (t + 1.)) - 1., times_pre)
            times_post = torch.max((o * (t + 1.)) - 1., times_post)
        output = torch.stack(output)
        return output, times_pre, times_post
    
    @staticmethod
    @torch.jit.script
    def backward_(grad_output, input, output, times_pre, times_post, weight, bias, weight_lateral, threshold, tau_m, tau_s, t_ref, use_bias, use_weight_lateral):
        input_firecount = torch.sum(input, dim = 0) # [batch_size, input_shape]
        output_firecount = torch.sum(output, dim = 0) # [batch_size, output_shape]

        # 定义返回值形状
        grad_input = torch.zeros_like(input) # [batch_size, input_shape]
        grad_weight = torch.zeros_like(weight) # [output_shape, input_shape]
        grad_bias = torch.zeros_like(bias) # [output_shape]
        grad_weight_lateral = torch.zeros_like(weight_lateral) # [output_shape]

        # ∂o / ∂a = 1 / ν
        partial_o_a = 1. / threshold
        # δ = ∂L / ∂a = ∂L / ∂o * ∂o / ∂a
        grad_a = grad_output * partial_o_a # [batch_size, output_shape]

        # ∂ai / ∂e_{i|j} = w_{ij}
        partial_a_e = weight # [output_shape, input_shape]

        batch_size = grad_output.shape[0]
        for b in range(batch_size):
            input_this_image = input[:, b] # [time_steps, input_shape]
            output_this_image = output[:, b] # [time_steps, output_shape]
            times_pre_this_image = times_pre[b] # [input_shape]
            times_post_this_image = times_post[b] # [output_shape]
            input_firecount_this_image = input_firecount[b] # [input_shape]
            output_firecount_this_image = output_firecount[b] # [output_shape]
            # δ = ∂L / ∂a
            grad_a_this_image = grad_a[b] # [output_shape]

            # e_{i|j}
            e = fun.single_stdp(
                weight,
                times_pre_this_image,
                times_post_this_image,
                input_firecount_this_image,
                output_firecount_this_image,
                tau_m,
                tau_s,
                t_ref
                ) # [output_shape, input_shape]

            if use_weight_lateral:
                e_lat = fun.double_stdp(
                    torch.diag(weight_lateral),
                    times_post_this_image,
                    times_post_this_image,
                    output_firecount_this_image,
                    output_firecount_this_image,
                    tau_m,
                    tau_s,
                    t_ref
                    ) # [output_shape, output_shape]
            else:
                e_lat = torch.tensor(0.)

            # ∂L / ∂e_{i|j} = δ_i * w_{ij}
            grad_e = fun.rmul(partial_a_e, grad_a_this_image) # [output_shape, input_shape]
            # δ_i * e_{i|j}
            delta_grad_weight_base = fun.rmul(e, grad_a_this_image) # [output_shape, input_shape]

            """
            梯度因子
            G = [
                δ_1 * w_{11} * e_{1|1}           δ_1 * w_{12} * e_{1|2}           ...   δ_1 * w_{1n_j} * e_{1|n_j}
                δ_2 * w_{21} * e_{2|1}           δ_2 * w_{22} * e_{2|2}           ...   δ_2 * w_{2n_j} * e_{2|n_j}
                         ...                              ...                     ...             ...
                δ_{n_i} * w_{n_i1} * e_{n_i|1}   δ_{n_i} * w_{n_i2} * e_{n_i|2}   ...   δ_{n_in_j} * w_{n_in_j} * e_{n_i|n_j}
            ]
            ，形状为[n_i(output_shape), n_j(input_shape)]；
            """
            G = grad_e * e # [output_shape, input_shape]

            """
            输出脉冲计数的倒数
            O = [1/o_1   1/o_2   ...   1/o_{n_i}]
            ，形状为[n_i(output_shape)]；
            输入脉冲计数的倒数
            I = [1/i_1   1/i_2   ...   1/i_{n_j}]
            ，形状为[n_j(input_shape)]。
            """
            O = 1. / torch.max(output_firecount_this_image, torch.ones_like(output_firecount_this_image)) # [output_shape]
            I = 1. / torch.max(input_firecount_this_image, torch.ones_like(input_firecount_this_image)) # [input_shape]

            """
            输入梯度因子
            IG = G * I = [
                δ_1 * w_{11} * e_{1|1} / i_1           δ_1 * w_{12} * e_{1|2} / i_2           ...   δ_1 * w_{1n_j} * e_{1|n_j} / i_{n_j}
                δ_2 * w_{21} * e_{2|1} / i_1           δ_2 * w_{22} * e_{2|2} / i_2           ...   δ_2 * w_{2n_j} * e_{2|n_j} / i_{n_j}
                               ...                                    ...                     ...                   ...
                δ_{n_i} * w_{n_i1} * e_{n_i|1} / i_1   δ_{n_i} * w_{n_i2} * e_{n_i|2} / i_2   ...   δ_{n_i} * w_{n_in_j} * e_{n_i|n_j} / i_{n_j}
            ]
            ，形状为[n_i(output_shape), n_j(input_shape)]，在第0个维度上求和得到矩阵
            GI = [\sum_{l}{δ_l * w_{l1} * e_{l|1} / i_1}   \sum_{l}{δ_l * w_{l2} * e_{l|2} / i_2}   ...   \sum_{l}{δ_l * w_{ln_j} * e_{l|n_j} / i_{n_j}}]
            ，形状为[n_j(input_shape)]，第j个元素为
            g_j = \sum_{l}{δ_l * w_{lj} * e_{l|j} / i_j}
            ；输出梯度因子
            OG = G^T * O = [
                δ_1 * w_{11} * e_{1|1} / o_1       δ_2 * w_{21} * e_{2|1} / o_2       ...   δ_{n_i} * w_{n_i1} * e_{n_i|1} / o_{n_i}
                δ_1 * w_{12} * e_{1|2} / o_1       δ_2 * w_{22} * e_{2|2} / o_2       ...   δ_{n_i} * w_{n_i2} * e_{n_i|2} / o_{n_i}
                            ...                                ...                    ...                   ...
                δ_1 * w_{1n_j} * e_{1|n_j} / o_1   δ_2 * w_{2n_j} * e_{2|n_j} / o_2   ...   δ_{n_i} * w_{n_in_j} * e_{n_i|n_j} / o_{n_i}
            ]
            ，形状为[n_j(input_shape), n_i(output_shape)]，在第0个维度上求和得到矩阵
            GO = [\sum_{l}{δ_1 * w_{1l} * e_{1|l} / o_1}   \sum_{l}{δ_2 * w_{2l} * e_{2|l} / o_2}   ...   \sum_{l}{δ_{n_i} * w_{n_il} * e_{n_i|l} / o_{n_i}}]
            ，形状为[n_i(output_shape)]，第i个元素为
            g_i = \sum_{l}{δ_i * w_{il} * e_{i|l} / o_i}
            。
            """
            IG = G * I # [output_shape, input_shape]
            GI = torch.sum(IG, dim = 0) # [input_shape]
            OG = G.T * O # [input_shape, output_shape]
            GO = torch.sum(OG, dim = 0) # [output_shape]

            # ∂L / ∂o_j^{k-1} = ∂L / ∂i_j = g_j
            grad_input[0, b] = GI

            # e_{i|j} * g_i
            e_GO = fun.rmul(e, GO) # [output_shape, input_shape]

            if use_weight_lateral:
                # 计算 g_i = \sum_{l != i}{e_{i|l} * e_{l|i} / o_{l}}
                # G_lat = torch.diag((e_lat @ fun.rmul(e_lat, O)) - fun.rmul(e_lat * e_lat, O)) # [output_shape]
                # 计算 gamma: gamma_i = 1 / (1 - w_{0}^2 / v^2 * g_i / o_i)
                # gamma = 1. / (1. - (weight_lateral * weight_lateral * partial_o_a * partial_o_a * G_lat * O)) # [output_shape]

                # 计算 gamma: gamma_i = 1 / (1 - w_{0}^2 * e_{i|i})
                gamma = 1. / (1. - (weight_lateral * weight_lateral * torch.diag(e_lat))) # [output_shape]
            else:
                gamma = torch.tensor(1.).to(grad_weight.device)
            
            # ∂L / ∂w_{ij} = γ * (δ_i * e_{i|j} + e_{i|j} * g_i / ν)
            delta_grad_weight = delta_grad_weight_base + e_GO * partial_o_a # [output_shape, input_shape]
            delta_grad_weight = fun.rmul(delta_grad_weight, gamma) # [output_shape, input_shape]
            grad_weight = grad_weight + delta_grad_weight # [output_shape, input_shape]
        return grad_input, grad_weight, grad_bias, grad_weight_lateral
        

    @staticmethod
    def forward(ctx, input, weight, bias, weight_lateral, threshold, tau_m, tau_s, t_ref):
        use_bias = bias is not None
        use_weight_lateral = weight_lateral is not None
        output, times_pre, times_post = mm_linear.forward_(
            input,
            weight,
            bias if use_bias else torch.tensor(0.),
            weight_lateral if use_weight_lateral else torch.tensor(0.),
            threshold,
            tau_m,
            tau_s,
            t_ref,
            use_bias = torch.tensor(use_bias, dtype = torch.bool),
            use_weight_lateral = torch.tensor(use_weight_lateral, dtype = torch.bool)
            )
        ctx.save_for_backward(input, output, times_pre, times_post, weight, bias, weight_lateral, threshold, tau_m, tau_s, t_ref)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        mmbp MLP的反向传播
        @params:
            grad_output: 从上一层传来的，关于输出的各个梯度dL/do，形状为[batch_size, output_shape]
        @return:
            grad_input: 关于输入的梯度dL/di，形状为[batch_size, input_shape]
            grad_weight: 关于权重的梯度dL/dw，形状为[output_shape, input_shape]
            grad_bias: 关于偏置的梯度dL/db，形状为[output_shape]
            grad_weight_lateral: 关于输出权重的梯度dL/dwl，形状为[output_shape]
        """
        # 取首个time_step的梯度
        grad_output = grad_output[0]
        input, output, times_pre, times_post, weight, bias, weight_lateral, threshold, tau_m, tau_s, t_ref = ctx.saved_tensors
        use_bias = bias is not None
        use_weight_lateral = weight_lateral is not None
        grad_input, grad_weight, grad_bias, grad_weight_lateral = mm_linear.backward_(
            grad_output,
            input,
            output,
            times_pre,
            times_post,
            weight,
            bias if use_bias else torch.tensor(0.),
            weight_lateral if use_weight_lateral else torch.tensor(0.),
            threshold,
            tau_m,
            tau_s,
            t_ref,
            use_bias = torch.tensor(use_bias, dtype = torch.bool),
            use_weight_lateral = torch.tensor(use_weight_lateral, dtype = torch.bool)
            )
        return grad_input, grad_weight, grad_bias if use_bias else None, grad_weight_lateral if use_weight_lateral else None, None, None, None, None

class Spiking(nn.Module):
    """
    mmbp全连接层
    """
    def __init__(self, input_shape, output_shape, threshold, tau_m, tau_s, t_ref, bias = None, weight_lateral = None):
        super().__init__()
        self.input_shape = nn.Parameter(torch.tensor(input_shape), requires_grad = False)
        self.output_shape = nn.Parameter(torch.tensor(output_shape), requires_grad = False)
        weight_mat = torch.Tensor(output_shape, input_shape)
        torch.nn.init.normal_(weight_mat)
        self.weight = nn.Parameter(weight_mat, requires_grad = True)
        self.tau_m = nn.Parameter(torch.tensor(tau_m), requires_grad = False)
        self.tau_s = nn.Parameter(torch.tensor(tau_s), requires_grad = False)
        self.t_ref = nn.Parameter(torch.tensor(t_ref), requires_grad = False)
        self.threshold = nn.Parameter(torch.tensor(threshold), requires_grad = False)
        if bias is None:
            self.bias = None
        else:
            self.bias = nn.Parameter(torch.ones(output_shape) * bias, requires_grad = False)
        if weight_lateral is None:
            self.weight_lateral = None
        else:
            self.weight_lateral = nn.Parameter(torch.ones(output_shape) * weight_lateral, requires_grad = False)
    
    def forward(self, x):
        return mm_linear.apply(x, self.weight, self.bias, self.weight_lateral, self.threshold, self.tau_m, self.tau_s, self.t_ref)