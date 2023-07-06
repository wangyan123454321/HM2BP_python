import torch
from torch.nn import functional as ff


@torch.jit.script
def add_by_row(mat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """
    逐行对元素相加，若一个矩阵为 A = [a_{ij}]mxn, 另一个向量为 b = [b_{i}]m, 则返回结果为 C = [a_{ij} + b_{i}]m×n
    @params:
        mat: torch.Tensor 矩阵 A
        vec: torch.Tensor 向量 b
    @return:
        res: torch.Tensor 结果 C
    """
    return (mat.T + vec).T


@torch.jit.script
def radd(mat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """
    逐行对元素相加，若一个矩阵为 A = [a_{ij}]mxn, 另一个向量为 b = [b_{i}]m, 则返回结果为 C = [a_{ij} + b_{i}]m×n
    @params:
        mat: torch.Tensor 矩阵 A
        vec: torch.Tensor 向量 b
    @return:
        res: torch.Tensor 结果 C
    """
    return add_by_row(mat, vec)


@torch.jit.script
def sub_by_row(mat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """
    逐行对元素相减，若一个矩阵为 A = [a_{ij}]mxn, 另一个向量为 b = [b_{i}]m, 则返回结果为 C = [a_{ij} - b_{i}]m×n
    @params:
        mat: torch.Tensor 矩阵 A
        vec: torch.Tensor 向量 b
    @return:
        res: torch.Tensor 结果 C
    """
    return (mat.T - vec).T


@torch.jit.script
def rsub(mat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """
    逐行对元素相减，若一个矩阵为 A = [a_{ij}]mxn, 另一个向量为 b = [b_{i}]m, 则返回结果为 C = [a_{ij} - b_{i}]m×n
    @params:
        mat: torch.Tensor 矩阵 A
        vec: torch.Tensor 向量 b
    @return:
        res: torch.Tensor 结果 C
    """
    return sub_by_row(mat, vec)


@torch.jit.script
def mul_by_row(mat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """
    逐行对元素相乘，若一个矩阵为 A = [a_{ij}]mxn, 另一个向量为 b = [b_{i}]m, 则返回结果为 C = [a_{ij} * b_{i}]m×n
    @params:
        mat: torch.Tensor 矩阵 A
        vec: torch.Tensor 向量 b
    @return:
        res: torch.Tensor 结果 C
    """
    return (mat.T * vec).T


@torch.jit.script
def rmul(mat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """
    逐行对元素相乘，若一个矩阵为 A = [a_{ij}]mxn, 另一个向量为 b = [b_{i}]m, 则返回结果为 C = [a_{ij} * b_{i}]m×n
    @params:
        mat: torch.Tensor 矩阵 A
        vec: torch.Tensor 向量 b
    @return:
        res: torch.Tensor 结果 C
    """
    return mul_by_row(mat, vec)


@torch.jit.script
def div_by_row(mat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """
    逐行对元素相除，若一个矩阵为 A = [a_{ij}]mxn, 另一个向量为 b = [b_{i}]m, 则返回结果为 C = [a_{ij} / b_{i}]m×n
    需要注意除以0的问题
    @params:
        mat: torch.Tensor 矩阵 A
        vec: torch.Tensor 向量 b
    @return:
        res: torch.Tensor 结果 C
    """
    assert (torch.nonzero(vec).flatten()).shape[0] == vec.shape[0]
    return (mat.T / vec).T


@torch.jit.script
def rdiv(mat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """
    逐行对元素相乘，若一个矩阵为 A = [a_{ij}]mxn, 另一个向量为 b = [b_{i}]m, 则返回结果为 C = [a_{ij} * b_{i}]m×n
    需要注意除以0的问题
    @params:
        mat: torch.Tensor 矩阵 A
        vec: torch.Tensor 向量 b
    @return:
        res: torch.Tensor 结果 C
    """
    return div_by_row(mat, vec)


@torch.jit.script
def add_by_col(mat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """
    逐列对元素相加，若一个矩阵为 A = [a_{ij}]mxn, 另一个向量为 b = [b_{j}]n, 则返回结果为 C = [a_{ij} + b_{j}]m×n
    @params:
        mat: torch.Tensor 矩阵 A
        vec: torch.Tensor 向量 b
    @return:
        res: torch.Tensor 结果 C
    """
    return mat + vec


@torch.jit.script
def cadd(mat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """
    逐列对元素相加，若一个矩阵为 A = [a_{ij}]mxn, 另一个向量为 b = [b_{j}]n, 则返回结果为 C = [a_{ij} + b_{j}]m×n
    @params:
        mat: torch.Tensor 矩阵 A
        vec: torch.Tensor 向量 b
    @return:
        res: torch.Tensor 结果 C
    """
    return add_by_col(mat, vec)


@torch.jit.script
def sub_by_col(mat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """
    逐列对元素相减，若一个矩阵为 A = [a_{ij}]mxn, 另一个向量为 b = [b_{j}]n, 则返回结果为 C = [a_{ij} - b_{j}]m×n
    @params:
        mat: torch.Tensor 矩阵 A
        vec: torch.Tensor 向量 b
    @return:
        res: torch.Tensor 结果 C
    """
    return mat - vec


@torch.jit.script
def csub(mat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """
    逐列对元素相减，若一个矩阵为 A = [a_{ij}]mxn, 另一个向量为 b = [b_{j}]n, 则返回结果为 C = [a_{ij} - b_{j}]m×n
    @params:
        mat: torch.Tensor 矩阵 A
        vec: torch.Tensor 向量 b
    @return:
        res: torch.Tensor 结果 C
    """
    return sub_by_col(mat, vec)


@torch.jit.script
def mul_by_col(mat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """
    逐列对元素相乘，若一个矩阵为 A = [a_{ij}]mxn, 另一个向量为 b = [b_{j}]n, 则返回结果为 C = [a_{ij} * b_{j}]m×n
    @params:
        mat: torch.Tensor 矩阵 A
        vec: torch.Tensor 向量 b
    @return:
        res: torch.Tensor 结果 C
    """
    return mat * vec


@torch.jit.script
def cmul(mat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """
    逐列对元素相乘，若一个矩阵为 A = [a_{ij}]mxn, 另一个向量为 b = [b_{j}]n, 则返回结果为 C = [a_{ij} * b_{j}]m×n
    @params:
        mat: torch.Tensor 矩阵 A
        vec: torch.Tensor 向量 b
    @return:
        res: torch.Tensor 结果 C
    """
    return mul_by_col(mat, vec)


@torch.jit.script
def div_by_col(mat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """
    逐列对元素相除，若一个矩阵为 A = [a_{ij}]mxn, 另一个向量为 b = [b_{j}]n, 则返回结果为 C = [a_{ij} / b_{j}]m×n
    需要注意除以0的问题
    @params:
        mat: torch.Tensor 矩阵 A
        vec: torch.Tensor 向量 b
    @return:
        res: torch.Tensor 结果 C
    """
    assert (torch.nonzero(vec).flatten()).shape[0] == vec.shape[0]
    return mat / vec


@torch.jit.script
def cdiv(mat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """
    逐列对元素相除，若一个矩阵为 A = [a_{ij}]mxn, 另一个向量为 b = [b_{j}]n, 则返回结果为 C = [a_{ij} / b_{j}]m×n
    需要注意除以0的问题
    @params:
        mat: torch.Tensor 矩阵 A
        vec: torch.Tensor 向量 b
    @return:
        res: torch.Tensor 结果 C
    """
    return div_by_col(mat, vec)


@torch.jit.script
def single_stdp(weight_mat, time_pres, time_posts, input_firecount, output_firecount, tau_m, tau_s, t_ref):
    """
    利用STDP计算权重更新因子
    @params:
        weight_mat: torch.Tensor 权重矩阵，形状为[output_shape(O), input_shape(I)]
        input_spike_train: torch.Tensor 输入脉冲序列，形状为[time_steps(T), input_shape(I)]
        output_spike_train: torch.Tensor 输出脉冲序列，形状为[time_steps(T), output_shape(O)]
        input_firecount: torch.Tensor 输入脉冲发射数量统计，形状为[input_shape(I)]
        output_firecount: torch.Tensor 输出脉冲发射数量统计，形状为[output_shape(O)]
        tau_m: float 参数τm
        tau_s: float 参数τs
        t_ref: float 不应期长度
    @return:
        dw: torch.Tensor 权重更新因子，叠加到原权重上
    """
    w_stdp_Delta = torch.zeros_like(weight_mat) # [output_shape, input_shape]
    
    output_not_empty = output_firecount.gt(0)
    input_not_empty = input_firecount.gt(0)

    # 判断是否有输入脉冲没输出脉冲
    output_false_input_true = torch.zeros_like(w_stdp_Delta)
    # V_{ij} = !o_{i} & o_{j}
    output_false_input_true = cmul(radd(output_false_input_true, 1. - output_not_empty.float()), input_not_empty.float()) # [output_shape, input_shape]

    # 若有输入没输出，则将权重差值设为0.1
    w_stdp_Delta = w_stdp_Delta + 0.1 * output_false_input_true # [output_shape, input_shape]

    # 计算STDP
    stdp_A_pos = 2
    stdp_A_neg = -0.5
    stdp_tau_pos = 64.
    stdp_tau_neg = 64.

    # Δt
    delta_t = torch.zeros_like(w_stdp_Delta) # [output_shape, input_shape]
    # Δt_{ij} = t_{i} - (t_{j} + t_{ref})
    delta_t = csub(radd(delta_t, time_posts), time_pres) - t_ref # [output_shape, input_shape]

    delta_t_positive = delta_t.gt(0).float() # [output_shape, input_shape]
    delta_t_positive = delta_t_positive * (1. - output_false_input_true) # [output_shape, input_shape]
    delta_t_negative = delta_t.lt(0).float() # [output_shape, input_shape]
    delta_t_negative = delta_t_negative * (1. - output_false_input_true) # [output_shape, input_shape]

    # input_firecount + output_firecount
    firecount_sum = torch.zeros_like(w_stdp_Delta) # [output_shape, input_shape]
    # S_{ij} = o_{i} + i_{j}
    firecount_sum = cadd(radd(firecount_sum, output_firecount), input_firecount) # [output_shape, input_shape]

    w_stdp_Delta = w_stdp_Delta + delta_t_positive * stdp_A_pos * firecount_sum * torch.exp(-delta_t / stdp_tau_pos) # [output_shape, input_shape]
    w_stdp_Delta = w_stdp_Delta + delta_t_negative * stdp_A_neg * firecount_sum * torch.exp(delta_t / stdp_tau_neg) # [output_shape, input_shape]

    return w_stdp_Delta


@torch.jit.script
def double_stdp(weight_mat, time_pres, time_posts, input_firecount, output_firecount, tau_m, tau_s, t_ref):
    """
    利用STDP计算权重更新因子
    @params:
        weight_mat: torch.Tensor 权重矩阵，形状为[output_shape(O), input_shape(I)]
        input_spike_train: torch.Tensor 输入脉冲序列，形状为[time_steps(T), input_shape(I)]
        output_spike_train: torch.Tensor 输出脉冲序列，形状为[time_steps(T), output_shape(O)]
        input_firecount: torch.Tensor 输入脉冲发射数量统计，形状为[input_shape(I)]
        output_firecount: torch.Tensor 输出脉冲发射数量统计，形状为[output_shape(O)]
        tau_m: float 参数τm
        tau_s: float 参数τs
        t_ref: float 不应期长度
    @return:
        dw: torch.Tensor 权重更新因子，叠加到原权重上
    """
    w_stdp_Delta = torch.zeros_like(weight_mat) # [output_shape, input_shape]
    
    output_not_empty = output_firecount.gt(0)
    input_not_empty = input_firecount.gt(0)

    # 判断是否有输入脉冲没输出脉冲
    output_false_input_true = torch.zeros_like(w_stdp_Delta)
    # V_{ij} = !o_{i} & o_{j}
    output_false_input_true = cmul(radd(output_false_input_true, 1. - output_not_empty.float()), input_not_empty.float()) # [output_shape, input_shape]

    # 若有输入没输出，则将权重差值设为0.1
    w_stdp_Delta = w_stdp_Delta + 0.1 * output_false_input_true # [output_shape, input_shape]

    # 计算STDP
    stdp_A_pos = 0.04
    stdp_A_neg = -0.01
    stdp_tau_pos = 55.
    stdp_tau_neg = 55.

    # Δt
    delta_t = torch.zeros_like(w_stdp_Delta) # [output_shape, input_shape]
    # Δt_{ij} = t_{i} - (t_{j} + t_{ref})
    delta_t = csub(radd(delta_t, time_posts), time_pres) - t_ref # [output_shape, input_shape]

    delta_t_positive = delta_t.gt(0).float() # [output_shape, input_shape]
    delta_t_positive = delta_t_positive * (1. - output_false_input_true) # [output_shape, input_shape]
    delta_t_negative = delta_t.lt(0).float() # [output_shape, input_shape]
    delta_t_negative = delta_t_negative * (1. - output_false_input_true) # [output_shape, input_shape]

    # input_firecount + output_firecount
    firecount_sum = torch.zeros_like(w_stdp_Delta) # [output_shape, input_shape]
    # S_{ij} = o_{i} + i_{j}
    firecount_sum = cadd(radd(firecount_sum, output_firecount), input_firecount) # [output_shape, input_shape]

    w_stdp_Delta = w_stdp_Delta + delta_t_positive * stdp_A_pos * torch.exp(-delta_t / stdp_tau_pos) # [output_shape, input_shape]
    w_stdp_Delta = w_stdp_Delta + delta_t_negative * stdp_A_neg * torch.exp(delta_t / stdp_tau_neg) # [output_shape, input_shape]

    return w_stdp_Delta


@torch.jit.script
def d_Spiking_accumulate_effect(weight_mat, input_spike_train, output_spike_train, input_firecount, output_firecount, tau_m, tau_s, t_ref):
    """
    计算电位
    @params:
        weight_mat: torch.Tensor 权重矩阵，形状为[output_shape(O), input_shape(I)]
        input_spike_train: torch.Tensor 输入脉冲序列，形状为[time_steps(T), input_shape(I)]
        output_spike_train: torch.Tensor 输出脉冲序列，形状为[time_steps(T), output_shape(O)]
        input_firecount: torch.Tensor 输入脉冲发射数量统计，形状为[input_shape(I)]
        output_firecount: torch.Tensor 输出脉冲发射数量统计，形状为[output_shape(O)]
        tau_m: float 参数τm
        tau_s: float 参数τs
        t_ref: float 不应期长度
    @return:
        e:S-PSP 脉冲序列级突触后膜电势
    """
    E = torch.zeros_like(weight_mat) # [output_shape, input_shape]
    
    time_posts_arr = []
    for i in range(output_firecount.shape[0]): # [output_shape]
        # ti的集合，一维
        time_posts = torch.flatten(output_spike_train[:, i].nonzero())
        time_posts_arr.append(time_posts)

    time_pres_arr = []
    for j in range(input_firecount.shape[0]): # [input_shape]
        time_pres = torch.flatten(input_spike_train[:, j].nonzero())
        time_pres_arr.append(time_pres)
    
    for i in range(output_firecount.shape[0]): # [output_shape]
        time_posts = time_posts_arr[i] # [output_times]
        if not time_posts.shape[0]:
            E[i] = 0.5 * (input_firecount.gt(0).float())
            continue
        for j in range(input_firecount.shape[0]): # [input_shape]
            time_pres = time_pres_arr[j] # [input_times]
            if not time_pres.shape[0]:
                continue

            """
            计算s的矩阵S
            S_{ij} = t_{i} - t_{j}
            """
            S = torch.zeros(time_posts.shape[0], time_pres.shape[0]).to(E.device) # [output_times, input_times]
            S = csub(radd(S, time_posts), time_pres) # [output_times, input_times]

            """
            需要满足下列条件才会计入response:
            (1) t_{j} >= t_{i} - 4. * tau_m -> t_{i} - t_{j} <= 4. * tau_m
            (2) t_{j} < t_{i} -> t_{i} - t_{j} > 0.
            (3) t_{j} + t_ref <= t_{i} -> t_{i} - t_{j} >= t_ref
            """
            condition_1 = S.le(4. * tau_m) # [output_times, input_times]
            condition_2 = S.gt(0.) # [output_times, input_times]
            condition_3 = S.ge(t_ref) # [output_times, input_times]
            valid = torch.logical_and(torch.logical_and(condition_1, condition_2), condition_3).float() # [output_times, input_times]

            if not torch.sum(valid):
                continue

            """
            计算t的矩阵T
            T_{ij} = t_{i} - \hat{t}_{i}
            """
            T = torch.zeros(time_posts.shape[0], time_pres.shape[0]).to(E.device) # [output_times, input_times]
            hat_time_posts = torch.zeros_like(time_posts) # [output_times]
            if time_posts.shape[0] > 1:
                hat_time_posts[1:] = time_posts[:-1]
            T = radd(T, time_posts - hat_time_posts) # [output_times]

            """
            计算公式：
            e_{i|j} = \sum_{t_i}{\sum_{t_j}{ε(s, t)}}
            其中
            s = t_i - t_j
            t = t_i - \hat{t}_i
            ε(s,t) = e^{-max(t - s, 0) / tau_s} / (1 - tau_s / tau_m) * [e^{-min(s, t) / tau_m} - e^{-min(s, t) / tau_s}]
            """
            epsilon = torch.exp(-torch.max(T - S, torch.zeros_like(S)) / tau_s) / (1. - tau_s / tau_m) * (torch.exp(-torch.min(S, T) / tau_m) - torch.exp(-torch.min(S, T) / tau_s)) # [output_times, input_times]
            E[i, j] = torch.sum(epsilon * valid)
    
    return E

if __name__ == "__main__":
    """
    M = [
        1    2    3    4
        5    6    7    8
        9    10   11   12
    ] 3x4
    """
    M = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

    """
    o = [
        2
        4
        3
    ] 3
    """
    o = torch.tensor([2, 4, 3])

    """
    i = [
        2    3    4    5
    ] 4
    """
    i = torch.tensor([2, 3, 4, 5])

    print("M_{ij} + o_{i}")
    print(radd(M, o))
    print("")

    print("M_{ij} - o_{i}")
    print(rsub(M, o))
    print("")

    print("M_{ij} * o_{i}")
    print(rmul(M, o))
    print("")

    print("M_{ij} / o_{i}")
    print(rdiv(M, o))
    print("")

    print("M_{ij} + i_{j}")
    print(cadd(M, i))
    print("")

    print("M_{ij} - i_{j}")
    print(csub(M, i))
    print("")

    print("M_{ij} * i_{j}")
    print(cmul(M, i))
    print("")
    
    print("M_{ij} / i_{j}")
    print(cdiv(M, i))
    print("")