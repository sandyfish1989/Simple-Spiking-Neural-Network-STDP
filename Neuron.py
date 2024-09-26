from Parameters import Parameters

class Neuron:
    def __init__(self, parameters: Parameters):
        self.parameters = parameters

        self.adaptive_spike_threshold = None # 适应性发放阈值，表示神经元发放（产生动作电位）的电压门限
        self.refractory_period = None # 不应期，表示神经元发放动作电位后，在这段时间内无法再次发放。这里默认初始化为 15 微秒。
        self.potential = None  # 神经元的膜电位，初始化为休息状态的电位. 默认给的是-70
        self.rest_until = None  # 表示神经元在何时恢复正常活动，通常是通过记录时间步 time_step 来判断。初始为 -1，表示神经元目前不处于抑制状态。

        self.initial()


    # 模拟神经元的超极化状态。超极化是指膜电位低于其静息电位，可能由抑制性输入引起
    # 该函数在神经元受到抑制或需要调整膜电位的场景下使用，使神经元在一段时间内无法发放新的动作电位。
    def hyperpolarization(self, time_step):
        self.potential = self.parameters.hyperpolarization_potential  # 将神经元的膜电位设置为超极化电位，即低于静息电位的水平。
        self.rest_until = time_step + self.refractory_period  # 设置神经元的恢复时间，表示它在接下来的 refractory_period 时间内将保持不活跃状态。


    # 抑制函数
    # 用于模拟神经元的抑制状态，与超极化类似，抑制是指降低神经元发放动作电位的能力
    def inhibit(self, time_step):
        self.potential = self.parameters.inhibitory_potential  # 将膜电位设置为抑制电位，通常是低于静息电位的一个电压值。
        self.rest_until = time_step + self.refractory_period  # 设置恢复时间，表示神经元在 refractory_period 时间内处于抑制状态。

    def initial(self):
        self.adaptive_spike_threshold = self.parameters.spike_threshold
        self.rest_until = -1
        self.refractory_period = 15  # (us)
        self.potential = self.parameters.resting_potential
