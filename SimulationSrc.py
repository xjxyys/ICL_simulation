import numpy as np
from scipy.stats import poisson
import random
import statsmodels.api as sm
import openai
from openai import OpenAI
import os
from dotenv import load_dotenv
from lifelines import KaplanMeierFitter


# 报童模型的脱销损失C1和滞销损失C2
C1 = 7
C2 = 4

# 设置代理，本机端口7890
# os.environ['http_proxy'] = 'http://127.0.0.1:7890'
# os.environ['https_proxy'] = 'https://127.0.0.1:7890'


class NewsboySimulator:
    def __init__(self, sample_mode='uniform'):
        self.DemandLambda = self.sampleLambda(sample_mode,1,10)
        self.OrderLambda = self.sampleLambda(sample_mode,1,20)

    def sampleLambda(self, sample_mode, lower_bound=1, upper_bound=10):
        if sample_mode == 'uniform':
            # Sample from a uniform distribution
            return np.random.uniform(low=lower_bound, high=upper_bound)
        
    def get_DemandLambda(self):
        return self.DemandLambda
    
    def get_OrderLambda(self):
        return self.OrderLambda
    
    def sample(self, Lambda, size: int) -> np.ndarray:
        # Sample from a Poisson distribution
        return np.random.poisson(Lambda, size)

    def simulate(self, days:int) -> list:
        # Initialize an empty list to store (order, demand) tuples
        order_demand_list = []
        
        # Generate samples for orders and demands
        orders = self.sample(self.get_OrderLambda(), days)
        demands = self.sample(self.get_DemandLambda(), days)
        
        # Generate (order, demand) tuples
        for day in range(days):
            order = orders[day]
            demand = demands[day]
            # If order is less than demand, observed demand is equal to order
            observed_demand = min(order, demand)
            order_demand_list.append((order, observed_demand))
        
        return order_demand_list

    def simulate_constOrder(self, days:int) -> list:
        """
        模拟报童模型，但是此时方便模型估计每天的订货量固定"""
        # Initialize an empty list to store (order, demand) tuples
        order_demand_list = []
        
        # Generate samples for demands
        demands = self.sample(self.get_DemandLambda(), days)
        order = self.sample(self.get_OrderLambda(), 1)[0]
        # Generate (order, demand) tuples
        for day in range(days):
            demand = demands[day]
            # If order is less than demand, observed demand is equal to order
            observed_demand = min(order, demand)
            order_demand_list.append((order, observed_demand))
        
        return order_demand_list
    
class Infer:
    def __init__(self, data):
        # [(order, observed_demand), ...]
        self.data = data
    
    def infer(self):
        """
        Infer the optimal order quantity using the observed data
        """
        pass

# 了解需求服从泊松分布的情况下的推断
class PoissonInfer(Infer):
    def estimate_lambda(self):
        # 只考虑观察到的需求等于订货量的天数
        filtered_demand = [observed_demand for order, observed_demand in self.data if order == observed_demand]
        # 如果过滤后没有数据，默认使用所有观察到的需求
        if not filtered_demand:
            filtered_demand = [observed_demand for _, observed_demand in self.data]
        # 计算平均需求作为λ的估计
        estimated_lambda = np.mean(filtered_demand)
        return estimated_lambda
    
    def grad_with_const_order(slef, x, Lambda, c, k):
        """
        计算每天订货量相同的情况下的梯度
        """
        numer = 0
        denom = np.exp(Lambda) # Initialize denom with e^Lambda
        # Precompute Lambda^j / j! for all j up to c
        tmp = [1] # tmp[0] = Lambda^0 / 0!

        for i in range(1, c+1):
            tmp.append(tmp[-1]*Lambda/i) # Compute Lambda^j / j! and store

        denom -= sum(tmp) # Subtract the sum of Lambda^j / j! terms from denom

        for i in range(0, c+1):
            numer += (i-Lambda)*tmp[i]/Lambda
            # numer += (i-Lambda)*tmp[i]
        # 从k+1到n的需求量的和
        return -k * numer/denom + np.sum(x[k:])/Lambda - (len(x) - k)

    def grad(self, x, Lambda, c, k):
        """
        计算每天订货量不同的情况下的梯度
        """
        n = len(x)
        total_sum_first_part = 0
        
        for i in range(1, k + 1):
            c_i = c[i - 1]  # Assuming c is a list with each c_i value
            numer = 0
            denom = np.exp(Lambda)  # Initialize denom with e^Lambda 
            tmp = [1]  # tmp[0] = Lambda^0 / 0!
            
            for j in range(1, c_i + 1):
                tmp.append(tmp[-1] * Lambda / j)  # Compute Lambda^j / j! from 1 to c_i and store
            
            denom -= sum(tmp)  # Subtract the sum of Lambda^j / j! terms from denom

            for j in range(c_i + 1):
                numer += (j - Lambda) * tmp[j] / Lambda # Compute the numerator
            
            total_sum_first_part += numer / denom
        
        sum_second_part = np.sum(x[k:n]) / Lambda - (n - k)
        
        return -total_sum_first_part + sum_second_part

    def estimate_lambda_with_censored_data(self, x, c, k, error=1e-4, eta=1e-2, rho=0):
        """
        估计泊松分布的λ，考虑了censored data
        参数:
        x -- 观察到的需求数据
        c -- 截尾值
        k -- 刚好被截尾的数据的个数
        error -- 误差
        eta -- 学习率
        rho -- 正则化参数
        """
        lambda0 = np.mean(x)
        tmp = 1
        step = 0
        while tmp > error:
            step += 1
            g = self.grad_with_const_order(x,lambda0,c,k)
            lambda0 = lambda0 + eta * (g - rho * lambda0)
            tmp = np.abs(g - rho * lambda0)
            if step % 1000 == 0:
                print(tmp)
                print(lambda0)
        return lambda0

    def estimate_lambda(self, x, c, k, error=1e-4, eta=1e-2, rho=0):
        """
        同上，不过c此时是一个列表，表示的截尾值
        """
        lambda0 = np.mean(x)
        tmp = 1
        step = 0
        while tmp > error:
            step += 1
            g = self.grad(x,lambda0,c,k)
            lambda0 = lambda0 + eta * (g - rho * lambda0)
            tmp = np.abs(g - rho * lambda0)
            if step % 1000 == 0:
                print(tmp)
                print(lambda0)
        return lambda0


    def infer_constOrder(self):
        # estimated_lambda = self.estimate_lambda()
        # 使用截尾数据估计λ
        const_order = self.data[0][0] # 订货量是固定的
        # 重新排列observed_demands，使得订货量等于观察到的需求的排在前面
        observed_demands = [observed_demand for _, observed_demand in self.data if observed_demand == const_order] + [observed_demand for _, observed_demand in self.data if observed_demand != const_order]
        censored_num = len([observed_demand for _, observed_demand in self.data if observed_demand == const_order])
        estimated_lambda = self.estimate_lambda_with_censored_data(observed_demands, const_order, censored_num)
        # 计算临界比率
        critical_ratio = C1 / (C1 + C2)
        # 找到使CDF(λ)等于临界比率的最小订货量
        optimal_order = poisson.ppf(critical_ratio, estimated_lambda)
        return optimal_order

    def infer(self):
        """
        这种推理的方式更加通用，可以不固定需求进行输入
        """
        # 提取observed_demands和order，使得订货量等于观察到的需求的排在前面，且分别与之对应
        orders = [order for order, observed_demand in self.data if observed_demand == order] + [order for order, observed_demand in self.data if observed_demand != order]
        observed_demands = [observed_demand for order, observed_demand in self.data if observed_demand == order] + [observed_demand for order, observed_demand in self.data if observed_demand != order]
        k = len([observed_demand for order, observed_demand in self.data if observed_demand == order])
        # check 前k个需求等于订货量
        if k >= 1:
            assert all([orders[i] == observed_demands[i] for i in range(k)]), "前k个需求不等于订货量"
        # 估计λ
        estimated_lambda = self.estimate_lambda(observed_demands, orders, k)
        # 计算临界比率
        critical_ratio = C1 / (C1 + C2)
        # 找到使CDF(λ)等于临界比率的最小订货量
        optimal_order = poisson.ppf(critical_ratio, estimated_lambda)
        return optimal_order

# 不了解需求服从泊松分布，利用经验分布进行推断   
class EmpiricalInfer(Infer):
    def infer(self):
        # 提取所有观察到的需求数据
        observed_demands = [observed_demand for _, observed_demand in self.data]
        
        # 构建经验累积分布函数
        ecdf = sm.distributions.ECDF(observed_demands)
        # 计算临界比率
        critical_ratio = C1 / (C1 + C2)  
        # 通过ECDF反查最佳订货量
        # 这是通过找到ECDF值刚好超过临界比率的最小需求量来完成的
        optimal_order = np.inf
        for demand in range(max(observed_demands) + 1):
            if ecdf(demand) >= critical_ratio:
                optimal_order = demand
                break
        return optimal_order
    
class KaplanMeierInfer(Infer):
    def estimate_survival_function(self):
        "Estimate the survival function using the Kaplan-Meier model"
        # Create arrays for the observed demands and a boolean for censoring
        survival_function = {}
        observations = sorted(self.data, key=lambda x: x[1])
        n = len(observations)
        censored = [d[0] == d[1] for d in observations]

        survival_prob = 1
        for i, (order, demand) in enumerate(observations):
            if not censored[i]:
                at_risk = n - i
                events = 1 # 当前点为事件发生点
                survival_prob *= (at_risk - events) / at_risk
            survival_function[demand] = survival_prob

        return survival_function

    def infer(self):
        """
        Infer the optimal order quantity using the Kaplan-Meier model
        """
        survival_function = self.estimate_survival_function()

        # convert the survival function to a cumulative distribution function
        # cdf = {}
        # prev_prob = 1
        # for demand, prob in survival_function.items():
        #     cdf[demand] = prev_prob - prob
        #     prev_prob = prob
        cdf = {demand: 1 - prob for demand, prob in survival_function.items()}
        
        # 计算临界比率
        critical_ratio = C1 / (C1 + C2)
        # 找到使生存函数等于临界比率的最小需求量
        optimal_order_quantity = None
        for demand, prob in sorted(cdf.items()):
            if prob >= critical_ratio:
                optimal_order_quantity = demand
                break

        return optimal_order_quantity

class ChatGPTInfer(Infer):

    def __init__(self, data, model="gpt-4-0125-preview"):
        self.data = data
        self.model = model
        
    def get_prompt(self):
    
        prompt = "Based on the historical order and demand data provided below, infer the optimal order quantity. "
        prompt += "The data consists of tuples representing the order quantity and the observed demand for each day: "
        prompt += str(self.data) + " "
        # 需要加上脱销损失和滞销损失
        prompt += "The cost of overstocking is " + str(C2) + " and the cost of understocking is " + str(C1) + "."
        prompt += "Please just output the optimal order quantity for next day."
        return prompt
    
    def get_prompt_new(self):
        """根据建议，用selling price和ordering cost的方式来构建prompt
        """
        prompt = "Based on the historical order and demand data provided below, infer the optimal order quantity. "
        prompt += "The data consists of tuples representing the order quantity and the observed demand for each day: "
        prompt += str(self.data) + " "
        # 使用销售价格和订货成本
        selling_price = C1 + C2  # 销售价格等于C1+C2
        ordering_cost = C2  # 订货成本等于过剩成本C2
        prompt += f"The selling price is {selling_price} and the ordering cost (overage cost) is {ordering_cost}. "
        prompt += "Please just output the optimal order quantity(a single number) for next day, omit the analysis process!"
        return prompt
    
    def get_prompt_cot(self):
        """告诉GPT这是报童模型，并利用COT的方式，让GPT think step by step
        """
        prompt = "Utilize the provided historical order and demand data to deduce the optimal order quantity according to the **newsboy model**. "
        # prompt += "Based on the historical order and demand data provided below, infer the optimal order quantity. "
        prompt += "The data consists of tuples representing the order quantity and the observed demand for each day: "
        prompt += str(self.data) + " "
        # 使用销售价格和订货成本
        selling_price = C1 + C2  # 销售价格等于C1+C2
        ordering_cost = C2  # 订货成本等于过剩成本C2
        prompt += f"The selling price is {selling_price} and the ordering cost (overage cost) is {ordering_cost}. "
        prompt += "You should think step by step and output the reasoning process."
        # think step by step输出推理过程，但是最后一行只输出最优订货量
        prompt += "Pay attention to the last line, which should only output the optimal order quantity for the next day(a single number)!"
        return prompt

    def infer_cot(self):
        client = OpenAI()
        prompt = self.get_prompt_cot()
        response = client.chat.completions.create(
        model = self.model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", 'content': prompt}
        ]
        )
        return response.choices[0].message.content
    
    def infer(self):
        client = OpenAI()
        # prompt = self.get_prompt()
        prompt = self.get_prompt_new()
        response = client.chat.completions.create(
        model = self.model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", 'content': prompt}
        ]
        )
        return response.choices[0].message.content
    
def get_average_cost(order, DemandLambda):
    """
    计算给定订货量、需求率和成本条件下的平均成本。
    
    参数:
    order -- 订货量
    DemandLambda -- 需求的泊松分布率（平均需求量）
    C1 -- 单位脱销成本
    C2 -- 单位滞销成本
    返回:
    avg_cost -- 平均成本
    """
    # 初始化平均成本
    avg_cost = 0
    # 计算最大可能需求，这里我们取3倍的平均需求，通常足以涵盖大部分概率
    max_demand = int(3 * DemandLambda)
    # 对所有可能的需求量求和
    for demand in range(max_demand + 1):
        # 计算泊松分布在该需求量的概率
        probability = poisson.pmf(demand, DemandLambda)
        # 如果需求大于订单，计算脱销成本
        if demand > order:
            cost = C1 * (demand - order)
        # 否则计算滞销成本
        else:
            cost = C2 * (order - demand)
        # 将概率加权的成本加到总成本中
        avg_cost += probability * cost
    return avg_cost