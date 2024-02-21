import numpy as np
from scipy.stats import poisson
import random
import statsmodels.api as sm
import openai
from openai import OpenAI
import os
from dotenv import load_dotenv

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

class Infer:
    def __init__(self, data):
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
    
    def infer(self):
        estimated_lambda = self.estimate_lambda()
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
    
class ChatGPTInfer(Infer):

    def __init__(self, data, model="gpt-4-0125-preview"):
        self.data = data
        self.model = model
        
    def get_prompt(self):
    
        prompt = "Based on the historical order and demand data provided below, infer the optimal order quantity. "
        prompt += "The data consists of tuples representing the order quantity and the observed demand for each day: "
        prompt += str(self.data) + " "
        # 需要加上脱销损失和滞销损失
        prompt += "The cost of overstocking is " + str(C1) + " and the cost of understocking is " + str(C2) + "."
        prompt += "Please just output the optimal order quantity for next day."
        return prompt
    
    def infer(self):
        client = OpenAI()
        prompt = self.get_prompt()
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