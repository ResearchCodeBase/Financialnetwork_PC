# 导入库
import argparse
from datetime import datetime
import csv
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from pulp import *
import matplotlib.pyplot as plt
# 比3_test增加了name索引
import pandas as pd




# 定义一个函数来读取数据
def load_data(L_path, e_path, name_path):
    l = np.loadtxt(L_path)
    e = np.loadtxt(e_path)
    with open(name_path, encoding='utf-8') as file:
        names = np.array(file.read().split('\t'))
    return l, e, names

# 定义函数来设置债务矩阵
def set_debt_matrix(l, n):
    L = np.sum(l, axis=1)
    π = np.zeros((n, n))
    # 设置债务矩阵
    for i in range(n):
      for j in range(n):
        if L[i] > 0:
          if l[i, j] > 0:
            π[i, j] = l[i, j] / L[i]
          else:
            π[i, j] = 0
        else:
          π[i, j] = 0
    return π

'''更新偿付能力银行组'''
def update_solvent_banks(P, S, l):
  # 提取数据
  P_new = P.copy()
  P_new[S, :] = l[S, :]
  return P_new


'''更新破产银行组'''
def solve_insolvent_banks(l, π, S, I, e, P,alpha,beta):
  n = len(I)

  # 参数
  # alpha是 外部资产的面值份额 可能会很低 （该行出售贷款组合，可能低价出售）
  # beta 银行间资产的面值份额  接近1 （有偿付能力的银行会全部支付）

  P_new = P.copy()
  e_new = e.copy()

  '''定义线性规划问题'''
  # 定义决策变量A b
  A = np.zeros((I.size, I.size))
  b = np.zeros(I.size)
  # 求解线性方程组Ax=b 的系数A
  for index1, value1 in enumerate(I):
      for index2, value2 in enumerate(I):
          if (index1 == index2):
              A[index1][index2] = 1 - beta * π[value1][value1]
          else:
              A[index1][index2] = -beta * π[value2][value1]
  # print('A\n',A)
  # 求解线性方程组Ax=b 的b
  for index, value in enumerate(I):
      if(e[value] < 0):
        b[index] =  e[value]+beta * l[S, value].sum()
      else:
        b[index] = e[value] * alpha + beta * l[S, value].sum()
  # print('B\n',b)

  x = np.linalg.solve(A, b)
  # print("Solution\n",x)

  # 验证线性方程组的解
  # print("Check\n",np.dot(A,x))

  # 更新
  for index, value in enumerate(I):
      P_new[value, :] = x[index] * π[value, :]
      if(e[value]<0):
        e_new[value] = e[value]
      else:
        e_new[value] = e[value] * alpha
  return P_new, e_new
# 算法主体
def compute_GVA(l, π, e,n,path,alpha,beta,names):

  # 参数
  v = np.zeros((n,1))
  # 初始化清算向量
  P = l.copy()
  epoch = 0
  # 添加变量存储P.sum
  liquidities = []
  updated = True
  # 获取当前时间并格式化为字符串
  L = np.sum(l, axis=1)
  print( 'alpha ={},beta={}',alpha,beta)
  # 创建保存结果的CSV文件
  # 在循环开始前初始化两个变量来保存第一个和最后一个epoch的破产银行索引
  # 创建一个与银行总数相同长度的全为0的列表
  first_epoch_bank_status = [0] * n
  last_epoch_bank_status = [0] * n
  while updated == True:
    # 银行间资产按列求和
    v = np.sum(P, axis=0)

    # 净资产 = 银行间资产 + 外部资产 - 银行间负债 注意这里的负债是最初的负债
    v_result = v + e - L
    print('*************************************')
    print('epoch:', epoch)
    print('v[125] 银行间资产',v[125])


    print('L[125] 银行间负债',L[125])
    print('e[125] 外部资产',e[125])
    print('v_result 净资产',v_result[125])
    # 划分银行组 np.where都会返回一个tuple,不是ndarray
    I = np.where(v_result <= 0)[0]
    S = np.where(v_result > 0)[0]

    # 更新偿付能力银行组、破产银行组的清算向量、注意不止更新P , e 也要更新
    P = update_solvent_banks(P, S, l)
    P, e = solve_insolvent_banks(l, π, S, I, e, P,alpha,beta)

    # 判断是否达到迭代终止条件，即破产银行集合没有再更新
    if (epoch != 0):
      if (I.size == prev_I.size):
        if ((prev_I == I).all() and (abs(P.sum() - prev_P.sum()) < 0.005)):
          print('P.sum() - prev_P.sum()',abs(P.sum() - prev_P.sum()))
          updated = False

    prev_I = I
    prev_P = P

    # 检查是否是第一个epoch，并保存破产银行索引
    if epoch == 0:
      # 将破产银行的位置标记为1
      first_epoch_bankrupt_banks = I.copy()
      for bank in first_epoch_bankrupt_banks:
        first_epoch_bank_status[bank] = 1


    print('破产银行集合', names[I])
    print('破产银行集合', I)
    print('破产银行数',len(I))
    print('净资产', v_result.sum())
    print('正常的银行集合', names[S])
    print('正常的银行集合', S)
    print('总流动性', P.sum())
    L_result = np.sum(P, axis=1)
    liquidities.append(P.sum())
    # Create a list of 0s and 1s based on the bankrupt bank set for each epoch
    bank_list = [1 if bank in I else 0 for bank in range(n)]

    # Append the epoch number and transposed bank list values to the CSV file


    epoch += 1
    # 在循环末尾更新最后一个epoch的破产银行索引
  last_epoch_bankrupt_banks = I.copy()
  for bank in last_epoch_bankrupt_banks:
    last_epoch_bank_status[bank] = 1

  # 获取负债矩阵文件的目录路径
  l_file_directory = os.path.dirname(path)

  # 获取当前时间，并格式化为字符串（例如：20240102_153045）
  current_time = datetime.now().strftime('%Y%m%d_%H%M%S')

  # 使用时间戳构建文件名
  output_filename = os.path.join(l_file_directory, f'违约标签_{current_time}.csv')

  with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
      writer = csv.writer(csvfile)
      writer.writerow(['First Epoch Bankrupt Banks', 'Last Epoch Bankrupt Banks'])
      # 为什么n-1 因为要去除其他银行的标签
      for i in range(n - 1):
          writer.writerow([first_epoch_bank_status[i], last_epoch_bank_status[i]])
      print('保存违约标签成功', output_filename)

  # 保存P值，文件名也加入当前时间戳
  np.savetxt(f'P_values_{current_time}.txt', P)


  return P,liquidities,output_filename

def plot_liquidity_over_iterations(liquidities, alpha, beta):
    """
    绘制流动性随迭代变化的曲线图。
    :param liquidities: 每次迭代后的总流动性列表。
    :param alpha: 参数alpha的值。
    :param beta: 参数beta的值。
    """
    plt.plot(liquidities)
    plt.title("Total Liquidity over Iterations")
    plt.text(1, 1, f"alpha = {alpha}, beta = {beta}", transform=plt.gca().transAxes)
    plt.xlabel('Iterations')
    plt.ylabel('Total Liquidity')
    plt.show()


# 主函数
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process Banking Network Features')
    parser.add_argument('--L_path', type=str, default='../data/2022-182/2022全-负债矩阵（有其他银行).txt', help='Path to the liability matrix file')
    parser.add_argument('--e_path', type=str, default='../data/2022-182/2022全-外部资产.txt', help='Path to the external assets file')
    parser.add_argument('--name_path', type=str, default='../data/2022-182/2022全-银行名字.txt', help='Path to the bank names file')
    parser.add_argument('--alpha', type=float, default='0.5',
                        help='Path to the bank names file')
    parser.add_argument('--beta', type=float, default='0.8')

    args = parser.parse_args()

    # 加载数据
    l, e, names = load_data(args.L_path, args.e_path, args.name_path)
    n = l.shape[0] # 银行总数
    π = set_debt_matrix(l, n)
    # 创建输出目录
    output_dir = os.path.dirname(args.L_path)
    os.makedirs(output_dir, exist_ok=True)
    # 执行GVA算法 output_filename是违约标签csv文件
    P, liquidities,output_filename = compute_GVA(l, π, e, n, args.L_path,args.alpha, args.beta,names)

    plot_liquidity_over_iterations(liquidities, args.alpha, args.beta)