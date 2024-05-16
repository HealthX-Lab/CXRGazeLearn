from scipy.stats import ttest_rel
import os

# 定义读取数据的函数
def read_data(file_path):
    with open(file_path, 'r') as file:
        data = [float(line.strip()) for line in file.readlines()]
    return data

# 文件路径基础部分
base_path = "/home/ziruiqiu/MscStudy/MT-UNet/stat_test/"

# data_mine_kl = read_data(os.path.join(base_path, "mine/KL_list.txt"))
# data_baseline_kl = read_data(os.path.join(base_path, "baseline/KL_list.txt"))

# data_mine_cc = read_data(os.path.join(base_path, "mine/CC_list.txt"))
# data_baseline_cc = read_data(os.path.join(base_path, "baseline/CC_list.txt"))

# data_mine_hs = read_data(os.path.join(base_path, "mine/HS_list.txt"))
# data_baseline_hs = read_data(os.path.join(base_path, "baseline/HS_list.txt"))


# data_mine_kl = read_data(os.path.join(base_path, "mine/KL_list.txt"))
# data_baseline_kl = read_data(os.path.join(base_path, "res_se/KL_list.txt"))

# data_mine_cc = read_data(os.path.join(base_path, "mine/CC_list.txt"))
# data_baseline_cc = read_data(os.path.join(base_path, "res_se/CC_list.txt"))

# data_mine_hs = read_data(os.path.join(base_path, "mine/HS_list.txt"))
# data_baseline_hs = read_data(os.path.join(base_path, "res_se/HS_list.txt"))

data_mine_kl = read_data(os.path.join(base_path, "mine/KL_list.txt"))
data_baseline_kl = read_data(os.path.join(base_path, "unets/KL_list.txt"))

data_mine_cc = read_data(os.path.join(base_path, "mine/CC_list.txt"))
data_baseline_cc = read_data(os.path.join(base_path, "unets/CC_list.txt"))

data_mine_hs = read_data(os.path.join(base_path, "mine/HS_list.txt"))
data_baseline_hs = read_data(os.path.join(base_path, "unets/HS_list.txt"))

# 执行Paired Samples t-test
t_statistic_kl, p_value_kl = ttest_rel(data_mine_kl, data_baseline_kl)
t_statistic_cc, p_value_cc = ttest_rel(data_mine_cc, data_baseline_cc)
t_statistic_hs, p_value_hs = ttest_rel(data_mine_hs, data_baseline_hs)
print("KLD: t-statistic =", t_statistic_kl, "p-value =", p_value_kl)
print("CC: t-statistic =", t_statistic_cc, "p-value =", p_value_cc)
print("HS: t-statistic =", t_statistic_hs, "p-value =", p_value_hs)

# 单侧检验的p值调整
# 假设: 您的模型表现不低于baseline
p_value_kl_one_sided = p_value_kl / 2 if t_statistic_kl < 0 else 1 - (p_value_kl / 2)
p_value_cc_one_sided = p_value_cc / 2 if t_statistic_cc > 0 else 1 - (p_value_cc / 2)
p_value_hs_one_sided = p_value_hs / 2 if t_statistic_hs > 0 else 1 - (p_value_hs / 2)

print("KLD: t-statistic =", t_statistic_kl, "one-sided p-value =", p_value_kl_one_sided)
print("CC: t-statistic =", t_statistic_cc, "one-sided p-value =", p_value_cc_one_sided)
print("HS: t-statistic =", t_statistic_hs, "one-sided p-value =", p_value_hs_one_sided)
