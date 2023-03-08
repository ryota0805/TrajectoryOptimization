from scipy.optimize import minimize
from objective_function import objective_function  # objective_function.pyから目的関数をインポート

# 初期値
x0 = [0, 0]

# minimize関数を呼び出し
result = minimize(objective_function, x0, method='SLSQP')

# 結果を表示
print(result)
