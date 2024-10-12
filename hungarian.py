import numpy as np

def pad_cost_matrix(cost_matrix):
    """
    如果矩阵不是方阵，将其扩展为方阵。
    使用非常大的值填充虚拟的行或列，使其不会被选中。
    """
    n, m = cost_matrix.shape
    if n == m:
        return cost_matrix  # 如果已经是方阵，直接返回
    
    # 扩展为方阵
    if n < m:
        # 行少列多，添加虚拟行
        padding = np.full((m - n, m), np.max(cost_matrix) * 10)
        cost_matrix = np.vstack((cost_matrix, padding))
    elif n > m:
        # 行多列少，添加虚拟列
        padding = np.full((n, n - m), np.max(cost_matrix) * 10)
        cost_matrix = np.hstack((cost_matrix, padding))
    
    return cost_matrix

def hungarian_algorithm(cost_matrix):
    """
    使用匈牙利算法解决最小化代价的指派问题。
    
    参数:
        cost_matrix: 输入的 n x m 代价矩阵
    返回:
        最优匹配的总代价和匹配方案
    """
    cost_matrix = np.array(cost_matrix)
    origin_cost_matrix = cost_matrix.copy()
    n, m = cost_matrix.shape

    # 扩展矩阵为方阵
    cost_matrix = pad_cost_matrix(cost_matrix)
    n, m = cost_matrix.shape
    
    for i in range(n):
        cost_matrix[i] -= np.min(cost_matrix[i])
    
    for j in range(m):
        cost_matrix[:, j] -= np.min(cost_matrix[:, j])
    
    star_matrix = np.zeros_like(cost_matrix, dtype=bool)
    prime_matrix = np.zeros_like(cost_matrix, dtype=bool)
    row_covered = np.zeros(n, dtype=bool)
    col_covered = np.zeros(m, dtype=bool)
    
    for i in range(n):
        for j in range(m):
            if cost_matrix[i, j] == 0 and not row_covered[i] and not col_covered[j]:
                star_matrix[i, j] = True
                row_covered[i] = True
                col_covered[j] = True
    
    row_covered[:] = False
    col_covered[:] = False
    
    def cover_columns_with_stars():
        for i in range(n):
            for j in range(m):
                if star_matrix[i, j]:
                    col_covered[j] = True

    cover_columns_with_stars()
    
    def find_zero():
        for i in range(n):
            if not row_covered[i]:
                for j in range(m):
                    if cost_matrix[i, j] == 0 and not col_covered[j]:
                        return i, j
        return None, None

    def find_star_in_row(row):
        for j in range(m):
            if star_matrix[row, j]:
                return j
        return None

    def find_prime_in_row(row):
        for j in range(m):
            if prime_matrix[row, j]:
                return j
        return None

    def augment_path(path):
        for r, c in path:
            star_matrix[r, c] = not star_matrix[r, c]

    def adjust_cost_matrix():
        min_uncovered_value = np.min(cost_matrix[~row_covered][:, ~col_covered])
        for i in range(n):
            if row_covered[i]:
                cost_matrix[i] += min_uncovered_value
        for j in range(m):
            if not col_covered[j]:
                cost_matrix[:, j] -= min_uncovered_value
    
    while np.sum(col_covered) < min(n, m):
        row, col = find_zero()
        if row is None:
            adjust_cost_matrix()
            row, col = find_zero()

        prime_matrix[row, col] = True
        star_col = find_star_in_row(row)
        if star_col is None:
            path = [(row, col)]
            while True:
                star_row = None
                for r in range(n):
                    if star_matrix[r, path[-1][1]]:
                        star_row = r
                        break
                if star_row is None:
                    break
                path.append((star_row, path[-1][1]))

                prime_col = find_prime_in_row(path[-1][0])
                path.append((path[-1][0], prime_col))
            
            augment_path(path)
            prime_matrix[:] = False
            row_covered[:] = False
            col_covered[:] = False
            cover_columns_with_stars()
        else:
            row_covered[row] = True
            col_covered[star_col] = False

    total_cost = 0
    result = []
    for i in range(len(origin_cost_matrix)):
        for j in range(len(origin_cost_matrix[0])):
            if star_matrix[i, j]:
                result.append((i, j))
                total_cost += origin_cost_matrix[i, j]
    return total_cost, result

def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    total_cost, result = hungarian_algorithm(cost_matrix)
    x, y = [], []
    n, m = cost_matrix.shape
    matches, unmatched_a, unmatched_b = [], [i for i in range(n)], [i for i in range(m)]
    a_i, b_i = [], []
    for i, j in result:
        if cost_matrix[i, j] <= thresh:
            x.append(i)
            y.append(j)
            matches.append((i, j))
            a_i.append(i)
            b_i.append(j)
    unmatched_a = list(set(unmatched_a) - set(a_i))
    unmatched_b = list(set(unmatched_b) - set(b_i))
    return matches, unmatched_a, unmatched_b
