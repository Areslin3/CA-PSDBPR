import random
import numpy as np
import math
import time
import argparse
from collections import defaultdict
from sklearn.cluster import KMeans


def formula_3_perturbation(y_ui, epsilon):
    """
    [算法实现：公式 3] 1-bit 随机响应机制 (Randomized Response)
    用于对二值交互数据进行本地差分隐私（LDP）扰动。
    """
    exp_eps = np.exp(epsilon)
    denominator = exp_eps + 1
    coefficient_num = exp_eps - 1.0

    # 计算扰动概率 Pr[y'=1]
    if y_ui == 1:
        prob_y_prime_is_1 = (1.0 / denominator) + (1.0 * coefficient_num / denominator)
    else:
        prob_y_prime_is_1 = (1.0 / denominator) + (0.0 * coefficient_num / denominator)

    if random.random() < prob_y_prime_is_1:
        return 1
    else:
        return 0


def formula_4_estimation(y_prime, epsilon):
    """
    [算法实现：公式 4] 差分隐私值的无偏估计 (Unbiased Estimation)
    对扰动后的比特流进行重构，以消除随机响应引入的偏差。
    """
    exp_eps = np.exp(epsilon)
    numerator = (y_prime * (exp_eps + 1.0)) - 1.0
    denominator = exp_eps - 1.0
    return numerator / denominator


def Lu_LDP(user_item_counts, Sort, userNum, itemNum, sortNum, epsilon):
    """
    [LDP 模块] 构建本地差分隐私保护下的类别偏好向量
    实现流程：原始交互提取 -> 随机响应扰动 -> 无偏估计重构 -> 类别特征聚合。
    """
    lu_vectors = {}

    for u_idx in range(1, userNum + 1):
        user_id_str = str(u_idx)
        category_vector = [0.0] * sortNum
        u_data = user_item_counts.get(user_id_str, {})

        for i_idx in range(1, itemNum + 1):
            cat_id = Sort.get(i_idx)
            if cat_id is None:
                continue
            cat_index = int(cat_id) - 1
            n_ui = int(u_data.get(str(i_idx), 0))

            if n_ui > 0:
                # 针对多次签到行为进行拆解处理
                for _ in range(n_ui):
                    y_p = formula_3_perturbation(1, epsilon)
                    est_val = formula_4_estimation(y_p, epsilon)
                    category_vector[cat_index] += est_val
            else:
                # 针对未观测样本进行扰动处理
                y_p = formula_3_perturbation(0, epsilon)
                est_val = formula_4_estimation(y_p, epsilon)
                category_vector[cat_index] += est_val

        lu_vectors[user_id_str] = category_vector
    return lu_vectors


def add_laplace_noise(data, sigma=0.5):
    """
    [隐私保护：公式 6] 拉普拉斯机制 (Laplace Mechanism)
    为梯度向量添加拉普拉斯噪声，实现梯度级别的差分隐私保护。
    """
    scale = sigma / np.sqrt(2)
    if isinstance(data, np.ndarray):
        noise = np.random.laplace(0, scale, data.shape)
    else:
        noise = np.random.laplace(0, scale)
    return data + noise


def K_means_fixed_constrained(lu_vectors, n_clusters=20):
    """
    [用户聚类] 基于偏好向量的 K-Means 聚类
    用于识别潜在的社会学邻居（Cluster-based Neighbors），为梯度交换提供拓扑结构。
    """
    X = np.array(list(lu_vectors.values()))
    users = list(lu_vectors.keys())

    print(f"聚类配置: 固定 K={n_clusters}, 标准 K-Means (无约束)")

    # 执行 K-Means 聚类
    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
    kmeans.fit(X)
    labels = kmeans.labels_

    # 构建邻居映射集合
    label_to_users = defaultdict(list)
    for user_id, label in zip(users, labels):
        label_to_users[label].append(int(user_id))

    Nu = {}
    for user_id, label in zip(users, labels):
        # 邻居定义为同簇内的其他用户
        neighbors = list(label_to_users[label])
        if int(user_id) in neighbors:
            neighbors.remove(int(user_id))
        Nu[int(user_id)] = neighbors

    # 打印聚类统计信息
    final_sizes = [len(label_to_users[l]) for l in label_to_users]
    if final_sizes:
        print(f"聚类统计: 最小簇={min(final_sizes)}, 最大簇={max(final_sizes)}")

    return Nu


def sigmoid(x):
    """数值稳定的 Sigmoid 激活函数"""
    if x >= 0:
        return 1.0 / (1 + np.exp(-x))
    else:
        return np.exp(x) / (1 + np.exp(x))


def favorite(Wc, userNum, sortNum, categoryNum):
    """根据权重矩阵 Wc 为用户筛选 Top-k 个偏好类别"""
    s = [[Wc[str(i + 1)][str(j + 1)] for j in range(sortNum)] for i in range(userNum)]
    argsorts = np.argsort(s)
    Ns = []
    for i in range(userNum):
        Ns.append([argsorts[i][j - categoryNum] + 1 for j in range(categoryNum)])
    return Ns


def predictRating(pu, qi, data):
    """计算预测评分：用户潜在向量与物品潜在向量的点积"""
    predictedRating = np.dot(pu, qi)
    if data == 1:
        maxR, minR = 1, 0
    else:
        maxR, minR = 5, 1
    return min(max(predictedRating, minR), maxR)


def computeError(realRating, predictedRating):
    """计算评分误差"""
    return realRating - predictedRating


def CategoryWeight(path, userNum, sortNum, L, trainSort, Lc, n):
    """
    [算法实现：公式 2] 类别偏好权重计算 (基于 TF-IDF 思想)
    计算用户对不同类别的显式偏好强度。
    """
    Wc = defaultdict(lambda: defaultdict(float))
    for i in range(userNum):
        user = str(i + 1)
        for j in range(sortNum):
            Wc[user][str(j + 1)] = 0
            if str(j + 1) in trainSort[user]:
                nc = trainSort[user][str(j + 1)]
                if n[user] > 0 and Lc[j + 1] > 0:
                    a = 2 * sigmoid(nc / n[user] * math.log(L / Lc[j + 1])) - 1
                    Wc[user][str(j + 1)] = float(format(a, '.2g'))
    return Wc


def sort_1(path):
    """加载物品-类别映射关系表"""
    list_data = np.genfromtxt(path, dtype=int, skip_header=0, delimiter='\t', usecols=(1, 2))
    return {line[0]: line[1] for line in list_data}


def sort_2(path, sortNum):
    """构建类别-物品集合的反向索引"""
    List = np.genfromtxt(path, dtype=int, skip_header=0, delimiter='\t', usecols=(1, 2))
    sort_item = defaultdict(list)
    Lc = {}
    for line in List:
        sort_item[str(line[1])].append(str(line[0]))
    for i in range(sortNum):
        sort_item[str(i + 1)] = set(sort_item[str(i + 1)])
        Lc[i + 1] = len(sort_item[str(i + 1)])
    return sort_item, Lc


def computePrecision(test4Rank, userVectors, itemComVectors, itemNum, data, itemBias, Sort, Wc):
    """评估模块：计算 Precision, Recall, F1, MAP 以及 MRR 指标"""
    precision5 = recall5 = precision10 = recall10 = f1score5 = f1score10 = ap_total = mrr_s = 0.0
    for userID in test4Rank:
        ratings = {}
        for itemID in range(itemNum):
            userVector = userVectors[userID - 1]
            itemVector = itemComVectors[userID - 1][itemID]
            predictedRating = predictRating(userVector, itemVector, data) + itemBias[userID - 1][itemID]
            ratings[itemID + 1] = predictedRating

        sortedRating = sorted(ratings, key=lambda k: ratings[k], reverse=True)
        items4Rank = [int(x.split(',')[0]) for x in test4Rank[userID]]

        # MAP (Mean Average Precision) 计算
        ap = 0.0
        hits = 0
        for rank, item_predict in enumerate(sortedRating[:10], 1):
            if item_predict in items4Rank:
                hits += 1
                ap += hits / rank
        ap_total += ap / len(items4Rank) if len(items4Rank) > 0 else 0

        # Precision & Recall @ K 计算
        hit5 = sum(1 for item in sortedRating[:5] if item in items4Rank)
        precision5 += hit5 / 5.0
        recall5 += hit5 / len(items4Rank)

        hit10 = sum(1 for item in sortedRating[:10] if item in items4Rank)
        precision10 += hit10 / 10.0
        recall10 += hit10 / len(items4Rank)

        # MRR (Mean Reciprocal Rank) 计算
        for rank, item_predict in enumerate(sortedRating, 1):
            if item_predict in items4Rank:
                mrr_s += 1.0 / rank
                break

    num_users = len(test4Rank)
    precision5 /= num_users
    recall5 /= num_users
    precision10 /= num_users
    recall10 /= num_users
    f1score5 = 2 * recall5 * precision5 / (recall5 + precision5 + 1e-10)
    f1score10 = 2 * recall10 * precision10 / (recall10 + precision10 + 1e-10)
    map_score = ap_total / num_users
    mrr = mrr_s / num_users

    return precision5, recall5, precision10, recall10, f1score5, f1score10, map_score, mrr


def initLatentVectors(rows, columns):
    """正态分布初始化用户潜在因子矩阵"""
    return np.random.randn(rows, columns) / math.sqrt(columns)


def initItemComVectors(userNum, itemNum, featureK):
    """初始化各用户的本地物品潜在因子矩阵备份"""
    itemVectors4Users = {}
    tmp = np.random.randn(itemNum, featureK) / math.sqrt(featureK)
    for user in range(userNum):
        itemVectors4Users[user] = tmp.copy()
    return itemVectors4Users


def initBiases(userNum, itemNum, featureK):
    """初始化物品偏置项"""
    itemBiases4Users = {}
    tmp = np.random.rand(itemNum) / math.sqrt(featureK)
    for user in range(userNum):
        itemBiases4Users[user] = tmp.copy()
    return itemBiases4Users


def train(userNum, itemNum, cityUsers, featureK, trainSet, testSet, testRank, epochs, LambdaU, LambdaV, LambdaZ, alpha,
          lrDecay, filePrefix, isSave, data, negativeNum, log, Nu, trainitem, Sort, Wc, Select, Rest,
          sigma, N1):
    """
    CA-PSDBPR 主训练流程
    包含：BPR 负采样更新、拉普拉斯梯度扰动、簇内邻居梯度共享。
    """
    userVectors = initLatentVectors(userNum, featureK)
    itemComVectors = initItemComVectors(userNum, itemNum, featureK)
    itemBias = initBiases(userNum, itemNum, featureK)

    print(f"训练开始... ")

    totalStart = time.time()
    for epoch in range(epochs):
        random.shuffle(trainSet)
        start = time.time()

        loss_total = 0
        for row in trainSet:
            userID = int(row['userID'])
            i = int(row['itemID'])

            if str(userID) not in Select or not Select[str(userID)]: continue
            if str(userID) not in Rest or not Rest[str(userID)]: continue

            for _ in range(negativeNum):
                # 三元组采样：正样本 (i), 偏好负样本 (p), 非偏好负样本 (j)
                p = int(random.choice(Select[str(userID)]))
                j = int(random.choice(Rest[str(userID)]))

                userVector = userVectors[userID - 1]
                itemComVectori = itemComVectors[userID - 1][i - 1]
                itemComVectorp = itemComVectors[userID - 1][p - 1]
                itemComVectorj = itemComVectors[userID - 1][j - 1]

                r_ui = np.dot(userVector, itemComVectori) + itemBias[userID - 1][i - 1]
                r_up = np.dot(userVector, itemComVectorp) + itemBias[userID - 1][p - 1]
                r_uj = np.dot(userVector, itemComVectorj) + itemBias[userID - 1][j - 1]

                r_upj = r_up - r_uj
                r_uip = r_ui - r_up

                # 计算累积训练损失 (Objective Loss)
                loss_total += -math.log(sigmoid(r_upj) + 1e-10) - math.log(sigmoid(r_uip) + 1e-10) + \
                              LambdaU / 2 * (np.linalg.norm(userVector, ord=2) ** 2) + \
                              LambdaV / 2 * (np.linalg.norm(itemComVectori, ord=2) ** 2 + \
                                             np.linalg.norm(itemComVectorp, ord=2) ** 2 + \
                                             np.linalg.norm(itemComVectorj, ord=2) ** 2) + \
                              LambdaZ / 2 * (itemBias[userID - 1][i - 1] ** 2 + itemBias[userID - 1][p - 1] ** 2 + \
                                             itemBias[userID - 1][j - 1] ** 2)

                loss_funcp = -sigmoid(-r_upj)
                loss_func1 = -sigmoid(-r_uip)

                # 计算用于共享的原始数据梯度 (Data Gradients)
                data_dVi = loss_func1 * userVector
                data_dVp = loss_funcp * userVector + loss_func1 * (-userVector)
                data_dVj = loss_funcp * (-userVector)
                data_dBi = loss_func1
                data_dBp = loss_funcp - loss_func1
                data_dBj = -loss_funcp

                # 叠加正则化项计算本地模型梯度 (Local Gradients)
                local_deltaU = loss_funcp * (itemComVectorp - itemComVectorj) + loss_func1 * (
                        itemComVectori - itemComVectorp) + LambdaU * userVector
                local_deltaVi = data_dVi + LambdaV * itemComVectori
                local_deltaVp = data_dVp + LambdaV * itemComVectorp
                local_deltaVj = data_dVj + LambdaV * itemComVectorj
                local_deltaBi = data_dBi + LambdaZ * itemBias[userID - 1][i - 1]
                local_deltaBp = data_dBp + LambdaZ * itemBias[userID - 1][p - 1]
                local_deltaBj = data_dBj + LambdaZ * itemBias[userID - 1][j - 1]

                # 更新本地模型参数
                userVectors[userID - 1] -= alpha * local_deltaU
                itemComVectors[userID - 1][i - 1] -= alpha * local_deltaVi
                itemComVectors[userID - 1][p - 1] -= alpha * local_deltaVp
                itemComVectors[userID - 1][j - 1] -= alpha * local_deltaVj
                itemBias[userID - 1][i - 1] -= alpha * local_deltaBi
                itemBias[userID - 1][p - 1] -= alpha * local_deltaBp
                itemBias[userID - 1][j - 1] -= alpha * local_deltaBj

                # 对共享梯度注入拉普拉斯噪声 (Gradient DP)
                noisy_deltaVi = add_laplace_noise(data_dVi, sigma)
                noisy_deltaVp = add_laplace_noise(data_dVp, sigma)
                noisy_deltaVj = add_laplace_noise(data_dVj, sigma)
                noisy_deltaBi = add_laplace_noise(data_dBi, sigma)
                noisy_deltaBp = add_laplace_noise(data_dBp, sigma)
                noisy_deltaBj = add_laplace_noise(data_dBj, sigma)

                # 簇内异步协作更新：将带噪梯度传输给邻居
                neighbors = Nu[userID]
                #if len(neighbors) > N1:
                    #neighbors = random.sample(neighbors, N1)

                for nb in neighbors:
                    idx = nb - 1
                    itemComVectors[idx][i - 1] -= alpha * noisy_deltaVi
                    itemComVectors[idx][p - 1] -= alpha * noisy_deltaVp
                    itemComVectors[idx][j - 1] -= alpha * noisy_deltaVj
                    itemBias[idx][i - 1] -= alpha * noisy_deltaBi
                    itemBias[idx][p - 1] -= alpha * noisy_deltaBp
                    itemBias[idx][j - 1] -= alpha * noisy_deltaBj

        end = time.time()
        # 评估当前 Epoch 性能（涵盖 8 个核心指标）
        p5, r5, p10, r10, f1_5, f1_10, map_score, mrr = computePrecision(
            testRank, userVectors, itemComVectors, itemNum, data, itemBias, Sort, Wc
        )

        # 打印至控制台
        print(f'Epoch:[{epoch:3d}][Loss:{loss_total:.2f}][P@5:{p5:.4f}][R@5:{r5:.4f}][P@10:{p10:.4f}][R@10:{r10:.4f}]'
              f'[F1@5:{f1_5:.4f}][F1@10:{f1_10:.4f}][MAP:{map_score:.4f}][MRR:{mrr:.4f}][Time:{end - start:.2f}s]')

        # 记录至日志文件
        log.write(f'Epoch:[{epoch}][Loss:{loss_total:.5f}][p5:{p5:.5f}][r5:{r5:.5f}][p10:{p10:.5f}][r10:{r10:.5f}]'
                  f'[f1_5:{f1_5:.5f}][f1_10:{f1_10:.5f}][map:{map_score:.5f}][mrr:{mrr:.5f}][time:{end - start:.5f}]\n')
        log.flush()

        if epoch % 10 == 0:
            alpha = alpha * lrDecay

    totalEnd = time.time()
    print('Total execution time: ', totalEnd - totalStart)


def readFile(path):
    """解析训练/测试数据集文件"""
    uiratings = []
    sortd = defaultdict(dict)
    itemd = defaultdict(list)
    C_hot = {}
    n = {}
    with open(path) as fd:
        for line in fd:
            rec = line.strip().split(',')
            uiratings.append(
                {'userID': rec[0], 'itemID': rec[1], 'sortID': rec[2], 'checknum': rec[3], 'city': rec[4], 'rating': 1,
                 'pui': rec[3]})
            if rec[1] not in itemd[rec[0]]:
                itemd[rec[0]].append(rec[1])
            sortd[rec[0]][rec[2]] = sortd[rec[0]].get(rec[2], 0) + int(rec[3])
            C_hot[rec[2]] = C_hot.get(rec[2], 0) + 1
            n[rec[0]] = n.get(rec[0], 0) + int(rec[3])
    return uiratings, sortd, itemd, C_hot, n


def generateUsers2Rank(userNum, userNum2Rank):
    """随机生成需要进行 Top-N 排名评估的用户列表"""
    users = []
    while len(users) < userNum2Rank:
        user = random.randint(1, userNum)
        if user not in users:
            users.append(user)
    return users


def readFile4RankRandom(path, users2Rank):
    """加载测试集中的用户实际访问记录用于性能指标计算"""
    uiRatings = {}
    with open(path) as fd:
        for line in fd:
            rec = line.strip().split(',')
            user = int(rec[0])
            item = int(rec[1])
            rating = 1
            if user in users2Rank:
                uiRatings.setdefault(user, []).append(f"{item},{rating}")
    return uiRatings


def readCityUser(path):
    """读取用户城市分布信息"""
    uiratings = {}
    with open(path) as fd:
        for line in fd:
            rec = line.strip().split(',')
            uiratings[rec[0]] = rec[1]
    return uiratings


def select(Sort_item, Ns, userNum, trainitem):
    """
    根据用户偏好类别筛选出“感兴趣但未访问”的候选项 (Preferred Candidates)
    """
    Select = {}
    for u in range(userNum):
        a = []
        for i in Ns[u]:
            a.extend(Sort_item[str(i)])
        Select[str(u + 1)] = sorted(list(set(a) - set(trainitem[str(u + 1)])))
    return Select


def rest(trainitem, Select, userNum, itemNum):
    """筛选出用户“既未访问也不感兴趣”的候选项 (Non-preferred Candidates)"""
    allitem = {str(i + 1): [str(j + 1) for j in range(itemNum)] for i in range(userNum)}
    return {str(i + 1): sorted(list(set(allitem[str(i + 1)]) - set(Select[str(i + 1)]) - set(trainitem[str(i + 1)])))
            for i in range(userNum)}


def main(args):
    """主程序入口：控制流程初始化及训练启动"""

    # 文件路径前缀
    filePrefix = r"C:\Users\Desktop\CA-PSDBPR\DATA\city123\\"

    # 1. 数据加载与预处理
    trainSet, trainSort, trainitem, trainC_hot, n_train = readFile(filePrefix + 'city123-city-train')
    testSet, testSort, testitem, testC_hot, n_test = readFile(filePrefix + 'city123-city-test')
    cityUsers = readCityUser(filePrefix + 'city123_cityusers')

    all_dataset_temp = trainSet + testSet

    userNum = 0
    itemNum = 0
    sortNum = 0

    for row in all_dataset_temp:
        uid = int(row['userID'])
        iid = int(row['itemID'])
        sid = int(row['sortID'])

        if uid > userNum: userNum = uid
        if iid > itemNum: itemNum = iid
        if sid > sortNum: sortNum = sid
    users2Rank = generateUsers2Rank(userNum, userNum)
    testRank = readFile4RankRandom(filePrefix + 'city123-city-test', users2Rank)

    # 日志输出配置
    output = filePrefix + f'CA-PSDBPR_results_K{args.featureK}_alpha{args.alpha}.txt'
    log = open(output, 'w')

    # 打印要求的头信息
    header = 'epoch, p5, r5, p10, r10, f1_5, f1_10, map, mrr, time per iter'
    print(header)
    log.write(header + '\n')

    # 2. 模型辅助组件初始化
    Sort = sort_1(filePrefix + 'city123-5k5k')
    Sort_item, Lc = sort_2(filePrefix + 'city123-5k5k', sortNum)

    user_item_counts = {}
    for row in trainSet:
        user_item_counts.setdefault(row['userID'], {})[row['itemID']] = int(row['checknum'])

    # 3. LDP 向量构建与用户聚类 (恢复进度打印)
    print("开始生成 LDP 向量 ")
    lu_vectors = Lu_LDP(user_item_counts, Sort, userNum, itemNum, sortNum, epsilon=args.threshold)
    print("LDP 向量生成完毕。")

    print("正在执行聚类 (K=20)...")
    Nu = K_means_fixed_constrained(lu_vectors, n_clusters=20)

    # 4. 偏好挖掘
    Wc = CategoryWeight(filePrefix + 'city123-city-train', userNum, sortNum, itemNum, trainSort, Lc, n_train)
    Ns = favorite(Wc, userNum, sortNum, args.categoryNum)
    Select = select(Sort_item, Ns, userNum, trainitem)
    Rest = rest(trainitem, Select, userNum, itemNum)

    # 5. 启动主训练程序
    train(userNum, itemNum, cityUsers, args.featureK, trainSet, testSet, testRank, args.epochs,
          args.LambdaU, args.LambdaV, args.LambdaZ, args.alpha, 0.9, filePrefix, False, 1,
          args.negativeNum, log, Nu, trainitem, Sort, Wc, Select, Rest,
          sigma=args.gradient_sigma, N1=args.N1)

    log.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CA-PSDBPR Model Training')
    parser.add_argument('--featureK', default=10, type=int)
    parser.add_argument('--LambdaU', default=0.1, type=float)
    parser.add_argument('--LambdaV', default=0.1, type=float)
    parser.add_argument('--LambdaZ', default=0.1, type=float)
    parser.add_argument('--alpha', default=0.05, type=float)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--negativeNum', default=1, type=int)
    parser.add_argument('--threshold', default=0.8, type=float)
    parser.add_argument('--categoryNum', default=5, type=int)
    parser.add_argument('--gradient_sigma', default=0.5, type=float)
    parser.add_argument('--N1', default=50, type=int)
    args = parser.parse_args()
    main(args)