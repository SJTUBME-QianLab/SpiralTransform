import random
import os


def k_fold_pre(filename, image_list_file, fold):  # 生成的是源文件的5-fold序号
    # num为原始样本数量
    # 数据集的 5-fold 数据准备
    file_all = open(image_list_file, "r").readlines()
    num = len(file_all)
    if os.path.exists(filename):
        print("5-fold-file has existed")
        train_index_5 = []
        test_index_5 = []
        flag = 0
        with open(filename, 'r') as f:
            for line in f:
                flag = flag + 1
                if flag <= fold:
                    a = line.split()
                    train_index_5.append(a)
                else:
                    a = line.split()
                    test_index_5.append(a)

        train_index_5_int = []
        for each in train_index_5:
            each_line = list(map(int, each))
            train_index_5_int.append(each_line)
        test_index_5_int = []
        for each in test_index_5:
            each_line = list(map(lambda x: int(x), each))
            test_index_5_int.append(each_line)
        return train_index_5_int, test_index_5_int


    label_0 = []
    label_1 = []
    for n in range(num):
        if file_all[n].split()[3] == '0':
            label_0.append(n)  # 保存label=0的序号
        else:
            label_1.append(n)  # 保存label=1的序号
    num_per_fold_0 = round(len(label_0) / fold)
    num_per_fold_1 = round(len(label_1) / fold)
    randnum_0 = random.sample(label_0, len(label_0))
    randnum_1 = random.sample(label_1, len(label_1))
    folds = []

    for i in range(fold):
        folds_fold = []
        if i != fold - 1:
            folds_fold.extend(randnum_0[i * num_per_fold_0: (i + 1) * num_per_fold_0])
            folds_fold.extend(randnum_1[i * num_per_fold_1: (i + 1) * num_per_fold_1])
            folds.append(folds_fold)  # 均是id
        else:
            folds_fold.extend(randnum_0[i * num_per_fold_0:])
            folds_fold.extend(randnum_1[i * num_per_fold_1:])
            folds.append(folds_fold)

    train_index_5 = []
    test_index_5 = []
    for i in range(fold):
        # training集
        featurej = []
        for j in range(fold):
            if j == i:  # 一共选取了4/5
                continue
            featurej.extend(folds[j])
        train_index_5.append(featurej)  # 5-fold中训练集的index

        # test集
        featurei = folds[i]
        test_index_5.append(featurei)  # 5-fold中测试集的index

    text_save(filename, data=train_index_5, state='w')
    text_save(filename, data=test_index_5, state='a')
    return train_index_5, test_index_5


def text_save(filename, data, state):  # filename为写入txt文件的路径，data为要写入数据列表.
    file = open(filename, state)  # 'a' 新建  'w+' 追加
    for i in range(len(data)):
        s = str(data[i]).replace('[', '').replace(']', '')  # 去除[],这两行按数据不同，可以选择
        s = s.replace("'", '').replace(',', '') + '\n'  # 去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()
    print("save 5-fold-file successfully")

