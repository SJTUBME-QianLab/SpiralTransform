from sklearn import linear_model, svm
import numpy as np
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score
from sklearn.model_selection import GridSearchCV
import joblib
from sklearn.preprocessing import StandardScaler
import random


def read_prediction_label(path, patient_file, slice_result_file,times_1,times_0):
    patient_fileline = open(path + patient_file, "r").readlines()
    slice_result_fileline = open(path + slice_result_file, "r").readlines()
    slice_result = [slice_result_fileline[i].split() for i in range(len(slice_result_fileline))]
    slice_result_0 = [slice_result[i] for i in range(len(slice_result)) if slice_result[i][1] == '0']
    slice_result_1 = [slice_result[i] for i in range(len(slice_result)) if slice_result[i][1] == '1']
    # num = len(patient_fileline)  # 12
    # # times_1=27 # times_0=45  取指定的扩增方式训练回归模型

    num_0 = len(slice_result_0)//times_0
    num_1 = len(slice_result_1)//times_1
    output_group_0 = [list() for _ in range(num_0)]
    target_group_0 = [None for _ in range(num_0)]
    output_group_1 = [list() for _ in range(num_1)]
    target_group_1 = [None for _ in range(num_1)]
    for i in range(len(slice_result_0)):
        if i % times_0 < times_1:  # 0-26
            output_group_0[i // times_0].append(float(slice_result_0[i][0]))
            target_group_0[i // times_0] = float(slice_result_0[i][1])
    for j in range(len(slice_result_1)):
        output_group_1[j // times_1].append(float(slice_result_1[j][0]))
        target_group_1[j // times_1] = float(slice_result_1[j][1])


    # 取三个值训练回归模型  每个角度都取三个值：原始/水平/垂直  num_0 = len(slice_result_0)//3
    '''
    num_0 = len(slice_result_0) // 3
    num_1 = len(slice_result_1) // 3
    output_group_0 = [list() for _ in range(num_0)]
    target_group_0 = [None for _ in range(num_0)]
    output_group_1 = [list() for _ in range(num_1)]
    target_group_1 = [None for _ in range(num_1)]
    for i in range(len(slice_result_0)):
        output_group_0[i // 3].append(float(slice_result_0[i][0]))
        target_group_0[i // 3] = float(slice_result_0[i][1])
    for j in range(len(slice_result_1)):
        output_group_1[j // 3].append(float(slice_result_1[j][0]))
        target_group_1[j // 3] = float(slice_result_1[j][1])
    '''

    # 阴性和阳性的样本合起来
    output_group_0.extend(output_group_1)
    target_group_0.extend(target_group_1)
    # index = random.sample(range(0, num_0+num_1), num_0+num_1)
    # output_group = [output_group_0[i] for i in index]
    # target_group = [target_group_0[i] for i in index]
    return output_group_0, target_group_0


def accuracy(output, target):
    output = np.where(output >= 0.5, output, 0)
    output = np.where(output < 0.5, output, 1)
    target = target.astype(np.int64)
    output = output.astype(np.int64)

    recall = recall_score(target, output)
    precision = precision_score(target, output)
    f1 = f1_score(target, output)

    error = target ^ output
    error_rate = error.sum() / output.shape[0]
    acc = 1 - error_rate

    # -----TN----- #
    TN = np.logical_and(output == 0, target == 0).sum()
    FN = np.logical_and(output == 0, target == 1).sum()
    FP = np.logical_and(output == 1, target == 0).sum()
    TP = np.logical_and(output == 1, target == 1).sum()

    # -----sensitivity=recall----- #
    sensitivity = TP/(TP+FN)
    specificity = TN/(TN+FP)

    return acc, recall, precision, f1, sensitivity, specificity


def aucrocs(output, target, path=None):  # 改准确度的计算方式

    """
    Returns:
    List of AUROCs of all classes.
    """
    if path is not None:
        filename = "{}/LinearRegression_pre_target.txt".format(path)
        file = open(filename, 'a')  # 'a' 新建  'w+' 追加
        for i in range(len(output)):
            out_write = str(output[i]) + ' ' + str(target[i]) + '\n'
            file.write(out_write)
        file.close()
    AUROCs = roc_auc_score(target, output)

    return AUROCs


def classification_LinearRegression(path, train_patient_file, train_slice_result_file,test_patient_file, test_slice_result_file,fold,times_1,times_0):
    # path = './runs/ResNet_modal_ADC_DWI_T2_SGD_multi_angle_slice_lr0.001_epoch20_40_64_pretrained_v2/'
    # train_patient_file = 'fold_0_image_names_train.txt'
    # train_slice_result_file = 'fold_0_result_train.txt'
    train_output_group, train_target_group = read_prediction_label(path, train_patient_file, train_slice_result_file,times_1,times_0)
    # test_patient_file = 'fold_0_image_names.txt'
    # test_slice_result_file = 'fold_0_result.txt'
    test_output_group, test_target_group = read_prediction_label(path, test_patient_file, test_slice_result_file,times_1,times_0)
    # 数据标准化
    # ss = StandardScaler()
    # train_output_group = ss.fit_transform(train_output_group)
    # test_output_group = ss.transform(test_output_group)

    model = linear_model.LinearRegression()
    model.fit(train_output_group, train_target_group)
    joblib.dump(model, '{}/checkpoint/LinearRegression{}.pkl'.format(path, fold))
    pred_train = model.predict(train_output_group)
    pred_test = model.predict(test_output_group)
    # <0的设为0；大于1的设为1
    pred_test = np.where(pred_test >= 0, pred_test, 0)
    pred_test = np.where(pred_test <= 1, pred_test, 1)
    # print(pred_train)
    # print(pred_test)
    print(model.intercept_)
    print(model.coef_)
    filename = "{}/LinearRegression_intercept_coef.txt".format(path)
    file = open(filename, 'a')  # 'a' 新建  'w+' 追加
    out_write = str(model.intercept_) + '\n' + str(model.coef_) + '\n'
    file.write(out_write)
    file.close()

    acc, recall, precision, f1,sensitivity, specificity = accuracy(pred_test, np.array(test_target_group))
    auc = aucrocs(pred_test, np.array(test_target_group), path)
    print('LinearRegression: fold{}: acc = {}, auc = {}'.format(fold, acc, auc))

    filename = "{}/classification_LinearRegression_result.txt".format(path)
    file = open(filename, 'a')  # 'a' 新建  'w+' 追加
    out_write = 'fold' + str(fold) + ' ' + str(acc) + ' ' + str(auc) + ' ' \
                + str(recall) + ' ' + str(precision) + ' ' + str(f1) + ' ' \
                + str(sensitivity) + ' ' + str(specificity) + '\n'
    file.write(out_write)
    file.close()


def classification_SVM(path, train_patient_file, train_slice_result_file, test_patient_file, test_slice_result_file, fold):
    # path = './runs/ResNet_modal_ADC_DWI_T2_SGD_multi_angle_slice_lr0.001_epoch20_40_64_pretrained_v2/'
    # train_patient_file = 'fold_0_image_names_train.txt'
    # train_slice_result_file = 'fold_0_result_train.txt'
    train_output_group, train_target_group = read_prediction_label(path, train_patient_file, train_slice_result_file)
    # test_patient_file = 'fold_0_image_names.txt'
    # test_slice_result_file = 'fold_0_result.txt'
    test_output_group, test_target_group = read_prediction_label(path, test_patient_file, test_slice_result_file)
    k = ['rbf', 'linear', 'poly', 'sigmoid']
    c = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 10.0, 100.0, 1000.0]
    g = np.arange(1e-4, 1e-2, 0.0001)
    g = g.tolist()
    param_grid = dict(kernel=k, C=c, gamma=g)
    svr = svm.SVC()
    grid = GridSearchCV(svr, param_grid, cv=5, scoring='accuracy')
    grid.fit(train_output_group, train_target_group)
    model = grid.best_estimator_
    # model = svm.SVC(C=grid.best_params_['C'], kernel=grid.best_params_['kernel'], gamma=grid.best_params_['gamma'], decision_function_shape='ovo')
    model.fit(train_output_group, train_target_group)
    # pred_train = model.predict(train_output_group)
    pred_test = model.predict(test_output_group)
    # print(pred_train)
    # print(pred_test)
    # print(model.intercept_)
    # print(model.coef_)

    acc, recall, precision, f1 = accuracy(pred_test, np.array(test_target_group))
    auc = aucrocs(pred_test, np.array(test_target_group))
    print('SVM: fold{}: acc = {}, auc = {}'.format(fold, acc, auc))

    filename = "{}/classification_SVM_result.txt".format(path)
    file = open(filename, 'a')  # 'a' 新建  'w+' 追加
    out_write = 'fold' + str(fold) + ' ' + str(acc) + ' ' + str(auc) + ' ' \
                + str(recall) + ' ' + str(precision) + ' ' + str(f1) + '\n'
    file.write(out_write)
    file.close()


