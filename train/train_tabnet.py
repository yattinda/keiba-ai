from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.pretraining import TabNetPretrainer
from torch.nn import CrossEntropyLoss
from sklearn.utils import class_weight
from sklearn.metrics import roc_auc_score

import pandas as pd
import numpy as np
import torch
import joblib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def split_date(df, test_size):
    sorted_id_list = df.sort_values('日付').index.unique()
    train_id_list = sorted_id_list[:round(len(sorted_id_list) * (1-test_size))]
    test_id_list = sorted_id_list[round(len(sorted_id_list) * (1-test_size)):]
    train = df.loc[train_id_list]
    test = df.loc[test_id_list]
    return train, test


# データの読み込み
print('loading...')
data = pd.read_csv('/work/yasuhito-m/workspace/keiba-ai/datasets/encoded/encoded_traindata.csv')
#data = pd.read_csv('../datasets/encoded/encoded_traindata.csv')
#着順を変換
data['着順'] = data['着順'].map(lambda x: 1 if x<4 else 0)
# tabnet cant handle NaN
data = data.dropna()

# 特徴量とターゲットの分割
train, test = split_date(data, 0.2)
X_train = train.drop(['着順','オッズ','人気','上がり','走破時間','通過順'], axis=1)
y_train = train['着順']
X_test = test.drop(['着順','オッズ','人気','上がり','走破時間','通過順'], axis=1)
y_test = test['着順']

X_train = torch.tensor(X_train.values).to(device)
y_train = torch.tensor(y_train.values).to(device)
X_test = torch.tensor(X_test.values).to(device)
y_test = torch.tensor(y_test.values).to(device)

class_weights = list(class_weight.compute_class_weight('balanced',
                    classes=np.unique(y_train.cpu()),
                    y=y_train.cpu().numpy())
                    )
weights = torch.tensor(class_weights, dtype=torch.float).to(device)



# lossを指定し重みを加える 重み無しならLossはNone
cross_entropy_loss_wight = CrossEntropyLoss(weight=weights)

# 事前学習の設定
unsupervised_model = TabNetPretrainer(
    n_d=64,
    n_a=64,
    n_steps=4,
    gamma=1.4,
    lambda_sparse=1e-5,
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
    mask_type='entmax',
    scheduler_params= {'mode': "min",'patience': 5,'min_lr': 1e-5,'factor': 0.9},
    scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
    verbose=1
)

print('pretraining...')
unsupervised_model.fit(
    X_train=X_train.cpu(),
    eval_set=[X_test.cpu()],
    batch_size=4096,
    virtual_batch_size=256,
    num_workers=0,
    drop_last=False,
    pretraining_ratio=0.5,
)

unsupervised_model.save_model('/work/yasuhito-m/workspace/keiba-ai/model/tabnet/pretrain')
loaded_pretrain = TabNetPretrainer()
loaded_pretrain.load_model('/work/yasuhito-m/workspace/keiba-ai/model/tabnet/pretrain.zip')

params = {
    'n_d': 64,
    'n_a': 64,
    'n_steps': 4,
    'gamma': 1.4,
    'lambda_sparse': 1e-5,
    'optimizer_fn': torch.optim.Adam,
    'optimizer_params': dict(lr=2e-2, weight_decay=1e-5),
    'mask_type': 'entmax',
    'scheduler_params': {'mode': "min",'patience': 5,'min_lr': 1e-5,'factor': 0.9},
    'scheduler_fn': torch.optim.lr_scheduler.ReduceLROnPlateau,
    'verbose': 1,
    'device_name': 'cuda' if torch.cuda.is_available() else 'cpu'
}


X_train_cpu = X_train.cpu().numpy()
X_test_cpu = X_test.cpu().numpy()
y_train_cpu = y_train.cpu().numpy()
y_test_cpu = y_test.cpu().numpy()
print('training...')
tabnet_clf = TabNetClassifier(**params)

tabnet_clf.fit(
    X_train=X_train_cpu,
    y_train=y_train_cpu,
    loss_fn = cross_entropy_loss_wight,
    eval_set=[(X_test_cpu, y_test_cpu)],
    eval_name=['test'],
    eval_metric=['balanced_accuracy'],
    patience=20,
    batch_size=4096,
    virtual_batch_size=128,
    num_workers=0,
    drop_last=False,
    from_unsupervised=loaded_pretrain,
)

joblib.dump(tabnet_clf,'/work/yasuhito-m/workspace/keiba-ai/model/tabnet/tabnet.pkl')
#joblib.dump(lgb_clf,'model/lightgbm/lightgbm.pkl')

y_pred_train = tabnet_clf.predict_proba(X_train_cpu)[:,1]
y_pred = tabnet_clf.predict_proba(X_test_cpu)[:,1]

#モデルの評価
#print(roc_auc_score(y_train,y_pred_train))
print(roc_auc_score(y_test_cpu, y_pred))
total_cases = len(y_test)  # テストデータの総数
TP = (y_test_cpu == 1) & (y_pred >= 0.5)  # True positives
FP = (y_test_cpu == 0) & (y_pred >= 0.5)  # False positives
TN = (y_test_cpu == 0) & (y_pred < 0.5)  # True negatives
FN = (y_test_cpu == 1) & (y_pred < 0.5)  # False negatives

TP_count = sum(TP)
FP_count = sum(FP)
TN_count = sum(TN)
FN_count = sum(FN)

accuracy_TP = TP_count / total_cases * 100
misclassification_rate_FP = FP_count / total_cases * 100
accuracy_TN = TN_count / total_cases * 100
misclassification_rate_FN = FN_count / total_cases * 100

print("Total cases:", total_cases)
print("True positives:", TP_count, "(", "{:.2f}".format(accuracy_TP), "%)")
print("False positives:", FP_count, "(", "{:.2f}".format(misclassification_rate_FP), "%)")
print("True negatives:", TN_count, "(", "{:.2f}".format(accuracy_TN), "%)")
print("False negatives:", FN_count, "(", "{:.2f}".format(misclassification_rate_FN), "%)")

# 特徴量の重要度を取得
importance = tabnet_clf.feature_importances_

# 特徴量の名前を取得
feature_data = train.drop(['着順','オッズ','人気','上がり','走破時間','通過順'], axis=1)
feature_names = feature_data.columns

# 特徴量の重要度を降順にソート
indices = np.argsort(importance)[::-1]

# 特徴量の重要度を降順に表示
for f in range(feature_data.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feature_names[indices[f]], importance[indices[f]]))