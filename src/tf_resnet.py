import os
import time
import pickle
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import log_loss, roc_auc_score

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

DATA_PATH = './jane-street-market-prediction/'

# GPU_NUM = 8
BATCH_SIZE = 2048 # * GPU_NUM
EPOCHS = 200
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
EARLYSTOP_NUM = 3
NFOLDS = 5

TRAIN = True
CACHE_PATH = './'

train = pd.read_csv(f'{DATA_PATH}/train.csv')

def save_pickle(dic, save_path):
    with open(save_path, 'wb') as f:
    # with gzip.open(save_path, 'wb') as f:
        pickle.dump(dic, f)

def load_pickle(load_path):
    with open(load_path, 'rb') as f:
    # with gzip.open(load_path, 'rb') as f:
        message_dict = pickle.load(f)
    return message_dict

class EarlyStopping:
    def __init__(self, patience=7, mode="max", delta=0.001):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    def __call__(self, epoch_score, model, model_path):

        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
        elif score < self.best_score: #  + self.delta
            self.counter += 1
            # print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            # ema.apply_shadow()
            self.save_checkpoint(epoch_score, model, model_path)
            # ema.restore()
            self.counter = 0

    def save_checkpoint(self, epoch_score, model, model_path):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            # print('Validation score improved ({} --> {}). Saving model!'.format(self.val_score, epoch_score))
            # if not DEBUG:
            torch.save(model.state_dict(), model_path)
        self.val_score = epoch_score

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(seed=42)

feat_cols = [f'feature_{i}' for i in range(130)]

if TRAIN:
    train = train.loc[train.date > 85].reset_index(drop=True)
    train['action'] = (train['resp'] > 0).astype('int')
    train['action_1'] = (train['resp_1'] > 0).astype('int')
    train['action_2'] = (train['resp_2'] > 0).astype('int')
    train['action_3'] = (train['resp_3'] > 0).astype('int')
    train['action_4'] = (train['resp_4'] > 0).astype('int')
    # valid = train.loc[(train.date >= 450) & (train.date < 500)].reset_index(drop=True)
    # train = train.loc[train.date < 450].reset_index(drop=True)
target_cols = ['action', 'action_1', 'action_2', 'action_3', 'action_4']

if TRAIN:
    #df = pd.concat([train[feat_cols], valid[feat_cols]]).reset_index(drop=True)
    #f_mean = df.mean()
    f_mean = train[feat_cols].mean()
    f_mean = f_mean.values
    np.save(f'{CACHE_PATH}/f_mean_online.npy', f_mean)
    train.fillna(train.mean(), inplace=True)
    #valid.fillna(df.mean(), inplace=True)
else:
    f_mean = np.load(f'{CACHE_PATH}/f_mean_online.npy')

##### Making features
# https://www.kaggle.com/lucasmorin/running-algos-fe-for-fast-inference/data
# eda:https://www.kaggle.com/carlmcbrideellis/jane-street-eda-of-day-0-and-feature-importance
# his example:https://www.kaggle.com/gracewan/plot-model
def fillna_npwhere_njit(array, values):
    if np.isnan(array.sum()):
        array = np.where(np.isnan(array), values, array)
    return array

class RunningEWMean:
    def __init__(self, WIN_SIZE=20, n_size=1, lt_mean=None):
        if lt_mean is not None:
            self.s = lt_mean
        else:
            self.s = np.zeros(n_size)
        self.past_value = np.zeros(n_size)
        self.alpha = 2 / (WIN_SIZE + 1)

    def clear(self):
        self.s = 0

    def push(self, x):

        x = fillna_npwhere_njit(x, self.past_value)
        self.past_value = x
        self.s = self.alpha * x + (1 - self.alpha) * self.s

    def get_mean(self):
        return self.s

if TRAIN:
    all_feat_cols = [col for col in feat_cols]

    train['cross_41_42_43'] = train['feature_41'] + train['feature_42'] + train['feature_43']
    train['cross_1_2'] = train['feature_1'] / (train['feature_2'] + 1e-5)
    #valid['cross_41_42_43'] = valid['feature_41'] + valid['feature_42'] + valid['feature_43']
    #valid['cross_1_2'] = valid['feature_1'] / (valid['feature_2'] + 1e-5)

    all_feat_cols.extend(['cross_41_42_43', 'cross_1_2'])

##### Model&Data fnc
class SmoothBCEwLogits(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth(targets:torch.Tensor, n_labels:int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = targets * (1.0 - smoothing) + 0.5 * smoothing
        return targets

    def forward(self, inputs, targets):
        targets = SmoothBCEwLogits._smooth(targets, inputs.size(-1),
            self.smoothing)
        loss = F.binary_cross_entropy_with_logits(inputs, targets,self.weight)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()

        return loss

class MarketDataset:
    def __init__(self, df):
        
        self.features = df[all_feat_cols].values
        self.label = df[target_cols].values.reshape(-1, len(target_cols))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return {
            'features': torch.tensor(self.features[idx], dtype=torch.float),
            'label': torch.tensor(self.label[idx], dtype=torch.float)
        }

def create_resnet_model(input_dim,output_dim):    
    inp = Input(shape=(input_dim,))
    x = BatchNormalization()(inp)
    x = Dropout(0.1)(x)
    
    x1 = Dense(1024)(x)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU(alpha=0.1)(x1)
    x1 = Dropout(0.3)(x1)
    x = Concatenate(axis=1)([x,x1])

    x2 = Dense(512)(x)
    x2 = BatchNormalization()(x2)
    x2 = LeakyReLU(alpha=0.1)(x2)
    x2 = Dropout(0.3)(x2)
    x = Concatenate(axis=1)([x1,x2])
    
    x3 = Dense(256)(x)
    x3 = BatchNormalization()(x3)
    x3 = LeakyReLU(alpha=0.1)(x3)
    x3 = Dropout(0.3)(x3)
    x = Concatenate(axis=1)([x2,x3])   
    
    x = Dense(output_dim,activation='sigmoid')(x)
    model = Model(inputs=inp,outputs=x)
    model.compile(optimizer=Adam(0.0001),loss=BinaryCrossentropy(label_smoothing=0.1),metrics=[tf.keras.metrics.AUC(name = 'auc')])
    return model

def create_mlp(
    num_columns, num_labels, hidden_units, dropout_rates, label_smoothing, learning_rate
):

    inp = tf.keras.layers.Input(shape=(num_columns,))
    x = tf.keras.layers.BatchNormalization()(inp)
    x = tf.keras.layers.Dropout(dropout_rates[0])(x)
    for i in range(len(hidden_units)):
        x = tf.keras.layers.Dense(hidden_units[i])(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(tf.keras.activations.swish)(x)
        x = tf.keras.layers.Dropout(dropout_rates[i + 1])(x)
    
    x = tf.keras.layers.Dense(num_labels)(x)
    out = tf.keras.layers.Activation("sigmoid")(x)

    model = tf.keras.models.Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=tfa.optimizers.RectifiedAdam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing),
        metrics=tf.keras.metrics.AUC(name="AUC"),
    )

    return model


def utility_score_bincount(date, weight, resp, action):
    count_i = len(np.unique(date))
    # print('weight: ', weight)
    # print('resp: ', resp)
    # print('action: ', action)
    # print('weight * resp * action: ', weight * resp * action)
    Pi = np.bincount(date, weight * resp * action)
    t = np.sum(Pi) / np.sqrt(np.sum(Pi ** 2)) * np.sqrt(250 / count_i)
    u = np.clip(t, 0, 6) * np.sum(Pi)
    return u

train.index = pd.to_datetime(train.date)

import itertools as itt

def cpcv_generator(t_span, n, k):
    # split data into N groups, with N << T
    # this will assign each index position to a group position
    group_num = np.arange(t_span) // (t_span // n)
    group_num[group_num == n] = n-1
    
    # generate the combinations 
    test_groups = np.array(list(itt.combinations(np.arange(n), k))).reshape(-1, k)
    C_nk = len(test_groups)
    n_paths = C_nk * k // n 
    
    # is_test is a T x C(n, k) array where each column is a logical array 
    # indicating which observation in in the test set
    is_test_group = np.full((n, C_nk), fill_value=False)
    is_test = np.full((t_span, C_nk), fill_value=False)
    
    # assign test folds for each of the C(n, k) simulations
    for k, pair in enumerate(test_groups):
        i, j = pair
        is_test_group[[i, j], k] = True
        
        # assigning the test folds
        mask = (group_num == i) | (group_num == j)
        is_test[mask, k] = True
        
    # for each path, connect the folds from different simulations to form a backtest path
    # the fold coordinates are: the fold number, and the simulation index e.g. simulation 0, fold 0 etc
    path_folds = np.full((n, n_paths), fill_value=np.nan)
    
    for i in range(n_paths):
        for j in range(n):
            s_idx = is_test_group[j, :].argmax().astype(int)
            path_folds[j, i] = s_idx
            is_test_group[j, s_idx] = False
            
    
    # finally, for each path we indicate which simulation we're building the path from and the time indices
    paths = np.full((t_span, n_paths), fill_value= np.nan)
    
    for p in range(n_paths):
        for i in range(n):
            mask = (group_num == i)
            paths[mask, p] = int(path_folds[i, p])
    # paths = paths_# .astype(int)

    return (is_test, paths, path_folds)    

# AFML, snippet 7.1
from tqdm import tqdm

def purge(t1, test_times): # whatever is not in the train set should be in the test set
    train_ = t1.copy(deep=True) # copy of the index
    train_ = train_.drop_duplicates()
    for start, end in tqdm(test_times.iteritems(), total=len(test_times)):
        df_0 = train_[(start <= train_.index) & (train_.index <= end)].index # train_ starts within test
        df_1 = train_[(start <= train_) & (train_ <= end)].index
        df_2 = train_[(train_.index <= start) & (end <= train_)].index
        train_ = train_.drop(df_0.union(df_1).union(df_2))
    return train_

# AFML, snippet 7.2
def embargo_(times, pct_embargo):
    step = int(times.shape[0] * pct_embargo) # more complicated logic if needed to use a time delta
    print(f'embargo step: {step}')
    print(times[0])
    print(times[step])
    quit
    if step == 0:
        ans = pd.Series(times, index=test_times)
    else:
        ans = pd.Series(times[step:].values, index=times[:-step].index)
        ans = ans.append(pd.Series(times[-1], index=times[-step:].index))
    return ans

def embargo(test_times, t1, pct_embargo=0.01): # done before purging
    # embargoed t1
    t1_embargo = embargo_(t1, pct_embargo)
    # test_start, test_end = test_times.index[0], test_times.index[-1]
    test_times_embargoed = t1_embargo.loc[test_times.index]
    return test_times_embargoed

# prediction and evalution times
# using business days, but the index is not holidays aware -- it can be fixed
t1_ = train.index
# recall that we are holding our position for 21 days
# normally t1 is important is there events such as stop losses, or take profit events
t1 = pd.Series(t1_[:], index=t1_[:]) # t1 is both the trade time and the event time
t1.head() # notice how the events (mark-to-market) take place 5 days later

num_paths = 5
num_groups_test = 2
num_groups = num_paths + 1 
num_ticks = len(train)
is_test, paths, _ = cpcv_generator(num_ticks, num_groups, num_groups_test)

num_sim = is_test.shape[1] # num of simulations needed to generate all backtest paths

## Cross Validation in Finance: Purging, Embargoing, Combination
## https://blog.quantinsti.com/cross-validation-embargo-purging-combinatorial/#embargoing






def run():
    torch.multiprocessing.freeze_support()

    for _fold in range(num_sim):

        test_idx = is_test[:,_fold]
        test_times = t1.loc[test_idx]
        test_times = test_times.drop_duplicates()
        
        #embargo
        test_times_embargoed = embargo(test_times, t1, pct_embargo=0.123)
        test_times_embargoed = test_times_embargoed.drop_duplicates()
        
        #purge
        train_times = purge(t1, test_times_embargoed)
        
        valid = train.loc[test_times.index, :]
        train_set = MarketDataset(train.loc[train_times.index, :])
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
        valid_set = MarketDataset(valid)
        valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)
        start_time = time.time()
        
        print(f'Fold: {_fold}')
        seed_everything(seed=42+_fold)
        torch.cuda.empty_cache()
        device = torch.device("cuda:0")
        model = Model()
        model.to(device)
        # model = nn.DataParallel(model)

        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        # optimizer = Nadam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        # optimizer = Lookahead(optimizer=optimizer, k=10, alpha=0.5)
        scheduler = None
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3,
        #                                                 max_lr=1e-2, epochs=EPOCHS, steps_per_epoch=len(train_loader))
        # loss_fn = nn.BCEWithLogitsLoss()
        loss_fn = SmoothBCEwLogits(smoothing=0.005)

        model_weights = f"{CACHE_PATH}/online_model{_fold}_va-"
        es = EarlyStopping(patience=EARLYSTOP_NUM, mode="max")
        for epoch in range(EPOCHS):
            train_loss = train_fn(model, optimizer, scheduler, loss_fn, train_loader, device)

            valid_pred = inference_fn(model, valid_loader, device)
            valid_auc = roc_auc_score(valid[target_cols].values, valid_pred)
            valid_logloss = log_loss(valid[target_cols].values, valid_pred)
            valid_pred = np.median(valid_pred, axis=1)
            valid_pred = np.where(valid_pred >= 0.5, 1, 0).astype(int)
            valid_u_score = utility_score_bincount(date=valid.date.values, weight=valid.weight.values,
                                                resp=valid.resp.values, action=valid_pred)
            print(f"FOLD{_fold} EPOCH:{epoch:3} train_loss={train_loss:.5f} "
                    f"valid_u_score={valid_u_score:.5f} valid_auc={valid_auc:.5f} "
                    f"time: {(time.time() - start_time) / 60:.2f}min")
            es(valid_auc, model, model_path=model_weights+f"{valid_auc}.pth")
            if es.early_stop:
                print("Early stopping")
                break
        # torch.save(model.state_dict(), model_weights)
    if True:
        valid_pred = np.zeros((len(valid), len(target_cols)))
        for _fold in range(num_sim):
            torch.cuda.empty_cache()
            device = torch.device("cuda:0")
            model = Model()
            model.to(device)
            model_weights = f"{CACHE_PATH}/online_model{_fold}.pth"
            model.load_state_dict(torch.load(model_weights))

            valid_pred += inference_fn(model, valid_loader, device) / num_sim
        auc_score = roc_auc_score(valid[target_cols].values, valid_pred)
        logloss_score = log_loss(valid[target_cols].values, valid_pred)

        valid_pred = np.median(valid_pred, axis=1)
        valid_pred = np.where(valid_pred >= 0.5, 1, 0).astype(int)
        valid_score = utility_score_bincount(date=valid.date.values, weight=valid.weight.values, resp=valid.resp.values,
                                            action=valid_pred)
        print(f'{NFOLDS} models valid score: {valid_score}\tauc_score: {auc_score:.4f}\tlogloss_score:{logloss_score:.4f}')



##############

hidden_units = [160, 160, 160]
dropout_rates = [0.2, 0.2, 0.2, 0.2]
label_smoothing = 1e-2
learning_rate = 1e-3

def utility_score_numba(date, weight, resp, action):
    Pi = np.bincount(date, weight * resp * action)
    t = np.sum(Pi) / np.sqrt(np.sum(Pi ** 2)) * np.sqrt(250 / len(Pi))
    u = min(max(t, 0), 6) * np.sum(Pi)
    return u

model = create_resnet_model(X.shape[-1],y.shape[-1])
FOLDS = 5

## Train smaller model to predict original model output

def create_small_nn(input_dim):
    inp = Input(shape=(input_dim,))
    x1 = Dense(128)(inp)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU(alpha=0.1)(x1)
    x1 = Dropout(0.3)(x1)

    x2 = Dense(64)(x1)
    x2 = BatchNormalization()(x2)
    x2 = LeakyReLU(alpha=0.1)(x2)
    x2 = Dropout(0.3)(x2)
    
    out = Dense(5,activation='sigmoid')(x2)
    model = Model(inputs=inp,outputs=out)
    return model


student= create_small_nn(132)
tr_util_scores = []
te_util_scores = []

def run():

    for _fold in range(num_sim):
            
        test_idx = is_test[:,_fold]
        test_times = t1.loc[test_idx]
        test_times = test_times.drop_duplicates()
        
        #embargo
        test_times_embargoed = embargo(test_times, t1, pct_embargo=0.123)
        test_times_embargoed = test_times_embargoed.drop_duplicates()
        
        #purge
        train_times = purge(t1, test_times_embargoed)
        valid = train.loc[test_times.index, :]
        train_set = MarketDataset(train.loc[train_times.index, :])
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
        valid_set = MarketDataset(valid)
        valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)
        start_time = time.time()
        
for fold, (train_indices, test_indices) in enumerate(splits):
    teacher_1 = create_resnet_model(X.shape[-1],y.shape[-1])
    teacher_1.load_weights(f'model_2222_{fold}_finetune.hdf5')
    teacher_2 = create_resnet_model(X.shape[-1],y.shape[-1])
    teacher_2.load_weights(f'model_3333_{fold}_finetune.hdf5')
    teacher_3 = model = create_mlp(X.shape[-1], 5, hidden_units, dropout_rates, label_smoothing, learning_rate)

    teacher_3.load_weights(f'model_4444_{fold}_finetune.hdf5')
    student= create_small_nn(132)
    distiller = Distiller(student=student, teacher=[teacher_1,teacher_2,teacher_3])
    distiller.compile(
        optimizer=keras.optimizers.Adam(),
        metrics=[keras.metrics.AUC(name = 'auc')],
        student_loss_fn=keras.losses.BinaryCrossentropy(label_smoothing=0.1),
        distillation_loss_fn=keras.losses.BinaryCrossentropy(label_smoothing=0.1),
        alpha=0.1,
        temperature=10,
    )

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    distiller.fit(X_train, y_train, validation_data=(X_test,y_test),epochs=300,batch_size=4096,callbacks=[EarlyStopping('val_auc',mode='max',patience=10,restore_best_weights=True)])
    distiller.student.save_weights(f'./model_2222_{fold}_sdistilled.hdf5')
    
    ##Evaluate utility scores
    # Train columns
    date_tr = train['date'].values[train_indices]
    weight_tr = train['weight'].values[train_indices]
    resp_tr = train['resp'].values[train_indices]
    action_tr_t = train['action'].values[train_indices]

    # Test columns
    date_te = train['date'].values[test_indices]
    weight_te = train['weight'].values[test_indices]
    resp_te = train['resp'].values[test_indices]
    action_te_t = train['action'].values[test_indices]
    
    action_tr = np.where(np.mean(distiller.student.predict(X_train),1)>= 0.5, 1, 0)
    action_te = np.where(np.mean(distiller.student.predict(X_test) ,1)>= 0.5, 1, 0)
    train_utility = utility_score_numba(date_tr, weight_tr, resp_tr, action_tr)
    test_utility = utility_score_numba(date_te, weight_te, resp_te, action_te)
    tr_util_scores.append(train_utility)
    te_util_scores.append(test_utility)
    print(train_utility)
    print(test_utility)
    print(utility_score_numba(date_tr, weight_tr, resp_tr, action_tr_t))
    print(utility_score_numba(date_te, weight_te, resp_te, action_te_t))


#####



if __name__ == '__main__':
    run()