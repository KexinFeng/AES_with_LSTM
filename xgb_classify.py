from helper_pad import Helper
import utils, math
import numpy as np
import xgboost as xgb
from xgboost import plot_importance
import sklearn

##
# Read in data
embedding_size = 10
# readfile = './data/small.tsv'
readfile = './data/training_set_rel3.tsv'

X_, _, y_ = Helper(set_num=0, file_name=readfile).get_embed(embedding_size)
# using helper_pad, the output X_ is already zero-padded and flattened rank-2 matrix.

print('reading file: ', readfile)
benchmark_score = 8 * np.ones_like(y_)
print('uniform guess error: ', np.sum(np.where(benchmark_score != y_, 1, 0)) / len(y_))
score = sklearn.metrics.cohen_kappa_score(benchmark_score, np.round(y_), weights='quadratic')
print("Uniform guess Kappa score: ", score)

data_size = y_.shape[0]
train_size = math.floor(0.9 * data_size)
test_size = data_size - train_size

X_train = X_[:train_size]
y_train = y_[:train_size]
X_test = X_[train_size:]
y_test = y_[train_size:]


##

kargin = {
    'max_depth': 10,
    'learning_rate': 0.1,
    'n_estimators': 20,  # number of rounds
    'objective': 'multi:softmax'
}
watchlist = [(X_train, y_train), (X_test, y_test)]
model = xgb.XGBClassifier(**kargin)
model.fit(X_train, y_train, eval_set=watchlist, verbose=True, eval_metric='merror')
##

pred_label = model.predict(X_test)
error_rate = np.sum(pred_label != y_test) / y_test.shape[0]
print('Test error using softprob = {}'.format(error_rate))
plot_importance(model)

print(pred_label)
print(y_test)

score = sklearn.metrics.cohen_kappa_score(np.round(pred_label), np.round(y_test), weights='quadratic')
print("Kappa score: ", score)


##

