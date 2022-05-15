import numpy as np
from sklearn.model_selection import train_test_split

col_ind = 0
X = np.load('./dataset/15/feature_15.npy')
y = np.load('./dataset/15/label_15.npy')[:, col_ind]
TT = np.load('./dataset/15/trade_table_15.npy')
val_size = 0.2
test_size = 0.1


def generate_feature(x, threshold):
    a = (x[:, 1] - x[:, 0]).reshape(-1, 1)
    b = (x[:, 3] - x[:, 2]).reshape(-1, 1)
    a[a > threshold] = threshold
    b[b < threshold] = threshold
    return np.concatenate([a, b], axis=1)


def process_data(X, y, TT, val_size, test_size, threshold=1, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=(
        val_size/(1-test_size)), shuffle=False, random_state=random_state)
    TT_train, TT_test = train_test_split(
        TT, test_size=test_size, shuffle=False, random_state=random_state)
    TT_train, TT_val = train_test_split(TT_train, test_size=(
        val_size/(1-test_size)), shuffle=False, random_state=random_state)

#     # Normalization
#     X_train = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)
#     X_val = (X_val - X_train.mean(axis=0)) / X_train.std(axis=0)
#     X_test = (X_test - np.concatenate([X_train, X_val], axis=0).mean(axis=0)) / np.concatenate([X_train, X_val], axis=0).std(axis=0)

    X_train = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)
    X_val = (X_val - X_val.mean(axis=0)) / X_val.std(axis=0)
    X_test = (X_test - X_test.mean(axis=0)) / X_test.std(axis=0)

    # Add two more features
    spread_train = generate_feature(TT_train, threshold)
    spread_val = generate_feature(TT_val, threshold)
    spread_test = generate_feature(TT_test, threshold)

    X_train = np.concatenate([X_train, spread_train], axis=1)
    X_val = np.concatenate([X_val, spread_val], axis=1)
    X_test = np.concatenate([X_test, spread_test], axis=1)

    return X_train, X_val, X_test, y_train, y_val, y_test


X_train, X_val, X_test, y_train, y_val, y_test = process_data(
    X, y, TT, val_size, test_size)

with open('data.npz', 'wb') as f:
    np.savez(f, X_train=X_train, X_val=X_val, X_test=X_test,
             y_train=y_train, y_val=y_val, y_test=y_test)
    f.close()
