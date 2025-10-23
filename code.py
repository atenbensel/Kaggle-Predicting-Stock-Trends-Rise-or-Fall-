import numpy as np
import pandas as pd
from pathlib import Path
from catboost import CatBoostClassifier, Pool

RANDOM_SEED = 42
N_FOLDS = 5
GAP_DAYS = 5
TRAIN_PATH = Path("train.csv")
TEST_PATH  = Path("test.csv")
SAMPLE_PATHS = ["sample_submission.csv", "SampleSubmission.csv", "sample.csv"]

if not TRAIN_PATH.exists() or not TEST_PATH.exists():
    raise FileNotFoundError("train.csv and/or test.csv not found in current directory.")

train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)

sample_sub = None
for p in SAMPLE_PATHS:
    if Path(p).exists():
        sample_sub = pd.read_csv(p); break

print("train shape:", train.shape, "| test shape:", test.shape)

train_lc = {c.lower(): c for c in train.columns}
test_lc  = {c.lower(): c for c in test.columns}

def pick(aliases, df="train", required=True):
    pool = train_lc if df == "train" else test_lc
    for a in aliases:
        if a in pool:
            return pool[a]
    if required:
        raise KeyError(f"Missing any of {aliases} in {df} columns.")
    return None

id_col = None
for k in ["id","row_id","rowid","prediction_id","index"]:
    if k in test_lc:
        id_col = test_lc[k]; break

symbol_col = pick(["symbol","ticker","stock_id","asset","name"], "train", required=False)
if symbol_col is None:
    symbol_col = "__SYMBOL__"
    train[symbol_col] = "ONE"; test[symbol_col] = "ONE"

date_col = pick(["date","datetime","timestamp","time","day"])
train[date_col] = pd.to_datetime(train[date_col])
test[date_col]  = pd.to_datetime(test[date_col])

close_col = pick(["close","closingprice","adjclose","adj_close"])
open_col  = pick(["open"])
high_col  = pick(["high"])
low_col   = pick(["low"])
vol_col   = pick(["volume","vol"], required=False)

target_col = None
for t in ["target","label","y","rise_fall","updown","class"]:
    if t in train_lc:
        target_col = train_lc[t]; break

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values([symbol_col, date_col]).copy()
    df["typical_price"] = (df[high_col] + df[low_col] + df[close_col]) / 3.0

    def per_symbol(g: pd.DataFrame):
        g = g.sort_values(date_col).copy()

        for w in [1,5,10,20,30,60,90]:
            g[f"ret_{w}"] = g[close_col].pct_change(w)

        for w in [5,10,20,30,60,90]:
            g[f"ma_{w}"]  = g[close_col].rolling(w, min_periods=max(2,w//2)).mean()
            g[f"std_{w}"] = g[close_col].rolling(w, min_periods=max(2,w//2)).std()

        win = 14
        d = g[close_col].diff()
        up, dn = d.clip(lower=0), -d.clip(upper=0)
        rs = up.rolling(win, min_periods=win//2).mean() / (dn.rolling(win, min_periods=win//2).mean() + 1e-9)
        g["rsi_14"] = 100 - (100/(1+rs))

        ema12 = g[close_col].ewm(span=12, adjust=False).mean()
        ema26 = g[close_col].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        sig  = macd.ewm(span=9, adjust=False).mean()
        g["macd"] = macd; g["macd_signal"] = sig; g["macd_hist"] = macd - sig

        ema8  = g[close_col].ewm(span=8, adjust=False).mean()
        ema21 = g[close_col].ewm(span=21, adjust=False).mean()
        g["ema8_over_ema21"] = ema8 / (ema21 + 1e-9)

        ma20 = g["ma_20"]; sd20 = g["std_20"]
        g["bb_b"] = (g[close_col] - (ma20 - 2*sd20)) / ((ma20 + 2*sd20) - (ma20 - 2*sd20) + 1e-9)

        tr = pd.concat([
            (g[high_col] - g[low_col]),
            (g[high_col] - g[close_col].shift()).abs(),
            (g[low_col]  - g[close_col].shift()).abs()
        ], axis=1).max(axis=1)
        g["atr_14"] = tr.rolling(14, min_periods=7).mean()

        if vol_col and vol_col in g.columns:
            direction = np.sign(g[close_col].diff()).fillna(0.0)
            g["obv"] = (direction * g[vol_col]).cumsum()
            vwap = (g["typical_price"] * g[vol_col]).cumsum() / (g[vol_col].cumsum() + 1e-9)
            g["vwap_ratio"] = g[close_col] / (vwap + 1e-9)
        else:
            g["obv"] = np.nan
            g["vwap_ratio"] = np.nan

        g["skew_20"] = g[close_col].pct_change().rolling(20, min_periods=10).skew()
        g["kurt_20"] = g[close_col].pct_change().rolling(20, min_periods=10).kurt()

        g["max_20"] = g[close_col].rolling(20, min_periods=10).max()
        g["min_20"] = g[close_col].rolling(20, min_periods=10).min()
        g["dist_to_max20"] = g[close_col] / (g["max_20"] + 1e-9)
        g["dist_to_min20"] = g[close_col] / (g["min_20"] + 1e-9)

        for w in [5,10,20,30,60,90]:
            g[f"prc_over_ma_{w}"] = g[close_col] / (g.get(f"ma_{w}") + 1e-9)

        day_ret = g[close_col].pct_change()
        for lag in range(1,6):
            g[f"day_ret_lag{lag}"] = day_ret.shift(lag)

        base_ex = {date_col, symbol_col, open_col, high_col, low_col, close_col, "typical_price"}
        if vol_col: base_ex.add(vol_col)
        if target_col: base_ex.add(target_col)
        feats = [c for c in g.columns if c not in base_ex]
        g[feats] = g[feats].shift(1)
        return g

    df = df.groupby(symbol_col, group_keys=False).apply(per_symbol)

    df["hl_spread"] = (df[high_col] - df[low_col]).shift(1)
    df["oc_spread"] = (df[close_col] - df[open_col]).shift(1)
    df["day_return"] = df[close_col].pct_change().shift(1)

    df["dow"] = df[date_col].dt.dayofweek
    df["month"] = df[date_col].dt.month
    df["dow_sin"] = np.sin(2*np.pi*df["dow"]/7);   df["dow_cos"] = np.cos(2*np.pi*df["dow"]/7)
    df["mon_sin"] = np.sin(2*np.pi*df["month"]/12);df["mon_cos"] = np.cos(2*np.pi*df["month"]/12)
    for c in ["dow","month","dow_sin","dow_cos","mon_sin","mon_cos"]:
        df[c] = df[c].shift(1)
    return df

train_fe = add_features(train)
test_fe  = add_features(test)

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values([symbol_col, date_col]).copy()
    df["typical_price"] = (df[high_col] + df[low_col] + df[close_col]) / 3.0

    def per_symbol(g: pd.DataFrame):
        g = g.sort_values(date_col).copy()

        for w in [1,5,10,20,30,60,90]:
            g[f"ret_{w}"] = g[close_col].pct_change(w)

        for w in [5,10,20,30,60,90]:
            g[f"ma_{w}"]  = g[close_col].rolling(w, min_periods=max(2,w//2)).mean()
            g[f"std_{w}"] = g[close_col].rolling(w, min_periods=max(2,w//2)).std()

        win = 14
        d = g[close_col].diff()
        up, dn = d.clip(lower=0), -d.clip(upper=0)
        rs = up.rolling(win, min_periods=win//2).mean() / (dn.rolling(win, min_periods=win//2).mean() + 1e-9)
        g["rsi_14"] = 100 - (100/(1+rs))

        ema12 = g[close_col].ewm(span=12, adjust=False).mean()
        ema26 = g[close_col].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        sig  = macd.ewm(span=9, adjust=False).mean()
        g["macd"] = macd; g["macd_signal"] = sig; g["macd_hist"] = macd - sig

        ema8  = g[close_col].ewm(span=8, adjust=False).mean()
        ema21 = g[close_col].ewm(span=21, adjust=False).mean()
        g["ema8_over_ema21"] = ema8 / (ema21 + 1e-9)

        ma20 = g["ma_20"]; sd20 = g["std_20"]
        g["bb_b"] = (g[close_col] - (ma20 - 2*sd20)) / ((ma20 + 2*sd20) - (ma20 - 2*sd20) + 1e-9)

        tr = pd.concat([
            (g[high_col] - g[low_col]),
            (g[high_col] - g[close_col].shift()).abs(),
            (g[low_col]  - g[close_col].shift()).abs()
        ], axis=1).max(axis=1)
        g["atr_14"] = tr.rolling(14, min_periods=7).mean()

        if vol_col and vol_col in g.columns:
            direction = np.sign(g[close_col].diff()).fillna(0.0)
            g["obv"] = (direction * g[vol_col]).cumsum()
            vwap = (g["typical_price"] * g[vol_col]).cumsum() / (g[vol_col].cumsum() + 1e-9)
            g["vwap_ratio"] = g[close_col] / (vwap + 1e-9)
        else:
            g["obv"] = np.nan
            g["vwap_ratio"] = np.nan

        g["skew_20"] = g[close_col].pct_change().rolling(20, min_periods=10).skew()
        g["kurt_20"] = g[close_col].pct_change().rolling(20, min_periods=10).kurt()

        g["max_20"] = g[close_col].rolling(20, min_periods=10).max()
        g["min_20"] = g[close_col].rolling(20, min_periods=10).min()
        g["dist_to_max20"] = g[close_col] / (g["max_20"] + 1e-9)
        g["dist_to_min20"] = g[close_col] / (g["min_20"] + 1e-9)

        for w in [5,10,20,30,60,90]:
            g[f"prc_over_ma_{w}"] = g[close_col] / (g.get(f"ma_{w}") + 1e-9)

        day_ret = g[close_col].pct_change()
        for lag in range(1,6):
            g[f"day_ret_lag{lag}"] = day_ret.shift(lag)

        base_ex = {date_col, symbol_col, open_col, high_col, low_col, close_col, "typical_price"}
        if vol_col: base_ex.add(vol_col)
        if target_col: base_ex.add(target_col)
        feats = [c for c in g.columns if c not in base_ex]
        g[feats] = g[feats].shift(1)
        return g

    df = df.groupby(symbol_col, group_keys=False).apply(per_symbol)

    df["hl_spread"] = (df[high_col] - df[low_col]).shift(1)
    df["oc_spread"] = (df[close_col] - df[open_col]).shift(1)
    df["day_return"] = df[close_col].pct_change().shift(1)

    df["dow"] = df[date_col].dt.dayofweek
    df["month"] = df[date_col].dt.month
    df["dow_sin"] = np.sin(2*np.pi*df["dow"]/7);   df["dow_cos"] = np.cos(2*np.pi*df["dow"]/7)
    df["mon_sin"] = np.sin(2*np.pi*df["month"]/12);df["mon_cos"] = np.cos(2*np.pi*df["month"]/12)
    for c in ["dow","month","dow_sin","dow_cos","mon_sin","mon_cos"]:
        df[c] = df[c].shift(1)
    return df

train_fe = add_features(train)
test_fe  = add_features(test)

if target_col is None:
    def derive_target(g: pd.DataFrame):
        g = g.sort_values(date_col).copy()
        fut = g[close_col].shift(-30)
        g["__target__"] = (fut > g[close_col]).astype(int)
        return g
    train_fe = train_fe.groupby(symbol_col, group_keys=False).apply(derive_target)
    target_col = "__target__"

train_fe = train_fe.dropna(subset=[target_col]).copy()

exclude = {target_col, date_col, symbol_col, open_col, high_col, low_col, close_col, "typical_price"}
if vol_col and vol_col in train_fe.columns: exclude.add(vol_col)
feat_cols = [c for c in train_fe.columns if c not in exclude]

for c in feat_cols:
    if train_fe[c].isna().any():
        train_fe[c] = train_fe.groupby(symbol_col)[c].transform(lambda s: s.fillna(s.median()))
        train_fe[c] = train_fe[c].fillna(train_fe[c].median())
    if c in test_fe.columns and test_fe[c].isna().any():
        test_fe[c] = test_fe.groupby(symbol_col)[c].transform(lambda s: s.fillna(s.median()))
        test_fe[c] = test_fe[c].fillna(train_fe[c].median() if c in train_fe else test_fe[c].median())

X_all = train_fe[feat_cols].values
y_all = train_fe[target_col].astype(int).values
dates_all = train_fe[date_col].values
X_test = test_fe[feat_cols].values

def ts_rolling_folds(dates: np.ndarray, n_folds=5, gap_days=5):
    d = pd.to_datetime(dates).astype("datetime64[D]").astype(int)
    qs = np.quantile(d, np.linspace(0,1,n_folds+1))
    folds = []
    for i in range(n_folds):
        v_start, v_end = int(qs[i]), int(qs[i+1])
        tr = d <= (v_start - gap_days)
        va = (d >= v_start) & (d <= v_end)
        if tr.sum() and va.sum():
            folds.append((tr, va))
    return folds

folds = ts_rolling_folds(dates_all, n_folds=N_FOLDS, gap_days=GAP_DAYS)
print(f"Using {len(folds)} purged folds.")

from catboost import CatBoostClassifier, Pool
cat_params = dict(
    loss_function="Logloss",
    eval_metric="Logloss",
    learning_rate=0.05,
    depth=8,
    l2_leaf_reg=3.0,
    random_seed=RANDOM_SEED,
    iterations=6000,
    od_type="Iter",
    od_wait=200,
    verbose=False,
    task_type="CPU"
)
oof_proba = np.zeros_like(y_all, dtype=float)
test_fold_preds = []

for i, (tr, va) in enumerate(folds, 1):
    Xtr, ytr = X_all[tr], y_all[tr]
    Xva, yva = X_all[va], y_all[va]
    m = CatBoostClassifier(**cat_params)
    m.fit(Pool(Xtr, ytr), eval_set=Pool(Xva, yva), use_best_model=True)
    oof_proba[va] = m.predict_proba(Pool(Xva, yva))[:,1]
    test_fold_preds.append(m.predict_proba(X_test)[:,1])
    print(f"Fold {i} done.")

ths = np.linspace(0.30, 0.70, 81)
accs = [((oof_proba >= t).astype(int) == y_all).mean() for t in ths]
best_t = ths[int(np.argmax(accs))]
print(f"OOF accuracy = {max(accs):.4f} @ threshold {best_t:.3f}")

test_proba = np.mean(test_fold_preds, axis=0)
test_pred = (test_proba >= best_t).astype(int)

if sample_sub is not None:
    sub = sample_sub.copy()
    pred_col = None
    for c in sub.columns:
        if c.lower() in ["target","prediction","pred","label","y"]:
            pred_col = c; break
    if pred_col is None and len(sub.columns) == 2:
        pred_col = sub.columns[1]
    if pred_col is None:
        pred_col = "Target"
        if pred_col not in sub.columns: sub[pred_col] = 0
    if id_col and (id_col in sub.columns) and (id_col in test.columns):
        sub = sub.set_index(id_col).reindex(test[id_col].values).reset_index()
    sub[pred_col] = test_pred
else:
    if id_col and id_col in test.columns:
        sub = pd.DataFrame({id_col: test[id_col].values, "Target": test_pred})
    else:
        sub = pd.DataFrame({"row_id": np.arange(len(test_pred)), "Target": test_pred})

sub.to_csv("submission.csv", index=False)
print("Saved submission.csv", sub.shape)
