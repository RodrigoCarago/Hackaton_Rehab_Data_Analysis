from __future__ import annotations

import io
import json
from typing import Any

import numpy as np
import scipy.io
import scipy.linalg
import scipy.signal
from mne.filter import filter_data as mne_filter_data
from mne.filter import notch_filter as mne_notch_filter
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold

app = FastAPI(title="Stroke Rehab BCI API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_mat(raw_bytes: bytes, source: str) -> tuple[np.ndarray, np.ndarray, float, list[str]]:
    mat = scipy.io.loadmat(io.BytesIO(raw_bytes), squeeze_me=True, struct_as_record=False)
    keys = [k for k in mat if not k.startswith("__")]
    if not keys:
        raise ValueError(f"No variable found in {source}")

    obj = mat[keys[0]]
    if hasattr(obj, "DataEEG"):
        obj = obj.DataEEG

    x = np.asarray(getattr(obj, "x"), dtype=np.float64)
    y = np.asarray(getattr(obj, "y"), dtype=np.int64).reshape(-1)
    s = float(np.asarray(getattr(obj, "s")).reshape(-1)[0])
    c = [str(ch).strip() for ch in np.asarray(getattr(obj, "c")).reshape(-1).tolist()]

    if x.ndim != 3:
        raise ValueError(f"{source}: x must be 3D, got {x.shape}")
    if x.shape[2] != y.shape[0]:
        raise ValueError(f"{source}: trials mismatch x={x.shape}, y={y.shape}")

    x_tct = np.transpose(x, (2, 1, 0))  # (trials, channels, time)
    return x_tct, y, s, c


def _select_channels(
    x: np.ndarray, src_chs: list[str], sel_chs: list[str]
) -> tuple[np.ndarray, list[str]]:
    src_map = {ch: i for i, ch in enumerate(src_chs)}
    valid = [ch for ch in sel_chs if ch in src_map]
    if not valid:
        return x, src_chs
    idxs = [src_map[ch] for ch in valid]
    return x[:, idxs, :], valid


# ── Signal processing ─────────────────────────────────────────────────────────

def _apply_notch(x: np.ndarray, sfreq: float, freqs: tuple[float, ...] = (50.0, 60.0)) -> np.ndarray:
    """Notch filter using MNE's FIR implementation (same as raw.notch_filter())."""
    nyq = sfreq / 2.0
    valid = [f for f in freqs if 0 < f < nyq]
    if not valid:
        return x.astype(np.float64)
    out = np.empty_like(x, dtype=np.float64)
    for i in range(x.shape[0]):
        # mne_notch_filter expects (n_channels, n_times)
        out[i] = mne_notch_filter(
            x[i].astype(np.float64), Fs=sfreq, freqs=valid, verbose=False
        )
    return out


def _apply_bandpass(x: np.ndarray, sfreq: float, lo: float, hi: float) -> np.ndarray:
    """Bandpass filter using MNE's FIR implementation (same as raw.filter())."""
    nyq = sfreq / 2.0
    l_freq = lo  if lo  > 0        else None
    h_freq = hi  if hi  < nyq      else None
    out = np.empty_like(x, dtype=np.float64)
    for i in range(x.shape[0]):
        # mne filter_data expects (..., n_times) — (n_channels, n_times) here
        out[i] = mne_filter_data(
            x[i].astype(np.float64),
            sfreq=sfreq,
            l_freq=l_freq,
            h_freq=h_freq,
            method="fir",
            verbose=False,
        )
    return out


def _zscore_normalize(
    train: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mu = train.mean(axis=(0, 2), keepdims=True)
    sd = train.std(axis=(0, 2), keepdims=True) + 1e-8
    return (train - mu) / sd, (test - mu) / sd


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _apply_optional_autoreject(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    sfreq: float,
    ch_names: list[str],
    enabled: bool,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if not enabled:
        keep_train = np.ones(x_train.shape[0], dtype=bool)
        keep_test = np.ones(x_test.shape[0], dtype=bool)
        return x_train, y_train, x_test, y_test, keep_train, keep_test, {
            "enabled": False,
            "applied": False,
            "dropped_train": 0,
            "dropped_test": 0,
            "modified_train": 0,
            "modified_test": 0,
        }

    try:
        import mne
        from autoreject import AutoReject
    except ImportError as exc:
        raise ValueError(
            "AutoReject requested but dependency is missing. Install with: pip install autoreject"
        ) from exc

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=["eeg"] * len(ch_names))

    def _build_epochs(x: np.ndarray) -> mne.EpochsArray:
        events = np.column_stack(
            [np.arange(x.shape[0], dtype=int), np.zeros(x.shape[0], dtype=int), np.ones(x.shape[0], dtype=int)]
        )
        return mne.EpochsArray(x, info, events=events, event_id={"mi": 1}, tmin=0.0, baseline=None, verbose=False)

    try:
        train_epochs = _build_epochs(x_train)
        test_epochs = _build_epochs(x_test)
        ar = AutoReject(random_state=seed, verbose=False)

        train_clean, train_log = ar.fit_transform(train_epochs, return_log=True)
        test_clean, test_log = ar.transform(test_epochs, return_log=True)
    except Exception as exc:
        raise ValueError(f"AutoReject failed: {exc}") from exc

    def _count_modified(log: Any) -> int:
        labels = np.asarray(getattr(log, "labels", []))
        if labels.ndim != 2 or labels.size == 0:
            return int(np.sum(getattr(log, "bad_epochs", np.array([], dtype=bool))))
        return int(np.sum(np.any(labels != 0, axis=1)))

    keep_train = ~train_log.bad_epochs
    keep_test = ~test_log.bad_epochs
    x_train_clean = train_clean.get_data(copy=True)
    x_test_clean = test_clean.get_data(copy=True)
    y_train_clean = y_train[keep_train]
    y_test_clean = y_test[keep_test]

    if x_train_clean.shape[0] < 10 or len(np.unique(y_train_clean)) < 2:
        raise ValueError("AutoReject removed too many training trials; insufficient data for classification")
    if x_test_clean.shape[0] < 2 or len(np.unique(y_test_clean)) < 2:
        raise ValueError("AutoReject removed too many test trials; insufficient class coverage")

    return x_train_clean, y_train_clean, x_test_clean, y_test_clean, keep_train, keep_test, {
        "enabled": True,
        "applied": True,
        "dropped_train": int(np.sum(~keep_train)),
        "dropped_test": int(np.sum(~keep_test)),
        "modified_train": _count_modified(train_log),
        "modified_test": _count_modified(test_log),
    }


# ── CSP ───────────────────────────────────────────────────────────────────────

def _fit_csp(x_train: np.ndarray, y_train: np.ndarray, n_components: int) -> np.ndarray:
    """Fit CSP spatial filters. Returns W: (n_components, channels)."""
    classes = np.unique(y_train)
    n_ch = x_train.shape[1]
    reg = 0.05

    covs = []
    for c in classes:
        xc = x_train[y_train == c]
        cov = np.mean([np.cov(t) + reg * np.eye(n_ch) for t in xc], axis=0)
        covs.append(cov)

    C0, C1 = covs[0], covs[1]
    Ctot = C0 + C1

    try:
        _, vecs = scipy.linalg.eigh(C0, Ctot)
    except Exception:
        _, vecs = np.linalg.eigh(C0)

    # eigh returns ascending order; select first (low eigenvalue) and last (high)
    half = max(1, n_components // 2)
    idx = np.concatenate([np.arange(half), np.arange(vecs.shape[1] - half, vecs.shape[1])])
    return vecs[:, idx].T  # (n_components, channels)


def _apply_csp(x: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Apply CSP and return log-variance features. Returns (trials, n_components)."""
    feats = []
    for trial in x:
        z = W @ trial
        feats.append(np.log(np.var(z, axis=1) + 1e-10))
    return np.array(feats)


def _fbcsp_features(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    sfreq: float,
    n_csp: int,
    k_best: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Filter Bank CSP with feature selection."""
    bands = [(8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32)]
    all_tr: list[np.ndarray] = []
    all_te: list[np.ndarray] = []

    for lo, hi in bands:
        if hi >= sfreq / 2:
            continue
        try:
            x_tr_f = _apply_bandpass(x_train, sfreq, lo, hi)
            x_te_f = _apply_bandpass(x_test, sfreq, lo, hi)
            W = _fit_csp(x_tr_f, y_train, n_csp)
            all_tr.append(_apply_csp(x_tr_f, W))
            all_te.append(_apply_csp(x_te_f, W))
        except Exception:
            continue

    if not all_tr:
        return None, None

    X_tr = np.hstack(all_tr)
    X_te = np.hstack(all_te)
    k = min(k_best, X_tr.shape[1])

    sel = SelectKBest(f_classif, k=k)
    return sel.fit_transform(X_tr, y_train), sel.transform(X_te)


# ── Metrics ───────────────────────────────────────────────────────────────────

def _compute_itr(accuracy_pct: float, n_classes: int, trial_s: float) -> float:
    """Information Transfer Rate (bits/min)."""
    p = accuracy_pct / 100.0
    n = max(2, n_classes)
    p = float(np.clip(p, 1.0 / n + 1e-6, 1.0 - 1e-6))
    if trial_s <= 0:
        return 0.0
    itr_trial = np.log2(n) + p * np.log2(p) + (1 - p) * np.log2((1 - p) / (n - 1))
    return float(max(0.0, itr_trial * 60.0 / trial_s))


def _pipeline_metrics(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    trial_s: float,
) -> dict[str, Any]:
    """Fit LDA and return metrics, roc, confusion."""
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    lda.fit(X_train, y_train)
    y_pred = lda.predict(X_test)
    n_classes = len(np.unique(np.concatenate([y_train, y_test])))

    acc = float(accuracy_score(y_test, y_pred) * 100)
    kappa = float(cohen_kappa_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    prec = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    rec = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    itr = _compute_itr(acc, n_classes, trial_s)

    roc_auc = 0.5
    fpr_d: list[float] = [0.0, 1.0]
    tpr_d: list[float] = [0.0, 1.0]

    try:
        y_proba = lda.predict_proba(X_test)
        if n_classes == 2:
            roc_auc = float(roc_auc_score(y_test, y_proba[:, 1]))
            fpr_arr, tpr_arr, _ = roc_curve(y_test, y_proba[:, 1])
            n_pts = min(100, len(fpr_arr))
            idx = np.linspace(0, len(fpr_arr) - 1, n_pts).astype(int)
            fpr_d = fpr_arr[idx].tolist()
            tpr_d = tpr_arr[idx].tolist()
        else:
            roc_auc = float(roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro"))
    except Exception:
        pass

    cm = confusion_matrix(y_test, y_pred).tolist()

    return {
        "metrics": {
            "accuracy": round(acc, 4),
            "kappa": round(kappa, 4),
            "f1": round(f1, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "roc_auc": round(roc_auc, 4),
            "itr": round(itr, 2),
        },
        "roc": {"fpr": fpr_d, "tpr": tpr_d},
        "confusion": cm,
    }


def _metrics_from_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    trial_s: float,
    labels: np.ndarray,
) -> dict[str, Any]:
    n_classes = len(labels)
    acc = float(accuracy_score(y_true, y_pred) * 100)
    kappa = float(cohen_kappa_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    itr = _compute_itr(acc, n_classes, trial_s)

    roc_auc = 0.5
    fpr_d: list[float] = [0.0, 1.0]
    tpr_d: list[float] = [0.0, 1.0]
    try:
        if n_classes == 2:
            roc_auc = float(roc_auc_score(y_true, y_proba[:, 1]))
            fpr_arr, tpr_arr, _ = roc_curve(y_true, y_proba[:, 1], pos_label=labels[1])
            n_pts = min(100, len(fpr_arr))
            idx = np.linspace(0, len(fpr_arr) - 1, n_pts).astype(int)
            fpr_d = fpr_arr[idx].tolist()
            tpr_d = tpr_arr[idx].tolist()
        else:
            roc_auc = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
    except Exception:
        pass

    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return {
        "metrics": {
            "accuracy": round(acc, 4),
            "kappa": round(kappa, 4),
            "f1": round(f1, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "roc_auc": round(roc_auc, 4),
            "itr": round(itr, 2),
        },
        "roc": {"fpr": fpr_d, "tpr": tpr_d},
        "confusion": cm,
    }


def _pipeline_cv_metrics(
    x: np.ndarray,
    y: np.ndarray,
    trial_s: float,
    seed: int,
    *,
    mode: str,
    n_comp: int,
    sfreq: float,
    fbcsp_k_best: int,
) -> dict[str, Any]:
    labels = np.unique(y)
    class_counts = np.bincount(np.searchsorted(labels, y))
    if np.min(class_counts) < 5:
        raise ValueError("Need at least 5 epochs per class to run 5-fold cross-validation")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    y_true_parts: list[np.ndarray] = []
    y_pred_parts: list[np.ndarray] = []
    y_proba_parts: list[np.ndarray] = []

    for train_idx, val_idx in skf.split(np.zeros(y.shape[0]), y):
        x_tr, y_tr = x[train_idx], y[train_idx]
        x_val, y_val = x[val_idx], y[val_idx]

        if mode == "csp":
            W = _fit_csp(x_tr, y_tr, n_comp)
            X_tr = _apply_csp(x_tr, W)
            X_val = _apply_csp(x_val, W)
        else:
            X_tr, X_val = _fbcsp_features(
                x_tr,
                y_tr,
                x_val,
                sfreq,
                max(2, n_comp // 2),
                fbcsp_k_best,
            )
            if X_tr is None or X_val is None:
                W = _fit_csp(x_tr, y_tr, n_comp)
                X_tr = _apply_csp(x_tr, W)
                X_val = _apply_csp(x_val, W)

        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        lda.fit(X_tr, y_tr)
        y_pred = lda.predict(X_val)
        y_proba_fold = np.zeros((X_val.shape[0], labels.shape[0]), dtype=np.float64)
        y_proba_lda = lda.predict_proba(X_val)
        for col, cls in enumerate(lda.classes_):
            label_idx = int(np.where(labels == cls)[0][0])
            y_proba_fold[:, label_idx] = y_proba_lda[:, col]

        y_true_parts.append(y_val)
        y_pred_parts.append(y_pred)
        y_proba_parts.append(y_proba_fold)

    y_true_all = np.concatenate(y_true_parts)
    y_pred_all = np.concatenate(y_pred_parts)
    y_proba_all = np.concatenate(y_proba_parts, axis=0)
    return _metrics_from_predictions(y_true_all, y_pred_all, y_proba_all, trial_s, labels)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _decimate(arr: np.ndarray, n: int = 200) -> list[float]:
    if arr.shape[-1] <= n:
        return arr.tolist()
    idx = np.linspace(0, arr.shape[-1] - 1, n).astype(int)
    return arr[..., idx].tolist()


def _decimate1d(arr: np.ndarray, n: int = 120) -> list[float]:
    if len(arr) <= n:
        return arr.tolist()
    idx = np.linspace(0, len(arr) - 1, n).astype(int)
    return arr[idx].tolist()


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/preprocess")
@app.post("/process")
async def process(
    training_file: UploadFile = File(...),
    test_file: UploadFile = File(...),
    high_pass_hz: float = Form(1.0),
    low_pass_hz: float = Form(40.0),
    epoch_tmin: float = Form(2.0),
    epoch_tmax: float = Form(6.0),
    csp_components: int = Form(4),
    fbcsp_k_best: int = Form(12),
    seed: int = Form(42),
    selected_channels: str = Form("[]"),
    use_autoreject: str = Form("false"),
) -> dict[str, Any]:
    try:
        tr_bytes = await training_file.read()
        te_bytes = await test_file.read()

        x_train, y_train, sfreq, ch_train = _load_mat(tr_bytes, training_file.filename or "train")
        x_test, y_test, sfreq_te, ch_test = _load_mat(te_bytes, test_file.filename or "test")

        if abs(sfreq - sfreq_te) > 1e-3:
            raise ValueError(f"Sampling rate mismatch: {sfreq} vs {sfreq_te}")

        # Channel selection
        try:
            sel_chs: list[str] = json.loads(selected_channels)
        except Exception:
            sel_chs = []

        if sel_chs:
            x_train, ch_train = _select_channels(x_train, ch_train, sel_chs)
            x_test, _ = _select_channels(x_test, ch_test, sel_chs)

        x_train_raw = x_train.copy()

        # Filtering
        x_train = _apply_notch(x_train, sfreq)
        x_test = _apply_notch(x_test, sfreq)
        x_train = _apply_bandpass(x_train, sfreq, high_pass_hz, low_pass_hz)
        x_test = _apply_bandpass(x_test, sfreq, high_pass_hz, low_pass_hz)

        # Epoch cropping
        t_start = max(0, int(epoch_tmin * sfreq))
        t_end = min(x_train.shape[2], int(epoch_tmax * sfreq))
        if t_end > t_start + 10:
            x_train = x_train[:, :, t_start:t_end]
            x_test = x_test[:, :, t_start:t_end]
            x_train_raw = x_train_raw[:, :, t_start:t_end]

        # Optional artifact rejection
        x_train, y_train, x_test, y_test, keep_train, keep_test, autoreject_info = _apply_optional_autoreject(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            sfreq=sfreq,
            ch_names=ch_train,
            enabled=_to_bool(use_autoreject),
            seed=seed,
        )
        # Keep RAW visualization aligned with the exact epochs used after AutoReject.
        x_train_raw = x_train_raw[keep_train]

        x_train, x_test = _zscore_normalize(x_train, x_test)

        trial_s = x_train.shape[2] / sfreq
        n_comp = int(np.clip(csp_components, 2, min(x_train.shape[1] - 1, 8)))

        # ── CSP + LDA ────────────────────────────────────────────
        W_csp = _fit_csp(x_train, y_train, n_comp)
        X_csp_tr = _apply_csp(x_train, W_csp)
        X_csp_te = _apply_csp(x_test, W_csp)
        csp_result = _pipeline_metrics(X_csp_tr, y_train, X_csp_te, y_test, trial_s)

        # ── FBCSP + LDA ──────────────────────────────────────────
        X_fb_tr, X_fb_te = _fbcsp_features(
            x_train, y_train, x_test, sfreq,
            max(2, n_comp // 2), fbcsp_k_best,
        )
        fb_result = (
            _pipeline_metrics(X_fb_tr, y_train, X_fb_te, y_test, trial_s)
            if X_fb_tr is not None and X_fb_te is not None
            else csp_result
        )

        # 5-fold CV metrics for session-level comparison
        x_all = np.concatenate([x_train, x_test], axis=0)
        y_all = np.concatenate([y_train, y_test], axis=0)
        csp_cv_result = _pipeline_cv_metrics(
            x=x_all,
            y=y_all,
            trial_s=trial_s,
            seed=seed,
            mode="csp",
            n_comp=n_comp,
            sfreq=sfreq,
            fbcsp_k_best=fbcsp_k_best,
        )
        fbcsp_cv_result = _pipeline_cv_metrics(
            x=x_all,
            y=y_all,
            trial_s=trial_s,
            seed=seed,
            mode="fbcsp",
            n_comp=n_comp,
            sfreq=sfreq,
            fbcsp_k_best=fbcsp_k_best,
        )

        # ── Visualization ────────────────────────────────────────
        vis_chs = min(4, x_train.shape[1])
        raw_lines = [_decimate(x_train_raw[0, i], 220) for i in range(vis_chs)]
        filt_lines = [_decimate(x_train[0, i], 220) for i in range(vis_chs)]

        ch0 = 0
        f_tr, p_tr = scipy.signal.welch(x_train[0, ch0], fs=sfreq, nperseg=min(256, x_train.shape[2]))
        f_te, p_te = scipy.signal.welch(x_test[0, ch0], fs=sfreq, nperseg=min(256, x_test.shape[2]))
        mask = (f_tr >= 0) & (f_tr <= 45)

        return {
            "fs": sfreq,
            "n_epochs": {"train": int(x_train.shape[0]), "test": int(x_test.shape[0])},
            "channel_names": ch_train[:vis_chs],
            "temporal": {
                "raw": raw_lines,
                "filtered": filt_lines,
                "n_samples": min(220, x_train.shape[2]),
                "duration_s": round(trial_s, 2),
            },
            "psd": {
                "freqs": _decimate1d(f_tr[mask]),
                "train": _decimate1d(p_tr[mask]),
                "test": _decimate1d(p_te[mask]),
            },
            "metrics": {
                "csp_lda": csp_cv_result["metrics"],
                "fbcsp_lda": fbcsp_cv_result["metrics"],
                "evaluation": {
                    "method": "stratified_kfold",
                    "folds": 5,
                    "dataset": "train+test",
                },
            },
            "confusion": {
                "csp_lda": csp_cv_result["confusion"],
                "fbcsp_lda": fbcsp_cv_result["confusion"],
            },
            "roc": {
                "csp_lda": csp_cv_result["roc"],
                "fbcsp_lda": fbcsp_cv_result["roc"],
            },
            "holdout": {
                "metrics": {
                    "csp_lda": csp_result["metrics"],
                    "fbcsp_lda": fb_result["metrics"],
                }
            },
            "preprocessing": {
                "autoreject": autoreject_info,
            },
        }

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}") from exc
