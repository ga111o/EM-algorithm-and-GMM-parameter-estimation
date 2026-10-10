"""Small GMM experiment (EXPERIMENTS.md): multi-start EM vs parametric DSM.

Usage: uv run python run_small.py
Writes data/, results/ next to this file. Figures/tables: report_assets.py.
"""
import csv
import hashlib
import itertools
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import scipy
import scipy.linalg
import sklearn
import torch
import torch.nn.functional as F
from scipy.special import logsumexp
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / "config.json").read_text())
DATA, RES = ROOT / "data", ROOT / "results"
PAR = RES / "parameters"
K, D, LAM0, R = CFG["K"], CFG["d"], CFG["lambda0"], CFG["n_restarts"]
SEEDS = CFG["seeds"]
LOG2PI = np.log(2 * np.pi)
torch.set_default_dtype(torch.float64)
torch.set_num_threads(CFG["torch_threads"])


class NumericalFailure(Exception):
    pass


# ---------------------------------------------------------------- GMM utilities
def log_comp(X, mu, Sigma):
    """log N(x; mu_k, Sigma_k), shape (n, K). Raises LinAlgError if not PD."""
    out = np.empty((len(X), K))
    for k in range(K):
        L = np.linalg.cholesky(Sigma[k])
        z = scipy.linalg.solve_triangular(L, (X - mu[k]).T, lower=True)
        out[:, k] = -0.5 * (z**2).sum(0) - np.log(np.diag(L)).sum() - 0.5 * D * LOG2PI
    return out


def log_density(X, pi, mu, Sigma):
    return logsumexp(np.log(pi) + log_comp(X, mu, Sigma), axis=1)


def min_eig(Sigma):
    return float(np.linalg.eigvalsh(Sigma).min())


def match_perm(pi_h, mu_h, Sig_h, pi, mu, Sig):
    """Single permutation rho (estimated index rho[k] for true k) minimizing total cost."""
    C = (((mu_h[:, None] - mu[None]) ** 2).sum(-1) / D
         + ((Sig_h[:, None] - Sig[None]) ** 2).sum((-1, -2)) / D**2
         + (pi_h[:, None] - pi[None]) ** 2)
    return min(itertools.permutations(range(K)), key=lambda p: sum(C[p[k], k] for k in range(K)))


def param_errors(est, true):
    p = list(match_perm(*est, *true))
    pi_h, mu_h, Sig_h = (x[p] for x in est)
    pi, mu, Sig = true
    return dict(E_pi=float(np.sqrt(((pi_h - pi) ** 2).sum() / K)),
                E_mu=float(np.sqrt(((mu_h - mu) ** 2).sum() / (K * D))),
                E_Sigma=float(np.sqrt(((Sig_h - Sig) ** 2).sum() / (K * D * D))))


def streams(seed):
    ss = np.random.SeedSequence([CFG["base_entropy"], seed])
    return dict(zip(CFG["rng_streams"], ss.spawn(len(CFG["rng_streams"]))))


# ---------------------------------------------------------------- data
def random_cov(rng):
    lam = rng.uniform(*CFG["cov_eig_range"], size=D)
    Q, Rm = np.linalg.qr(rng.standard_normal((D, D)))
    Q = Q * np.where(np.diag(Rm) < 0, -1.0, 1.0)
    return Q @ np.diag(lam) @ Q.T


def sample_gmm(rng, n, pi, mu, Sigma):
    z = rng.choice(K, size=n, p=pi)
    L = np.linalg.cholesky(Sigma)
    return mu[z] + np.einsum("nij,nj->ni", L[z], rng.standard_normal((n, D)))


def make_data(name, seed):
    c = CFG["conditions"][name]
    st = streams(seed)
    pi, a = np.array(c["pi"]), c["a"]
    mu = np.array([[-a, 0.0], [0.0, 0.0], [a, 0.0]])
    rc = np.random.default_rng(st["covariance"])
    Sigma = np.stack([random_cov(rc) for _ in range(K)])
    rd = np.random.default_rng(st["data"])
    Xtr = sample_gmm(rd, CFG["n_train"], pi, mu, Sigma)
    Xva = sample_gmm(rd, CFG["n_val"], pi, mu, Sigma)
    Xte = sample_gmm(rd, CFG["n_test"], pi, mu, Sigma)

    # initial values: shared covariance (1/n), eigenvalues floored strictly above lambda0
    s, U = np.linalg.eigh(np.cov(Xtr.T, bias=True))
    floor_active = bool((s < CFG["init_cov_eig_floor"]).any())
    S0 = (U * np.maximum(s, CFG["init_cov_eig_floor"])) @ U.T
    ri = np.random.default_rng(st["init"])
    init_idx = np.stack([ri.choice(CFG["n_train"], K, replace=False) for _ in range(R)])

    # controlled anomalies from the true density
    ac = CFG["anomaly"]
    sd = np.sqrt(np.diagonal(Sigma, axis1=1, axis2=2))
    lo = (mu - ac["box_halfwidth_sd"] * sd).min(0)
    hi = (mu + ac["box_halfwidth_sd"] * sd).max(0)
    cstar = float(np.quantile(np.exp(log_density(Xva, pi, mu, Sigma)), ac["density_quantile"]))
    ra = np.random.default_rng(st["anomaly"])
    acc, n_acc, attempts = [], 0, 0
    while n_acc < CFG["n_anomaly"]:
        cand = ra.uniform(lo, hi, size=(ac["candidate_batch"], D))
        ok = np.flatnonzero(np.exp(log_density(cand, pi, mu, Sigma)) < cstar)
        need = CFG["n_anomaly"] - n_acc
        if len(ok) >= need:
            attempts += int(ok[need - 1]) + 1
            acc.append(cand[ok[:need]])
            n_acc += need
        else:
            attempts += len(cand)
            acc.append(cand[ok])
            n_acc += len(ok)
    d = dict(pi=pi, mu=mu, Sigma=Sigma, Xtr=Xtr, Xva=Xva, Xte=Xte, Xan=np.concatenate(acc),
             init_idx=init_idx, pi0=np.full(K, 1.0 / K), Sigma0=S0, init_floor_active=floor_active,
             box_lo=lo, box_hi=hi, cstar=cstar, anomaly_attempts=attempts)
    np.savez(DATA / f"{name}_seed{seed}.npz", **d)
    return d


def init_params(d, r):
    return d["pi0"].copy(), d["Xtr"][d["init_idx"][r]].copy(), np.repeat(d["Sigma0"][None], K, 0)


# ---------------------------------------------------------------- EM
def e_step(X, pi, mu, Sigma):
    lc = np.log(pi) + log_comp(X, mu, Sigma)
    lp = logsumexp(lc, axis=1)
    return lp.sum(), np.exp(lc - lp[:, None])


def m_step(X, Rsp):
    Nk = Rsp.sum(0)
    if not np.all(np.isfinite(Nk)) or np.any(Nk <= 0):
        raise NumericalFailure("N_k = 0 or non-finite")
    pi = Nk / len(X)
    mu = (Rsp.T @ X) / Nk[:, None]
    Sigma = np.empty((K, D, D))
    for k in range(K):
        Xc = X - mu[k]
        s, U = np.linalg.eigh((Rsp[:, k, None] * Xc).T @ Xc / Nk[k])
        Sigma[k] = (U * np.maximum(s, LAM0)) @ U.T  # constrained M-step, Sigma >= lambda0 I
    if not all(np.all(np.isfinite(x)) for x in (pi, mu, Sigma)):
        raise NumericalFailure("non-finite parameter")
    return pi, mu, Sigma


def run_em(name, seed, d, r):
    ec = CFG["em"]
    X, n = d["Xtr"], len(d["Xtr"])
    pi, mu, Sigma = init_params(d, r)
    t0 = time.perf_counter()
    rows = []

    def record(it, ll, delta):
        flag = ""
        if np.isfinite(delta) and delta < 0:
            flag = "decrease_beyond_tol" if delta < -ec["delta_num_rel"] * (1 + abs(ll - delta)) else "roundoff_negative"
        rows.append(dict(condition=name, seed=seed, restart=r, iteration=it, loglik=ll, delta=delta,
                         train_nll=-ll / n, val_nll=-log_density(d["Xva"], pi, mu, Sigma).mean(),
                         min_cov_eig=min(min_eig(S) for S in Sigma), min_weight=pi.min(),
                         elapsed_s=time.perf_counter() - t0, flag=flag))

    ll, Rsp = e_step(X, pi, mu, Sigma)
    record(0, ll, np.nan)
    reason = "max_iter"
    for it in range(1, ec["max_iter"] + 1):
        try:
            pi_n, mu_n, Sig_n = m_step(X, Rsp)
            ll_n, Rsp = e_step(X, pi_n, mu_n, Sig_n)
            if not np.isfinite(ll_n):
                raise NumericalFailure("non-finite log-likelihood")
        except (NumericalFailure, np.linalg.LinAlgError):
            reason = "numerical_failure"
            break
        pi, mu, Sigma = pi_n, mu_n, Sig_n
        delta = ll_n - ll
        record(it, ll_n, delta)
        if 0 <= delta <= ec["tol_rel"] * (1 + abs(ll)):
            ll = ll_n
            reason = "tolerance"
            break
        ll = ll_n
    return rows, (pi, mu, Sigma), reason, time.perf_counter() - t0


def em_one_more(X, params):
    """Changes after one extra EM update from the final parameters."""
    ll, Rsp = e_step(X, *params)
    new = m_step(X, Rsp)
    ll2, _ = e_step(X, *new)
    return dict(extra_dpi=float(np.abs(new[0] - params[0]).max()),
                extra_dmu=float(np.abs(new[1] - params[1]).max()),
                extra_dSigma=float(np.abs(new[2] - params[2]).max()),
                extra_dloglik=float(ll2 - ll))


# ---------------------------------------------------------------- DSM
def inv_softplus(y):
    return np.log(np.expm1(y))


def dsm_unpack(u, mu, Lraw):
    L = torch.tril(Lraw, -1) + torch.diag_embed(F.softplus(torch.diagonal(Lraw, dim1=-2, dim2=-1)))
    return F.log_softmax(u, 0), mu, L @ L.mT + LAM0 * torch.eye(D)


def dsm_loss(u, mu, Lraw, x0, t, eps):
    logpi, mu, Sigma = dsm_unpack(u, mu, Lraw)
    a, b = np.exp(-t), np.sqrt(1 - np.exp(-2 * t))
    xt = a * x0 + b * eps
    Lc = torch.linalg.cholesky(a**2 * Sigma + b**2 * torch.eye(D))  # C_{k,t}
    diff = (a * mu[None] - xt[:, None]).unsqueeze(-1)  # (n,K,D,1)
    z = torch.linalg.solve_triangular(Lc, diff, upper=False)
    logN = (-0.5 * (z**2).sum((-1, -2)) - torch.log(torch.diagonal(Lc, dim1=-2, dim2=-1)).sum(-1)
            - 0.5 * D * LOG2PI)
    rk = torch.softmax(logpi + logN, dim=1)
    score = (rk[..., None] * torch.cholesky_solve(diff, Lc).squeeze(-1)).sum(1)
    return (b**2 * ((score + eps / b) ** 2).sum(1)).mean()


def to_numpy_params(u, mu, Lraw):
    with torch.no_grad():
        logpi, mu, Sigma = dsm_unpack(u, mu, Lraw)
    return np.exp(logpi.numpy()), mu.detach().numpy().copy(), Sigma.numpy().copy()


def run_dsm(name, seed, d, r):
    dc = CFG["dsm"]
    tlev = CFG["conditions"][name]["dsm_t"]
    children = streams(seed)["dsm"].spawn(R + 1)
    rng = np.random.default_rng(children[r])
    eval_eps = torch.tensor(np.random.default_rng(children[R]).standard_normal((len(tlev), CFG["n_val"], D)))
    Xtr, Xva = torch.tensor(d["Xtr"]), torch.tensor(d["Xva"])
    pi0, mu0, S0 = init_params(d, r)
    Lraw0 = np.linalg.cholesky(S0 - LAM0 * np.eye(D))
    idx = np.arange(D)
    Lraw0[:, idx, idx] = inv_softplus(Lraw0[:, idx, idx])
    u = torch.tensor(np.log(pi0), requires_grad=True)
    mu = torch.tensor(mu0, requires_grad=True)
    Lraw = torch.tensor(Lraw0, requires_grad=True)
    opt = torch.optim.Adam([u, mu, Lraw], lr=dc["lr"])
    t0 = time.perf_counter()
    rows, ckpts = [], []

    def checkpoint(step, train_loss):
        p = to_numpy_params(u, mu, Lraw)
        with torch.no_grad():
            ev = float(np.mean([dsm_loss(u, mu, Lraw, Xva, t, eval_eps[i]).item() for i, t in enumerate(tlev)]))
        rows.append(dict(condition=name, seed=seed, restart=r, update=step, train_loss_window=train_loss,
                         eval_dsm_loss=ev, train_nll=-log_density(d["Xtr"], *p).mean(),
                         val_nll=-log_density(d["Xva"], *p).mean(),
                         min_cov_eig=min(min_eig(S) for S in p[2]), min_weight=p[0].min(),
                         elapsed_s=time.perf_counter() - t0))
        ckpts.append((step, p))

    checkpoint(0, np.nan)
    status, acc = "completed", 0.0
    for step in range(1, dc["n_updates"] + 1):
        bi = rng.choice(CFG["n_train"], dc["batch_size"], replace=False)
        t = tlev[rng.integers(len(tlev))]
        eps = torch.tensor(rng.standard_normal((dc["batch_size"], D)))
        try:
            loss = dsm_loss(u, mu, Lraw, Xtr[bi], t, eps)
            if not torch.isfinite(loss):
                raise NumericalFailure("non-finite loss")
            opt.zero_grad()
            loss.backward()
            opt.step()
            if not all(torch.isfinite(x).all() for x in (u, mu, Lraw)):
                raise NumericalFailure("non-finite parameter")
        except (NumericalFailure, torch.linalg.LinAlgError):
            status = "numerical_failure"
            break
        acc += loss.item()
        if step % dc["ckpt_every"] == 0:
            checkpoint(step, acc / dc["ckpt_every"])
            acc = 0.0
    return rows, ckpts, status, time.perf_counter() - t0


# ---------------------------------------------------------------- evaluation
def detection(d, params):
    ac = CFG["anomaly"]
    s_val = -log_density(d["Xva"], *params)
    s_te, s_an = -log_density(d["Xte"], *params), -log_density(d["Xan"], *params)
    thr = float(np.quantile(s_val, ac["threshold_quantile"]))
    tp, fp = int((s_an > thr).sum()), int((s_te > thr).sum())
    y = np.r_[np.zeros(len(s_te)), np.ones(len(s_an))]
    s = np.r_[s_te, s_an]
    prec, rec, _ = precision_recall_curve(y, s)
    return dict(threshold=thr, test_fpr=fp / len(s_te), recall=tp / len(s_an),
                precision=tp / (tp + fp) if tp + fp > 0 else float("nan"), n_detected=tp + fp,
                auroc=float(roc_auc_score(y, s)), pr_auc_trapz=float(auc(rec, prec)))


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for row in rows:
            w.writerow({k: repr(float(v)) if isinstance(v, (float, np.floating)) else v for k, v in row.items()})


def environment():
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "")
    sha = lambda p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
    return dict(python=sys.version, platform=platform.platform(), numpy=np.__version__, scipy=scipy.__version__,
                torch=torch.__version__, sklearn=sklearn.__version__, cpu=cpu, gpu_present=gpu,
                device_used="cpu", dtype=CFG["dtype"], torch_threads=torch.get_num_threads(),
                thread_env={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
                code_sha256={p: sha(p) for p in ("run_small.py", "config.json")},
                started=time.strftime("%Y-%m-%dT%H:%M:%S%z"))


def save_params(path, **arrays):
    np.savez(PAR / path, **arrays)


def main():
    for p in (DATA, RES, PAR):
        p.mkdir(parents=True, exist_ok=True)
    env = environment()
    (RES / "environment.json").write_text(json.dumps(env, indent=2))
    t_all = time.perf_counter()

    em_hist, dsm_hist, runs, diag, pairs, by_seed, det = [], [], [], [], [], [], []
    datasets, em_final = {}, {}
    for name, c in CFG["conditions"].items():
        for seed in SEEDS:
            src = c.get("reuse", name)
            if (src, seed) not in datasets:
                datasets[(src, seed)] = make_data(src, seed)
            d = datasets[(src, seed)]
            true = (d["pi"], d["mu"], d["Sigma"])

            def metrics(params):
                return dict(**param_errors(params, true), train_loglik=float(log_density(d["Xtr"], *params).sum()),
                            val_nll=float(-log_density(d["Xva"], *params).mean()),
                            test_nll=float(-log_density(d["Xte"], *params).mean()))

            # ---- EM (experiment A); reused for conditions with "reuse"
            if "reuse" not in c:
                em_final[(name, seed)] = []
                for r in range(R):
                    rows, params, reason, el = run_em(name, seed, d, r)
                    em_hist += rows
                    pi0, mu0, S0 = init_params(d, r)
                    save_params(f"{name}_seed{seed}_em_r{r}.npz", pi0=pi0, mu0=mu0, Sigma0=S0,
                                pi=params[0], mu=params[1], Sigma=params[2])
                    me = min(min_eig(S) for S in params[2])
                    row = dict(condition=name, method="EM", seed=seed, restart=r, status=reason,
                               n_updates=rows[-1]["iteration"], ckpt="final", elapsed_s=el,
                               min_cov_eig=me, boundary_active=bool(me <= LAM0 * (1 + 1e-9)),
                               weight_sum_err=abs(params[0].sum() - 1), **metrics(params))
                    runs.append(row)
                    em_final[(name, seed)].append((row, params))
                    if reason != "numerical_failure":
                        diag.append(dict(condition=name, seed=seed, restart=r, status=reason,
                                         **em_one_more(d["Xtr"], params)))
                ok = [(row, p) for row, p in em_final[(name, seed)] if row["status"] != "numerical_failure"]
                for (ra, pa), (rb, pb) in itertools.combinations(ok, 2):
                    perm = list(match_perm(*pb, *pa))
                    pb_al = tuple(x[perm] for x in pb)
                    dl = np.abs(log_density(d["Xte"], *pa) - log_density(d["Xte"], *pb))
                    mx = max(float(np.abs(x - y).max()) for x, y in zip(pa, pb_al))
                    pairs.append(dict(condition=name, seed=seed, restart_a=ra["restart"], restart_b=rb["restart"],
                                      dloglik=ra["train_loglik"] - rb["train_loglik"],
                                      max_abs_param_diff=mx, mean_abs_dlogp_test=float(dl.mean()),
                                      max_abs_dlogp_test=float(dl.max()),
                                      same_solution=bool(mx <= CFG["same_solution_param_tol"])))
            em_ok = [(row, p) for row, p in em_final[(src, seed)] if row["status"] != "numerical_failure"]

            # ---- DSM (experiment B), same data and initial values
            dsm_cands = []
            for r in range(R):
                rows, ckpts, status, el = run_dsm(name, seed, d, r)
                dsm_hist += rows
                pi0, mu0, S0 = init_params(d, r)
                save_params(f"{name}_seed{seed}_dsm_r{r}.npz", pi0=pi0, mu0=mu0, Sigma0=S0,
                            updates=np.array([s for s, _ in ckpts]),
                            pi=np.stack([p[0] for _, p in ckpts]), mu=np.stack([p[1] for _, p in ckpts]),
                            Sigma=np.stack([p[2] for _, p in ckpts]))
                step, params = ckpts[-1]
                runs.append(dict(condition=name, method="DSM", seed=seed, restart=r, status=status,
                                 n_updates=step, ckpt="final", elapsed_s=el,
                                 min_cov_eig=min(min_eig(S) for S in params[2]), boundary_active=False,
                                 weight_sum_err=abs(params[0].sum() - 1), **metrics(params)))
                if status == "completed":
                    dsm_cands += [(r, s, p, row["val_nll"]) for (s, p), row in zip(ckpts, rows)
                                  if s >= CFG["dsm"]["select_min_update"]]

            # ---- model selection (validation NLL only) and evaluation
            models = {}
            if em_ok:
                models["em_best_train"] = max(em_ok, key=lambda x: x[0]["train_loglik"])
                models["em_worst_train"] = min(em_ok, key=lambda x: x[0]["train_loglik"])
                models["em_selected"] = min(em_ok, key=lambda x: x[0]["val_nll"])
            if dsm_cands:
                r, s, p, _ = min(dsm_cands, key=lambda x: x[3])
                models["dsm_selected"] = (dict(restart=r, ckpt=s), p)
            for m, (row, params) in models.items():
                if m.startswith("em") and "reuse" in c:
                    continue
                ck = row["ckpt"] if m == "dsm_selected" else "final"
                if m in ("em_selected", "dsm_selected"):
                    by_seed.append(dict(condition=name, method=m.split("_")[0].upper(), seed=seed,
                                        restart=row["restart"], ckpt=ck, **metrics(params)))
                    save_params(f"{name}_seed{seed}_{m}.npz", pi=params[0], mu=params[1], Sigma=params[2],
                                restart=row["restart"], ckpt=str(ck))
                det.append(dict(condition=name, seed=seed, model=m, restart=row["restart"], ckpt=ck,
                                **detection(d, params)))
            print(f"{name} seed {seed} done ({time.perf_counter() - t_all:.0f}s)", flush=True)

    write_csv(RES / "em_history.csv", em_hist)
    write_csv(RES / "dsm_history.csv", dsm_hist)
    write_csv(RES / "metrics_by_run.csv", runs)
    write_csv(RES / "metrics_by_seed.csv", by_seed)
    write_csv(RES / "detection_by_seed.csv", det)
    write_csv(RES / "em_diagnostics.csv", diag)
    write_csv(RES / "em_restart_pairs.csv", pairs)

    # ---- seed-level summary (mean, sample sd over seeds)
    summ = []
    def agg(rows, keys, metrics_, group):
        for g in sorted({tuple(r[k] for k in keys) for r in rows}):
            sel = [r for r in rows if tuple(r[k] for k in keys) == g]
            for mtr in metrics_:
                v = np.array([r[mtr] for r in sel], float)
                v = v[np.isfinite(v)]
                summ.append(dict(table=group, condition=g[0], method=g[1], metric=mtr, mean=v.mean() if len(v) else np.nan,
                                 sd=v.std(ddof=1) if len(v) > 1 else np.nan, n_seeds=len(v)))
    agg(by_seed, ("condition", "method"), ("E_pi", "E_mu", "E_Sigma", "test_nll", "val_nll"), "selected_model")
    agg(det, ("condition", "model"), ("threshold", "test_fpr", "recall", "precision", "auroc", "pr_auc_trapz"),
        "detection")
    write_csv(RES / "summary.csv", summ)

    # ---- checks
    dnum = CFG["em"]["delta_num_rel"]
    checks = {"total_runtime_s": time.perf_counter() - t_all, "conditions": {}}
    for name in CFG["conditions"]:
        er = [r for r in runs if r["condition"] == name and r["method"] == "EM"]
        dr = [r for r in runs if r["condition"] == name and r["method"] == "DSM"]
        eh = [h for h in em_hist if h["condition"] == name]
        dh = [h for h in dsm_hist if h["condition"] == name]
        ch = {"dsm": {"runs": len(dr), "status": {s: sum(r["status"] == s for r in dr) for s in {r["status"] for r in dr}},
                      "max_weight_sum_err": max(r["weight_sum_err"] for r in dr),
                      "min_cov_eig_minus_lambda0": min(h["min_cov_eig"] for h in dh) - LAM0,
                      "all_finite": bool(all(np.isfinite([h["train_nll"], h["val_nll"], h["eval_dsm_loss"]]).all() for h in dh)),
                      "seeds_with_selected_model": sum(r["condition"] == name and r["method"] == "DSM" for r in by_seed)}}
        if er:
            ch["em"] = {"runs": len(er), "stop_reason": {s: sum(r["status"] == s for r in er) for s in {r["status"] for r in er}},
                        "decrease_beyond_tol": sum(h["flag"] == "decrease_beyond_tol" for h in eh),
                        "roundoff_negative": sum(h["flag"] == "roundoff_negative" for h in eh),
                        "delta_num_rule": f"delta < -{dnum}*(1+|l_t|)",
                        "max_weight_sum_err": max(r["weight_sum_err"] for r in er),
                        "min_cov_eig_minus_lambda0": min(h["min_cov_eig"] for h in eh) - LAM0,
                        "boundary_active_runs": sum(r["boundary_active"] for r in er),
                        "all_finite": bool(all(np.isfinite([h["loglik"], h["val_nll"]]).all() for h in eh)),
                        "seeds_with_selected_model": sum(r["condition"] == name and r["method"] == "EM" for r in by_seed),
                        "same_solution_pairs": f"{sum(p['same_solution'] for p in pairs if p['condition'] == name)}/"
                                               f"{sum(p['condition'] == name for p in pairs)}"}
        checks["conditions"][name] = ch
    checks["init_cov_floor_activated"] = sum(bool(d["init_floor_active"]) for d in datasets.values())
    checks["anomaly_attempts"] = {f"{k[0]}_seed{k[1]}": int(d["anomaly_attempts"]) for k, d in datasets.items()}
    (RES / "checks.json").write_text(json.dumps(checks, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(json.dumps(checks, indent=2, default=str))


if __name__ == "__main__":
    main()
