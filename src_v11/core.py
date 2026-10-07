from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import shutil
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image, ImageFile
from scipy.fft import dctn
from scipy.optimize import minimize_scalar
from scipy.special import softmax, logsumexp
from scipy.stats import binomtest
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, label_binarize
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

# Existing ViruSafe-MPX modules are intentionally reused lazily. v1.1 never retrains the CNN.
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def ensure_kaggle_dataset(*args, **kwargs):
    from src.data import ensure_kaggle_dataset as _f
    return _f(*args, **kwargs)

def load_best_model(*args, **kwargs):
    from src.modeling import load_best_model as _f
    return _f(*args, **kwargs)

ImageFile.LOAD_TRUNCATED_IMAGES = True

CLASSES = ["Mpox", "Chickenpox", "Measles", "Cowpox", "HFMD", "Healthy"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def atomic_json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    os.replace(tmp, path)


def _json_default(x):
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating,)): return float(x)
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, Path): return str(x)
    raise TypeError(type(x).__name__)


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logger(root: Path) -> logging.Logger:
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ViruSafe-MPX-v1.1")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_dir / "v1.1_pipeline.log", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(sh)
    return logger


class V11State:
    def __init__(self, path: Path):
        self.path = path
        self.data: Dict[str, Any] = {}
        if path.exists():
            try: self.data = json.loads(path.read_text(encoding="utf-8"))
            except Exception: self.data = {}

    def done(self, stage: str) -> bool:
        return self.data.get(stage, {}).get("status") == "done"

    def mark(self, stage: str, status: str, detail: Dict[str, Any] | None = None):
        self.data[stage] = {"status": status, **(detail or {})}
        atomic_json_dump(self.data, self.path)


class Stage:
    def __init__(self, state: V11State, name: str, logger: logging.Logger):
        self.state=state; self.name=name; self.logger=logger
    def __enter__(self):
        self.logger.info("=== v1.1 Stage: %s ===", self.name)
        self.state.mark(self.name,"running")
        return self
    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.state.mark(self.name,"done")
            self.logger.info("=== v1.1 Completed: %s ===", self.name)
        else:
            self.state.mark(self.name,"failed",{"error":str(exc)})
            self.logger.error("v1.1 stage %s failed: %s", self.name, exc)
        return False


def device_auto() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def npz_load(path: Path) -> Dict[str,np.ndarray]:
    with np.load(path, allow_pickle=True) as z:
        return {k:z[k] for k in z.files}


def npz_save(path: Path, **kwargs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+".tmp.npz")
    np.savez_compressed(tmp, **kwargs)
    os.replace(tmp,path)


def patient_group_from_path(p: str | Path) -> str:
    stem=Path(str(p)).stem
    parts=stem.split("_")
    return "_".join(parts[:2]) if len(parts)>=2 else stem


def make_eval_transform(image_size:int):
    return transforms.Compose([
        transforms.Resize((image_size,image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406),std=(0.229,0.224,0.225)),
    ])


class ManifestEvalDataset(Dataset):
    def __init__(self,csv_path:Path,image_size:int):
        self.df=pd.read_csv(csv_path)
        self.tf=make_eval_transform(image_size)
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        r=self.df.iloc[i]; p=Path(str(r["path"]))
        with Image.open(p) as im: x=self.tf(im.convert("RGB"))
        return x,int(r["label_index"]),str(p)


class PathLabelDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, image_size:int):
        self.df=rows.reset_index(drop=True); self.tf=make_eval_transform(image_size)
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        r=self.df.iloc[i]; p=Path(str(r["path"]))
        with Image.open(p) as im: x=self.tf(im.convert("RGB"))
        return x,int(r.get("label_index",-1)),str(p)


@torch.no_grad()
def infer_logits_features(model, loader, device:torch.device, amp:bool=True, hflip_tta:bool=True):
    model.eval(); L=[];F=[];Y=[];P=[]
    for x,y,p in tqdm(loader,desc="v1.1 inference",leave=False):
        x=x.to(device,non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=(amp and device.type=="cuda")):
            logits=model(x)
            if hflip_tta:
                logits=(logits+model(torch.flip(x,dims=[3])))/2.0
            feat=model.forward_features(x)
            if feat.ndim==4: feat=feat.mean(dim=(-2,-1))
            elif feat.ndim==3: feat=feat.mean(dim=1)
            elif feat.ndim>2: feat=feat.flatten(1)
        L.append(logits.detach().float().cpu().numpy())
        F.append(feat.detach().float().cpu().numpy())
        Y.append(np.asarray(y)); P.extend(list(p))
    return {"logits":np.concatenate(L),"features":np.concatenate(F),"y":np.concatenate(Y),"paths":np.asarray(P,dtype=object)}


def aligned_predict_proba(clf, X:np.ndarray, n_classes:int=6)->np.ndarray:
    p=clf.predict_proba(X)
    out=np.full((len(X),n_classes),1e-12,dtype=float)
    for j,c in enumerate(clf.classes_.astype(int)): out[:,c]=p[:,j]
    out/=out.sum(1,keepdims=True)
    return out


def ece(probs:np.ndarray,y:np.ndarray,n_bins:int=15)->float:
    conf=probs.max(1); pred=probs.argmax(1); correct=(pred==y).astype(float)
    edges=np.linspace(0,1,n_bins+1); v=0.0
    for a,b in zip(edges[:-1],edges[1:]):
        m=(conf>=a)&((conf<b) if b<1 else (conf<=b))
        if m.any(): v += m.mean()*abs(correct[m].mean()-conf[m].mean())
    return float(v)


def multiclass_brier(probs:np.ndarray,y:np.ndarray,n_classes:int=6)->float:
    oh=np.eye(n_classes)[y]
    return float(np.mean(np.sum((probs-oh)**2,axis=1)))


def safe_macro_auc(probs:np.ndarray,y:np.ndarray,n_classes:int=6)->float:
    try:
        yb=label_binarize(y,classes=np.arange(n_classes))
        vals=[]
        for c in range(n_classes):
            if len(np.unique(yb[:,c]))>1: vals.append(roc_auc_score(yb[:,c],probs[:,c]))
        return float(np.mean(vals)) if vals else float("nan")
    except Exception:return float("nan")


def metrics_summary(probs:np.ndarray,y:np.ndarray)->Dict[str,float]:
    pred=probs.argmax(1)
    return {
        "accuracy":float(accuracy_score(y,pred)),
        "balanced_accuracy":float(balanced_accuracy_score(y,pred)),
        "macro_f1":float(f1_score(y,pred,average="macro",zero_division=0)),
        "macro_auroc":safe_macro_auc(probs,y,probs.shape[1]),
        "ece":ece(probs,y),
        "brier":multiclass_brier(probs,y,probs.shape[1]),
        "n":int(len(y)),
    }


def per_class_table(probs:np.ndarray,y:np.ndarray,class_names:Sequence[str]=CLASSES)->pd.DataFrame:
    pred=probs.argmax(1)
    pr,re,f1,sup=precision_recall_fscore_support(y,pred,labels=np.arange(len(class_names)),zero_division=0)
    rows=[]
    cm=confusion_matrix(y,pred,labels=np.arange(len(class_names)))
    for i,c in enumerate(class_names):
        tn=cm.sum()-cm[i,:].sum()-cm[:,i].sum()+cm[i,i]
        fp=cm[:,i].sum()-cm[i,i]
        spec=tn/max(tn+fp,1)
        rows.append({"class":c,"precision":pr[i],"recall":re[i],"specificity":spec,"f1":f1[i],"support":int(sup[i])})
    return pd.DataFrame(rows)


def fit_prob_temperature(probs:np.ndarray,y:np.ndarray)->float:
    logp=np.log(np.clip(probs,1e-12,1.0))
    def nll(logT):
        T=float(np.exp(logT)); p=softmax(logp/T,axis=1)
        return log_loss(y,p,labels=np.arange(probs.shape[1]))
    res=minimize_scalar(nll,bounds=(math.log(0.25),math.log(4.0)),method="bounded",options={"xatol":1e-4})
    return float(np.exp(res.x)) if res.success else 1.0


def apply_prob_temperature(probs:np.ndarray,T:float)->np.ndarray:
    return softmax(np.log(np.clip(probs,1e-12,1.0))/max(T,1e-6),axis=1)


def select_feature_head(train_features,train_y,train_groups,cfg,logger):
    C_grid=[float(x) for x in cfg["reliability_head"]["candidate_C"]]
    cw_grid=[None if str(x).lower()=="none" else str(x) for x in cfg["reliability_head"]["candidate_class_weight"]]
    requested=int(cfg["reliability_head"].get("inner_group_folds",3))
    # Ensure every class has enough groups for StratifiedGroupKFold.
    group_counts=[]
    for c in np.unique(train_y): group_counts.append(len(np.unique(train_groups[train_y==c])))
    n_splits=max(2,min(requested,min(group_counts)))
    cv=StratifiedGroupKFold(n_splits=n_splits,shuffle=True,random_state=int(cfg["project"]["protocol_seed"]))
    rows=[]
    for C in C_grid:
        for cw in cw_grid:
            f1s=[]; losses=[]; ok=True
            for tr,va in cv.split(train_features,train_y,train_groups):
                if len(np.unique(train_y[tr]))<len(np.unique(train_y)):
                    ok=False;break
                scaler=StandardScaler()
                Xtr=scaler.fit_transform(train_features[tr]); Xva=scaler.transform(train_features[va])
                clf=LogisticRegression(C=C,class_weight=cw,solver="lbfgs",max_iter=int(cfg["reliability_head"].get("max_iter",2500)))
                clf.fit(Xtr,train_y[tr])
                p=aligned_predict_proba(clf,Xva,len(CLASSES))
                f1s.append(f1_score(train_y[va],p.argmax(1),average="macro",zero_division=0))
                losses.append(log_loss(train_y[va],p,labels=np.arange(len(CLASSES))))
            if ok:
                rows.append({"C":C,"class_weight":"none" if cw is None else cw,"cv_macro_f1":float(np.mean(f1s)),"cv_log_loss":float(np.mean(losses))})
    if not rows: raise RuntimeError("No valid frozen-feature head candidate could be fitted.")
    tab=pd.DataFrame(rows).sort_values(["cv_macro_f1","cv_log_loss"],ascending=[False,True]).reset_index(drop=True)
    best=tab.iloc[0]
    cw=None if best["class_weight"]=="none" else str(best["class_weight"])
    scaler=StandardScaler(); X=scaler.fit_transform(train_features)
    clf=LogisticRegression(C=float(best["C"]),class_weight=cw,solver="lbfgs",max_iter=int(cfg["reliability_head"].get("max_iter",2500)))
    clf.fit(X,train_y)
    logger.info("Selected frozen-feature head: C=%s class_weight=%s CV macro-F1=%.4f",best["C"],best["class_weight"],best["cv_macro_f1"])
    return {"scaler":scaler,"clf":clf},tab


def feature_head_probs(head,features):
    return aligned_predict_proba(head["clf"],head["scaler"].transform(features),len(CLASSES))


def choose_blend(base_val,head_val,y_val,cfg)->Tuple[float,pd.DataFrame]:
    rows=[]
    for a in [float(x) for x in cfg["reliability_head"]["blend_weights"]]:
        p=(1-a)*base_val+a*head_val
        rows.append({"head_weight":a,"macro_f1":float(f1_score(y_val,p.argmax(1),average="macro",zero_division=0)),"accuracy":float(accuracy_score(y_val,p.argmax(1))),"ece":ece(p,y_val),"log_loss":float(log_loss(y_val,p,labels=np.arange(len(CLASSES))))})
    tab=pd.DataFrame(rows).sort_values(["macro_f1","log_loss"],ascending=[False,True]).reset_index(drop=True)
    base_f1=float(f1_score(y_val,base_val.argmax(1),average="macro",zero_division=0))
    best=tab.iloc[0]
    min_gain=float(cfg["reliability_head"].get("minimum_validation_macro_f1_gain",0.0))
    alpha=float(best["head_weight"]) if float(best["macro_f1"])>=base_f1+min_gain else 0.0
    return alpha,tab


def mondrian_sets(probs:np.ndarray,q_by_class:np.ndarray)->np.ndarray:
    return (1.0-probs)<=q_by_class.reshape(1,-1)


def conformal_quantile(scores:np.ndarray,alpha:float)->float:
    n=len(scores)
    if n==0:return 1.0
    level=min(1.0,math.ceil((n+1)*(1-alpha))/n)
    try:return float(np.quantile(scores,level,method="higher"))
    except TypeError:return float(np.quantile(scores,level,interpolation="higher"))


def mondrian_qhat(probs:np.ndarray,y:np.ndarray,alpha:float,n_classes:int=6)->np.ndarray:
    out=[]
    for c in range(n_classes):
        m=y==c; scores=1.0-probs[m,c]
        out.append(conformal_quantile(scores,alpha))
    return np.asarray(out,dtype=float)


def conformal_eval(probs:np.ndarray,y:np.ndarray,q:np.ndarray,class_names=CLASSES)->Tuple[Dict[str,float],pd.DataFrame,np.ndarray]:
    sets=mondrian_sets(probs,q)
    # Never return an empty prediction set: fall back to top-1; report fallback rate separately.
    empty=sets.sum(1)==0
    if empty.any(): sets[empty,probs[empty].argmax(1)]=True
    covered=sets[np.arange(len(y)),y]
    rows=[]
    for c,name in enumerate(class_names):
        m=y==c
        rows.append({"class":name,"coverage":float(covered[m].mean()) if m.any() else np.nan,"mean_set_size":float(sets[m].sum(1).mean()) if m.any() else np.nan,"n":int(m.sum()),"qhat":float(q[c])})
    overall={"coverage":float(covered.mean()),"mean_set_size":float(sets.sum(1).mean()),"singleton_rate":float((sets.sum(1)==1).mean()),"empty_fallback_rate":float(empty.mean())}
    return overall,pd.DataFrame(rows),sets


def selective_table(probs:np.ndarray,y:np.ndarray,coverages=(0.7,0.8,0.9,0.95))->pd.DataFrame:
    conf=probs.max(1); pred=probs.argmax(1); order=np.argsort(-conf); rows=[]
    for cov in coverages:
        n=max(1,int(math.ceil(len(y)*cov))); idx=order[:n]
        rows.append({"target_coverage":cov,"realized_coverage":len(idx)/len(y),"selective_accuracy":float((pred[idx]==y[idx]).mean()),"selective_macro_f1":float(f1_score(y[idx],pred[idx],average="macro",zero_division=0)),"threshold":float(conf[idx].min())})
    return pd.DataFrame(rows)


def risk_coverage(probs,y)->pd.DataFrame:
    conf=probs.max(1); correct=(probs.argmax(1)==y).astype(float); order=np.argsort(-conf); c=correct[order]
    n=np.arange(1,len(y)+1); acc=np.cumsum(c)/n
    return pd.DataFrame({"coverage":n/len(y),"risk":1-acc,"accuracy":acc,"threshold":conf[order]})


def phash64(path:Path)->int:
    with Image.open(path) as im:
        arr=np.asarray(im.convert("L").resize((32,32),Image.Resampling.LANCZOS),dtype=np.float32)
    coeff=dctn(arr,type=2,norm="ortho")[:8,:8]
    flat=coeff.flatten(); med=np.median(flat[1:]); bits=flat>med
    h=0
    for b in bits: h=(h<<1)|int(bool(b))
    return int(h)


def sha256(path:Path)->str:
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for ch in iter(lambda:f.read(1024*1024),b""):h.update(ch)
    return h.hexdigest()


def hamming64(a:int,b:int)->int:return int((a^b).bit_count())


def infer_external_label(path:Path)->str|None:
    aliases={
        "monkeypox":"Mpox","monkey pox":"Mpox","mpox":"Mpox",
        "chickenpox":"Chickenpox","chicken pox":"Chickenpox",
        "measles":"Measles","normal":"Healthy","healthy":"Healthy",
    }
    for part in reversed(path.parts):
        n=part.lower().replace("_"," ").replace("-"," ").strip()
        if n in aliases:return aliases[n]
    return None


def discover_external_images(root:Path,prefer_original:bool=True)->pd.DataFrame:
    rows=[]
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            lab=infer_external_label(p)
            if lab: rows.append({"path":str(p.resolve()),"class":lab,"label_index":CLASS_TO_IDX[lab],"is_augmented":any("augment" in x.lower() for x in p.parts),"is_original_hint":any("original" in x.lower() for x in p.parts)})
    df=pd.DataFrame(rows)
    if df.empty: raise FileNotFoundError(f"Could not identify labeled MSID images under {root}")
    if prefer_original:
        if (~df["is_augmented"]).sum()>=20: df=df[~df["is_augmented"]].copy()
        if df["is_original_hint"].sum()>=20: df=df[df["is_original_hint"]].copy()
    return df.drop_duplicates("path").reset_index(drop=True)


def deduplicate_external(external:pd.DataFrame, msld_paths:Sequence[str], threshold:int, cache_dir:Path, logger)->Tuple[pd.DataFrame,pd.DataFrame]:
    cache_dir.mkdir(parents=True,exist_ok=True)
    cache=cache_dir/"dedup_hash_cache.csv"
    known={}
    if cache.exists():
        try:
            c=pd.read_csv(cache)
            known={r["path"]:(r["sha256"],int(str(r["phash"]),16)) for _,r in c.iterrows()}
        except Exception: known={}
    all_paths=[Path(str(p)) for p in list(msld_paths)+external["path"].astype(str).tolist()]
    rows=[]
    for p in tqdm(all_paths,desc="Duplicate audit",leave=False):
        s=str(p)
        if s in known: sh,ph=known[s]
        else:
            try: sh,ph=sha256(p),phash64(p)
            except Exception as e:
                logger.warning("Hashing failed for %s: %s",p,e); continue
            known[s]=(sh,ph)
        rows.append({"path":s,"sha256":sh,"phash":format(ph,"016x")})
    pd.DataFrame(rows).drop_duplicates("path").to_csv(cache,index=False)
    msld_set=set(str(Path(str(x))) for x in msld_paths)
    ref=[(s,sh,ph) for s,(sh,ph) in known.items() if s in msld_set]
    decisions=[]; keep=[]
    for _,r in external.iterrows():
        p=str(Path(str(r["path"]))); sh,ph=known.get(p,(None,None))
        exact=False; near=False; mind=65; match=""
        if sh is not None:
            for rp,rsh,rph in ref:
                if sh==rsh: exact=True; mind=0; match=rp; break
                d=hamming64(ph,rph)
                if d<mind: mind=d; match=rp
            near=(mind<=threshold)
        drop=bool(exact or near)
        decisions.append({"external_path":p,"class":r["class"],"exact_duplicate":exact,"nearest_phash_distance":int(mind) if mind<=64 else np.nan,"near_duplicate":near,"matched_msld_path":match,"excluded":drop})
        if not drop: keep.append(r.to_dict())
    audit=pd.DataFrame(decisions)
    clean=pd.DataFrame(keep)
    if not clean.empty:
        # Deduplicate external file hashes among themselves.
        clean["sha256"]=[known[str(Path(p))][0] for p in clean["path"]]
        clean=clean.drop_duplicates("sha256").drop(columns=["sha256"]).reset_index(drop=True)
    return clean,audit


def score_auroc(id_scores,ood_scores)->float:
    y=np.concatenate([np.ones(len(id_scores)),np.zeros(len(ood_scores))]); s=np.concatenate([id_scores,ood_scores])
    return float(roc_auc_score(y,s))


def fpr95(id_scores,ood_scores)->float:
    y=np.concatenate([np.ones(len(id_scores)),np.zeros(len(ood_scores))]); s=np.concatenate([id_scores,ood_scores])
    fpr,tpr,_=roc_curve(y,s)
    idx=np.where(tpr>=0.95)[0]
    return float(fpr[idx[0]]) if len(idx) else 1.0


def fit_feature_ood(train_features,train_y,pca_dim:int):
    scaler=StandardScaler(); Xs=scaler.fit_transform(train_features)
    dim=min(int(pca_dim),Xs.shape[0]-1,Xs.shape[1])
    pca=PCA(n_components=dim,svd_solver="randomized",random_state=42); X=pca.fit_transform(Xs)
    cent=np.vstack([X[train_y==c].mean(0) for c in range(len(CLASSES))])
    global_cent=X.mean(0)
    residual=np.vstack([X[train_y==c]-cent[c] for c in range(len(CLASSES)) if (train_y==c).any()])
    cov=LedoitWolf().fit(residual); precision=cov.precision_
    return {"scaler":scaler,"pca":pca,"centroids":cent,"global_centroid":global_cent,"precision":precision,"train_pca":X}


def feature_ood_scores(model,features):
    X=model["pca"].transform(model["scaler"].transform(features)); C=model["centroids"]; P=model["precision"]
    # Cosine prototype
    xn=X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-12); cn=C/(np.linalg.norm(C,axis=1,keepdims=True)+1e-12)
    cosine=(xn@cn.T).max(1)
    # Mahalanobis
    diff=X[:,None,:]-C[None,:,:]
    md=np.einsum("nci,ij,ncj->nc",diff,P,diff,optimize=True)
    maha=-md.min(1)
    gd=X-model["global_centroid"]; global_md=np.einsum("ni,ij,nj->n",gd,P,gd,optimize=True)
    rel=global_md-md.min(1)
    return {"cosine_prototype":cosine,"mahalanobis":maha,"relative_mahalanobis":rel},X


def softmax_ood_scores(probs:np.ndarray, logits:np.ndarray|None=None)->Dict[str,np.ndarray]:
    ps=np.sort(probs,axis=1)
    out={
        "msp":probs.max(1),
        "neg_entropy":np.sum(probs*np.log(np.clip(probs,1e-12,1.0)),axis=1),
        "margin":ps[:,-1]-ps[:,-2],
    }
    if logits is not None: out["energy"]=logsumexp(logits,axis=1)
    return out


def bootstrap_delta(y,base_pred,enh_pred,groups,iters:int,seed:int)->pd.DataFrame:
    rng=np.random.default_rng(seed); ug=np.unique(groups); rows=[]
    vals_acc=[];vals_f1=[]
    for _ in range(iters):
        sample=rng.choice(ug,size=len(ug),replace=True); idx=np.concatenate([np.where(groups==g)[0] for g in sample])
        yy=y[idx]; b=base_pred[idx]; e=enh_pred[idx]
        vals_acc.append(accuracy_score(yy,e)-accuracy_score(yy,b))
        vals_f1.append(f1_score(yy,e,average="macro",zero_division=0)-f1_score(yy,b,average="macro",zero_division=0))
    for name,v in [("delta_accuracy",vals_acc),("delta_macro_f1",vals_f1)]:
        a=np.asarray(v); rows.append({"metric":name,"mean":float(a.mean()),"ci_low":float(np.quantile(a,.025)),"ci_high":float(np.quantile(a,.975))})
    return pd.DataFrame(rows)


def mcnemar_exact(y,base_pred,enh_pred)->Dict[str,Any]:
    b_ok=base_pred==y; e_ok=enh_pred==y
    b=int(np.sum(b_ok & ~e_ok)); c=int(np.sum(~b_ok & e_ok)); n=b+c
    p=float(binomtest(min(b,c),n=n,p=0.5,alternative="two-sided").pvalue) if n else 1.0
    return {"baseline_correct_enhanced_wrong":b,"baseline_wrong_enhanced_correct":c,"discordant_pairs":n,"exact_p_value":p}


def save_excel_from_csvs(tables_dir:Path,out:Path):
    with pd.ExcelWriter(out,engine="xlsxwriter") as writer:
        used=set()
        for csv in sorted(tables_dir.glob("*.csv")):
            try:df=pd.read_csv(csv)
            except Exception:continue
            name=csv.stem[:31]; base=name; i=1
            while name in used:
                tail=f"_{i}"; name=base[:31-len(tail)]+tail;i+=1
            used.add(name); df.to_excel(writer,sheet_name=name,index=False)


def write_protocol_lock(cfg:Dict[str,Any],out:Path):
    obj={
        "version":"1.1",
        "purpose":"Pre-specified reliability enhancement and external validation; no CNN retraining; no test-label model selection.",
        "class_order":CLASSES,
        "reliability_head":cfg["reliability_head"],
        "conformal":cfg["conformal"],
        "ood_v1_1":cfg["ood_v1_1"],
        "external_validation":{k:v for k,v in cfg["external_validation"].items() if k!="enabled"},
        "selection_rules":{
            "feature_head":"hyperparameters chosen by patient-grouped CV on training embeddings only",
            "fusion_weight":"chosen from fixed grid on validation data; outer test labels never used",
            "enhancement_guard":"enhanced predictor activated only when pre-specified validation macro-F1 gain is met",
            "ood_method":"chosen using held-out DermNet validation categories; reported OOD performance uses disjoint DermNet test categories",
            "external":"MSID excluded from all tuning; duplicate/near-duplicate filtering is applied before evaluation",
        },
    }
    atomic_json_dump(obj,out)


def validate_base_project(root:Path,cfg:Dict[str,Any])->Dict[str,Any]:
    missing=[]; folds=cfg["required_base"]["folds"]
    for f in folds:
        for p in [root/"outputs"/"predictions"/f"fold_{f}.npz",root/"outputs"/"checkpoints"/f"fold_{f}"/"best.pt",root/"data"/"generated_patient_folds"/f"fold_{f}"/"train.csv",root/"data"/"generated_patient_folds"/f"fold_{f}"/"val.csv",root/"data"/"generated_patient_folds"/f"fold_{f}"/"test.csv"]:
            if not p.exists():missing.append(str(p.relative_to(root)))
    if not (root/"outputs"/"tables"/"dermnet_ood_manifest.csv").exists(): missing.append("outputs/tables/dermnet_ood_manifest.csv")
    if missing: raise RuntimeError("v1.1 requires the completed v1.0.4 project. Missing:\n - "+"\n - ".join(missing))
    cmap_path=root/"outputs"/"class_mapping.json"
    if cmap_path.exists():
        cmap=json.loads(cmap_path.read_text(encoding="utf-8"))
        actual=[c for c,_ in sorted(cmap.items(),key=lambda kv:kv[1])]
        if actual!=CLASSES:
            raise RuntimeError(f"Class order mismatch. Expected {CLASSES}, found {actual}")
    # Detect moved projects whose generated manifests still contain stale absolute image paths.
    probe=pd.read_csv(root/"data"/"generated_patient_folds"/"fold_1"/"train.csv").head(5)
    stale=[p for p in probe["path"].astype(str) if not Path(p).exists()]
    if stale:
        raise RuntimeError("Generated manifest image paths are stale/missing. Keep v1.1 in the same existing project location or regenerate the base data manifests first. Example missing path: "+stale[0])
    return {"folds":folds,"cuda":torch.cuda.is_available(),"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}
