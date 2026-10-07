from __future__ import annotations

import html
import json
import math
import os
import shutil
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.special import softmax
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader

from .core import (
    CLASSES, CLASS_TO_IDX, ManifestEvalDataset, PathLabelDataset, Stage, V11State,
    aligned_predict_proba, apply_prob_temperature, atomic_json_dump, bootstrap_delta,
    choose_blend, conformal_eval, deduplicate_external, device_auto, discover_external_images,
    ece, ensure_kaggle_dataset, feature_head_probs, feature_ood_scores, fit_feature_ood,
    fit_prob_temperature, fpr95, infer_logits_features, load_best_model, mcnemar_exact,
    metrics_summary, mondrian_qhat, npz_load, npz_save, patient_group_from_path,
    per_class_table, risk_coverage, save_excel_from_csvs, score_auroc, selective_table,
    softmax_ood_scores, validate_base_project, write_protocol_lock, select_feature_head,
)


def _dirs(root:Path):
    out=root/"outputs_v1.1"
    d={
        "out":out,"tables":out/"tables","figures":out/"figures","cache":out/"cache",
        "models":out/"models","explanations":out/"explanations","report":out/"report",
        "audit":out/"audit",
    }
    for p in d.values():p.mkdir(parents=True,exist_ok=True)
    return d


def _savefig(fig,path:Path):
    fig.tight_layout(); fig.savefig(path.with_suffix(".png"),dpi=220,bbox_inches="tight"); fig.savefig(path.with_suffix(".pdf"),bbox_inches="tight"); plt.close(fig)


def _model_cfg(cfg): return cfg["required_base"]


def _load_fold_model(root,cfg,fold,device):
    mc=_model_cfg(cfg)
    return load_best_model(root/"outputs"/"checkpoints"/f"fold_{fold}"/"best.pt",mc["model_name"],len(CLASSES),device)


def _loader(csv_path:Path,cfg,shuffle=False):
    mc=_model_cfg(cfg); ds=ManifestEvalDataset(csv_path,int(mc["image_size"]))
    return DataLoader(ds,batch_size=int(mc["batch_size"]),shuffle=shuffle,num_workers=int(mc["num_workers"]),pin_memory=torch.cuda.is_available(),persistent_workers=(int(mc["num_workers"])>0))


def preflight(root,cfg,state,logger):
    d=_dirs(root); marker=d["audit"]/"preflight.json"
    if state.done("preflight") and marker.exists(): return json.loads(marker.read_text(encoding="utf-8"))
    with Stage(state,"preflight",logger):
        info=validate_base_project(root,cfg); info["version"]="1.1"; info["no_cnn_retraining"]=True
        write_protocol_lock(cfg,d["audit"]/"PROTOCOL_LOCK_v1.1.json")
        atomic_json_dump(info,marker)
        logger.info("Preflight PASS. Existing five CNN checkpoints will be reused; no CNN retraining will occur.")
        return info


def _extract_train_val_cache(root,cfg,d,fold,device,logger):
    cp=d["cache"]/f"fold_{fold}_trainval_features.npz"
    if cp.exists(): return npz_load(cp)
    model=_load_fold_model(root,cfg,fold,device); mc=_model_cfg(cfg)
    mroot=root/"data"/"generated_patient_folds"/f"fold_{fold}"
    tr=infer_logits_features(model,_loader(mroot/"train.csv",cfg),device,amp=bool(mc.get("use_amp",True)),hflip_tta=True)
    va=infer_logits_features(model,_loader(mroot/"val.csv",cfg),device,amp=bool(mc.get("use_amp",True)),hflip_tta=True)
    npz_save(cp,train_logits=tr["logits"],train_features=tr["features"],train_y=tr["y"],train_paths=tr["paths"],val_logits=va["logits"],val_features=va["features"],val_y=va["y"],val_paths=va["paths"])
    del model
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return npz_load(cp)


def run_internal_enhancement(root,cfg,state,device,logger):
    d=_dirs(root); summary_path=d["out"]/"internal_enhancement_summary.json"
    if state.done("internal_enhancement") and summary_path.exists():return json.loads(summary_path.read_text(encoding="utf-8"))
    with Stage(state,"internal_enhancement",logger):
        pooled_y=[]; pooled_base=[]; pooled_enh=[]; pooled_paths=[]; fold_rows=[]; conf_rows=[]; set_arrays=[]
        alpha=float(cfg["conformal"].get("alpha",0.10))
        for fold in cfg["required_base"]["folds"]:
            logger.info("v1.1 internal enhancement: fold %d",fold)
            basez=npz_load(root/"outputs"/"predictions"/f"fold_{fold}.npz")
            cache=_extract_train_val_cache(root,cfg,d,fold,device,logger)
            train_groups=np.asarray([patient_group_from_path(x) for x in cache["train_paths"]],dtype=object)
            head,cvtab=select_feature_head(cache["train_features"],cache["train_y"],train_groups,cfg,logger)
            cvtab.insert(0,"fold",fold); cvtab.to_csv(d["tables"]/f"feature_head_cv_fold_{fold}.csv",index=False)
            joblib.dump(head,d["models"]/f"feature_head_fold_{fold}.joblib")

            if not np.array_equal(cache["val_paths"].astype(str), basez["val_paths"].astype(str)):
                raise RuntimeError(f"Fold {fold}: cached validation feature order does not match original prediction order.")
            temp0=float(basez["temperature"][0])
            base_val=softmax(basez["val_logits"]/temp0,axis=1)
            base_test=softmax(basez["test_logits"]/temp0,axis=1)
            head_val=feature_head_probs(head,cache["val_features"])
            head_test=feature_head_probs(head,basez["test_features"])
            a,blendtab=choose_blend(base_val,head_val,basez["val_y"],cfg)
            blendtab.insert(0,"fold",fold); blendtab.to_csv(d["tables"]/f"blend_selection_fold_{fold}.csv",index=False)
            val_mix=(1-a)*base_val+a*head_val; test_mix=(1-a)*base_test+a*head_test
            T=fit_prob_temperature(val_mix,basez["val_y"])
            val_enh=apply_prob_temperature(val_mix,T); test_enh=apply_prob_temperature(test_mix,T)

            q=mondrian_qhat(val_enh,basez["val_y"],alpha,len(CLASSES))
            conf_overall,confpc,sets=conformal_eval(test_enh,basez["test_y"],q,CLASSES)
            confpc.insert(0,"fold",fold); conf_rows.append(confpc); set_arrays.append(sets)

            bm=metrics_summary(base_test,basez["test_y"]); em=metrics_summary(test_enh,basez["test_y"])
            fold_rows.append({"fold":fold,"head_weight":a,"probability_temperature":T,
                              **{f"baseline_{k}":v for k,v in bm.items() if k!="n"},
                              **{f"enhanced_{k}":v for k,v in em.items() if k!="n"},
                              "mondrian_coverage":conf_overall["coverage"],"mondrian_mean_set_size":conf_overall["mean_set_size"]})
            params={"fold":fold,"head_weight":a,"probability_temperature":T,"original_temperature":temp0,"mondrian_qhat":q.tolist()}
            atomic_json_dump(params,d["models"]/f"enhancement_params_fold_{fold}.json")
            npz_save(d["cache"]/f"enhanced_fold_{fold}.npz",val_probs=val_enh,val_y=basez["val_y"],test_probs=test_enh,test_y=basez["test_y"],test_paths=basez["test_paths"],mondrian_sets=sets.astype(np.uint8),qhat=q)
            pooled_y.append(basez["test_y"]); pooled_base.append(base_test); pooled_enh.append(test_enh);pooled_paths.extend(basez["test_paths"].tolist())

        y=np.concatenate(pooled_y); bp=np.concatenate(pooled_base); ep=np.concatenate(pooled_enh); paths=np.asarray(pooled_paths,dtype=object)
        base_m=metrics_summary(bp,y); enh_m=metrics_summary(ep,y)
        base_pc=per_class_table(bp,y); base_pc.insert(0,"variant","Baseline")
        enh_pc=per_class_table(ep,y); enh_pc.insert(0,"variant","ViruSafe-MPX v1.1")
        pd.concat([base_pc,enh_pc],ignore_index=True).to_csv(d["tables"]/"internal_per_class_comparison.csv",index=False)
        pd.DataFrame(fold_rows).to_csv(d["tables"]/"internal_fold_comparison.csv",index=False)
        confdf=pd.concat(conf_rows,ignore_index=True); confdf.to_csv(d["tables"]/"mondrian_conformal_by_fold_class.csv",index=False)

        sel_b=selective_table(bp,y);sel_b.insert(0,"variant","Baseline")
        sel_e=selective_table(ep,y);sel_e.insert(0,"variant","ViruSafe-MPX v1.1")
        pd.concat([sel_b,sel_e],ignore_index=True).to_csv(d["tables"]/"selective_prediction_comparison.csv",index=False)
        rc_b=risk_coverage(bp,y);rc_b["variant"]="Baseline"
        rc_e=risk_coverage(ep,y);rc_e["variant"]="ViruSafe-MPX v1.1"
        pd.concat([rc_b,rc_e],ignore_index=True).to_csv(d["tables"]/"risk_coverage_comparison.csv",index=False)

        groups=np.asarray([patient_group_from_path(x) for x in paths],dtype=object)
        bpred=bp.argmax(1); epred=ep.argmax(1)
        boot=bootstrap_delta(y,bpred,epred,groups,int(cfg["statistics"]["bootstrap_iterations"]),int(cfg["statistics"]["bootstrap_seed"]));boot.to_csv(d["tables"]/"paired_bootstrap_improvement.csv",index=False)
        mc=mcnemar_exact(y,bpred,epred);pd.DataFrame([mc]).to_csv(d["tables"]/"mcnemar_exact.csv",index=False)

        # Pooled empirical Mondrian coverage from fold-specific sets.
        all_sets=np.concatenate(set_arrays); covered=all_sets[np.arange(len(y)),y].astype(bool)
        pooled_conf={"coverage":float(covered.mean()),"mean_set_size":float(all_sets.sum(1).mean()),"singleton_rate":float((all_sets.sum(1)==1).mean()),"target_coverage":1-alpha}
        conf_class=[]
        for c,name in enumerate(CLASSES):
            m=y==c;conf_class.append({"class":name,"coverage":float(covered[m].mean()),"mean_set_size":float(all_sets[m].sum(1).mean()),"n":int(m.sum())})
        pd.DataFrame(conf_class).to_csv(d["tables"]/"mondrian_conformal_pooled_by_class.csv",index=False)
        pd.DataFrame([base_m,enh_m],index=["Baseline","ViruSafe-MPX v1.1"]).reset_index(names="variant").to_csv(d["tables"]/"internal_pooled_comparison.csv",index=False)
        predtab=pd.DataFrame({"path":paths,"patient_group":groups,"true_index":y,"true_class":[CLASSES[i] for i in y],"baseline_pred":[CLASSES[i] for i in bpred],"enhanced_pred":[CLASSES[i] for i in epred],"baseline_confidence":bp.max(1),"enhanced_confidence":ep.max(1),"mondrian_set_size":all_sets.sum(1)})
        predtab.to_csv(d["tables"]/"v1.1_pooled_oof_predictions.csv",index=False)

        # Figures
        vals=["accuracy","balanced_accuracy","macro_f1"]
        mpox_base=float(base_pc[base_pc["class"]=="Mpox"]["recall"].iloc[0]);mpox_enh=float(enh_pc[enh_pc["class"]=="Mpox"]["recall"].iloc[0])
        labels=["Accuracy","Balanced accuracy","Macro-F1","Mpox recall"]
        bvals=[base_m[x] for x in vals]+[mpox_base]; evals=[enh_m[x] for x in vals]+[mpox_enh]
        x=np.arange(len(labels));w=.36;fig,ax=plt.subplots(figsize=(9,5));ax.bar(x-w/2,bvals,w,label="Baseline");ax.bar(x+w/2,evals,w,label="v1.1");ax.set_xticks(x,labels,rotation=15);ax.set_ylim(0,1);ax.set_ylabel("Score");ax.set_title("Internal patient-grouped out-of-fold performance");ax.legend();ax.grid(axis="y",alpha=.25);_savefig(fig,d["figures"]/"17_internal_performance_comparison")
        s70b=float(sel_b.loc[np.argmin(abs(sel_b.target_coverage-.70)),"selective_accuracy"]);s70e=float(sel_e.loc[np.argmin(abs(sel_e.target_coverage-.70)),"selective_accuracy"])
        s90b=float(sel_b.loc[np.argmin(abs(sel_b.target_coverage-.90)),"selective_accuracy"]);s90e=float(sel_e.loc[np.argmin(abs(sel_e.target_coverage-.90)),"selective_accuracy"])
        fig,ax=plt.subplots(figsize=(8,5));labs=["ECE ↓","Brier ↓","Sel. acc @70% ↑","Sel. acc @90% ↑"];ax.bar(np.arange(4)-w/2,[base_m["ece"],base_m["brier"],s70b,s90b],w,label="Baseline");ax.bar(np.arange(4)+w/2,[enh_m["ece"],enh_m["brier"],s70e,s90e],w,label="v1.1");ax.set_xticks(np.arange(4),labs,rotation=15);ax.set_title("Calibration and selective reliability");ax.legend();ax.grid(axis="y",alpha=.25);_savefig(fig,d["figures"]/"18_calibration_selective_reliability")
        cdf=pd.DataFrame(conf_class);fig,ax=plt.subplots(figsize=(9,5));ax.bar(cdf["class"],cdf["coverage"]);ax.axhline(1-alpha,linestyle="--",label=f"Target {1-alpha:.0%}");ax.set_ylim(0,1.05);ax.set_ylabel("Empirical coverage");ax.set_title("Class-conditional conformal coverage");ax.legend();ax.grid(axis="y",alpha=.25);_savefig(fig,d["figures"]/"19_mondrian_conformal_coverage")
        fig,ax=plt.subplots(figsize=(8,5));for_df=pd.concat([rc_b,rc_e]);
        for name,g in for_df.groupby("variant"):ax.plot(g["coverage"],g["risk"],label=name)
        ax.set_xlabel("Coverage");ax.set_ylabel("Risk (1 - accuracy)");ax.set_title("Risk-coverage comparison");ax.legend();ax.grid(alpha=.25);_savefig(fig,d["figures"]/"20_risk_coverage_v1.1")

        summary={"baseline":base_m,"enhanced":enh_m,"mpox_recall_baseline":mpox_base,"mpox_recall_enhanced":mpox_enh,"mondrian":pooled_conf,"mcnemar":mc,"n_oof":int(len(y))}
        atomic_json_dump(summary,summary_path);return summary


def run_ood_v11(root,cfg,state,device,logger):
    d=_dirs(root);sp=d["out"]/"ood_v1.1_summary.json"
    if state.done("ood_v1_1") and sp.exists():return json.loads(sp.read_text(encoding="utf-8"))
    with Stage(state,"ood_v1_1",logger):
        man=pd.read_csv(root/"outputs"/"tables"/"dermnet_ood_manifest.csv")
        cats=sorted(man["category"].astype(str).unique())
        seed=int(cfg["project"]["protocol_seed"])
        cats=sorted(cats,key=lambda x:__import__("hashlib").sha256(f"{seed}|{x}".encode()).hexdigest())
        nval=max(1,min(len(cats)-1,int(math.ceil(len(cats)*float(cfg["ood_v1_1"]["ood_validation_category_fraction"])))))
        valcats=set(cats[:nval]); man["ood_split"]=np.where(man["category"].isin(valcats),"validation","test");man.to_csv(d["tables"]/"dermnet_category_holdout_split.csv",index=False)
        rows_by_fold={}; candidate_rows=[]
        for fold in cfg["required_base"]["folds"]:
            logger.info("Feature-space OOD analysis fold %d",fold)
            cache=_extract_train_val_cache(root,cfg,d,fold,device,logger); basez=npz_load(root/"outputs"/"predictions"/f"fold_{fold}.npz"); enz=npz_load(d["cache"]/f"enhanced_fold_{fold}.npz")
            params=json.loads((d["models"]/f"enhancement_params_fold_{fold}.json").read_text());head=joblib.load(d["models"]/f"feature_head_fold_{fold}.joblib")
            oc=d["cache"]/f"dermnet_features_fold_{fold}.npz"
            if oc.exists():odz=npz_load(oc)
            else:
                model=_load_fold_model(root,cfg,fold,device); ds=PathLabelDataset(man.assign(label_index=-1),int(cfg["required_base"]["image_size"]));loader=DataLoader(ds,batch_size=16,shuffle=False,num_workers=int(cfg["required_base"]["num_workers"]),pin_memory=torch.cuda.is_available(),persistent_workers=(int(cfg["required_base"]["num_workers"])>0));pred=infer_logits_features(model,loader,device,amp=bool(cfg["required_base"].get("use_amp",True)),hflip_tta=True);npz_save(oc,**pred);odz=pred;del model
                if torch.cuda.is_available():torch.cuda.empty_cache()
            temp0=float(basez["temperature"][0]);a=float(params["head_weight"]);T=float(params["probability_temperature"])
            def enh_probs(logits,features):
                b=softmax(logits/temp0,axis=1);h=feature_head_probs(head,features);return apply_prob_temperature((1-a)*b+a*h,T)
            id_val_probs=enz["val_probs"];id_test_probs=enz["test_probs"];ood_probs=enh_probs(odz["logits"],odz["features"])
            val_mask=man["ood_split"].values=="validation";test_mask=~val_mask
            scores={}
            for k,v in softmax_ood_scores(id_val_probs,basez["val_logits"]).items():scores.setdefault(k,{})["id_val"]=v
            for k,v in softmax_ood_scores(id_test_probs,basez["test_logits"]).items():scores.setdefault(k,{})["id_test"]=v
            oscores=softmax_ood_scores(ood_probs,odz["logits"])
            for k,v in oscores.items():scores.setdefault(k,{})["ood_val"]=v[val_mask];scores[k]["ood_test"]=v[test_mask]
            for dim in cfg["ood_v1_1"]["pca_dimensions"]:
                fm=fit_feature_ood(cache["train_features"],cache["train_y"],int(dim));sv,Xv=feature_ood_scores(fm,cache["val_features"]);st,Xt=feature_ood_scores(fm,basez["test_features"]);so,Xo=feature_ood_scores(fm,odz["features"])
                for base_name in sv:
                    name=f"{base_name}_pca{dim}";scores[name]={"id_val":sv[base_name],"id_test":st[base_name],"ood_val":so[base_name][val_mask],"ood_test":so[base_name][test_mask]}
            # kNN cosine on standardized, L2-normalized frozen features
            scaler=__import__("sklearn.preprocessing",fromlist=["StandardScaler"]).StandardScaler();Xtr=scaler.fit_transform(cache["train_features"]);Xv=scaler.transform(cache["val_features"]);Xt=scaler.transform(basez["test_features"]);Xo=scaler.transform(odz["features"])
            norm=lambda x:x/(np.linalg.norm(x,axis=1,keepdims=True)+1e-12);Xtr=norm(Xtr);Xv=norm(Xv);Xt=norm(Xt);Xo=norm(Xo)
            for k in cfg["ood_v1_1"]["knn_k"]:
                kk=min(int(k),len(Xtr));nn=NearestNeighbors(n_neighbors=kk,metric="cosine",n_jobs=-1).fit(Xtr)
                fn=lambda x:1-nn.kneighbors(x,return_distance=True)[0].mean(1)
                name=f"knn_cosine_k{kk}";scores[name]={"id_val":fn(Xv),"id_test":fn(Xt),"ood_val":fn(Xo[val_mask]),"ood_test":fn(Xo[test_mask])}
            # Fixed score fusions. Each component is standardized using ID validation only;
            # OOD validation labels are used only later to choose among the pre-specified candidates.
            fusion_pairs=[
                ("msp","knn_cosine_k10"),
                ("msp","cosine_prototype_pca64"),
                ("knn_cosine_k10","mahalanobis_pca64"),
                ("msp","relative_mahalanobis_pca64"),
            ]
            for n1,n2 in fusion_pairs:
                if n1 not in scores or n2 not in scores: continue
                fused={}
                for name in (n1,n2):
                    mu=float(np.mean(scores[name]["id_val"]));sd=float(np.std(scores[name]["id_val"]))+1e-12
                    for split in ("id_val","id_test","ood_val","ood_test"):
                        fused.setdefault(split,[]).append((scores[name][split]-mu)/sd)
                scores[f"fusion__{n1}__{n2}"]={k:0.5*(v[0]+v[1]) for k,v in fused.items()}
            # Cache score arrays for selected-method replay.
            rows_by_fold[fold]=scores
            for name,s in scores.items():candidate_rows.append({"fold":fold,"method":name,"validation_auroc":score_auroc(s["id_val"],s["ood_val"]),"validation_fpr95":fpr95(s["id_val"],s["ood_val"])})
        cdf=pd.DataFrame(candidate_rows);cdf.to_csv(d["tables"]/"ood_candidate_validation_metrics.csv",index=False)
        agg=cdf.groupby("method",as_index=False).agg(validation_auroc_mean=("validation_auroc","mean"),validation_auroc_std=("validation_auroc","std"),validation_fpr95_mean=("validation_fpr95","mean")).sort_values(["validation_auroc_mean","validation_fpr95_mean"],ascending=[False,True]);agg.to_csv(d["tables"]/"ood_method_selection.csv",index=False)
        selected=str(agg.iloc[0]["method"]);logger.info("Selected OOD method from disjoint DermNet validation categories: %s",selected)
        testrows=[];target=float(cfg["ood_v1_1"]["id_acceptance_target"])
        for fold,scores in rows_by_fold.items():
            s=scores[selected];thr=float(np.quantile(s["id_val"],1-target));testrows.append({"fold":fold,"method":selected,"test_auroc":score_auroc(s["id_test"],s["ood_test"]),"test_fpr95":fpr95(s["id_test"],s["ood_test"]),"threshold_from_id_validation":thr,"test_id_acceptance":float((s["id_test"]>=thr).mean()),"test_ood_rejection":float((s["ood_test"]<thr).mean())})
        tdf=pd.DataFrame(testrows);tdf.to_csv(d["tables"]/"ood_selected_method_test_metrics.csv",index=False)
        summary={"selected_method":selected,"n_ood_validation_categories":len(valcats),"n_ood_test_categories":len(cats)-len(valcats),"test_auroc_mean":float(tdf.test_auroc.mean()),"test_auroc_std":float(tdf.test_auroc.std()),"test_fpr95_mean":float(tdf.test_fpr95.mean()),"test_id_acceptance_mean":float(tdf.test_id_acceptance.mean()),"test_ood_rejection_mean":float(tdf.test_ood_rejection.mean())}
        atomic_json_dump(summary,sp)
        top=agg.head(10).sort_values("validation_auroc_mean");fig,ax=plt.subplots(figsize=(9,6));ax.barh(top.method,top.validation_auroc_mean,xerr=top.validation_auroc_std.fillna(0));ax.set_xlim(.5,1);ax.set_xlabel("OOD AUROC on validation categories");ax.set_title("Feature-space OOD method selection (validation only)");ax.grid(axis="x",alpha=.25);_savefig(fig,d["figures"]/"21_ood_method_selection")
        fig,ax=plt.subplots(figsize=(7,5));ax.errorbar([0],[summary["test_auroc_mean"]],yerr=[summary["test_auroc_std"]],fmt="o",capsize=6);ax.set_xlim(-1,1);ax.set_ylim(.5,1);ax.set_xticks([0],[selected]);ax.set_ylabel("AUROC");ax.set_title("Selected OOD method on held-out DermNet categories");ax.grid(axis="y",alpha=.25);_savefig(fig,d["figures"]/"22_ood_heldout_performance")
        return summary


def run_external_validation(root,cfg,state,device,logger):
    d=_dirs(root);sp=d["out"]/"external_validation_summary.json"
    if not cfg["external_validation"].get("enabled",True):return {"status":"disabled"}
    if state.done("external_validation") and sp.exists():return json.loads(sp.read_text(encoding="utf-8"))
    with Stage(state,"external_validation",logger):
        ecfg=cfg["external_validation"];ensure_kaggle_dataset(ecfg["kaggle_slug"],root/ecfg["downloads_dir"],root/ecfg["extract_dir"],logger)
        ext=discover_external_images(root/ecfg["extract_dir"],bool(ecfg.get("prefer_original_images",True)));ext.to_csv(d["tables"]/"external_msid_discovered.csv",index=False)
        msld=[]
        for f in cfg["required_base"]["folds"]:
            for split in ["train","val","test"]:
                msld.extend(pd.read_csv(root/"data"/"generated_patient_folds"/f"fold_{f}"/f"{split}.csv")["path"].astype(str).tolist())
        msld=list(dict.fromkeys(msld))
        clean,audit=deduplicate_external(ext,msld,int(ecfg["perceptual_hash_distance_threshold"]),d["cache"],logger);audit.to_csv(d["tables"]/"external_duplicate_audit.csv",index=False);clean.to_csv(d["tables"]/"external_msid_clean_manifest.csv",index=False)
        if len(clean)<10:raise RuntimeError(f"Only {len(clean)} external MSID images remain after duplicate filtering; external validation is not reliable.")
        fold_base=[];fold_enh=[]
        for fold in cfg["required_base"]["folds"]:
            cp=d["cache"]/f"external_msid_fold_{fold}.npz"
            if cp.exists():z=npz_load(cp)
            else:
                model=_load_fold_model(root,cfg,fold,device);ds=PathLabelDataset(clean,int(cfg["required_base"]["image_size"]));loader=DataLoader(ds,batch_size=16,shuffle=False,num_workers=int(cfg["required_base"]["num_workers"]),pin_memory=torch.cuda.is_available(),persistent_workers=(int(cfg["required_base"]["num_workers"])>0));z=infer_logits_features(model,loader,device,amp=bool(cfg["required_base"].get("use_amp",True)),hflip_tta=True);npz_save(cp,**z);del model
                if torch.cuda.is_available():torch.cuda.empty_cache()
            basez=npz_load(root/"outputs"/"predictions"/f"fold_{fold}.npz");temp0=float(basez["temperature"][0]);params=json.loads((d["models"]/f"enhancement_params_fold_{fold}.json").read_text());head=joblib.load(d["models"]/f"feature_head_fold_{fold}.joblib");a=float(params["head_weight"]);T=float(params["probability_temperature"])
            b=softmax(z["logits"]/temp0,axis=1);h=feature_head_probs(head,z["features"]);e=apply_prob_temperature((1-a)*b+a*h,T);fold_base.append(b);fold_enh.append(e)
        bp=np.mean(np.stack(fold_base),axis=0);ep=np.mean(np.stack(fold_enh),axis=0);y=clean["label_index"].to_numpy(int)
        shared=[CLASS_TO_IDX[x] for x in ["Mpox","Chickenpox","Measles","Healthy"]];shared_names=[CLASSES[i] for i in shared]
        def eval_external(p):
            pred=p.argmax(1); strict_acc=float((pred==y).mean()); macro=float(f1_score(y,pred,labels=shared,average="macro",zero_division=0));bal=float(np.mean([((pred[y==c])==c).mean() for c in shared if (y==c).any()]))
            pr=p[:,shared];pr=pr/pr.sum(1,keepdims=True);yp=np.asarray([shared[i] for i in pr.argmax(1)]);restricted_acc=float((yp==y).mean());restricted_f1=float(f1_score(y,yp,labels=shared,average="macro",zero_division=0));return {"strict_six_class_accuracy":strict_acc,"shared_class_macro_f1_with_six_class_predictions":macro,"shared_class_balanced_accuracy_with_six_class_predictions":bal,"restricted_four_class_accuracy":restricted_acc,"restricted_four_class_macro_f1":restricted_f1},pred,yp,pr
        bm,bpred,brpred,bpr=eval_external(bp);em,epred,erpred,epr=eval_external(ep)
        rows=[]
        for variant,m in [("Baseline ensemble",bm),("ViruSafe-MPX v1.1 ensemble",em)]:rows.append({"variant":variant,**m})
        pd.DataFrame(rows).to_csv(d["tables"]/"external_validation_metrics.csv",index=False)
        predtab=clean[["path","class","label_index"]].copy();predtab["baseline_pred_six"]=[CLASSES[i] for i in bpred];predtab["enhanced_pred_six"]=[CLASSES[i] for i in epred];predtab["enhanced_pred_restricted4"]=[CLASSES[i] for i in erpred];predtab["enhanced_confidence"] = ep.max(1);predtab.to_csv(d["tables"]/"external_msid_predictions.csv",index=False)
        # restricted 4-class confusion
        idxmap={c:i for i,c in enumerate(shared)};yt=np.asarray([idxmap[i] for i in y]);yp=np.asarray([idxmap[i] for i in erpred]);cm=confusion_matrix(yt,yp,labels=np.arange(4));fig,ax=plt.subplots(figsize=(7,6));im=ax.imshow(cm,cmap="Blues");ax.set_xticks(np.arange(4),shared_names,rotation=30,ha="right");ax.set_yticks(np.arange(4),shared_names);ax.set_xlabel("Predicted");ax.set_ylabel("True");ax.set_title("External MSID validation — restricted shared-label protocol");
        for i in range(4):
            for j in range(4):ax.text(j,i,str(cm[i,j]),ha="center",va="center")
        fig.colorbar(im,ax=ax,fraction=.046,pad=.04);_savefig(fig,d["figures"]/"23_external_msid_confusion_matrix")
        # per-class F1 restricted
        pc=[]
        from sklearn.metrics import precision_recall_fscore_support
        prc,rec,f1,sup=precision_recall_fscore_support(yt,yp,labels=np.arange(4),zero_division=0)
        for i,n in enumerate(shared_names):pc.append({"class":n,"precision":prc[i],"recall":rec[i],"f1":f1[i],"support":int(sup[i])})
        pcdf=pd.DataFrame(pc);pcdf.to_csv(d["tables"]/"external_msid_per_class.csv",index=False);fig,ax=plt.subplots(figsize=(8,5));ax.bar(pcdf["class"],pcdf["f1"]);ax.set_ylim(0,1);ax.set_ylabel("F1");ax.set_title("External MSID per-class performance");ax.grid(axis="y",alpha=.25);_savefig(fig,d["figures"]/"24_external_msid_per_class_f1")
        summary={"discovered_images":int(len(ext)),"images_after_duplicate_filter":int(len(clean)),"excluded_duplicates_or_near_duplicates":int(audit["excluded"].sum()),"baseline":bm,"enhanced":em,"protocol":"MSID is external-only; exact/perceptual duplicates against MSLD removed before evaluation; both strict six-class and restricted shared-four-class metrics are reported."}
        atomic_json_dump(summary,sp);return summary


def run_explainability_v11(root,cfg,state,device,logger):
    d=_dirs(root);manifest=d["tables"]/"gradcam_test_only_v1.1.csv"
    if not cfg["explainability_v1_1"].get("enabled",True):return {"status":"disabled"}
    if state.done("explainability_v1_1") and manifest.exists():return {"n":len(pd.read_csv(manifest))}
    with Stage(state,"explainability_v1_1",logger):
        try:
            from pytorch_grad_cam import GradCAM
            from pytorch_grad_cam.utils.image import show_cam_on_image
            from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        except Exception as e:
            logger.warning("Grad-CAM unavailable: %s",e);return {"status":"skipped","reason":str(e)}
        fold=int(cfg["explainability_v1_1"].get("fold",1));basez=npz_load(root/"outputs"/"predictions"/f"fold_{fold}.npz");enz=npz_load(d["cache"]/f"enhanced_fold_{fold}.npz");p=enz["test_probs"];y=enz["test_y"];pred=p.argmax(1);conf=p.max(1);n=int(cfg["explainability_v1_1"].get("examples_per_type",4))
        cor=np.where(pred==y)[0];err=np.where(pred!=y)[0];sel=[]
        sel += [(int(i),"high_confidence_correct") for i in cor[np.argsort(-conf[cor])[:n]]]
        sel += [(int(i),"uncertain") for i in np.argsort(conf)[:n]]
        sel += [(int(i),"error") for i in err[np.argsort(-conf[err])[:n]]]
        seen=set();sel=[x for x in sel if not (x[0] in seen or seen.add(x[0]))]
        model=_load_fold_model(root,cfg,fold,device);target_layer=None
        for m in model.modules():
            if isinstance(m,torch.nn.Conv2d):target_layer=m
        cam=GradCAM(model=model,target_layers=[target_layer]);tf=__import__("src.data",fromlist=["make_transforms"]).make_transforms(int(cfg["required_base"]["image_size"]))[1]
        params=json.loads((d["models"]/f"enhancement_params_fold_{fold}.json").read_text());head=joblib.load(d["models"]/f"feature_head_fold_{fold}.joblib");temp0=float(basez["temperature"][0]);a=float(params["head_weight"]);T=float(params["probability_temperature"])
        def enhanced_one(img:Image.Image):
            x=tf(img).unsqueeze(0).to(device)
            with torch.no_grad(),torch.autocast(device_type=device.type,enabled=(device.type=="cuda")):
                l=model(x);ft=model.forward_features(x);ft=ft.mean(dim=(-2,-1)) if ft.ndim==4 else (ft.mean(dim=1) if ft.ndim==3 else ft.flatten(1))
            b=softmax(l.float().cpu().numpy()/temp0,axis=1);h=feature_head_probs(head,ft.float().cpu().numpy());return apply_prob_temperature((1-a)*b+a*h,T)[0]
        outrows=[];faith=[];rng=np.random.default_rng(int(cfg["project"]["protocol_seed"]));size=int(cfg["required_base"]["image_size"])
        for rank,(idx,kind) in enumerate(sel,1):
            path=Path(str(basez["test_paths"][idx]));target=int(pred[idx])
            with Image.open(path) as im:rgb=im.convert("RGB").resize((size,size));rf=np.asarray(rgb).astype(np.float32)/255.;tensor=tf(rgb).unsqueeze(0).to(device)
            gray=cam(input_tensor=tensor,targets=[ClassifierOutputTarget(target)])[0];overlay=show_cam_on_image(rf,gray,use_rgb=True);op=d["explanations"]/f"gradcam_v11_{rank:02d}_{kind}_{path.stem}.png";Image.fromarray(overlay).save(op)
            base_conf=float(enhanced_one(rgb)[target]);outrows.append({"kind":kind,"source":str(path),"output":str(op.relative_to(root)),"true_class":CLASSES[int(y[idx])],"pred_class":CLASSES[target],"confidence":float(conf[idx]),"split":"held-out test"})
            arr=np.asarray(rgb).copy();flat=gray.flatten();order=np.argsort(-flat);N=len(flat)
            for frac in cfg["explainability_v1_1"]["faithfulness_mask_fractions"]:
                k=max(1,int(N*float(frac)));mask=np.zeros(N,bool);mask[order[:k]]=True;mask=mask.reshape(size,size);fill=np.median(arr.reshape(-1,3),axis=0).astype(np.uint8);camarr=arr.copy();camarr[mask]=fill;randarr=arr.copy();ridx=rng.choice(N,size=k,replace=False);rmask=np.zeros(N,bool);rmask[ridx]=True;randarr[rmask.reshape(size,size)]=fill
                c1=float(enhanced_one(Image.fromarray(camarr))[target]);c2=float(enhanced_one(Image.fromarray(randarr))[target]);faith.append({"example":rank,"kind":kind,"mask_fraction":float(frac),"original_confidence":base_conf,"cam_mask_confidence":c1,"random_mask_confidence":c2,"cam_confidence_drop":base_conf-c1,"random_confidence_drop":base_conf-c2})
        pd.DataFrame(outrows).to_csv(manifest,index=False);fdf=pd.DataFrame(faith);fdf.to_csv(d["tables"]/"gradcam_deletion_faithfulness.csv",index=False)
        agg=fdf.groupby("mask_fraction",as_index=False)[["cam_confidence_drop","random_confidence_drop"]].mean();fig,ax=plt.subplots(figsize=(7,5));ax.plot(agg.mask_fraction,agg.cam_confidence_drop,marker="o",label="CAM-guided masking");ax.plot(agg.mask_fraction,agg.random_confidence_drop,marker="o",label="Random masking");ax.set_xlabel("Masked image fraction");ax.set_ylabel("Mean confidence drop");ax.set_title("Grad-CAM deletion faithfulness on held-out test images");ax.legend();ax.grid(alpha=.25);_savefig(fig,d["figures"]/"25_gradcam_deletion_faithfulness")
        del model
        if torch.cuda.is_available():torch.cuda.empty_cache()
        return {"n_examples":len(outrows),"mean_cam_drop_30pct":float(agg.iloc[-1].cam_confidence_drop),"mean_random_drop_30pct":float(agg.iloc[-1].random_confidence_drop)}


def run_ablation_v11(root,cfg,state,logger,internal,ood,external):
    d=_dirs(root);p=d["tables"]/"ablation_v1.1.csv"
    if state.done("ablation_v1_1") and p.exists():return pd.read_csv(p).to_dict("records")
    with Stage(state,"ablation_v1_1",logger):
        sel=pd.read_csv(d["tables"]/"selective_prediction_comparison.csv");s90={r.variant:float(r.selective_accuracy) for _,r in sel.iloc[(sel.target_coverage-.9).abs().groupby(sel.variant).idxmin()].iterrows()}
        rows=[
            {"component":"Base EfficientNetV2-S + temperature scaling","macro_f1":internal["baseline"]["macro_f1"],"accuracy":internal["baseline"]["accuracy"],"ece":internal["baseline"]["ece"],"mpox_recall":internal["mpox_recall_baseline"],"selective_accuracy_at_90":s90.get("Baseline"),"ood_auroc_heldout":np.nan,"external_restricted4_accuracy":external.get("baseline",{}).get("restricted_four_class_accuracy",np.nan)},
            {"component":"+ Frozen-feature reliability head + validation-selected fusion","macro_f1":internal["enhanced"]["macro_f1"],"accuracy":internal["enhanced"]["accuracy"],"ece":internal["enhanced"]["ece"],"mpox_recall":internal["mpox_recall_enhanced"],"selective_accuracy_at_90":s90.get("ViruSafe-MPX v1.1"),"ood_auroc_heldout":np.nan,"external_restricted4_accuracy":external.get("enhanced",{}).get("restricted_four_class_accuracy",np.nan)},
            {"component":"+ Class-conditional conformal prediction","macro_f1":internal["enhanced"]["macro_f1"],"accuracy":internal["enhanced"]["accuracy"],"ece":internal["enhanced"]["ece"],"mpox_recall":internal["mpox_recall_enhanced"],"selective_accuracy_at_90":s90.get("ViruSafe-MPX v1.1"),"conformal_coverage":internal["mondrian"]["coverage"],"ood_auroc_heldout":np.nan,"external_restricted4_accuracy":external.get("enhanced",{}).get("restricted_four_class_accuracy",np.nan)},
            {"component":"+ Feature-space OOD gate (full v1.1)","macro_f1":internal["enhanced"]["macro_f1"],"accuracy":internal["enhanced"]["accuracy"],"ece":internal["enhanced"]["ece"],"mpox_recall":internal["mpox_recall_enhanced"],"selective_accuracy_at_90":s90.get("ViruSafe-MPX v1.1"),"conformal_coverage":internal["mondrian"]["coverage"],"ood_auroc_heldout":ood.get("test_auroc_mean",np.nan),"external_restricted4_accuracy":external.get("enhanced",{}).get("restricted_four_class_accuracy",np.nan)},
        ];df=pd.DataFrame(rows);df.to_csv(p,index=False)
        # Native-metric panel, no artificial composite score.
        metrics=["macro_f1","mpox_recall","selective_accuracy_at_90"];fig,ax=plt.subplots(figsize=(10,5));x=np.arange(len(rows));w=.24
        for j,m in enumerate(metrics):ax.bar(x+(j-1)*w,df[m],w,label=m.replace("_"," "))
        ax.set_xticks(x,["Baseline","Reliability head","+ Conformal","Full v1.1"],rotation=15);ax.set_ylim(0,1);ax.set_title("Ablation using native performance and safety metrics");ax.legend();ax.grid(axis="y",alpha=.25);_savefig(fig,d["figures"]/"26_ablation_v1.1_native_metrics")
        return rows


def build_report(root,cfg,state,logger,internal,ood,external,explain):
    d=_dirs(root);rp=d["report"]/"index.html"
    if state.done("report_v1_1") and rp.exists():return
    with Stage(state,"report_v1_1",logger):
        save_excel_from_csvs(d["tables"],d["tables"]/"ViruSafe_MPX_v1.1_All_Tables.xlsx")
        final={"project":cfg["project"]["name"],"internal":internal,"ood":ood,"external":external,"explainability":explain,"methodological_guardrails":["No CNN retraining in v1.1","Outer test labels are not used for reliability-head hyperparameter fitting","OOD method is selected on disjoint DermNet categories and evaluated on held-out categories","MSID is external-only and duplicate-filtered against MSLD"]}
        atomic_json_dump(final,d["out"]/"results_summary_v1.1.json")
        # Manuscript-ready markdown
        md=f"""# ViruSafe-MPX v1.1 — Manuscript-Ready Results\n\n## Internal patient-grouped five-fold evaluation\n- Baseline accuracy: **{internal['baseline']['accuracy']:.4f}**\n- v1.1 accuracy: **{internal['enhanced']['accuracy']:.4f}**\n- Baseline Macro-F1: **{internal['baseline']['macro_f1']:.4f}**\n- v1.1 Macro-F1: **{internal['enhanced']['macro_f1']:.4f}**\n- Baseline Mpox recall: **{internal['mpox_recall_baseline']:.4f}**\n- v1.1 Mpox recall: **{internal['mpox_recall_enhanced']:.4f}**\n- Baseline ECE: **{internal['baseline']['ece']:.4f}**\n- v1.1 ECE: **{internal['enhanced']['ece']:.4f}**\n- Mondrian empirical coverage: **{internal['mondrian']['coverage']:.4f}** (target {internal['mondrian']['target_coverage']:.2f})\n\n## Held-out-category OOD evaluation\n- Selected method: **{ood.get('selected_method','n/a')}**\n- Mean AUROC: **{ood.get('test_auroc_mean',float('nan')):.4f}**\n- Mean FPR@95TPR: **{ood.get('test_fpr95_mean',float('nan')):.4f}**\n- Mean OOD rejection at validation-derived threshold: **{ood.get('test_ood_rejection_mean',float('nan')):.4f}**\n\n## External MSID validation after duplicate filtering\n- Images retained: **{external.get('images_after_duplicate_filter','n/a')}**\n- Strict six-class accuracy: **{external.get('enhanced',{}).get('strict_six_class_accuracy',float('nan')):.4f}**\n- Restricted shared-four-class accuracy: **{external.get('enhanced',{}).get('restricted_four_class_accuracy',float('nan')):.4f}**\n- Restricted shared-four-class Macro-F1: **{external.get('enhanced',{}).get('restricted_four_class_macro_f1',float('nan')):.4f}**\n\n## Statistical comparison\n- McNemar exact p-value (baseline vs v1.1): **{internal['mcnemar']['exact_p_value']:.6g}**\n\n## Interpretation guardrail\nThese results are intended for research reporting. External validation and uncertainty/OOD analyses do not establish clinical diagnostic readiness.\n"""
        (d["out"]/"MANUSCRIPT_READY_RESULTS_v1.1.md").write_text(md,encoding="utf-8")
        imgs=[("Internal performance","17_internal_performance_comparison.png"),("Calibration and selective reliability","18_calibration_selective_reliability.png"),("Class-conditional conformal","19_mondrian_conformal_coverage.png"),("Held-out OOD","22_ood_heldout_performance.png"),("External MSID","23_external_msid_confusion_matrix.png"),("Ablation","26_ablation_v1.1_native_metrics.png")]
        cards="".join(f'<h3>{html.escape(t)}</h3><img src="../figures/{f}" style="max-width:100%;border:1px solid #ddd">' for t,f in imgs if (d["figures"]/f).exists())
        h=f"""<!doctype html><html><head><meta charset='utf-8'><title>ViruSafe-MPX v1.1</title><style>body{{font-family:Arial,sans-serif;max-width:1050px;margin:35px auto;line-height:1.5}}table{{border-collapse:collapse}}td,th{{padding:7px;border:1px solid #ddd}}code{{background:#f3f3f3;padding:2px 5px}}</style></head><body><h1>ViruSafe-MPX v1.1</h1><h2>Reliability Enhancement & External Validation</h2><p>This report preserves the original v1.0.4 results and writes all v1.1 outputs separately under <code>outputs_v1.1</code>.</p><h2>Key internal metrics</h2><table><tr><th>Metric</th><th>Baseline</th><th>v1.1</th></tr><tr><td>Accuracy</td><td>{internal['baseline']['accuracy']:.4f}</td><td>{internal['enhanced']['accuracy']:.4f}</td></tr><tr><td>Macro-F1</td><td>{internal['baseline']['macro_f1']:.4f}</td><td>{internal['enhanced']['macro_f1']:.4f}</td></tr><tr><td>Mpox recall</td><td>{internal['mpox_recall_baseline']:.4f}</td><td>{internal['mpox_recall_enhanced']:.4f}</td></tr><tr><td>ECE</td><td>{internal['baseline']['ece']:.4f}</td><td>{internal['enhanced']['ece']:.4f}</td></tr></table><h2>OOD</h2><p>Selected on disjoint DermNet validation categories: <b>{html.escape(str(ood.get('selected_method')))}</b>; held-out category AUROC = <b>{ood.get('test_auroc_mean',float('nan')):.4f}</b>.</p><h2>External MSID</h2><p>{external.get('images_after_duplicate_filter','n/a')} images retained after duplicate filtering. Restricted four-class accuracy = <b>{external.get('enhanced',{}).get('restricted_four_class_accuracy',float('nan')):.4f}</b>.</p>{cards}<h2>Files</h2><p>See <code>tables/ViruSafe_MPX_v1.1_All_Tables.xlsx</code>, <code>results_summary_v1.1.json</code>, and <code>MANUSCRIPT_READY_RESULTS_v1.1.md</code>.</p></body></html>"""
        rp.write_text(h,encoding="utf-8")
        if cfg.get("report",{}).get("open_when_finished",True):
            try:webbrowser.open(rp.resolve().as_uri())
            except Exception:pass


def run_all(root:Path,cfg:Dict[str,Any],state:V11State,logger):
    device=device_auto();logger.info("ViruSafe-MPX v1.1 started | device=%s | GPU=%s",device,torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    preflight(root,cfg,state,logger)
    internal=run_internal_enhancement(root,cfg,state,device,logger)
    ood=run_ood_v11(root,cfg,state,device,logger)
    external=run_external_validation(root,cfg,state,device,logger)
    explain=run_explainability_v11(root,cfg,state,device,logger)
    run_ablation_v11(root,cfg,state,logger,internal,ood,external)
    build_report(root,cfg,state,logger,internal,ood,external,explain)
    logger.info("ViruSafe-MPX v1.1 COMPLETE. Open outputs_v1.1\\report\\index.html")
    return {"internal":internal,"ood":ood,"external":external,"explainability":explain}
