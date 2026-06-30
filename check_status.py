import json, os

for v in ["v0-08d2","v0-08d3","v0-08d4","v0-08d5","v0-08d6","v0-08d7",
          "v0-08d8","v0-08d9","v0-08d10","v0-08d11","v0-08d12","v0-08d14","v0-08d15","v0-08d16"]:
    name = "pokemon-20260627-" + v
    path = "/kaggle/working/" + name + "/" + name + "-unknown0_mlp_report.json"
    if os.path.exists(path):
        r = json.load(open(path))
        m = r.get("unknown0_lgbm_decision_metrics", {})
        t1 = m.get("top1")
        bi = r.get("best_iteration")
        auc = r.get("row_auc")
        print(v + ": top1=" + str(round(t1,4)) + " iter=" + str(bi) + " auc=" + str(round(auc,4)))
    elif os.path.exists("/kaggle/working/" + name + "-submission.tar.gz"):
        print(v + ": DONE (no report)")
    elif os.path.exists("/kaggle/working/" + name):
        nf = len(os.listdir("/kaggle/working/" + name))
        print(v + ": in progress (" + str(nf) + " files)")
    else:
        print(v + ": not started")
