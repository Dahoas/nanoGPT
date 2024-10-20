import pathlib
import os
import shutil
import json
import numpy as np


def find():
    fields = {
        "dataset": "openwebtext_1_10",
    }
    res = {}
    logs = list(pathlib.Path("logs/tml_train").glob("*"))
    for log in logs:
        config = os.path.join(log, "config.json")
        with open(config, "r") as f:
            config = json.load(f)
        flag = True
        for k, v in fields.items():
            if v not in config[k]:
                flag = False
                break
        if flag:
            try:
                with open(os.path.join(log, "stats.jsonl"), "r") as f:
                    lines = f.readlines()
                    stats = [json.loads(s) for s in lines]
            except FileNotFoundError:
                continue
            vals = np.array([s["val/loss"] for s in stats])
            best_val = np.mean(np.sort(vals)[:5])
            last_val = vals[-1]
            res[str(log)] = {"config": config, "val_loss": best_val, "last_val": last_val}
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    find()