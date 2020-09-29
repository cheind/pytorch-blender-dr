#!/bin/bash
for i in {01..10}
do
    python -m py_torch.main
    python -m py_torch.inference
    cp evaluation/pred.json evaluation/pred_$i.json
    echo "Welcome $i times"
done