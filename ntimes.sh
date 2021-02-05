#!/bin/bash
for i in {01..10}
do
    python -m py_torch.main --config example_train.txt
    python -m py_torch.main --config example_test.txt
    cp evaluation/pred.json evaluation/pred_$i.json
    echo "Welcome $i times"
done