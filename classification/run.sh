CUDA_VISIBLE_DEVICES=1 python3 main.py --mode train --model res_18 --lr 3e-3 --batch-size 16 --data-path ../data/gt_crop1 --save-path ./log/out1
CUDA_VISIBLE_DEVICES=1 python3 main.py --mode train --model res_18 --lr 1e-3 --batch-size 16 --data-path ../data/gt_crop1 --save-path ./log/out2
CUDA_VISIBLE_DEVICES=1 python3 main.py --mode train --model res_18 --lr 3e-4 --batch-size 16 --data-path ../data/gt_crop1 --save-path ./log/out3
CUDA_VISIBLE_DEVICES=1 python3 main.py --mode train --model res_18 --lr 1e-4 --batch-size 16 --data-path ../data/gt_crop1 --save-path ./log/out4
CUDA_VISIBLE_DEVICES=1 python3 main.py --mode train --model res_18 --lr 3e-5 --batch-size 16 --data-path ../data/gt_crop1 --save-path ./log/out5
