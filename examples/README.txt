Two jupyter notebooks are provided to demonstrate the use of the package.

- prediction.ipynb: This notebook demonstrates how to use the package to predict the fire spread on a simulated dataset.

- calibration.ipynb: This notebook demonstrates how to calibrate the model on a simulated dataset.

Here are some scripts to reproduce the results in the paper.

Commands:
---
Timing MPI-CA:
time mpirun -n 64 python mpica.py --size 50
time mpirun -n 64 python mpica.py --size 100

sometimes ' --allow-run-as-root' is needed, so it becomes
time mpirun --allow-run-as-root -n 64 python mpica.py --size 100


---
Timing PyTorchFire:
time python speedtest.py --size 50 --device cpu
time python speedtest.py --size 50 --device cuda
python speedtest.py --size 100 --device cpu
python speedtest.py --size 100 --device cuda
python speedtest.py --size 1000 --device cuda


---
The next commands require datasets to be downloaded and stored in current folder.

---
Parameter calibration on simulated dataset (as shown in table):

Repeat target:
python simulated_tbl.py --mode predict --exp_id s0_0 --params_from_exp True --device cuda:0 --run_name target_s0_0

Before:
python simulated_tbl.py --mode predict --exp_id s0_0 --device cuda:0 --run_name before_s0_0

Performing calibration:
python simulated_tbl.py --mode train --exp_id s0_0 --device cuda:0 --max_epochs 30 --run_name trained_s0_0

After:
python simulated_tbl.py --mode predict_from_result --exp_id s0_0 --device cuda:0 --run_name s0_0


---
The following two scripts were written earlier and are not as clean as the previous ones.


---
Parameter calibration on simulated dataset (as shown in figure):

Repeat target:
python simulated_fig.py --mode repeat --exp_id Bear_2020 --device cuda:0 --run_name repeat_0_0

Before:
python simulated_fig.py --mode predict --exp_id Bear_2020 --p_h 0.15 --steps_update_interval 10 --device cuda:0 --run_name 0_0.15

Performing calibration:
python simulated_fig.py --mode train --exp_id Bear_2020 --p_h 0.15 --steps_update_interval 10 --max_epochs 30 --device cuda:0 --run_name calibrated_0_0.15

After:
python simulated_fig.py --mode predict --exp_id Bear_2020 --a 0.07944133877754211 --c_1 0.1871010810136795 --c_2 0.0030385444406419992 --p_h 0.2542177438735962 --seed 18324160971470526000 --device cuda:0 --run_name after_0_0.15


---
Parameter calibration on real fire (as shown in figure):

Before:
python real.py --mode predict --exp_id Bear_2020 --p_h 0.15 --device cuda:0 --run_name real_Bear_2020_0.15_before

Performing calibration:
python real.py --mode train --exp_id Bear_2020 --p_h 0.15 --max_epochs 30 --device cuda:0 --run_name real_Bear_2020_0.15

After:
python real.py --mode predict --exp_id Bear_2020 --a 0 --c_1 0.00200487463735044 --c_2 0.2683422863483429 --p_continue 0.30000001192092896 --p_h 0.20000000298023224 --seed 14381954276780055000 --device cuda:0 --run_name after_Bear_2020_0.15
