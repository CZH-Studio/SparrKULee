@REM python -m utils.preprocessing.preprocessing

@REM python -m match_mismatch.experiments.baseline -p 1.0 -c 2 -n p100nc2
@REM python -m match_mismatch.experiments.baseline -p 1.0 -c 5 -n p100nc5
@REM python -m match_mismatch.experiments.baseline -p 0.1 -c 2 -n p10nc2
@REM python -m match_mismatch.experiments.baseline -p 0.1 -c 5 -n p10nc5

python -m match_mismatch.experiments.BMMSNet -p 1.0 -c 2 -n p100nc2
python -m match_mismatch.experiments.BMMSNet -p 1.0 -c 5 -n p100nc5
python -m match_mismatch.experiments.BMMSNet -p 0.1 -c 2 -n p10nc2
python -m match_mismatch.experiments.BMMSNet -p 0.1 -c 5 -n p10nc5

python -m match_mismatch.experiments.sota -p 1.0 -c 2 -n p100nc2
python -m match_mismatch.experiments.sota -p 1.0 -c 5 -n p100nc5 -b 32
@REM python -m match_mismatch.experiments.sota -p 0.1 -c 2 -n p10nc2
@REM python -m match_mismatch.experiments.sota -p 0.1 -c 5 -n p10nc5
