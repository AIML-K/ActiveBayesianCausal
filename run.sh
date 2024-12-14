python3 run/benchmark.py acic Naive
python3 run/benchmark.py acic Leverage
python3 run/benchmark.py acic Mackay
python3 run/benchmark.py acic ACE
python3 run/benchmark.py acic ABC3

python3 src/plot.py bench acic

python3 run/benchmark.py ihdp Naive
python3 run/benchmark.py ihdp Leverage
python3 run/benchmark.py ihdp Mackay
python3 run/benchmark.py ihdp ACE
python3 run/benchmark.py ihdp ABC3

python3 src/plot.py bench ihdp

python3 run/benchmark.py lalonde Naive
python3 run/benchmark.py lalonde Leverage
python3 run/benchmark.py lalonde Mackay
python3 run/benchmark.py lalonde ACE
python3 run/benchmark.py lalonde ABC3

python3 src/plot.py bench lalonde

python3 run/benchmark.py boston Naive
python3 run/benchmark.py boston Leverage
python3 run/benchmark.py boston Mackay
python3 run/benchmark.py boston ACE
python3 run/benchmark.py boston ABC3

python3 src/plot.py bench boston