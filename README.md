# Paper: Robust P2P Personalized Learning

## Install requirements
`pip install -r requirements.txt`

## Run
`python run.py`

## Experiments

### Communication rounds
```
# MP, confidence False
mp_exp.communication_rounds(**config)
plots.figure(file, config)
```
> Generates a file with name `communication_rounds_{number_of_nodes}_N` 

```
# MP, confidence True
mp_exp.communication_rounds(**config)
plots.figure(file, config)
```
> Generates a file with name `communication_rounds_{number_of_nodes}_C` 

```
# CDPL, confidence True
mp_exp.communication_rounds(**config)
plots.figure(file, config)
```
> Generates a file with name `communication_rounds_{number_of_nodes}_F` 

Plot the result using the function `plot(number_of_nodes, analysis)`
```
plots.plot(50, "communication_rounds"
```

### Byzantine resilience
Same analogy
```
# MP/CDPL, confidence True/False
mp_exp.byzantine(**config)
```

### Contribution factor
```
# CDPL
mp_exp.contribution_factor(**config)
plots.contribution_factor(file)
```

### Byzantine detection precision
```
# CDPL
file = mp_exp.byzantine_metrics(**config)
plots.byzantine_metrics(file)
```

### data unbalancedness
```
# MP/CDPL, confidence True/False
mp_exp.data_unbalancedness(**config)
```
Plot the result using the function `plot(20, "data_unbalancedness")`

### Graph sparsity
```
# MP/CDPL, confidence True/False
mp_exp.graph_sparsity(**config)
```
Plot the result using the function `plot(20, "graph_sparsity")`

## Customization
To customize the setting of the prototype edit the parameters in `peernet/constants.py`

Default params:

```
DEBUG_LEVEL = 0
START_PORT = 15000
STOP_CONDITION = 100
TEST_SAMPLE = 1000
CF_THRESHOLD = 0.8
EPSILON_FAIRNESS = -0.1
CONFIDENCE_MEASURE = "max"
ACCURACY_METRIC = "accuracy"
...
```