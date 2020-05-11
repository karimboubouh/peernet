# Controlled Model Propagation

## Install requirements
`pip install -r requirements.txt`

## Run
`python run.py`

## Experiments

### Communication rounds
```
# MP, confidence False
mp_exp.communication_rounds(**config)
```
> Generates a file with name `communication_rounds_{number_of_nodes}_N` 

```
# MP, confidence True
mp_exp.communication_rounds(**config)
```
> Generates a file with name `communication_rounds_{number_of_nodes}_C` 

```
# CMP, confidence True
mp_exp.communication_rounds(**config)
```
> Generates a file with name `communication_rounds_{number_of_nodes}_F` 

Plot the result using the function `plot(number_of_nodes, analysis)`
```
plot(20, "communication_rounds")
```

### Contribution factor
Same analogy
```
# MP/CMP, confidence True/False
mp_exp.contribution_factor(**config)
```
Plot the result using the function `plot(20, "contribution_factor")`


### data unbalancedness
```
# MP/CMP, confidence True/False
mp_exp.data_unbalancedness(**config)
```
Plot the result using the function `plot(20, "data_unbalancedness")`

### Graph sparsity
```
# MP/CMP, confidence True/False
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
CF_THRESHOLD = 0.5
EPSILON_FAIRNESS = -0.2
CONFIDENCE_MEASURE = "mean"
ACCURACY_METRIC = "accuracy"
...
```