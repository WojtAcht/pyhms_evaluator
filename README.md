# pyhms_evaluator

Hyperparameter tuning and evaluation of `pyhms`.

Install dependencies and use shell:
```
poetry install --with dev
poetry shell
```

Run:
```
python3 -m pyhms_evaluator
```

To tune parameters:
```
python3 -m pyhms_evaluator.tuner
```

To investigate hyperparameter optimization:
```
deepcave --open
```

Please use `results.ipynb` to investigate results.
