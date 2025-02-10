# Data pipelines


Data pipelines uses cache to minimize api request and also caches to save calculated features. Caches can be disabled. See notebooks for usage. 

Caches saved in `.cache` folder. 

## Installation

install using python virtual env

```
cd koray/
python3 -m venv .venv
source .venv/bin/activate.fish  # or something else
pip install -r requirements.txt

cd ..
python -m koray.scripts.get-features
```

---
 
## Calculating Features 

you can add new features in `feature_calculation/feature_funcs.py`. remember to delete delete cache if you want to see the new features. 


## Visualization

see `notebooks/`
