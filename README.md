# Diffusion Models Inference

## Usage
### Environment
```python
pip install -r requirements.txt
```

```yaml
    "which_dataset": {
        "name": "PainValidationDataset", // import Dataset() class / function(not recommend) from default file
        "args":{
            "data_root": "/media/ziyi/Dataset/OAI_pain/full/ap/*",
            "mask_config": {
                "mask_mode": "hybrid"
            },
            "ids": [1223, 1245, 2698, 5591, 6909, 9255, 9351, 9528] // if want all data ids: "all"
        }
    },
```


### Testing
1. Modify the configure file to point to your data following the steps in **Data Prepare** part.
2. Set your model path following the steps in **Resume Training** part.
```yaml
"path": { //set every part file path
	"resume_state": "submodels/cond_model/150"
},
```

3. Run the script:

```python
python eval.py -p test -c config/pain-cond.json
```
