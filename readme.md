
# make env
```sh
conda create -n agent_fill_online_form python=3.11
conda activate agent_fill_online_form
pip install -r requirements.txt
```


# Run
```sh
uvicorn api:app --reload --port 8000
```

Then open http://localhost:8000


