## LIB
``` bash
# if conda installed: 
conda activate

# installing virtualenv 
virtualenv env

source env/bin/activate #for unix 
env/Script/activate #for window

# required package
pip install -r req.txt

```

## FLOW:
- Download a EDF file, put it input edf folder
- Go to config.ini, change `path_edf`, and `path_stage` to path to `*.edf` file and `STAGE.csv` file
- run the program `main.py` 
```bash
# run program to label data:
python main.py
```
- Program will promt sessionid
- Program will display ICA component of the chunk & topomap of each component
- After that, it'll promt you which component(s) to reject, enter list of components sperated by space. If non-refjected, leave bank
- Program will display entire EEG signal of session before and after clean.
- If satisfied, press y to save this chunk to training data, other wise press n. 
- Repeat until sufficient sample collected.
- After finish running, data will be saved at `data.json`