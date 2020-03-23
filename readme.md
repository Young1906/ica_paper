To run the experiment of labelling ICA components:

## Library requirements
``` bash
# if conda is installed: 
conda activate

# Install virtualenv 
virtualenv env

# Activate virtualenv
source env/bin/activate #for unix 
env/Script/activate #for window

# Install required packages
pip install -r req.txt
```

## Program flow:
- Download an EDF file and put it into `edf` folder
- Go to config.ini, change `path_edf`, and `path_stage` into paths to `*.edf` file and `STAGE.csv` file
- Run `main.py` 
```bash
# run program to label data:
python main.py
```
- The program will ask for session id.
- After typing session id, the program will display ICA component of the chunk & topomap of each component.
- After that, it'll promt you to reject a component(s), user will enter list of components-to-reject sperated by space. If there is no component to reject, just leave it bank.
- The program will then display the entire EEG signal of session before and after EOG cleaning.
- If the result is satisfying, press `y` to save this chunk to training data, otherwise press `n`. 
- Repeat until sufficient amount of samples is collected.
- Upon finish, data will be saved in the `csvs` folder, where each component record is a `csv` file.
