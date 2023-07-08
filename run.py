import os

n_estimators = [110,100,150,200]
max_depth=[20,30,60,70]

for n in n_estimators:
    for m in max_depth:
        #print(n,m)
        os.system(f"python basic_ml_model.py -n{n} -a{m}")