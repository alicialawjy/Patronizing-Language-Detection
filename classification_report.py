import pandas as pd
from sklearn.metrics import classification_report
import sys

try:
    file_name = sys.argv[1]
except:
    file_name = "eval_results.csv"


the_file = "output-files/" + file_name

print("Accessing the file :" + file_name)

df = pd.read_csv(the_file,index_col=0)

pred = df['pred']
actual = df['actual']

predictions = []
for ls in pred:
    ls = ls.replace("[", "")
    ls = ls.replace("]", "")
    ls_split = ls.split(" ")
    predictions.extend(ls_split)
    #print(the_list)
    #print(type(the_list))
    #predictions.extend(i)


actual_list = []
for ls_a in actual:
    ls_a = ls_a.replace("[", "")
    ls_a = ls_a.replace("]", "")
    ls_a_split = ls_a.split(" ")
    actual_list.extend(ls_a_split)
    #print(the_list)
    #print(type(the_list))
    #predictions.extend(i)

print(classification_report(actual_list,predictions ))