import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
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
    ls = ls.replace("\n", "")
    ls_split = ls.split(" ")
    map_object = map(int, ls_split)
    ls_split = list(map_object)
    predictions.extend(ls_split)
    #print(the_list)
    #print(type(the_list))
    #predictions.extend(i)


actual_list = []
for ls_a in actual:
    ls_a = ls_a.replace("[", "")
    ls_a = ls_a.replace("]", "")
    ls_a = ls_a.replace("\n", "")
    ls_a_split = ls_a.split(" ")
    map_object = map(int, ls_a_split)
    ls_a_split = list(map_object)
    actual_list.extend(ls_a_split)
    #print(the_list)
    #print(type(the_list))
    #predictions.extend(i)

print(classification_report(actual_list,predictions ))
print(confusion_matrix(actual_list,predictions))