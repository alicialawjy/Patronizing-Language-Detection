import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc

from nlpaug.util import Action
import pandas as pd

df = pd.read_csv("datasets/balanced_data/df_train.csv")
print(df.head())

# Insert word by contextual word embeddings (BERT, DistilBERT, RoBERTA or XLNet)
aug = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="substitute")

df_aug = df['text'].apply(lambda x: aug.augment(x))
df_aug.to_csv('df_aug.zip', index=False)

print("Sucessfully saved augmented data using BERT contextual word embeddings")