from bertopic import BERTopic
import pandas as pd

df = pd.read_json("C:/Users/c3007818/Desktop/BERTopic/dataset/News_Category_Dataset_v3.json", lines=True)
docs = (df["headline"] + ". " + df["short_description"]).tolist()
docs = docs[:2000]

topic_model = BERTopic()
topics, probs = topic_model.fit_transform(docs)

print(topic_model.get_topic_info())