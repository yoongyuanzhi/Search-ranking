from flashrank.Ranker import Ranker, RerankRequest

import pandas as pd

# ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="xxx")


# Instantiate Ranker with absolute path
ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/opt")

# Metadata is optional, Id can be your DB ids from your retrieval stage or simple numeric indices.
query = "retail shops approved"
passages = []
rerankrequest = RerankRequest(query=query, passages=passages)
results = ranker.rerank(rerankrequest)
# Convert the list of dictionaries to a pandas DataFrame
df = pd.DataFrame(results)

# Display the DataFrame
print(df)


