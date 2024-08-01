# Aim : Ranking search results (open sourced version)

This code enables the user to do 2 things:

(1) Retrieve the top k number of documents from the database based on semantic similarity to the query 
(2) The retrieve document is then further ranked by FlashRank

So the flow of events is :

(1) Provide a query
(2) Retrieve top documents from the database 
(3) Provide further ordering/ranking of the retrieved results, and show it to the user

Benefit of this code:

(1) Semantic based search 
(2) Ranking helps to prioritise search results , so that user only has to look at the most relevant results
