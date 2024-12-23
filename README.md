# prouting
Smart (probabilistic) routing for chatbots using just vector embeddings

This is currently just a demo using langchain and OpenAI to focus on the
concepts, but there is nothing stopping you form implementing this without
them as long as you have:

- utterances per route
- embeddings for each utterance
- the ability to embed a new input that needs to be routed

once you have those, it's just a case of:
- using PCA to reduce the dimensionality of the embedded utterances to the number of classes
- embed the new input and apply the same PCA transformation
- calculate the distances between the new input and the closest utterance of each class
- softmax the (negative) distances to get a categorical probability distribution of class occupancy

Now we can do probabalistic routing, because we have a probability distribution over routes.
