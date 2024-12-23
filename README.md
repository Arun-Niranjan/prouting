# prouting
Smart (probabilistic) routing for chatbots using just vector embeddings

This is currently just a demo using langchain and OpenAI to focus on the
concepts, but there is nothing stopping you form implementing this without
them as long as you have:

- utterances per route
- embeddings for each utterance
- the ability to embed a new input that needs to be routed

once you have those, it's just a case of:
- using PCA to reduce the dimensionality of the embedded utterances (80% variance explained for now)
- embed the new input and apply the same PCA transformation
- calculate the distances between the new input and the closest utterance of each class
- either:
  - softmax the (negative) distances to get a categorical probability distribution of class occupancy
  - or do a weighted sum of the inverse distance

Now we can do probabalistic routing, because we have a categorical probability distribution over routes.
