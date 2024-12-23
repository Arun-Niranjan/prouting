from langchain_openai import OpenAIEmbeddings
from typing import List, Dict
import numpy as np
from sklearn.decomposition import PCA


class Prouter:
    def __init__(
        self,
        utterances: Dict[str, List[str]],
        embedder: OpenAIEmbeddings,
        use_pca: bool = True,
    ):
        self.embbeder = embedder
        embedded_utterances = {}
        for route, values in utterances.items():
            embedded_utterances[route] = embedder.embed_documents(utterances[route])
        self.embedded_utterances = embedded_utterances
        self.use_pca = use_pca
        if use_pca:
            flat_embedded_vectors = [
                x for xs in list(self.embedded_utterances.values()) for x in xs
            ]
            pca = PCA(n_components=0.8)  # variance explained
            self.pca = pca.fit(flat_embedded_vectors)
            for route, values in self.embedded_utterances.items():
                for idx, vec in enumerate(values):
                    self.embedded_utterances[route][idx] = self.pca.transform(
                        np.array(vec).reshape(-1, 1).T
                    )

    def _get_distances(self, input: str) -> Dict[str, float]:
        embedded_input = np.array(self.embbeder.embed_query(input))
        if self.use_pca:
            embedded_input = self.pca.transform(embedded_input.reshape(-1, 1).T)
        res = {}
        for route, vals in self.embedded_utterances.items():
            dists = []
            for v in vals:
                dists.append(np.linalg.norm(embedded_input - np.array(v)))
            res[route] = np.min(dists)
        return res

    def predict_route_softmax(self, input: str) -> Dict[str, float]:
        res = {}
        dists = self._get_distances(input)
        norm = np.sum(np.exp(-np.array(list(dists.values()))))
        for route, dist in dists.items():
            res[route] = np.exp(-dist) / norm
        return res

    def predict_route(self, input: str) -> Dict[str, float]:
        res = {}
        dists = self._get_distances(input)
        epsilon = 1.0 / 1000000000.0

        norm = np.sum(1.0 / (np.array(list(dists.values())) + epsilon))
        for route, dist in dists.items():
            res[route] = (1.0 / (dist + epsilon)) / norm

        return res
