from langchain_openai import OpenAIEmbeddings
from typing import List, Dict
import numpy as np


class Prouter:
    def __init__(
        self,
        utterances: Dict[str, List[str]],
        embedder: OpenAIEmbeddings,
    ):
        self.embbeder = embedder
        embedded_utterances = {}
        for route, values in utterances.items():
            embedded_utterances[route] = embedder.embed_documents(utterances[route])
        self.embedded_utterances = embedded_utterances

    def _get_distances(self, input: str) -> Dict[str, float]:
        embedded_input = np.array(self.embbeder.embed_query(input))
        res = {}
        for route, vals in self.embedded_utterances.items():
            dists = []
            for v in vals:
                dists.append(np.linalg.norm(embedded_input - np.array(v)))
            res[route] = np.min(dists)
        return res

    def predict_route(self, input: str) -> Dict[str, float]:
        res = {}
        dists = self._get_distances(input)
        norm = np.sum(np.exp(-np.array(list(dists.values()))))
        for route, dist in dists.items():
            res[route] = np.exp(-dist) / norm
        return res
