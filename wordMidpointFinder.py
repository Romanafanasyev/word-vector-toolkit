from typing import List, Any

import numpy as np
import faiss
from navec import Navec
from sklearn.metrics.pairwise import cosine_similarity


class WordMidpointFinder:
    def __init__(self, model_path: str, log: bool = False):
        self.navec = Navec.load(model_path)
        self.log = log

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Normalize the vector to unit length."""
        return vector / np.linalg.norm(vector)

    def _compute_midpoint_vector(self, vector1: np.ndarray, vector2: np.ndarray) -> np.ndarray:
        """Compute the midpoint vector between two vectors."""
        midpoint_vector = (vector1 + vector2) / 2
        return self._normalize_vector(midpoint_vector)

    def find_words_closest_to_midpoint(self, word1: str, word2: str, top_k: int = 10) -> List[Any]:
        """Find the top_k words closest to the midpoint vector between word1 and word2."""
        if word1 not in self.navec:
            print(f"No word \"{word1}\" in Navec")
            return []
        if word2 not in self.navec:
            print(f"No word \"{word2}\" in Navec")
            return []

        vector1 = self.navec[word1]
        vector2 = self.navec[word2]
        midpoint_vector = self._compute_midpoint_vector(vector1, vector2)

        vectors = np.array([self.navec[word] for word in self.navec.vocab.words])
        words = np.array(self.navec.vocab.words)

        # Exclude vectors of word1 and word2 from the search
        exclude_indices = [i for i, word in enumerate(words) if word in {word1, word2}]
        include_indices = [i for i in range(len(words)) if i not in exclude_indices]

        vectors = vectors[include_indices]
        words = words[include_indices]

        # Normalize vectors to unit length
        vectors = vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]

        # Create FAISS index
        index = faiss.IndexFlatIP(vectors.shape[1])  # Inner Product (cosine similarity)
        index.add(vectors)

        # Search for the top_k closest vectors
        _, indices = index.search(np.array([midpoint_vector]), top_k)
        closest_words = words[indices[0]]

        if self.log:
            self._log_similarity(closest_words, word1, word2, midpoint_vector)

        return closest_words

    def _log_similarity(self, closest_words: np.ndarray, word1: str, word2: str, midpoint_vector: np.ndarray):
        """Log cosine similarity for each word with the midpoint vector."""
        # ANSI escape sequences for coloring
        HEADER = '\033[95m'
        OKBLUE = '\033[94m'
        OKCYAN = '\033[96m'
        OKGREEN = '\033[92m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'

        similarity_word1_word2 = cosine_similarity([self.navec[word1]], [self.navec[word2]])[0][0]
        print(f"{HEADER}===============================")
        print(f"    Cosine Similarity Log")
        print(f"==============================={ENDC}")

        print(f"{BOLD}{OKCYAN}Word 1: {word1}{ENDC}")
        print(f"{BOLD}{OKCYAN}Word 2: {word2}{ENDC}")
        print(f"{OKGREEN}Similarity between {word1} and {word2}: {similarity_word1_word2:.4f}{ENDC}")

        print(f"{HEADER}-------------------------------{ENDC}")

        words_to_log = [word1, word2] + list(closest_words)
        for word in words_to_log:
            vector = self.navec[word]
            similarity = cosine_similarity([midpoint_vector], [vector])[0][0]
            print(f"{OKBLUE}{word:<20} -> {OKGREEN}{similarity:.4f}{ENDC}")

        print(f"{HEADER}==============================={ENDC}")
