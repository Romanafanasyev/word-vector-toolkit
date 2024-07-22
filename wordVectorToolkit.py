import numpy as np
from typing import List, Any
import faiss
from navec import Navec
from sklearn.metrics.pairwise import cosine_similarity


class WordVectorToolkit:
    def __init__(self, model_path: str, log: bool = False):
        self.navec = Navec.load(model_path)
        self.log = log

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Normalize the vector to unit length."""
        return vector / np.linalg.norm(vector)

    def _compute_midpoint_vector(self, vector1: np.ndarray, vector2: np.ndarray) -> np.ndarray:
        """Compute the midpoint vector between two vectors."""
        norm_vector1 = self._normalize_vector(vector1)
        norm_vector2 = self._normalize_vector(vector2)

        midpoint_vector = (norm_vector1 + norm_vector2) / 2
        return self._normalize_vector(midpoint_vector)

    def find_closest_words(self, target_vector: np.ndarray, exclude_words: List[str] = [], top_k: int = 10) -> List[Any]:
        """Find the top_k words closest to the target vector."""
        vectors = np.array([self.navec[word] for word in self.navec.vocab.words])
        words = np.array(self.navec.vocab.words)

        if exclude_words:
            exclude_indices = [i for i, word in enumerate(words) if word in exclude_words]
            include_indices = [i for i in range(len(words)) if i not in exclude_indices]
            vectors = vectors[include_indices]
            words = words[include_indices]

        vectors = vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]

        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)

        _, indices = index.search(np.array([target_vector]), top_k)
        closest_words = words[indices[0]]

        return closest_words

    def find_avg_words(self, word1: str, word2: str, top_k: int = 10) -> List[Any]:
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

        closest_words = self.find_closest_words(midpoint_vector, exclude_words=[word1, word2], top_k=top_k)

        if self.log:
            self._log_similarity(closest_words, [word1, word2], midpoint_vector)

        return closest_words

    def find_opposite_words(self, word: str, top_k: int = 10) -> List[Any]:
        """Find the top_k words closest to the opposite vector of the given word."""
        if word not in self.navec:
            print(f"No word \"{word}\" in Navec")
            return []

        vector = self.navec[word]
        opposite_vector = -vector

        opposite_words = self.find_closest_words(opposite_vector, exclude_words=[word], top_k=top_k)

        if self.log:
            self._log_similarity(opposite_words, [word], opposite_vector)

        return opposite_words

    def find_analogy(self, word_a: str, word_b: str, word_c: str, top_k: int = 1) -> List[Any]:
        """Find the word that completes the analogy: a:b::c:?. """
        if word_a not in self.navec or word_b not in self.navec or word_c not in self.navec:
            print(f"One or more words not in Navec")
            return []

        vector_a = self.navec[word_a]
        vector_b = self.navec[word_b]
        vector_c = self.navec[word_c]

        analogy_vector = vector_b - vector_a + vector_c

        analogy_words = self.find_closest_words(analogy_vector, exclude_words=[word_a, word_b, word_c], top_k=top_k)

        if self.log:
            self._log_similarity(analogy_words, [word_a, word_b, word_c], analogy_vector)

        return analogy_words

    def _log_similarity(self, closest_words: List[str], reference_words: List[str], target_vector: np.ndarray):
        """Log cosine similarity for each word with the target vector."""
        HEADER = '\033[95m'
        OKBLUE = '\033[94m'
        OKCYAN = '\033[96m'
        OKGREEN = '\033[92m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'

        print(f"{HEADER}===============================")
        print(f"    Cosine Similarity Log")
        print(f"==============================={ENDC}")

        # Log reference words and their similarities
        for i, word in enumerate(reference_words):
            print(f"{BOLD}{OKCYAN}Word {i + 1}: {word}{ENDC}")

        if len(reference_words) >= 2:
            similarity_word1_word2 = \
            cosine_similarity([self.navec[reference_words[0]]], [self.navec[reference_words[1]]])[0][0]
            print(
                f"{OKGREEN}Similarity between {reference_words[0]} and {reference_words[1]}: {similarity_word1_word2:.4f}{ENDC}")

        print(f"{HEADER}-------------------------------{ENDC}")

        # Log similarities with the midpoint vector
        closest_words_list = list(closest_words)
        words_to_log = reference_words + closest_words_list
        for word in words_to_log:
            vector = self.navec[word]
            similarity = cosine_similarity([target_vector], [vector])[0][0]
            print(f"{OKBLUE}{word:<20} -> {OKGREEN}{similarity:.4f}{ENDC}")

        print(f"{HEADER}-------------------------------{ENDC}")

        if closest_words_list:
            # Log similarities between the reference words and the best guess
            best_guess = closest_words_list[0]
            print(f"{OKGREEN}Similarity with best guess ({OKBLUE}{best_guess}{OKGREEN}):{ENDC}")
            for word in reference_words:
                similarity = cosine_similarity([self.navec[word]], [self.navec[best_guess]])[0][0]
                print(f"{OKBLUE}{word:<20} -> {OKGREEN}{similarity:.4f}{ENDC}")

        print(f"{HEADER}==============================={ENDC}")