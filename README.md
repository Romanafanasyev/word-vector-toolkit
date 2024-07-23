# Project Title: Word Vector Toolkit
## 📝 Description

Word Vector Toolkit is a Python project that leverages the Navec word vectors to perform various semantic operations such as finding average words between two given words, identifying opposite words, and solving analogies. This toolkit is specifically designed to work with Russian words.

## 📂 Files

- **main.py**: Demonstrates how to use the toolkit for finding average words, opposite words, and analogies.
- **game.py**: Interactive game using the toolkit to find average words between user inputs.
- **wordVectorToolkit.py**: Contains the WordVectorToolkit class with methods for semantic operations.
- **requirements.txt**: Lists the dependencies required to run the project.

## 🔧 Installation

1. Download the Navec models:
    ```sh
    wget https://storage.yandexcloud.net/natasha-navec/packs/navec_hudlit_v1_12B_500K_300d_100q.tar
    wget https://storage.yandexcloud.net/natasha-navec/packs/navec_news_v1_1B_250K_300d_100q.tar
    ```
2. Install the dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## 🚀 Usage
### ✨ Key Features

The **WordVectorToolkit** provides several key methods for working with word vectors:
1. **Find Average Words**
    - **Method**: `find_avg_words(word1: str, word2: str, top_k: int = 10) -> List[Any]`
    - **Description**: Finds words semantically in between `word1` and `word2`.
2. **Find Opposite Words**
    - **Method**: `find_opposite_words(word: str, top_k: int = 10) -> List[Any]`
    - **Description**: Finds words semantically opposite to the given `word`.
3. **Find Analogies**
    - **Method**: `find_analogy(word_a: str, word_b: str, word_c: str, top_k: int = 1) -> List[Any]`
    - **Description**: Solves analogies like "man is to king as woman is to ?".

### 📖 Usage Details

To use the toolkit, initialize the `WordVectorToolkit` class:

**Example**:
```python
finder = WordVectorToolkit(model_path, log=True)
```
**Parameters**:
- `model_path`: Path to the Navec model file.
- `log`: Boolean flag to enable logging of cosine similarities.

### 💡 Suggested Examples to Try

Here are some interesting examples to explore using the `WordVectorToolkit` class:

1. **Find Average Words**:
    - Words between "эмблема" and "храбрость".
    - Words between "напиток" and "кровь".
2. **Find Opposite Words**:
    - Opposites of "айти".
    - Opposites of "добро" (good).
3. **Find Analogies**:
    - "Врач" is to "больница" as "учитель" is to ?.
    - "Мужчина" is to "король" as "женщина" is to ?.
    - "Париж" is to "Франция" as "Москва" is to ?.

These examples will help you get started with exploring the semantic relationships between words using the toolkit.

## 🔍 Dependencies

- [Navec](https://github.com/natasha/navec): Library of pretrained word embeddings for Russian language.
- [Faiss](https://github.com/facebookresearch/faiss): Library for efficient similarity search and clustering of dense vectors.

## 🤝 Contribution

Please feel free to contribute to this project. You can open issues, suggest features, or submit pull requests on the project's repository.