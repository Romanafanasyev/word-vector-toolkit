# wget https://storage.yandexcloud.net/natasha-navec/packs/navec_hudlit_v1_12B_500K_300d_100q.tar
# wget https://storage.yandexcloud.net/natasha-navec/packs/navec_news_v1_1B_250K_300d_100q.tar

from wordVectorToolkit import WordVectorToolkit

model_path = 'navec_hudlit_v1_12B_500K_300d_100q.tar'

finder = WordVectorToolkit(model_path, log=True)

# Finding the average word
avg_words = finder.find_avg_words("эмблема", "храбрость", top_k=10)
print(avg_words)


# Finding the opposite words
opposite_words = finder.find_opposite_words('война', top_k=10)
print(opposite_words)


# Finding an analogy
analogy_result = finder.find_analogy('мужчина', 'король', 'женщина', top_k=10)
print(analogy_result)
