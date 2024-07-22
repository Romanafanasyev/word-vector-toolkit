# wget https://storage.yandexcloud.net/natasha-navec/packs/navec_hudlit_v1_12B_500K_300d_100q.tar
# wget https://storage.yandexcloud.net/natasha-navec/packs/navec_news_v1_1B_250K_300d_100q.tar

from wordMidpointFinder import WordMidpointFinder

model_path = 'navec_hudlit_v1_12B_500K_300d_100q.tar'
# model_path = 'navec_news_v1_1B_250K_300d_100q.tar'
finder = WordMidpointFinder(model_path, log=True)

counter = 1
while True:
    word1 = input()
    word2 = input()

    closest_words = finder.find_words_closest_to_midpoint(word1, word2, top_k=10)
    print(f"STEP №'{counter}': {closest_words}")
    counter += 1
