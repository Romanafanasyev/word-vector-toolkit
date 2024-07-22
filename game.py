from wordVectorToolkit import WordVectorToolkit

model_path = 'navec_hudlit_v1_12B_500K_300d_100q.tar'

finder = WordVectorToolkit(model_path, log=True)

# Game Say The Same Thing
counter = 1
while True:
    word1 = input()
    word2 = input()

    print("Hmmm...")

    avg_words = finder.find_avg_words(word1, word2, top_k=10)
    print(f"STEP №'{counter}': {avg_words}")
    counter += 1
