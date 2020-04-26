from nltk.translate.bleu_score import corpus_bleu

reference = [['this', 'is', 'a', 'test'], ['this', 'is' 'test']]
candidate = ['this', 'is', 'a', 'test']
score = corpus_bleu(reference, candidate)
print(score)