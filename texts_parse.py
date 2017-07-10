from __future__ import unicode_literals

import nltk
import itertools
import numpy as np
# from sklearn.feature_extraction.text import CountVectorizer


unknown_token = "UNKNOWN_TOKEN"
paragraph_start_token = "SENTENCE_START"
paragraph_end_token = "SENTENCE_END"


def parse(texts, vocabulary_size=10000, sequence_length=100, pad='right'):

    padding_token = vocabulary_size

    paragraphs = [text.lower() for text in texts]
    paragraphs = [
        "%s %s %s" % (
            paragraph_start_token, x, paragraph_end_token) for x in paragraphs
    ]
    # Tokenize the sentences into words
    tokenized_paragraphs = [par.split(' ') for par in paragraphs]

    # Count the word frequencies
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_paragraphs))
    # print "Found %d unique words tokens." % len(word_freq.items())
    # print ' ? in freqdist ', '?' in word_freq

    # Get the most common words and build index_to_word and word_to_index vectors
    vocab = word_freq.most_common(vocabulary_size - 1)
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unknown_token)
    word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

    # print "Using vocabulary size %d." % vocabulary_size
    # print("The least frequent word in our vocabulary is ", vocab[-1][0],
    #       " and appeared ", vocab[-1][1], " times.")

    # Replace all words not in our vocabulary with the unknown token
    for i, par in enumerate(tokenized_paragraphs):
        tokenized_paragraphs[i] = [
            w if w in word_to_index else unknown_token for w in par
        ]
    # print "\nExample sentence: '%s'" % sentences[0]
    # print "\nExample sentence after Pre-processing: '%s'" % tokenized_sentences[0]


    """ 
    tf_vectorizer = CountVectorizer(
      max_df=1.0, min_df=0.0, max_features=vocabulary_size
    )
    tf_vec = tf_vectorizer.fit_transform([q[1] for q in Q])  # titles only
    q_idx = questions_index(Q, tf_vec)
    """
    # Create the training data
    X_train = np.asarray(
        [[word_to_index[w] for w in par] for par in tokenized_paragraphs]
    )

    if pad == 'right':
        X_train = np.vstack(
            [np.pad(
                x,
                (0, sequence_length - len(x)),
                'constant',
                constant_values=padding_token
             ) for x in X_train]
        )
    else:
        X_train = np.vstack(
            [np.pad(
                x,
                (sequence_length - len(x), 0),
                'constant',
                constant_values=padding_token
             ) for x in X_train]
        )

    new_voc_size = len(word_to_index.keys())
    if new_voc_size < vocabulary_size:
        print 'voc size {} instead of {}'.format(new_voc_size, vocabulary_size)
    return X_train, new_voc_size
