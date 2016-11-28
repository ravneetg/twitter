import os
import random
import numpy as np
import pandas as pd
from collections import Hashable, defaultdict
import cPickle as pickle

import gensim
import sklearn.feature_extraction.text as sktext
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

import preprocessing
from textauger.exceptions import NotFittedError


class BaseTopicModel(object):
    """
    Base class for topic models such as LDA and LSI.
    """

    def __init__(self, vectorizer=None, num_topics=15, chunksize=1000, passes=10, update_every=1,
                 iterations=1000, random_seed=None, load_model=None):
        if vectorizer is None and load_model is None:
            raise TypeError("Either vectorizer or load_model must be specified")
        if not isinstance(vectorizer, sktext.VectorizerMixin) and vectorizer is not None:
            raise TypeError("vectorizer should be a scikit-learn text vectorizer")
        if load_model is None:
            self.vectorizer = vectorizer
            self.num_topics = num_topics
            self.chunksize = chunksize
            self.passes = passes
            self.update_every = update_every
            self.iterations = iterations
            self.random_seed = random_seed
            self._topic_model = None

            self.id2word = {v: k for k, v in self.vectorizer.vocabulary_.items()}
            if self.random_seed is not None:
                random.seed(self.random_seed)
                np.random.seed(abs(hash(self.random_seed)) % 4294967295)
        else:
            self.load(load_model)


    def score_doc(self, doc):
        """
        Score a new document based on the fitted model.

        Args:
            doc (str): Document to be scored for the model topics.

        Returns:
            Dictionary mapping each topic number to a probability.

        Raises:
            NotFittedError: If model has not been fit to corpus.

        """
        if self._topic_model is None:
            raise NotFittedError('The model needs to be fit before scoring a new document.')
        term_doc_matrix = self.vectorizer.transform([doc])
        corpus = gensim.matutils.Sparse2Corpus(term_doc_matrix, documents_columns=False)

        topic_scores = list(self._topic_model[corpus])[0]
        return {topic: prob for topic, prob in topic_scores}

    def score_corpus(self, corpus):
        """
        Score a new corpus based on the fitted model.

        Args:
            corpus (List[str]): Corpus to be scored for the model topics.
                Each document in the corpus will be given a score for each topic in the model.

        Returns:
            List[Dict]: One dictionary for each document, where the dictionary maps each topic
            number to a probability.

        Raises:
            NotFittedError: If model has not been fit to corpus.

        """
        if self._topic_model is None:
            raise NotFittedError('The model needs to be fit before scoring a new corpus.')

        return [self.score_doc(doc) for doc in corpus]


    def load(self, filepath):
        """
        Load a saved topic model.

        Args:
            filepath (str): Location of the pickle file that contains the model.
        """
        # Currently the topic models need to be initialized with a vectorizer, which is not ideal
        # for a user that will want to load a previously saved model instance.
        # How should we fix?
        if not isinstance(filepath, str):
            raise TypeError("Path to file {} is not a string".format(filepath))
        if not os.path.isfile(filepath):
            raise IOError("{} is not an existing file".format(filepath))

        with open(filepath, 'r') as f:
            self.__dict__ = pickle.load(f)
        if self.random_seed is not None:
            random.seed(self.random_seed)
            np.random.seed(abs(hash(self.random_seed)) % 4294967295)

    def save(self, filepath):
        """
        Save the topic model to filepath.

        Args:
            filepath (str): Location to save the pickle file.
        """
        if not isinstance(filepath, str):
            raise TypeError("Path to file {} is not a string".format(filepath))

        with open(filepath, 'w') as f:
            pickle.dump(self.__dict__, f)


    def topic_words(self, n_words=10, probs=True):
        """
        Returns the top words for each topic with probability scores.

        Args:
            n_words: Number of words to return for each topic.
            probs: If True, also returns the probability score for each word.

        Returns:
            Dict: Mapping topic number to list of Tuple(word, prob) (if probs=True)
            or list of words (if probs=False) in decreasing order of importance.

        Raises:
            NotFittedError: If model has not been fit to corpus.
            TypeError: If `n_words` is not an int.
            ValueError: If `n_words` is less than 1.

        """
        if self._topic_model is None:
            raise NotFittedError('The model needs to be fit before retrieving words for each topic.')
        if not isinstance(n_words, int):
            raise TypeError('n_words must be int')
        if (n_words<1):
            raise ValueError('n_words must be greater than 0')

        topics = self._topic_model.show_topics(self.num_topics, n_words, formatted=False)
        if probs:
            topic_word_dict = {topic: word_list for topic, word_list in topics}
        else:
            topic_word_dict = {topic: list(zip(*word_list)[0]) for topic, word_list in topics}

        return topic_word_dict


    def assign_topics(self, docs, cutoff=None, num_topics=2):
        """
        Returns the top topics for each document in `docs` with probabilities.

        Only topics with probabilities above the `cutoff` value for a given document are included.
        If no topics meet the `cutoff` threshold, then an empty list will be returned for that
        document.

        Args:
            docs (List[str])
            cutoff (Optional[float]): Cutoff threshold for probability score of included topics.
                Must be in the range [0.0,1.0].
            num_topics: Max number of topics to be included for each document.

        Returns:
            List with one item for each document.  Each item is a list of tuples with
            (topic, probability) for top topics in document.

        Raises:
            NotFittedError: If model has not been fit to a corpus.
            TypeError: If `cutoff` is not a float.
            ValueError: If `cutoff` is outside of the range [0,1].
            TypeError: If `num_topics` is not an int.

        """
        if self._topic_model is None:
            raise NotFittedError('The model needs to be fit before assigning topics.')
        if cutoff is None:
            cutoff = 0.0
        if not 0.0 <= cutoff <= 1.0:
            raise ValueError('cutoff should be between 0.0 and 1.0')

        docs = preprocessing._u(docs)

        topic_scores_alldocs = self.score_corpus(docs)

        top_topics_alldocs = []
        for topic_scores in topic_scores_alldocs:
            top_topics = sorted(topic_scores.iteritems(), key=lambda x: x[1], reverse=True)
            top_topics_alldocs.append([topic for topic in top_topics[:num_topics]
                                             if topic[1] >= cutoff])

        return top_topics_alldocs


    def fit(self, corpus):
        raise NotImplementedError


class LdaModel(BaseTopicModel):
    """
    Latent Dirichlet Allocation (LDA) model based on gensim.

    Fits the LDA model to a training corpus for determining topics.  Can apply model to other
    documents for finding topics.  Uses the
    `gensim <https://radimrehurek.com/gensim/models/ldamodel.html>`_ package.

    Args:
        vectorizer: Trained vectorizer, for example, the output of
            :func:`textfeatures.tfidf_vectorizer` or :func:`textfeatures.count_vectorizer`.
        num_topics: Number of topics to train for in model.
        update_every: Outputs model update every number of `chunksize` documents.
        iterations: Maximum number of iterations for model.
        random_seed (Optional[hashable object]): Sets the random seed for the model.
            If None, the model in gensim chooses the seed.

    """

    def __init__(self, *args, **kwargs):
        super(LdaModel, self).__init__(*args, **kwargs)

    def fit(self, docs):
        """
        Fits the LDA model to the `docs`.

        Args:
            docs (List[str]): List of documents to fit model to.

        Returns:
            An instance of self.

        """
        docs = preprocessing._u(docs)
        train_term_doc_matrix = self.vectorizer.transform(docs)
        corpus = gensim.matutils.Sparse2Corpus(train_term_doc_matrix, documents_columns=False)
        self._topic_model = gensim.models.ldamodel.LdaModel(corpus,
                                                    num_topics=self.num_topics,
                                                    id2word=self.id2word,
                                                    chunksize=self.chunksize,
                                                    passes=self.passes,
                                                    update_every=self.update_every,
                                                    iterations=self.iterations)

        return self


class LsiModel(BaseTopicModel):
    """
    Latent Semantic Indexing (LSI) model based on gensim.

    Fits the LSI model to a training corpus for determining topics.  Can apply model to other
    documents for finding topics.  Uses the
    `gensim <https://radimrehurek.com/gensim/models/lsimodel.html>`_ package. Performs TF-IDF
    prior to model fit.

    Args:
        vectorizer: Trained vectorizer, for example, the output of
            :func:`textfeatures.tfidf_vectorizer` or :func:`textfeatures.count_vectorizer`.
        num_topics: Number of topics to train for in model.
        update_every: Outputs model update every number of `chunksize` documents.
        iterations: Maximum number of iterations for model.
        random_seed (Optional[hashable object]): Sets the random seed for the model.
            If None, the model in gensim chooses the seed.

    """

    def __init__(self, *args, **kwargs):
        super(LsiModel, self).__init__(*args, **kwargs)

    def fit(self, docs):
        """
        Fits the LSI model to the `docs`.

        Args:
            docs (List[str]): List of documents to fit model to.

        Returns:
            An instance of self.

        """
        docs = preprocessing._u(docs)
        train_term_doc_matrix = self.vectorizer.transform(docs)
        corpus = gensim.matutils.Sparse2Corpus(train_term_doc_matrix, documents_columns=False)
        self._topic_model = gensim.models.lsimodel.LsiModel(corpus,
                                                    num_topics=self.num_topics,
                                                    id2word=self.id2word,
                                                    chunksize=self.chunksize)

        return self


class BinaryClassificationModel(object):
    """
    BinaryClassification model

    This model fits a binary classification model of your choosing.  It also calculates
    various model evaluation metrics.  This uses the
    `sklearn documentation <http://goo.gl/d4vGwr>`_ package.

    Args:
        model_type (str): The type of model to build.  Accepted values are "LogisticRegression",
            "DecisionTree", "RandomForest", or "SupportVectorMachine".
        model_parmeters (dict): A dictionary of accepted model parameters.  Refer to the
            `sklearn documentation <http://goo.gl/d4vGwr>`_ for accepted parameters.
    """
    def __init__(self, model_type, model_parameters=None):
        if model_parameters is None:
            model_parameters = {}

        self.model_type = model_type
        self.model_parameters = model_parameters

        if self.model_type == "LogisticRegression":
            self.model = LogisticRegression(**self.model_parameters)
        elif self.model_type == "DecisionTree":
            self.model = DecisionTreeClassifier(**self.model_parameters)
        elif self.model_type == "RandomForest":
            self.model = RandomForestClassifier(**self.model_parameters)
        elif self.model_type == "SupportVectorMachine":
            self.model = LinearSVC(**self.model_parameters)
        else:
            raise ValueError("model_type must be one of the following:" \
                                "\n    LogisticRegression" \
                                "\n    DecisionTree" \
                                "\n    RandomForest" \
                                "\n    SupportVectorMachine")

        self.columns_ = None
        self.coef_ = None
        self.preds_ = None
        self.probs_ = None
        self.model_evaluation_ = None


    def fit(self, features, target):
        """
        Fits a model given the features and target.  The model is stored in the object.

        Args:
            features (Pandas DataFrame): The training set features
            target (1d array/Pandas Series): The target labels for the given training features

        Returns:
            None
        """
        self.columns_ = features.columns.tolist()
        self.model.fit(features, target)
        self.coef_ = self.model.coef_.tolist()[0]


    def get_coefficients(self):
        """
        Returns the coefficients of a fitted model

        Args:
            None

        Returns:
            dict
                Column name-coefficient key value pairs

        Raises:
            NotFittedError: If model has not been fit
        """
        if self.coef_ is None:
            raise NotFittedError('The model needs to be fit before getting coefficients.')

        return dict(zip(self.columns_, self.coef_))


    def predict(self, features):
        """"
        Calculate the class label given a set of features

        Args:
            features (Pandas DataFrame): The features to score

        Returns:
            numpy 1d array
                The scored class labels corresponding to the given features

        Raises:
            NotFittedError: If model has not been fit
        """
        if self.coef_ is None:
            raise NotFittedError('The model needs to be fit before making class predictions.')

        self.preds_ = self.model.predict(features)

        return self.preds_


    def predict_proba(self, features):
        """
        Calculate the predicted probability of each class

        Args:
            features (Pandas DataFrame): The features to score

        Returns:
            numpy 2d array
                The probabilities of both classes

        Raises:
            NotFittedError: If model has not been fit
        """
        if self.coef_ is None:
            raise NotFittedError('The model needs to be fit before making probability predictions.')

        self.probs_ = self.model.predict_proba(features)

        return self.probs_


    def evaluate(self, test_features, test_target_labels):
        """
        Evaluate a trained model's performance.  This assumes you've fit the model already.

        Args:
            test_features (Pandas DataFrame): The test features used to evaluate the model
            test_target_labels (numpy 1d array): The associated test target to evaluate the model

        Returns:
            dict
                A dictionary of various model performance metrics

        Raises:
            NotFittedError: If model has not been fit
        """
        if self.coef_ is None:
            raise NotFittedError('The model needs to be fit before model evaluation.')

        # Calculate probabilities and lables
        self.predict(test_features)
        self.predict_proba(test_features)

        # Model evaluation metrics
        self.model_evaluation_ = {
            'accuracy_score': metrics.accuracy_score(test_target_labels, self.preds_),
            'confusion_matrix': metrics.confusion_matrix(test_target_labels, self.preds_),
            'f1_score': metrics.f1_score(test_target_labels, self.preds_),
            'precision_score': metrics.precision_score(test_target_labels, self.preds_),
            'recall_score': metrics.recall_score(test_target_labels, self.preds_),
            'roc_auc_score': metrics.roc_auc_score(test_target_labels, self.probs_[:,1]),
            'somers_d': (metrics.roc_auc_score(test_target_labels, self.probs_[:,1]) - 0.5) * 2.0,
            'zero_one_loss': metrics.zero_one_loss(test_target_labels, self.preds_)
        }
        # Note: Somers' D is related to ROC-AUC by AUC = Somers' D / 2 + 0.5
        #       Thus, Somers' D = (AUC - 0.5) * 2.0
        # Source: https://en.wikipedia.org/wiki/Somers%27_D#Somers.E2.80.99_D_for_logistic_regression

        return self.model_evaluation_


    def save(self, file_dir):
        """
        Save the BinaryClassificationModel as a pickle file

        Args:
            file_dir (str): The directory to save the file

        Returns:
            None

        Raises:
            AssertionError: If the directory doesn't exist and cannot be created
        """
        # Check for existance of directory
        if not os.path.exists(file_dir) or not os.path.isdir(file_dir):
            os.makedirs(file_dir)

        assert(os.path.exists(file_dir)), "The given directory, %s, does not exist and cannot be created." % file_dir

        # Write the self dict
        with open(os.path.join(file_dir, "%s.pkl" % self.model_type), 'wb') as pickle_file:
            pickle.dump(self.__dict__, pickle_file)


    def load(self, file_path):
        """
        Loads a pickeled BinaryClassificationModel, updating the object __dict__ to return
        to the previously saved state.

        Args:
            file_path (str): The path to the pickeled BinaryClassificationModel file

        Returns:
            None

        Raises:
            AssertionError: If the pickle file doesn't exist
        """
        assert(os.path.exists(file_path)), "The given pickle file, %s, does not exist." % file_path

        with open(file_path) as pickle_file:
            self.__dict__.update(pickle.load(pickle_file))

