import unittest
import pandas as pd
import numpy as np

from ..naive_bayes import log_prior, cc_mean_ignore_missing, cc_std_ignore_missing, \
    cc_mean_consider_missing, cc_std_consider_missing, log_prob, NBClassifier, NBClassifierWithMissing


class TestNaiveBayes(unittest.TestCase):

    def setUp(self):
        self.df = pd.read_csv('diabetes.csv')
        # Let's generate the split ourselves.
        np_random = np.random.RandomState(seed=12345)
        rand_unifs = np_random.uniform(0, 1, size=self.df.shape[0])
        division_thresh = np.percentile(rand_unifs, 80)
        train_indicator = rand_unifs < division_thresh
        eval_indicator = rand_unifs >= division_thresh

        # In[4]:

        train_df = self.df[train_indicator].reset_index(drop=True)
        self.train_features = train_df.loc[:, train_df.columns != 'Outcome'].values
        self.train_labels = train_df['Outcome'].values
        train_df.head()

        # In[5]:

        eval_df = self.df[eval_indicator].reset_index(drop=True)
        self.eval_features = eval_df.loc[:, eval_df.columns != 'Outcome'].values
        self.eval_labels = eval_df['Outcome'].values
        eval_df.head()

        # ## 0.2 Pre-processing The Data

        # Some of the columns exhibit missing values. We will use a Naive Bayes Classifier later that will treat such missing values in a special way.  To be specific, for attribute 3 (Diastolic blood pressure), attribute 4 (Triceps skin fold thickness), attribute 6 (Body mass index), and attribute 8 (Age), we should regard a value of 0 as a missing value.
        #
        # Therefore, we will be creating the `train_featues_with_nans` and `eval_features_with_nans` numpy arrays to be just like their `train_features` and `eval_features` counter-parts, but with the zero-values in such columns replaced with nans.

        # In[7]:

        train_df_with_nans = train_df.copy(deep=True)
        eval_df_with_nans = eval_df.copy(deep=True)
        for col_with_nans in ['BloodPressure', 'SkinThickness', 'BMI', 'Age']:
            train_df_with_nans[col_with_nans] = train_df_with_nans[col_with_nans].replace(0, np.nan)
            eval_df_with_nans[col_with_nans] = eval_df_with_nans[col_with_nans].replace(0, np.nan)
        self.train_features_with_nans = train_df_with_nans.loc[:, train_df_with_nans.columns != 'Outcome'].values
        self.eval_features_with_nans = eval_df_with_nans.loc[:, eval_df_with_nans.columns != 'Outcome'].values

        # In[8]:

        print('Here are the training rows with at least one missing values.')
        print('')
        print('You can see that such incomplete data points constitute a substantial part of the data.')
        print('')
        nan_training_data = train_df_with_nans[train_df_with_nans.isna().any(axis=1)]


    def test_log_prior(self):
        some_labels = np.array([0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1])
        some_log_py = log_prior(some_labels)
        assert np.array_equal(some_log_py.round(3), np.array([[-0.916], [-0.511]]))

        log_py = log_prior(self.train_labels)

    def test_mean_ignore_missing(self):
        some_feats = np.array([[1., 85., 66., 29., 0., 26.6, 0.4, 31.],
                               [8., 183., 64., 0., 0., 23.3, 0.7, 32.],
                               [1., 89., 66., 23., 94., 28.1, 0.2, 21.],
                               [0., 137., 40., 35., 168., 43.1, 2.3, 33.],
                               [5., 116., 74., 0., 0., 25.6, 0.2, 30.]])
        some_labels = np.array([0, 1, 0, 1, 0])

        some_mu_y = cc_mean_ignore_missing(some_feats, some_labels)

        assert np.array_equal(some_mu_y.round(2), np.array([[2.33, 4.],
                                                            [96.67, 160.],
                                                            [68.67, 52.],
                                                            [17.33, 17.5],
                                                            [31.33, 84.],
                                                            [26.77, 33.2],
                                                            [0.27, 1.5],
                                                            [27.33, 32.5]]))

    def test_std_ignore_missing(self):
        some_feats = np.array([[1., 85., 66., 29., 0., 26.6, 0.4, 31.],
                               [8., 183., 64., 0., 0., 23.3, 0.7, 32.],
                               [1., 89., 66., 23., 94., 28.1, 0.2, 21.],
                               [0., 137., 40., 35., 168., 43.1, 2.3, 33.],
                               [5., 116., 74., 0., 0., 25.6, 0.2, 30.]])
        some_labels = np.array([0, 1, 0, 1, 0])

        some_std_y = cc_std_ignore_missing(some_feats, some_labels)

        assert np.array_equal(some_std_y.round(3), np.array([[1.886, 4.],
                                                             [13.768, 23.],
                                                             [3.771, 12.],
                                                             [12.499, 17.5],
                                                             [44.312, 84.],
                                                             [1.027, 9.9],
                                                             [0.094, 0.8],
                                                             [4.497, 0.5]]))

    def test_log_prob(self):
        some_feats = np.array([[1., 85., 66., 29., 0., 26.6, 0.4, 31.],
                               [8., 183., 64., 0., 0., 23.3, 0.7, 32.],
                               [1., 89., 66., 23., 94., 28.1, 0.2, 21.],
                               [0., 137., 40., 35., 168., 43.1, 2.3, 33.],
                               [5., 116., 74., 0., 0., 25.6, 0.2, 30.]])
        some_labels = np.array([0, 1, 0, 1, 0])

        some_mu_y = cc_mean_ignore_missing(some_feats, some_labels)
        some_std_y = cc_std_ignore_missing(some_feats, some_labels)
        some_log_py = log_prior(some_labels)

        some_log_p_x_y = log_prob(some_feats, some_mu_y, some_std_y, some_log_py)

        assert np.array_equal(some_log_p_x_y.round(3), np.array([[-20.822, -36.606],
                                                                 [-60.879, -27.944],
                                                                 [-21.774, -295.68],
                                                                 [-417.359, -27.944],
                                                                 [-23.2, -42.6]]))


    def test_nb_classifier(self):
        diabetes_classifier = NBClassifier(self.train_features, self.train_labels)
        train_pred = diabetes_classifier.predict(self.train_features)
        eval_pred = diabetes_classifier.predict(self.eval_features)

        train_acc = (train_pred == self.train_labels).mean()
        eval_acc = (eval_pred == self.eval_labels).mean()
        print(f'The training data accuracy of your trained model is {train_acc}')
        print(f'The evaluation data accuracy of your trained model is {eval_acc}')


    def test_mean_consider_missing(self):
        some_feats = np.array([[1., 85., 66., 29., 0., 26.6, 0.4, 31.],
                               [8., 183., 64., 0., 0., 23.3, 0.7, 32.],
                               [1., 89., 66., 23., 94., 28.1, 0.2, 21.],
                               [0., 137., 40., 35., 168., 43.1, 2.3, 33.],
                               [5., 116., 74., 0., 0., 25.6, 0.2, 30.]])
        some_labels = np.array([0, 1, 0, 1, 0])

        for i, j in [(0, 0), (1, 1), (2, 3), (3, 4), (4, 2)]:
            some_feats[i, j] = np.nan

        some_mu_y = cc_mean_consider_missing(some_feats, some_labels)

        assert np.array_equal(some_mu_y.round(2), np.array([[3., 4.],
                                                            [96.67, 137.],
                                                            [66., 52.],
                                                            [14.5, 17.5],
                                                            [31.33, 0.],
                                                            [26.77, 33.2],
                                                            [0.27, 1.5],
                                                            [27.33, 32.5]]))

    def test_std_consider_missing(self):
        some_feats = np.array([[1., 85., 66., 29., 0., 26.6, 0.4, 31.],
                               [8., 183., 64., 0., 0., 23.3, 0.7, 32.],
                               [1., 89., 66., 23., 94., 28.1, 0.2, 21.],
                               [0., 137., 40., 35., 168., 43.1, 2.3, 33.],
                               [5., 116., 74., 0., 0., 25.6, 0.2, 30.]])
        some_labels = np.array([0, 1, 0, 1, 0])

        for i, j in [(0, 0), (1, 1), (2, 3), (3, 4), (4, 2)]:
            some_feats[i, j] = np.nan

        some_std_y = cc_std_consider_missing(some_feats, some_labels)

        assert np.array_equal(some_std_y.round(2), np.array([[2., 4.],
                                                             [13.77, 0.],
                                                             [0., 12.],
                                                             [14.5, 17.5],
                                                             [44.31, 0.],
                                                             [1.03, 9.9],
                                                             [0.09, 0.8],
                                                             [4.5, 0.5]]))


    def test_nb_classifier_with_missing_values(self):
        diabetes_classifier_nans = NBClassifierWithMissing(self.train_features_with_nans, self.train_labels)
        train_pred = diabetes_classifier_nans.predict(self.train_features_with_nans)
        eval_pred = diabetes_classifier_nans.predict(self.eval_features_with_nans)

        train_acc = (train_pred == self.train_labels).mean()
        eval_acc = (eval_pred == self.eval_labels).mean()
        print('The training data accuracy of your trained model is {train_acc}'.format(train_acc=train_acc))
        print('The evaluation data accuracy of your trained model is {eval_acc}'.format(eval_acc=eval_acc))
