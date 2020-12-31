import numpy as np


def log_prior(train_labels):
    """
    # $$\log p_y =\begin{bmatrix}\log p(y=0)\\\log p(y=1)\end{bmatrix}$$
    :param train_labels:
    :return:
    """
    comparason_matrix = (np.array([0,1]).reshape(2,1) == train_labels).astype(np.int)
    zeros = np.log(comparason_matrix[0].sum() / len(train_labels))
    ones = np.log(comparason_matrix[1].sum() / len(train_labels))
    log_py = np.array([[zeros],[ones]])
    assert log_py.shape == (2,1)
    return log_py


def cc_mean_ignore_missing(train_features, train_labels):
    """
    # $$\mu_y = \begin{bmatrix} \mathbb{E}[x^{(0)}|y=0] & \mathbb{E}[x^{(0)}|y=1]\\
    # \mathbb{E}[x^{(1)}|y=0] & \mathbb{E}[x^{(1)}|y=1] \\
    # \cdots & \cdots\\
    # \mathbb{E}[x^{(7)}|y=0] & \mathbb{E}[x^{(7)}|y=1]\end{bmatrix}$$

    :param train_features:
    :param train_labels:
    :return:
    """
    N, d = train_features.shape
    zeros_length = np.count_nonzero(train_labels == 0)
    ones_length = np.count_nonzero(train_labels == 1)

    labels = (train_labels.reshape(1,-1) == np.array([0,1]).reshape(2,1)).astype(np.int)
    
    zero_indexes = np.argwhere(labels[0])
    one_indexes = np.argwhere(labels[1])
    
    zeros = np.take(train_features, zero_indexes, axis=0).reshape(zeros_length, d)
    zeros_mean = np.mean(zeros, axis=0)

    ones = np.take(train_features, one_indexes, axis=0).reshape(ones_length, d)
    ones_mean = np.mean(ones, axis=0)
    
    mu_y = np.column_stack([zeros_mean,ones_mean])

    assert mu_y.shape == (d, 2)
    return mu_y


def cc_std_ignore_missing(train_features, train_labels):
    """
    # $$\sigma_y = \begin{bmatrix} \text{std}[x^{(0)}|y=0] & \text{std}[x^{(0)}|y=1]\\
    # \text{std}[x^{(1)}|y=0] & \text{std}[x^{(1)}|y=1] \\
    # \cdots & \cdots\\
    # \text{std}[x^{(7)}|y=0] & \text{std}[x^{(7)}|y=1]\end{bmatrix}$$

    :param train_features:
    :param train_labels:
    :return:
    """
    N, d = train_features.shape
    zeros_length = np.count_nonzero(train_labels == 0)
    ones_length = np.count_nonzero(train_labels == 1)

    labels = (train_labels.reshape(1,-1) == np.array([0,1]).reshape(2,1)).astype(np.int)
    
    zero_indexes = np.argwhere(labels[0])
    one_indexes = np.argwhere(labels[1])
    
    zeros = np.take(train_features, zero_indexes, axis=0).reshape(zeros_length, d)
    zeros_std = np.std(zeros, axis=0)

    ones = np.take(train_features, one_indexes, axis=0).reshape(ones_length, d)
    ones_std = np.std(ones, axis=0)

    sigma_y = np.column_stack([zeros_std,ones_std])
    assert sigma_y.shape == (d, 2)
    
    return sigma_y


def log_prob(features, mu_y, sigma_y, log_py):
    """
    # $$\log p_{x,y} = \begin{bmatrix} \bigg[\log p(y=0) + \sum_{j=0}^{7} \log p(x_1^{(j)}|y=0) \bigg] & \bigg[\log p(y=1) + \sum_{j=0}^{7} \log p(x_1^{(j)}|y=1) \bigg] \\
    # \bigg[\log p(y=0) + \sum_{j=0}^{7} \log p(x_2^{(j)}|y=0) \bigg] & \bigg[\log p(y=1) + \sum_{j=0}^{7} \log p(x_2^{(j)}|y=1) \bigg] \\
    # \cdots & \cdots \\
    # \bigg[\log p(y=0) + \sum_{j=0}^{7} \log p(x_N^{(j)}|y=0) \bigg] & \bigg[\log p(y=1) + \sum_{j=0}^{7} \log p(x_N^{(j)}|y=1) \bigg] \\
    # \end{bmatrix}$$

    :param features:
    :param mu_y:
    :param sigma_y:
    :param log_py:
    :return:
    """
    N, d = features.shape
    prefix = (- np.log(sigma_y)) - ( 1 / 2 * np.log(2 * np.pi))
    #(1/(2sigma^2)) * (x-mu)^2
    pdf_zeros = features - mu_y[:,0]
    pdf_zeros = np.square(pdf_zeros)
    pdf_zeros *= (1 / (2 * np.square(sigma_y[:,0])))
    total_zeros = prefix[:,0] - pdf_zeros
    total_zeros = total_zeros.sum(axis=1)
    total_zeros = total_zeros + log_py[0]
    
    pdf_ones = features - mu_y[:,1]
    pdf_ones = np.square(pdf_ones)
    pdf_ones *= (1 / (2 * np.square(sigma_y[:,1])))
    total_ones = prefix[:,1] - pdf_ones
    total_ones = total_ones.sum(axis=1)
    total_ones = total_ones + log_py[1]
    
    log_p_x_y = np.column_stack([total_zeros, total_ones])
    assert log_p_x_y.shape == (N,2)
    return log_p_x_y


class NBClassifier():
    def __init__(self, train_features, train_labels):
        self.train_features = train_features
        self.train_labels = train_labels
        self.log_py = log_prior(train_labels)
        self.mu_y = self.get_cc_means()
        self.sigma_y = self.get_cc_std()
        
    def get_cc_means(self):
        mu_y = cc_mean_ignore_missing(self.train_features, self.train_labels)
        return mu_y
    
    def get_cc_std(self):
        sigma_y = cc_std_ignore_missing(self.train_features, self.train_labels)
        return sigma_y
    
    def predict(self, features):
        log_p_x_y = log_prob(features, self.mu_y, self.sigma_y, self.log_py)
        return log_p_x_y.argmax(axis=1)


def cc_mean_consider_missing(train_features_with_nans, train_labels):
    """
    # $$\mu_y = \begin{bmatrix} \mathbb{E}[x^{(0)}|y=0] & \mathbb{E}[x^{(0)}|y=1]\\
    # \mathbb{E}[x^{(1)}|y=0] & \mathbb{E}[x^{(1)}|y=1] \\
    # \cdots & \cdots\\
    # \mathbb{E}[x^{(7)}|y=0] & \mathbb{E}[x^{(7)}|y=1]\end{bmatrix}$$

    :param train_features_with_nans:
    :param train_labels:
    :return:
    """
    N, d = train_features_with_nans.shape
    zeros_length = np.count_nonzero(train_labels == 0)
    ones_length = np.count_nonzero(train_labels == 1)

    labels = (train_labels.reshape(1,-1) == np.array([0,1]).reshape(2,1)).astype(np.int)
    
    zero_indexes = np.argwhere(labels[0])
    one_indexes = np.argwhere(labels[1])
    
    zeros = np.take(train_features_with_nans, zero_indexes, axis=0).reshape(zeros_length, d)
    zeros_mean = np.nanmean(zeros, axis=0)

    ones = np.take(train_features_with_nans, one_indexes, axis=0).reshape(ones_length, d)
    ones_mean = np.nanmean(ones, axis=0)
    
    mu_y = np.column_stack([zeros_mean,ones_mean]) 
    
    assert not np.isnan(mu_y).any()
    assert mu_y.shape == (d, 2)
    return mu_y


def cc_std_consider_missing(train_features_with_nans, train_labels):
    """
    # $$\sigma_y = \begin{bmatrix} \text{std}[x^{(0)}|y=0] & \text{std}[x^{(0)}|y=1]\\
    # \text{std}[x^{(1)}|y=0] & \text{std}[x^{(1)}|y=1] \\
    # \cdots & \cdots\\
    # \text{std}[x^{(7)}|y=0] & \text{std}[x^{(7)}|y=1]\end{bmatrix}$$

    :param train_features_with_nans:
    :param train_labels:
    :return:
    """
    N, d = train_features_with_nans.shape
    
    zeros_length = np.count_nonzero(train_labels == 0)
    ones_length = np.count_nonzero(train_labels == 1)

    labels = (train_labels.reshape(1,-1) == np.array([0,1]).reshape(2,1)).astype(np.int)
    
    zero_indexes = np.argwhere(labels[0])
    one_indexes = np.argwhere(labels[1])
    
    zeros = np.take(train_features_with_nans, zero_indexes, axis=0).reshape(zeros_length, d)
    zeros_std = np.nanstd(zeros, axis=0)

    ones = np.take(train_features_with_nans, one_indexes, axis=0).reshape(ones_length, d)
    ones_std = np.nanstd(ones, axis=0)

    sigma_y = np.column_stack([zeros_std,ones_std])
    
    assert not np.isnan(sigma_y).any()
    assert sigma_y.shape == (d, 2)
    return sigma_y


class NBClassifierWithMissing(NBClassifier):
    """
    Compare to accuracy with SciKit version:

    # from sklearn.naive_bayes import GaussianNB
    # gnb = GaussianNB().fit(train_features, train_labels)
    # train_pred_sk = gnb.predict(train_features)
    # eval_pred_sk = gnb.predict(eval_features)
    # print(f'The training data accuracy of your trained model is {(train_pred_sk == train_labels).mean()}')
    # print(f'The evaluation data accuracy of your trained model is {(eval_pred_sk == eval_labels).mean()}')
    """
    def get_cc_means(self):
        mu_y = cc_mean_consider_missing(self.train_features, self.train_labels)
        return mu_y
    
    def get_cc_std(self):
        sigma_y = cc_std_consider_missing(self.train_features, self.train_labels)
        return sigma_y


