""""""  		  	   		 	   			  		 			     			  	 
"""  		  	   		 	   			  		 			     			  	 
A simple wrapper for linear regression.  (c) 2015 Tucker Balch  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		 	   			  		 			     			  	 
Atlanta, Georgia 30332  		  	   		 	   			  		 			     			  	 
All Rights Reserved  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
Template code for CS 4646/7646  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		 	   			  		 			     			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		 	   			  		 			     			  	 
and other users of this template code are advised not to share it with others  		  	   		 	   			  		 			     			  	 
or to make it available on publicly viewable websites including repositories  		  	   		 	   			  		 			     			  	 
such as github and gitlab.  This copyright statement should not be removed  		  	   		 	   			  		 			     			  	 
or edited.  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
We do grant permission to share solutions privately with non-students such  		  	   		 	   			  		 			     			  	 
as potential employers. However, sharing with other current or future  		  	   		 	   			  		 			     			  	 
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		 	   			  		 			     			  	 
GT honor code violation.  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
-----do not edit anything above this line---  		  	   		 	   			  		 			     			  	 
"""  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
import numpy as np
import scipy.stats as st


def find_best_feature_random(data):
    return np.random.randint(data.shape[1] - 1)


def build_tree(data, leaf_size):
    if (data.shape[0] <= leaf_size):
        return np.array([['leaf', st.mode(data[:, -1]).mode, None, None]])
    elif len(set(data[:, -1])) == 1:
        return np.array([['leaf', data[:, -1][0], None, None]])
    else:
        # determine the best feature
        feature = find_best_feature_random(data)
        split_val = np.median(data[:, feature])

        # edge case [0, 0, 1, 1, 1] median = 1,
        if data[:, feature].max() <= split_val:
            # all data goes to left tree, right tree empty, change median to mean for split_val
            split_val = np.mean(data[:, feature])

        # edge case [0, 0, 0, 1, 1] median = 0
        if data[:, feature].min() >= split_val:
            # all data goes to right tree, left tree empty, change median to mean for split_val
            split_val = np.mean(data[:, feature])

        if len(set(data[:, feature])) == 1:
            return np.array([['leaf', st.mode(data[:, -1]).mode, None, None]])

        left_tree = build_tree(data[data[:, feature] <= split_val], leaf_size)
        right_tree = build_tree(data[data[:, feature] > split_val], leaf_size)
        root = np.array([feature, split_val, 1, left_tree.shape[0] + 1])
        return np.vstack((root, left_tree, right_tree))


class RTLearner(object):
    """  		  	   		 	   			  		 			     			  	 
    This is a Linear Regression Learner. It is implemented correctly.  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		 	   			  		 			     			  	 
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.  		  	   		 	   			  		 			     			  	 
    :type verbose: bool  		  	   		 	   			  		 			     			  	 
    """  		  	   		 	   			  		 			     			  	 
    def __init__(self, leaf_size=1, verbose=False):
        """  		  	   		 	   			  		 			     			  	 
        Constructor method  		  	   		 	   			  		 			     			  	 
        """  		  	   		 	   			  		 			     			  	 
        self.tree = None
        self.leaf_size = leaf_size
        self.verbose = verbose
  		  	   		 	   			  		 			     			  	 
    def author(self):  		  	   		 	   			  		 			     			  	 
        """  		  	   		 	   			  		 			     			  	 
        :return: The GT username of the student  		  	   		 	   			  		 			     			  	 
        :rtype: str  		  	   		 	   			  		 			     			  	 
        """  		  	   		 	   			  		 			     			  	 
        return "rchen613"  # replace tb34 with your Georgia Tech username
  		  	   		 	   			  		 			     			  	 
    def add_evidence(self, data_x, data_y):  		  	   		 	   			  		 			     			  	 
        """  		  	   		 	   			  		 			     			  	 
        Add training data to learner  		  	   		 	   			  		 			     			  	 

        :param data_x: A set of feature values used to train the learner  		  	   		 	   			  		 			     			  	 
        :type data_x: numpy.ndarray  		  	   		 	   			  		 			     			  	 
        :param data_y: The value we are attempting to predict given the X data  		  	   		 	   			  		 			     			  	 
        :type data_y: numpy.ndarray  		  	   		 	   			  		 			     			  	 
        """
        # build and save the model
        data = np.c_[data_x, data_y]
        self.tree = build_tree(data, self.leaf_size)


    def query(self, points):  		  	   		 	   			  		 			     			  	 
        """  		  	   		 	   			  		 			     			  	 
        Estimate a set of test points given the model we built.  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
        :param points: A numpy array with each row corresponding to a specific query.  		  	   		 	   			  		 			     			  	 
        :type points: numpy.ndarray  		  	   		 	   			  		 			     			  	 
        :return: The predicted result of the input data according to the trained model  		  	   		 	   			  		 			     			  	 
        :rtype: numpy.ndarray  		  	   		 	   			  		 			     			  	 
        """
        pred_y = []
        for point in points:
            node = 0
            cur_feature = None
            while cur_feature != 'leaf':
                if self.tree[node,1] < point[int(self.tree[node,0])]:
                    node += int(self.tree[node, 3])
                else:
                    node += 1
                cur_feature = self.tree[node,0]
            pred_y.append(self.tree[node, 1])
        return pred_y


if __name__ == "__main__":
    print("the secret clue is 'zzyzx'")  		  	   		 	   			  		 			     			  	 
