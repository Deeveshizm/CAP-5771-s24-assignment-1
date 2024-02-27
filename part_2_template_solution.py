# Add your imports here.
# Note: only sklearn, numpy, utils and new_utils are allowed.
import utils as u
import new_utils as nu
from sklearn.model_selection import ShuffleSplit, KFold
import numpy as np
from numpy.typing import NDArray
from typing import Any
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import collections

# ======================================================================

# I could make Section 2 a subclass of Section 1, which would facilitate code reuse.
# However, both classes have the same function names. Better to pass Section 1 instance
# as an argument to Section 2 class constructor.


class Section2:
    def __init__(
        self,
        normalize: bool = True,
        seed: int | None = None,
        frac_train: float = 0.2,
    ):
        """
        Initializes an instance of MyClass.

        Args:
            normalize (bool, optional): Indicates whether to normalize the data. Defaults to True.
            seed (int, optional): The seed value for randomization. If None, each call will be randomized.
                If an integer is provided, calls will be repeatable.

        Returns:
            None
        """
        self.normalize = normalize
        self.seed = seed
        self.frac_train = frac_train

    # ---------------------------------------------------------

    """
    A. Repeat part 1.B but make sure that your data matrix (and labels) consists of
        all 10 classes by also printing out the number of elements in each class y and 
        print out the number of classes for both training and testing datasets. 
    """

    def partA(
        self,
    ) -> tuple[
        dict[str, Any],
        NDArray[np.floating],
        NDArray[np.int32],
        NDArray[np.floating],
        NDArray[np.int32],
    ]:
        Xtrain, ytrain, Xtest, ytest = u.prepare_data() 

        answer = {'nb_classes_train': len(np.unique(ytrain)),
                  'nb_classes_test': len(np.unique(ytest)),
                  'class_count_train': collections.Counter(ytrain),
                  'class_count_test': collections.Counter(ytest),
                  'length_Xtrain': len(Xtrain),
                  'length_Xest': len(Xtest),
                  'length_ytrain': len(ytrain),
                  'length_ytest': len(ytest),
                  'max_Xtrain':np.max(Xtrain),
                  'max_Xtest':np.max(Xtest)
                  }

        # Enter your code and fill the `answer`` dictionary

        # `answer` is a dictionary with the following keys:
        # - nb_classes_train: number of classes in the training set
        # - nb_classes_test: number of classes in the testing set
        # - class_count_train: number of elements in each class in the training set
        # - class_count_test: number of elements in each class in the testing set
        # - length_Xtrain: number of elements in the training set
        # - length_Xtest: number of elements in the testing set
        # - length_ytrain: number of labels in the training set
        # - length_ytest: number of labels in the testing set
        # - max_Xtrain: maximum value in the training set
        # - max_Xtest: maximum value in the testing set

        # return values:
        # Xtrain, ytrain, Xtest, ytest: the data used to fill the `answer`` dictionary


        return answer, Xtrain, ytrain, Xtest, ytest

    """
    B.  Repeat part 1.C, 1.D, and 1.F, for the multiclass problem. 
        Use the Logistic Regression for part F with 300 iterations. 
        Explain how multi-class logistic regression works (inherent, 
        one-vs-one, one-vs-the-rest, etc.).
        Repeat the experiment for ntrain=1000, 5000, 10000, ntest = 200, 1000, 2000.
        Comment on the results. Is the accuracy higher for the training or testing set?
        What is the scores as a function of ntrain.

        Given X, y from mnist, use:
        Xtrain = X[0:ntrain, :]
        ytrain = y[0:ntrain]
        Xtest = X[ntrain:ntrain+test]
        ytest = y[ntrain:ntrain+test]
    """

    def partB(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
        ntrain_list: list[int] = [],
        ntest_list: list[int] = [],
    ) -> dict[int, dict[str, Any]]:
        """ """
        #X, y, Xtest, ytest = u.prepare_data()
        answer = {}
        #ntrain_list = [1000,5000,10000]
        #test_list = [ 200, 1000, 2000]
        for i in range(0,len(ntrain_list)):
            train_val = ntrain_list[i]
            test_val= ntest_list[i]
            X_train = X[:train_val]
            y_train = y[:train_val]
            X_test = Xtest[:test_val]
            y_test = ytest[:test_val]
            #answer
            ##Part C 
            partC ={}
            clf = DecisionTreeClassifier(random_state=self.seed)
            cv = KFold(n_splits=5)
            scores = u.train_simple_classifier_with_cv(Xtrain=X_train,ytrain=y_train,clf=clf,cv=KFold(n_splits=5))
            score_dict={}
            for key,array in scores.items():
                if(key=='fit_time'):
                    score_dict['mean_fit_time'] = array.mean()
                    score_dict['std_fit_time'] = array.std()
                if(key=='test_score'):
                    score_dict['mean_accuracy'] = array.mean()
                    score_dict['std_accuracy'] = array.std()
            partC["clf"] = clf  # the estimator (classifier instance)
            partC["cv"] = cv
            partC["scores_C"] = score_dict

            ##Part D
            partD = {}
            cv_D = ShuffleSplit(n_splits=5,random_state=self.seed)
            scores_dt= u.train_simple_classifier_with_cv(Xtrain=X_train,ytrain=y_train,clf=clf,cv=cv_D)
            partD["mean_fit_time"] = np.mean(scores_dt['fit_time'])
            partD["std_fit_time"] = np.std(scores_dt['fit_time'])
            partD["mean_accuracy"] = np.mean(scores_dt['test_score'])    
            partD["std_accuracy"] = np.std(scores_dt['test_score']) 
            
            partD["clf"] = clf  # the estimator (classifier instance)
            partD["cv"] = cv_D
            partD["scores_D"] = score_dict

            ##Part F
            partF = {}
            clf_F = LogisticRegression(max_iter=300,random_state=self.seed)
            cv_F = ShuffleSplit(n_splits=5,random_state=self.seed)
            scores_trainlr = u.train_simple_classifier_with_cv(Xtrain=X_train, ytrain=y_train, clf=LogisticRegression(max_iter=300,random_state=self.seed), cv=ShuffleSplit(n_splits=5,random_state=self.seed))
            clf_F.fit(X_train,y_train)
            y_train_pred = clf_F.predict(X_train)
            y_test_pred = clf_F.predict(X_test)
            confusion_matrix_train = confusion_matrix(y_train, y_train_pred)
            confusion_matrix_test = confusion_matrix(y_test,y_test_pred)
            partF["scores_train_F"] = clf_F.score(X_train,y_train)
            partF["scores_test_F"] = clf_F.score(X_test,y_test)
            partF["mean_cv_accuracy_F"] = np.mean(scores_trainlr['test_score']) 
            partF["clf"] =  clf_F
            partF["cv"] = cv_F
            partF["conf_mat_train"] = confusion_matrix_train
            partF["conf_mat_test"] = confusion_matrix_test
            answer[ntrain_list[i]] = {}
            answer[ntrain_list[i]]["partC"] = partC
            answer[ntrain_list[i]]["partD"] = partD
            answer[ntrain_list[i]]["partF"] = partF
            answer[ntrain_list[i]]["ntrain"] = ntrain_list[i]
            answer[ntrain_list[i]]["ntest"] = ntest_list[i]
            answer[ntrain_list[i]]["class_count_train"] = list(np.bincount(y_train))
            answer[ntrain_list[i]]["class_count_test"] = list(np.bincount(y_test))
        
        # Enter your code and fill the `answer`` dictionary
        """
        `answer` is a dictionary with the following keys:
           - 1000, 5000, 10000: each key is the number of training samples

           answer[k] is itself a dictionary with the following keys
            - "partC": dictionary returned by partC section 1
            - "partD": dictionary returned by partD section 1
            - "partF": dictionary returned by partF section 1
            - "ntrain": number of training samples
            - "ntest": number of test samples
            - "class_count_train": number of elements in each class in
                               the training set (a list, not a numpy array)
            - "class_count_test": number of elements in each class in
                               the training set (a list, not a numpy array)
        """
        return answer
