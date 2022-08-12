# Abstract
Learning project coding my own logistic regression algorithm in Python. In the file "WritingLogisticRegressionAlgorithm.ipynb" each function used to implement logistic regression is coded and tested step-by-step. I learnt the theory of logistic regression algorithms in "Machine Learning" Coursera specialisation. Using the theoretical description, I coded a vectorised version of logistic regression using modules numpy with compatibility with dataframes from module pandas. Added hyperparameters include polynomial features and regularisation. After placing the algorithm in a custom module "mylogregression.py", in the file "TestingLogisticRegressionAlgorithm.ipynb", I tested my logistic regression algorithm by performing initial diagnosis of chronic kidney disease (CKD). I demonstrate the picking and choosing of hyperparameters on a cross-validation dataset after initial training. As initial diagnosis is usually a precursor to a more rigorous diagnosis, the model is trained to detect when someone is more than 10% likely to have CKD - on the cross-validation dataset this yielded perfect detection of true positives with a small percentage of false positives. The final algorithm achieved 100% detection of positive cases of CKD and correctly identified 76% of negative CKD cases. An acceptable result for initial diagnosis. 
