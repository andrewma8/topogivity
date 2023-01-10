from sklearn.svm import SVC

def create_instance_of_linear_SVC_using_gamma_and_N(gamma,num_train_samples):
    
    # relates our hyperparameter gamma and our number of train samples (i.e., N in ERM) to the inverse
    # regularization parameter C used in sklearn
    inverse_reg_param_C = 1 / (2* gamma * num_train_samples)
    
    clf = SVC(C=inverse_reg_param_C,kernel="linear",tol=0.0005)
    
    return clf