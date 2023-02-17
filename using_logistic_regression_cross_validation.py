import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
import material_representation
import metrics


    
def main():
    
    #load data
    with open('processed_data/set_A_data.pkl', 'rb') as f:
        list_of_lists_of_material_dicts_for_cv = pickle.load(f)
    with open('processed_data/featurization_data.pkl', 'rb') as f:
        featurization_data = pickle.load(f)
    list_of_atomic_numbers_for_featurization = featurization_data["list_of_atomic_numbers_for_featurization"]
    atomic_number_to_drop = featurization_data["atomic_number_to_drop"]
    
    #convert list_of_lists_of_material_dicts_for_cv into corresponding list of matrices of inputs as well as a
    # corresponding list of vectors of labels
    num_folds_for_cv = len(list_of_lists_of_material_dicts_for_cv)
    list_of_matrices_of_inputs_for_cv = []
    list_of_vectors_of_labels_for_cv = []
    for i in range(num_folds_for_cv):
        list_of_material_dicts_for_this_fold = list_of_lists_of_material_dicts_for_cv[i]
        matrix_of_inputs_for_this_fold, vector_of_labels_for_this_fold = \
                material_representation.build_matrix_of_inputs_and_vector_of_labels(
                        list_of_material_dicts_for_this_fold, list_of_atomic_numbers_for_featurization, atomic_number_to_drop)
        list_of_matrices_of_inputs_for_cv.append(matrix_of_inputs_for_this_fold)
        list_of_vectors_of_labels_for_cv.append(vector_of_labels_for_this_fold)
    
    train_accuracy_for_each_split = np.zeros(num_folds_for_cv)
    train_recall_for_each_split = np.zeros(num_folds_for_cv)
    train_precision_for_each_split = np.zeros(num_folds_for_cv)
    train_F1_for_each_split = np.zeros(num_folds_for_cv)
    test_accuracy_for_each_split = np.zeros(num_folds_for_cv)
    test_recall_for_each_split = np.zeros(num_folds_for_cv)
    test_precision_for_each_split = np.zeros(num_folds_for_cv)
    test_F1_for_each_split = np.zeros(num_folds_for_cv)
    
    #iterate through all the possible splits
    for i in range(num_folds_for_cv):
        
        #partition into train and test for this split
        matrix_of_inputs_for_test = list_of_matrices_of_inputs_for_cv[i]
        vector_of_labels_for_test = list_of_vectors_of_labels_for_cv[i]
        list_of_matrices_of_inputs_for_train =\
                            list_of_matrices_of_inputs_for_cv[:i] + list_of_matrices_of_inputs_for_cv[(i+1):]
        list_of_vectors_of_labels_for_train =\
                            list_of_vectors_of_labels_for_cv[:i] + list_of_vectors_of_labels_for_cv[(i+1):]
        matrix_of_inputs_for_train = np.concatenate(list_of_matrices_of_inputs_for_train)
        vector_of_labels_for_train = np.concatenate(list_of_vectors_of_labels_for_train)
        
        #create the instance of logistic regression
        clf = LogisticRegression(penalty='none',tol=1e-5,max_iter=200)
        
        #fit model on training set
        clf.fit(matrix_of_inputs_for_train,vector_of_labels_for_train)
        
        #classify both the train inputs and the test inputs
        predicted_labels_on_train_inputs = clf.predict(matrix_of_inputs_for_train)
        predicted_labels_on_test_inputs = clf.predict(matrix_of_inputs_for_test)
        
        #compute train metrics and test metrics
        train_accuracy, train_recall, train_precision, train_F1 = metrics.compute_all_metrics(
                                                            vector_of_labels_for_train, predicted_labels_on_train_inputs)
        test_accuracy, test_recall, test_precision, test_F1 = metrics.compute_all_metrics(
                                                            vector_of_labels_for_test, predicted_labels_on_test_inputs)
        
        #put the metrics into the vectors
        train_accuracy_for_each_split[i] = train_accuracy
        train_recall_for_each_split[i] = train_recall
        train_precision_for_each_split[i] = train_precision
        train_F1_for_each_split[i] = train_F1
        test_accuracy_for_each_split[i] = test_accuracy
        test_recall_for_each_split[i] = test_recall
        test_precision_for_each_split[i] = test_precision
        test_F1_for_each_split[i] = test_F1
    
    #compute mean and stdev for each metric; and print them
    
    mean_train_accuracy = np.mean(train_accuracy_for_each_split)
    std_train_accuracy = np.std(train_accuracy_for_each_split,ddof=1)
    print("mean train accuracy:", mean_train_accuracy)
    print("std train accuracy:", std_train_accuracy)
    
    mean_train_recall = np.mean(train_recall_for_each_split)
    std_train_recall = np.std(train_recall_for_each_split,ddof=1)
    print("mean train recall:", mean_train_recall)
    print("std train recall:", std_train_recall)
    
    mean_train_precision = np.mean(train_precision_for_each_split)
    std_train_precision = np.std(train_precision_for_each_split,ddof=1)
    print("mean train precision:", mean_train_precision)
    print("std train precision:", std_train_precision)
    
    mean_train_F1 = np.mean(train_F1_for_each_split)
    std_train_F1 = np.std(train_F1_for_each_split,ddof=1)
    print("mean train F1:", mean_train_F1)
    print("std train F1:", std_train_F1)
    
    mean_test_accuracy = np.mean(test_accuracy_for_each_split)
    std_test_accuracy = np.std(test_accuracy_for_each_split,ddof=1)
    print("\nmean test accuracy:", mean_test_accuracy)
    print("std test accuracy:", std_test_accuracy)
    
    mean_test_recall = np.mean(test_recall_for_each_split)
    std_test_recall = np.std(test_recall_for_each_split,ddof=1)
    print("mean test recall:", mean_test_recall)
    print("std test recall:", std_test_recall)
    
    mean_test_precision = np.mean(test_precision_for_each_split)
    std_test_precision = np.std(test_precision_for_each_split,ddof=1)
    print("mean test precision:", mean_test_precision)
    print("std test precision:", std_test_precision)
    
    mean_test_F1 = np.mean(test_F1_for_each_split)
    std_test_F1 = np.std(test_F1_for_each_split,ddof=1)
    print("mean test F1:", mean_test_F1)
    print("std test F1:", std_test_F1)
    
    
if __name__ == '__main__':
    main()