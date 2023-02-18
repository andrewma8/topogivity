import numpy as np
import pickle
import matplotlib.pyplot as plt
import material_representation
import metrics
import model



def compute_inner_cross_val_scores_for_fixed_gamma(list_of_matrices_of_inputs_for_inner_cv, list_of_vectors_of_labels_for_inner_cv, gamma):
#does inner cv for a fixed gamma.  I.e., does train and validate for each of the possible splits, and computes the train and validation metrics.  Returns the train and validation metrics in a dict (i.e., does not aggregate them, just returns the value of each metric for every single split)
    
    num_folds_for_inner_cv = len(list_of_matrices_of_inputs_for_inner_cv)
    
    train_accuracy_for_each_split = np.zeros(num_folds_for_inner_cv)
    train_recall_for_each_split = np.zeros(num_folds_for_inner_cv)
    train_precision_for_each_split = np.zeros(num_folds_for_inner_cv)
    train_F1_for_each_split = np.zeros(num_folds_for_inner_cv)
    val_accuracy_for_each_split = np.zeros(num_folds_for_inner_cv)
    val_recall_for_each_split = np.zeros(num_folds_for_inner_cv)
    val_precision_for_each_split = np.zeros(num_folds_for_inner_cv)
    val_F1_for_each_split = np.zeros(num_folds_for_inner_cv)
    
    #iterate through all the possible splits
    for i in range(num_folds_for_inner_cv):
        
        #partition into train and validation for this split
        matrix_of_inputs_for_val = list_of_matrices_of_inputs_for_inner_cv[i]
        vector_of_labels_for_val = list_of_vectors_of_labels_for_inner_cv[i]
        list_of_matrices_of_inputs_for_train =\
                            list_of_matrices_of_inputs_for_inner_cv[:i] + list_of_matrices_of_inputs_for_inner_cv[(i+1):]
        list_of_vectors_of_labels_for_train =\
                            list_of_vectors_of_labels_for_inner_cv[:i] + list_of_vectors_of_labels_for_inner_cv[(i+1):]
        matrix_of_inputs_for_train = np.concatenate(list_of_matrices_of_inputs_for_train)
        vector_of_labels_for_train = np.concatenate(list_of_vectors_of_labels_for_train)
        
        #create the instance of linear SVM
        clf = model.create_instance_of_linear_SVC_using_gamma_and_N(gamma,len(vector_of_labels_for_train))
        
        #fit on the training data
        clf.fit(matrix_of_inputs_for_train, vector_of_labels_for_train)
        
        #classify both the train inputs and the val inputs
        predicted_labels_on_train_inputs = clf.predict(matrix_of_inputs_for_train)
        predicted_labels_on_val_inputs = clf.predict(matrix_of_inputs_for_val)
        
        # acc, recall, precision, F1 metrics for both train and val
        train_accuracy, train_recall, train_precision, train_F1 = metrics.compute_all_metrics(vector_of_labels_for_train, 
                                                                                              predicted_labels_on_train_inputs)
        val_accuracy, val_recall, val_precision, val_F1 = metrics.compute_all_metrics(vector_of_labels_for_val,
                                                                                             predicted_labels_on_val_inputs)
        
        train_accuracy_for_each_split[i] = train_accuracy
        train_recall_for_each_split[i] = train_recall
        train_precision_for_each_split[i] = train_precision
        train_F1_for_each_split[i] = train_F1
        val_accuracy_for_each_split[i] = val_accuracy
        val_recall_for_each_split[i] = val_recall
        val_precision_for_each_split[i] = val_precision
        val_F1_for_each_split[i] = val_F1
    
    #put the results into a dictionary
    inner_cv_results_for_fixed_gamma = {}
    inner_cv_results_for_fixed_gamma["gamma"] = gamma
    inner_cv_results_for_fixed_gamma["train_accuracy"] = train_accuracy_for_each_split
    inner_cv_results_for_fixed_gamma["train_recall"] = train_recall_for_each_split
    inner_cv_results_for_fixed_gamma["train_precision"] = train_precision_for_each_split
    inner_cv_results_for_fixed_gamma["train_F1"] = train_F1_for_each_split
    inner_cv_results_for_fixed_gamma["val_accuracy"] = val_accuracy_for_each_split
    inner_cv_results_for_fixed_gamma["val_recall"] = val_recall_for_each_split
    inner_cv_results_for_fixed_gamma["val_precision"] = val_precision_for_each_split
    inner_cv_results_for_fixed_gamma["val_F1"] = val_F1_for_each_split
    
    return inner_cv_results_for_fixed_gamma
    


def perform_inner_cross_val_for_multiple_values_of_gamma(list_of_matrices_of_inputs_for_inner_cv, list_of_vectors_of_labels_for_inner_cv, values_of_gamma):
#takes as input a list where the i-th entry is a np matrix that represents the materials in the (i+1)-th fold of the inner cv, and another list where the i-th entry is a np vector that represents the corresponding labels for the (i+1)-th fold of he inner cv.  For each hyperparameter gamma in values_of_gamma, performs cv. Determines the best hyperparameter gamma as the one that gives the greatest mean F1 score.  Returns a dict that contains the best hyperparameter as well as the corresponding stats
    
    num_of_hyperparam_settings = len(values_of_gamma)
    
    #we make a list of dicts, where the dict for each entry corresponds to a single gamma
    #also we make a vector which is just the inner cv mean val F1 score for each gamma
    inner_cv_results_for_each_gamma = []
    inner_cv_mean_val_F1_for_each_gamma = np.zeros(num_of_hyperparam_settings)
    
    for i in range(num_of_hyperparam_settings):
        
        inner_cv_results_for_this_gamma = compute_inner_cross_val_scores_for_fixed_gamma(
                            list_of_matrices_of_inputs_for_inner_cv, list_of_vectors_of_labels_for_inner_cv, values_of_gamma[i])
        inner_cv_results_for_each_gamma.append(inner_cv_results_for_this_gamma)
        
        inner_cv_mean_val_F1_for_this_gamma = np.mean(inner_cv_results_for_this_gamma["val_F1"])
        inner_cv_mean_val_F1_for_each_gamma[i] = inner_cv_mean_val_F1_for_this_gamma
        
        print("\nHyperparameter setting number", i)
        print("gamma =", values_of_gamma[i], ";  the inner CV mean validation F1 score was:", 
                                                                  inner_cv_mean_val_F1_for_this_gamma)
        
    # determine the best hyperparameter based on F1 score
    index_with_best_F1 = np.argmax(inner_cv_mean_val_F1_for_each_gamma)
    
    # the dict from the list of dicts that had the best F1, i.e., the best hypeparameter and corresponding train and val results
    inner_cv_results_for_best_gamma = inner_cv_results_for_each_gamma[index_with_best_F1]
    
    return inner_cv_results_for_best_gamma
    
    
    
def main():
    
    #load data
    with open('processed_data/set_A_data.pkl', 'rb') as f:
        list_of_lists_of_material_dicts_for_nested_cv = pickle.load(f)
    with open('processed_data/featurization_data.pkl', 'rb') as f:
        featurization_data = pickle.load(f)
    list_of_atomic_numbers_for_featurization = featurization_data["list_of_atomic_numbers_for_featurization"]
    atomic_number_to_drop = featurization_data["atomic_number_to_drop"]
    
    #convert list_of_lists_of_material_dicts_for_nested_cv into corresponding list of matrices of inputs as well as a
    # corresponding list of vectors of labels
    num_folds_for_nested_cv = len(list_of_lists_of_material_dicts_for_nested_cv)
    list_of_matrices_of_inputs_for_nested_cv = []
    list_of_vectors_of_labels_for_nested_cv = []
    for i in range(num_folds_for_nested_cv):
        list_of_material_dicts_for_this_fold = list_of_lists_of_material_dicts_for_nested_cv[i]
        matrix_of_inputs_for_this_fold, vector_of_labels_for_this_fold = \
                    material_representation.build_matrix_of_inputs_and_vector_of_labels(
                        list_of_material_dicts_for_this_fold, list_of_atomic_numbers_for_featurization, atomic_number_to_drop)
        list_of_matrices_of_inputs_for_nested_cv.append(matrix_of_inputs_for_this_fold)
        list_of_vectors_of_labels_for_nested_cv.append(vector_of_labels_for_this_fold)
    
    #hyperparameters to try
    values_of_gamma = np.geomspace(1e-6, 1e-4, 75)
    
    # decision function bin boundary points
    decision_function_bin_bps = np.arange(-2.0,2.5,0.5)
    print("decision_function_bin_bps:", decision_function_bin_bps)
    
    #threshold for which we will do an extra computation of the precision
    threshold = 1.0
    print("threshold:", threshold)
    
    #each entry of below list corresponds to one of fold of nested cv (i.e., each entry contains exactly one test set of outer
    #loop) in each entry, all of the results are only for the best hyperparameter for that fold (which was selected via inner cv)
    comprehensive_results_for_nested_cv = []
    
    #outer loop of nested CV
    for i in range(num_folds_for_nested_cv):
        
        print("\n\n\nOUTER FOLD INDEX", i)
        
        #for this fold, partition into materials for test and materials for inner cv
        matrix_of_inputs_for_test = list_of_matrices_of_inputs_for_nested_cv[i]
        vector_of_labels_for_test = list_of_vectors_of_labels_for_nested_cv[i]
        list_of_matrices_of_inputs_for_inner_cv = \
                            list_of_matrices_of_inputs_for_nested_cv[:i] + list_of_matrices_of_inputs_for_nested_cv[(i+1):]
        list_of_vectors_of_labels_for_inner_cv =\
                            list_of_vectors_of_labels_for_nested_cv[:i] + list_of_vectors_of_labels_for_nested_cv[(i+1):]
        
        #do the inner CV, which will determine a best hyperparameter and its corresponding train and validation results, which
        # will all be returned in the results_for_best_gamma dictionary
        results_for_best_gamma = perform_inner_cross_val_for_multiple_values_of_gamma(
                            list_of_matrices_of_inputs_for_inner_cv, list_of_vectors_of_labels_for_inner_cv, values_of_gamma)
        
        #using the best hyperparameter determined above, retrain on all materials that went into the inner CV, and then test on
        # the test samples for this fold
        matrix_of_inputs_for_retrain = np.concatenate(list_of_matrices_of_inputs_for_inner_cv)
        vector_of_labels_for_retrain = np.concatenate(list_of_vectors_of_labels_for_inner_cv)
        clf = model.create_instance_of_linear_SVC_using_gamma_and_N(
                                                    results_for_best_gamma["gamma"],len(vector_of_labels_for_retrain))
        clf.fit(matrix_of_inputs_for_retrain,vector_of_labels_for_retrain)
        predicted_labels_on_retrain_inputs = clf.predict(matrix_of_inputs_for_retrain)
        predicted_labels_on_test_inputs = clf.predict(matrix_of_inputs_for_test)
        
        #basic metrics for retrain and test
        retrain_accuracy, retrain_recall, retrain_precision, retrain_F1 = metrics.compute_all_metrics(
                                                                vector_of_labels_for_retrain, predicted_labels_on_retrain_inputs)
        test_accuracy, test_recall, test_precision, test_F1 = metrics.compute_all_metrics(
                                                                vector_of_labels_for_test, predicted_labels_on_test_inputs)
        
        #compute some additional metrics for the test set.
        test_frac_classified_as_topological = np.sum(predicted_labels_on_test_inputs) / len(predicted_labels_on_test_inputs)
        decision_function_values_on_test_inputs = clf.decision_function(matrix_of_inputs_for_test)
        test_precision_using_threshold = metrics.compute_precision_using_threshold(
                                                vector_of_labels_for_test, decision_function_values_on_test_inputs, threshold)
        test_frac_above_threshold = metrics.compute_frac_above_threshold(decision_function_values_on_test_inputs, threshold)
        test_frac_ground_truth_topological_for_each_decision_function_bin =\
                        metrics.compute_frac_ground_truth_topological_for_each_decision_function_bin(
                                vector_of_labels_for_test, decision_function_values_on_test_inputs, decision_function_bin_bps)
        
        # add the retrain and test results to the dict that already contains the train and val results 
        #(which also correspond to this best gamma)
        results_for_best_gamma["retrain_accuracy"] = retrain_accuracy
        results_for_best_gamma["retrain_recall"] = retrain_recall
        results_for_best_gamma["retrain_precision"] = retrain_precision
        results_for_best_gamma["retrain_F1"] = retrain_F1
        results_for_best_gamma["test_accuracy"] = test_accuracy
        results_for_best_gamma["test_recall"] = test_recall
        results_for_best_gamma["test_precision"] = test_precision
        results_for_best_gamma["test_F1"] = test_F1
        results_for_best_gamma["test_frac_classified_as_topological"] = test_frac_classified_as_topological
        results_for_best_gamma["test_precision_using_threshold"] = test_precision_using_threshold
        results_for_best_gamma["test_frac_above_threshold"] = test_frac_above_threshold
        results_for_best_gamma["test_frac_ground_truth_topological_for_each_decision_function_bin"] =\
                                            test_frac_ground_truth_topological_for_each_decision_function_bin
        
        # compute balanced accuracy overall 
        test_balanced_acc = metrics.compute_balanced_accuracy(vector_of_labels_for_test, predicted_labels_on_test_inputs)
        
        #compute balanced accuracy for individual subsets
        list_of_material_dicts_for_test = list_of_lists_of_material_dicts_for_nested_cv[i]
        test_one_elt_subset_balanced_acc = metrics.balanced_accuracy_for_materials_with_given_number_of_distinct_elements(
                    list_of_material_dicts_for_test, clf, list_of_atomic_numbers_for_featurization, atomic_number_to_drop, 1)
        test_two_elt_subset_balanced_acc = metrics.balanced_accuracy_for_materials_with_given_number_of_distinct_elements(
                    list_of_material_dicts_for_test, clf, list_of_atomic_numbers_for_featurization, atomic_number_to_drop, 2)
        test_three_elt_subset_balanced_acc = metrics.balanced_accuracy_for_materials_with_given_number_of_distinct_elements(
                    list_of_material_dicts_for_test, clf, list_of_atomic_numbers_for_featurization, atomic_number_to_drop, 3)
        test_four_elt_subset_balanced_acc = metrics.balanced_accuracy_for_materials_with_given_number_of_distinct_elements(
                    list_of_material_dicts_for_test, clf, list_of_atomic_numbers_for_featurization, atomic_number_to_drop, 4)
        
        #compute accuracy for individual subsets
        test_one_elt_subset_acc = metrics.accuracy_for_materials_with_given_number_of_distinct_elements(
                    list_of_material_dicts_for_test, clf, list_of_atomic_numbers_for_featurization, atomic_number_to_drop, 1)
        test_two_elt_subset_acc = metrics.accuracy_for_materials_with_given_number_of_distinct_elements(
                    list_of_material_dicts_for_test, clf, list_of_atomic_numbers_for_featurization, atomic_number_to_drop, 2)
        test_three_elt_subset_acc = metrics.accuracy_for_materials_with_given_number_of_distinct_elements(
                    list_of_material_dicts_for_test, clf, list_of_atomic_numbers_for_featurization, atomic_number_to_drop, 3)
        test_four_elt_subset_acc = metrics.accuracy_for_materials_with_given_number_of_distinct_elements(
                    list_of_material_dicts_for_test, clf, list_of_atomic_numbers_for_featurization, atomic_number_to_drop, 4)
        
        #add the test balanced accuracy results (both overall and subsets) to the dict that already contains train results,
        #validation results, retrain results, and the rest of the test results
        results_for_best_gamma["test_balanced_acc"] = test_balanced_acc
        results_for_best_gamma["test_one_elt_subset_balanced_acc"] = test_one_elt_subset_balanced_acc
        results_for_best_gamma["test_two_elt_subset_balanced_acc"] = test_two_elt_subset_balanced_acc
        results_for_best_gamma["test_three_elt_subset_balanced_acc"] = test_three_elt_subset_balanced_acc
        results_for_best_gamma["test_four_elt_subset_balanced_acc"] = test_four_elt_subset_balanced_acc
        
        #add the accuracy on subset results to that dict as well
        results_for_best_gamma["test_one_elt_subset_acc"] = test_one_elt_subset_acc
        results_for_best_gamma["test_two_elt_subset_acc"] = test_two_elt_subset_acc
        results_for_best_gamma["test_three_elt_subset_acc"] = test_three_elt_subset_acc
        results_for_best_gamma["test_four_elt_subset_acc"] = test_four_elt_subset_acc
        
        #put all of these results --which all correspond to best hyperparmaeter setting -- (train, validation, retrain, test)
        # into comprehensive_results_for_nested_cv
        comprehensive_results_for_nested_cv.append(results_for_best_gamma)
        
        #print some stuff
        print("results_for_best_gamma:")
        print(results_for_best_gamma)
    
    # compute means and stdevs, and print results
    print("\n\n\nAGGREGATED RESULTS FOR THE ENTIRE NESTED CROSS-VALIDATION:")
    for key in ["retrain_accuracy", "retrain_recall", "retrain_precision", "retrain_F1",
                    "test_accuracy", "test_recall", "test_precision", "test_F1",
                        "test_frac_classified_as_topological", "test_precision_using_threshold", "test_frac_above_threshold",
                           "test_balanced_acc", "test_one_elt_subset_balanced_acc", "test_two_elt_subset_balanced_acc",
                               "test_three_elt_subset_balanced_acc", "test_four_elt_subset_balanced_acc",
                                   "test_one_elt_subset_acc", "test_two_elt_subset_acc", "test_three_elt_subset_acc",
                                       "test_four_elt_subset_acc"]:
        list_form_result_for_each_split = [] # only contains the values that are defined (i.e., if np.NaN occurs, then exclude)
        num_folds_for_which_this_metric_was_undefined = 0
        for i in range(num_folds_for_nested_cv):
            result_for_this_split = comprehensive_results_for_nested_cv[i][key]
            if result_for_this_split is np.NaN:
                num_folds_for_which_this_metric_was_undefined += 1
            else:
                list_form_result_for_each_split.append(result_for_this_split)
        result_for_each_split = np.array(list_form_result_for_each_split)
        mean_result = np.mean(result_for_each_split)
        std_result = np.std(result_for_each_split,ddof=1)
        print("\nmean_"+key+":", mean_result)
        print("std_"+key+":", std_result)
        if num_folds_for_which_this_metric_was_undefined != 0:
            print("the value of this metric was undefined for", num_folds_for_which_this_metric_was_undefined,
                     "folds, and so the mean and standard deviation were calculated using the remaining",
                         num_folds_for_nested_cv - num_folds_for_which_this_metric_was_undefined, "folds")
    num_decision_function_bins = len(decision_function_bin_bps) + 1
    test_frac_ground_truth_topological_for_each_decision_function_bin_for_each_split =\
                                                            np.zeros((num_folds_for_nested_cv,num_decision_function_bins))
    for i in range(num_folds_for_nested_cv):
        test_frac_ground_truth_topological_for_each_decision_function_bin_for_each_split[i] =\
                comprehensive_results_for_nested_cv[i]["test_frac_ground_truth_topological_for_each_decision_function_bin"]
    mean_test_frac_ground_truth_topological_for_each_decision_function_bin =\
                np.mean(test_frac_ground_truth_topological_for_each_decision_function_bin_for_each_split,0)
    std_test_frac_ground_truth_topological_for_each_decision_function_bin =\
                np.std(test_frac_ground_truth_topological_for_each_decision_function_bin_for_each_split,0,ddof=1)
    print("\nmean_test_frac_ground_truth_topological_for_each_decision_function_bin:", 
                                 mean_test_frac_ground_truth_topological_for_each_decision_function_bin)
    print("std_test_frac_ground_truth_topological_for_each_decision_function_bin:",
                                 std_test_frac_ground_truth_topological_for_each_decision_function_bin)
    
    #selected value of hyperparameter for each split
    selected_gamma_for_each_split = np.zeros(num_folds_for_nested_cv)
    for i in range(num_folds_for_nested_cv):
        selected_gamma_for_each_split[i] = comprehensive_results_for_nested_cv[i]["gamma"]
    print("selected gamma for each split, in order of split:", selected_gamma_for_each_split)
    print("selected gamma for each split, sorted in ascending order:", np.sort(selected_gamma_for_each_split))
    median_selected_gamma = np.median(selected_gamma_for_each_split)
    print("median selected gamma:", median_selected_gamma)
    
    #save median gamma
    filename_with_path_for_median_selected_gamma = "results/median_selected_gamma.pkl"
    output_for_median_selected_gamma = open(filename_with_path_for_median_selected_gamma, "wb")
    pickle.dump(median_selected_gamma, output_for_median_selected_gamma, pickle.HIGHEST_PROTOCOL)
    output_for_median_selected_gamma.close()
    
    #plotting the bar graph that shows the test topological fraction for each g(M) bin
    ind = np.arange(0,len(mean_test_frac_ground_truth_topological_for_each_decision_function_bin))
    tuple_of_decision_function_bin_strings = ('(-\u221E,-2)','[-2,-1.5)','[-1.5,-1)','[-1,-0.5)',
                                                  '[-0.5,0)','[0,0.5)','[0.5,1)','[1,1.5)','[1.5,2)','[2,\u221E)')
    vertical_tick_marks = np.arange(0,1.1,0.1)
    plt.figure(figsize=(15,7.5))
    plt.bar(ind, mean_test_frac_ground_truth_topological_for_each_decision_function_bin,
                        yerr=std_test_frac_ground_truth_topological_for_each_decision_function_bin, capsize=8)
    plt.ylim([0,1])
    plt.ylabel("test topological fraction $\sigma(B)$", fontsize=23)
    plt.xlabel('$g(M)$ bin $B$', fontsize=23)
    plt.xticks(ind,tuple_of_decision_function_bin_strings,fontsize=17)
    plt.yticks(vertical_tick_marks,fontsize=17)
    plt.grid(True,axis='y')
    plt.show()

    
    
if __name__ == '__main__':
    main()