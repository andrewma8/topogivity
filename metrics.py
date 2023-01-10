import numpy as np
import material_representation
import chemistry

    
def compute_confusion_matrix(ground_truth_labels, predicted_labels):
#computes the number of true positive, number of false positives, number of false negatives, and number of true negatives; where positive classification means topological
    
    if len(ground_truth_labels) != len(predicted_labels):
        raise
    num_samples = len(ground_truth_labels)
    
    num_true_positives = 0
    num_false_positives = 0
    num_false_negatives = 0
    num_true_negatives = 0
    
    for i in range(num_samples):
        if ground_truth_labels[i] == 1:
            if predicted_labels[i] == 1:
                num_true_positives  += 1
            elif predicted_labels[i] == 0:
                num_false_negatives += 1
            else:
                raise
        elif ground_truth_labels[i] == 0:
            if predicted_labels[i] == 1:
                num_false_positives += 1
            elif predicted_labels[i] == 0:
                num_true_negatives +=  1
            else:
                raise
        else:
            raise
    
    return num_true_positives, num_false_positives, num_false_negatives, num_true_negatives



def compute_all_metrics(ground_truth_labels, predicted_labels):
#computes the accuracy, recall, precision, and F1 score; where positive classification means topological
    
    if len(ground_truth_labels) != len(predicted_labels):
        raise
    num_samples = len(ground_truth_labels)
    
    num_true_positives, num_false_positives, num_false_negatives, num_true_negatives = compute_confusion_matrix(
                                                                                ground_truth_labels, predicted_labels)
    
    accuracy = (num_true_positives + num_true_negatives) / num_samples
    recall = num_true_positives / (num_true_positives + num_false_negatives)
    precision = num_true_positives / (num_true_positives + num_false_positives)
    F1 = 2 * precision * recall / (precision + recall)
    

    return accuracy, recall, precision, F1



def compute_precision_using_threshold(ground_truth_labels, decision_function_values, threshold):
#takes as input a vector of ground truth labels, a vector of decision function values, and some threshold for decision function value.  Returns what the precision is when using being \geq this threshold as the criterion for a positive (topological) classification.  

    # first, converts the decision function values to 1 or 0 based on this threshold, and then uses compute all metrics
    predicted_labels_using_threshold = (decision_function_values >= threshold).astype(np.int64)
    _, _, precision_using_threshold, _ = compute_all_metrics(ground_truth_labels, 
                                                                 predicted_labels_using_threshold)
    return precision_using_threshold

    
    
def compute_frac_above_threshold(decision_function_values, threshold):
# computes the fraction of decision function values that are above some threshold
    
    num_of_samples_above_threshold = np.sum(decision_function_values >= threshold)
    frac_above_threshold = num_of_samples_above_threshold / len(decision_function_values)
    
    return frac_above_threshold



def compute_frac_ground_truth_topological_for_each_decision_function_bin(ground_truth_labels, decision_function_values, decision_function_bin_bps):
#for each decision function bin, consider the samples for which the decision function value is in that decision function bin.  This method will compute the fraction of those samples which have a ground truth label of topological.  In other words, this allows us to estimate the probability that a given material is topological by looking at what decision function bin an ML model assigns it to.  decision_function_bin_bps represent the boundaries between different decision function bins, where the first decision function bin goes from -infinity to the first bouondary point, and the last decision function bin goes from the last boundary point to +infinity.
    
    if len(ground_truth_labels) != len(decision_function_values):
        raise
    
    num_samples = len(decision_function_values)

    num_decision_function_bins = len(decision_function_bin_bps)+1
    frac_ground_truth_topological_for_each_decision_function_bin = np.zeros(num_decision_function_bins)
    
    for i in range(num_decision_function_bins):
        
        if i == 0:
            is_it_in_the_bin = (decision_function_values < decision_function_bin_bps[i])
        elif i == (num_decision_function_bins-1):
            is_it_in_the_bin = (decision_function_values >= decision_function_bin_bps[i-1])
        else:
            is_it_in_the_bin = np.logical_and(
                (decision_function_values >= decision_function_bin_bps[i-1]),
                (decision_function_values < decision_function_bin_bps[i])
                )
        
        num_samples_in_the_bin = 0
        num_samples_in_the_bin_that_have_ground_truth_pos = 0
        
        for j in range(num_samples):
            
            if is_it_in_the_bin[j]:
                num_samples_in_the_bin += 1
                if ground_truth_labels[j] == 1:
                    num_samples_in_the_bin_that_have_ground_truth_pos += 1
        
        frac_ground_truth_topological_for_each_decision_function_bin[i] = \
                     num_samples_in_the_bin_that_have_ground_truth_pos / num_samples_in_the_bin
    
    return frac_ground_truth_topological_for_each_decision_function_bin



def compute_balanced_accuracy(ground_truth_labels, predicted_labels):
    
    if len(ground_truth_labels) != len(predicted_labels):
        raise
    
    num_true_positives, num_false_positives, num_false_negatives, num_true_negatives = compute_confusion_matrix(
                                                                                ground_truth_labels, predicted_labels)
    balanced_accuracy = 0.5 * num_true_positives / (num_true_positives + num_false_negatives) +\
                            0.5 * num_true_negatives / (num_true_negatives + num_false_positives)
    
    return balanced_accuracy



def balanced_accuracy_for_materials_with_given_number_of_distinct_elements(list_of_material_dicts, clf, list_of_atomic_numbers_for_featurization, atomic_number_to_drop, num_distinct_elements):
#a method which given a list of material dicts and the fitted model and the featurization info computes the balanced accuracy for subset with given number of distinct elements.

    list_of_material_dicts_with_given_number_of_distinct_elements =\
            chemistry.material_dicts_with_given_number_of_distinct_elements(list_of_material_dicts, num_distinct_elements)
    
    matrix_of_inputs_with_given_number_of_distinct_elements, vector_of_labels_with_given_number_of_distinct_elements =\
            material_representation.build_matrix_of_inputs_and_vector_of_labels(
                list_of_material_dicts_with_given_number_of_distinct_elements, list_of_atomic_numbers_for_featurization,
                                                                                                        atomic_number_to_drop)
    
    predicted_labels_on_inputs_with_given_number_of_distinct_elements = clf.predict(
                                                            matrix_of_inputs_with_given_number_of_distinct_elements)
    
    balanced_accuracy_for_subset_with_given_number_of_distinct_elements =\
                                    compute_balanced_accuracy(vector_of_labels_with_given_number_of_distinct_elements,
                                                          predicted_labels_on_inputs_with_given_number_of_distinct_elements)
    
    return balanced_accuracy_for_subset_with_given_number_of_distinct_elements



def accuracy_for_materials_with_given_number_of_distinct_elements(list_of_material_dicts, clf, list_of_atomic_numbers_for_featurization, atomic_number_to_drop, num_distinct_elements):
#a method which given a list of material dicts and the fitted model and the featurization info computes the accuracy for subset with given number of distinct elements.
#TODO: IN FUTURE VERSION OF CODE, MERGE THIS WITH THE ABOVE METHOD

    list_of_material_dicts_with_given_number_of_distinct_elements =\
            chemistry.material_dicts_with_given_number_of_distinct_elements(list_of_material_dicts, num_distinct_elements)
    
    matrix_of_inputs_with_given_number_of_distinct_elements, vector_of_labels_with_given_number_of_distinct_elements =\
            material_representation.build_matrix_of_inputs_and_vector_of_labels(
                list_of_material_dicts_with_given_number_of_distinct_elements, list_of_atomic_numbers_for_featurization,
                                                                                                        atomic_number_to_drop)
    
    predicted_labels_on_inputs_with_given_number_of_distinct_elements = clf.predict(
                                                            matrix_of_inputs_with_given_number_of_distinct_elements)
    
    accuracy_for_subset_with_given_number_of_distinct_elements, _, _, _ =\
                    compute_all_metrics(vector_of_labels_with_given_number_of_distinct_elements,
                                                predicted_labels_on_inputs_with_given_number_of_distinct_elements)
    
    return accuracy_for_subset_with_given_number_of_distinct_elements