import numpy as np
import pickle
import matplotlib.pyplot as plt
from pymatgen.util.plotting import periodic_table_heatmap

import chemistry
import material_representation
import model
import metrics

    
def main():
    
    # load data
    with open('processed_data/set_A_data.pkl', 'rb') as f:
        list_of_lists_of_material_dicts_for_nested_cv = pickle.load(f)
    with open('processed_data/set_B_data.pkl', 'rb') as f:
        set_B_data = pickle.load(f)
    with open('processed_data/featurization_data.pkl', 'rb') as f:
        featurization_data = pickle.load(f)
    list_of_set_B_case1_material_dicts = set_B_data["case1_material_dicts"]
    list_of_set_B_case3_material_dicts = set_B_data["case3_material_dicts"]
    list_of_atomic_numbers_for_featurization = featurization_data["list_of_atomic_numbers_for_featurization"]
    atomic_number_to_drop = featurization_data["atomic_number_to_drop"]
    
    #convert set A data into matrix and vector forms
    list_of_material_dicts_for_reretrain = []
    for i in range(len(list_of_lists_of_material_dicts_for_nested_cv)):
        list_of_material_dicts_for_reretrain.extend(list_of_lists_of_material_dicts_for_nested_cv[i])
    matrix_of_inputs_for_reretrain, vector_of_labels_for_reretrain = \
                material_representation.build_matrix_of_inputs_and_vector_of_labels(
                        list_of_material_dicts_for_reretrain, list_of_atomic_numbers_for_featurization, atomic_number_to_drop)
    
    #convert set B data into matrix and vector forms
    matrix_of_inputs_for_sanity_check, _ = \
                material_representation.build_matrix_of_inputs_and_vector_of_labels(
                        list_of_set_B_case3_material_dicts, list_of_atomic_numbers_for_featurization, atomic_number_to_drop)
    matrix_of_inputs_for_discovery, _ = \
                material_representation.build_matrix_of_inputs_and_vector_of_labels(
                        list_of_set_B_case1_material_dicts, list_of_atomic_numbers_for_featurization, atomic_number_to_drop)
    
    #hyperparameter that we reretrain with
    with open('results/median_selected_gamma.pkl', 'rb') as f:
        selected_gamma = pickle.load(f)
    print("\nselected_gamma:",selected_gamma)
    
    #reretrain on all of set A
    clf = model.create_instance_of_linear_SVC_using_gamma_and_N(selected_gamma,len(vector_of_labels_for_reretrain))
    clf.fit(matrix_of_inputs_for_reretrain,vector_of_labels_for_reretrain)
    predicted_labels_on_reretrain_inputs = clf.predict(matrix_of_inputs_for_reretrain)
    reretrain_accuracy, reretrain_recall, reretrain_precision, reretrain_F1 = \
                            metrics.compute_all_metrics(vector_of_labels_for_reretrain, predicted_labels_on_reretrain_inputs)
    print("\nreretrain_accuracy:",reretrain_accuracy)
    print("reretrain_recall:",reretrain_recall)
    print("reretrain_precision:",reretrain_precision)
    print("reretrain_F1:",reretrain_F1)
    
    #print learned weights
    print("\nclf.coef_:",clf.coef_)
    print("clf.intercept_:",clf.intercept_)
    
    #visualization
    decision_function_for_each_elt = {}
    for atomic_number in list_of_atomic_numbers_for_featurization:
        str_elt = chemistry.get_str_elt(atomic_number)
        vec_elt = material_representation.convert_element_to_vector_rep(
                                        str_elt,list_of_atomic_numbers_for_featurization,atomic_number_to_drop)
        decision_function_for_this_elt = clf.decision_function(vec_elt.reshape(-1,len(vec_elt)))
        decision_function_for_each_elt[str_elt] = decision_function_for_this_elt[0]
    periodic_table_heatmap(decision_function_for_each_elt, cbar_label="f(v_E; w,b)",
                                                               show_plot=True,cmap="bwr",cmap_range=(-8,8),value_format='%.3f')
    
    #sanity check on set B case 3 materials
    predicted_labels_on_sanity_check_inputs = clf.predict(matrix_of_inputs_for_sanity_check)
    num_of_sanity_check_samples = len(predicted_labels_on_sanity_check_inputs)
    print("\nnumber of sanity check samples:", num_of_sanity_check_samples)
    print("fraction of sanity check samples classified as topological:", 
                                                  np.sum(predicted_labels_on_sanity_check_inputs) / num_of_sanity_check_samples)
    
    #discovery on set B case 1 materials
    decision_function_values_on_discovery_inputs = clf.decision_function(matrix_of_inputs_for_discovery)
    num_of_discovery_inputs_with_decision_function_geq_1 = np.sum(decision_function_values_on_discovery_inputs >= 1)
    num_of_discovery_inputs_with_decision_function_geq_0 = np.sum(decision_function_values_on_discovery_inputs >= 0)
    print("number of discovery inputs with decision function value greater than or equal to 1.0:",
                                                          num_of_discovery_inputs_with_decision_function_geq_1)
    print("number of discovery inputs with decision function value greater than or equal to 0.0:",
                                                          num_of_discovery_inputs_with_decision_function_geq_0)
    indices_that_would_sort_high2low = np.flip(np.argsort(decision_function_values_on_discovery_inputs))
    
    #print discovered materials that the model is confident about (i.e. those with decision function geq 1)
    for i in range(num_of_discovery_inputs_with_decision_function_geq_1):
        print("\nrank:",i+1)
        index = indices_that_would_sort_high2low[i]
        print("list_of_set_B_case1_material_dicts[index]:",list_of_set_B_case1_material_dicts[index])
        print("decision_function_values_on_discovery_inputs[index]:",decision_function_values_on_discovery_inputs[index])
    
    write_results_to_txt = True
    
    if write_results_to_txt:
    
        #put all of the case1 set B materials into a form that's writeable to a txt file
        sorted_list_of_discovery_inputs_w_decision_function_writable = []
        for i in range(len(indices_that_would_sort_high2low)):
            index = indices_that_would_sort_high2low[i]
            decision_function_value = decision_function_values_on_discovery_inputs[index]
            material_dict = list_of_set_B_case1_material_dicts[index]
            space_group = material_dict["space_group"]
            reduced_formula = material_dict["reduced_formula"]
            str_form_of_material_and_decision_function = \
                            str(space_group) + ", " + reduced_formula + ", " + str(decision_function_value) + "\n"
            sorted_list_of_discovery_inputs_w_decision_function_writable.append(str_form_of_material_and_decision_function)

        #make a text file that's all set B case 1 materials, sorted by decision function
        filename_with_path_all_discovery_inputs = "results/all_decision_function_values.txt"
        f = open(filename_with_path_all_discovery_inputs, "w")
        f.writelines(sorted_list_of_discovery_inputs_w_decision_function_writable)
        f.close()
        
        #make an text file that's just the set B case 1 materials with decision function above 1.0, sorted by decision function
        confident_sorted_list_of_discovery_inputs_w_decision_function_writable = \
            sorted_list_of_discovery_inputs_w_decision_function_writable[:num_of_discovery_inputs_with_decision_function_geq_1]
        filename_with_path_confidently_topological_discovery_inputs = "results/all_confident_predictions.txt"
        f = open(filename_with_path_confidently_topological_discovery_inputs, "w")
        f.writelines(confident_sorted_list_of_discovery_inputs_w_decision_function_writable)
        f.close()
        
        print("Results were written to txt files")
    
    else:
        print("Results were NOT written to txt files")
        
        
    save_learned_topogivities = True
    
    if save_learned_topogivities:
        filename_with_path_for_learned_topogivities = "results/learned_topogivities_svm.pkl"
        output_for_learned_topogivities = open(filename_with_path_for_learned_topogivities, "wb")
        pickle.dump(decision_function_for_each_elt, output_for_learned_topogivities, pickle.HIGHEST_PROTOCOL)
        output_for_learned_topogivities.close()
        print("\nLearned topogivities SAVED")
    else:
        print("\nLearned topogivities NOT SAVED")
    
    
    
if __name__ == '__main__':
    main()