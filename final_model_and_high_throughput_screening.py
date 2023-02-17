import numpy as np
import pickle
from pymatgen.util.plotting import periodic_table_heatmap
import chemistry
import material_representation
import model
import metrics



def create_txt_file_containing_materials_with_decision_function_geq_thresh(list_of_material_dicts, vector_of_decision_function_values, threshold_value, name_of_txt_file):
#takes as input a list of material dicts, a corresponding vector containing the decision function values for that list of material dicts, a threshold value, and a string that is the name of the txt file to create.  Creates a txt file that contains only those materials with decision function value ≥ the threshold value.  (The entries in this txt file are sorted by decision function value.)
    
    indices_that_would_sort_high2low = np.flip(np.argsort(vector_of_decision_function_values))
    num_of_materials_with_decision_function_geq_thresh = np.sum(vector_of_decision_function_values >= threshold_value)
    
    sorted_list_of_geq_threshold_materials_w_decision_function_writable = []
    for i in range(num_of_materials_with_decision_function_geq_thresh):
        index = indices_that_would_sort_high2low[i]
        decision_function_value = vector_of_decision_function_values[index]
        material_dict = list_of_material_dicts[index]
        space_group = material_dict["space_group"]
        reduced_formula = material_dict["reduced_formula"]
        str_form_of_material_and_decision_function = \
                        str(space_group) + ", " + reduced_formula + ", " + str(decision_function_value) + "\n"
        sorted_list_of_geq_threshold_materials_w_decision_function_writable.append(str_form_of_material_and_decision_function)

    filename_with_path_for_materials_geq_threshold = "results/" + name_of_txt_file
    f = open(filename_with_path_for_materials_geq_threshold, "w")
    f.writelines(sorted_list_of_geq_threshold_materials_w_decision_function_writable)
    f.close()

        
    
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
    matrix_of_inputs_for_additional_evaluation, _ = \
                material_representation.build_matrix_of_inputs_and_vector_of_labels(
                        list_of_set_B_case3_material_dicts, list_of_atomic_numbers_for_featurization, atomic_number_to_drop)
    matrix_of_inputs_for_discovery, _ = \
                material_representation.build_matrix_of_inputs_and_vector_of_labels(
                        list_of_set_B_case1_material_dicts, list_of_atomic_numbers_for_featurization, atomic_number_to_drop)
    
    #hyperparameter that we reretrain with
    with open('results/median_selected_gamma.pkl', 'rb') as f:
        selected_gamma = pickle.load(f)
    print("\nselected_gamma:",selected_gamma)
    
    #reretrain on all of set A to obtain the final model
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
    
    #First, map w and b to topogivities (note that the method of mapping implemented below is mathematically equivalent to
    #the formulation described in the supplementary material of the paper.  Then, visualize these topogivities on periodic table
    decision_function_for_each_elt = {}
    for atomic_number in list_of_atomic_numbers_for_featurization:
        str_elt = chemistry.get_str_elt(atomic_number)
        vec_elt = material_representation.convert_element_to_vector_rep(
                                        str_elt,list_of_atomic_numbers_for_featurization,atomic_number_to_drop)
        decision_function_for_this_elt = clf.decision_function(vec_elt.reshape(-1,len(vec_elt)))
        decision_function_for_each_elt[str_elt] = decision_function_for_this_elt[0]
    periodic_table_heatmap(decision_function_for_each_elt, cbar_label="topogivity",
                                                           show_plot=True,cmap="bwr",cmap_range=(-8,8),value_format='%.3f')
    
    #save learned topogivities
    filename_with_path_for_learned_topogivities = "results/learned_topogivities_svm.pkl"
    output_for_learned_topogivities = open(filename_with_path_for_learned_topogivities, "wb")
    pickle.dump(decision_function_for_each_elt, output_for_learned_topogivities, pickle.HIGHEST_PROTOCOL)
    output_for_learned_topogivities.close()
    
    #additional evaluation of model peformance on set B case 3 materials
    predicted_labels_on_additional_evaluation_inputs = clf.predict(matrix_of_inputs_for_additional_evaluation)
    num_of_additional_evaluation_samples = len(predicted_labels_on_additional_evaluation_inputs)
    print("\nnumber of additional evaluation samples:", num_of_additional_evaluation_samples)
    print("fraction of additional evaluation samples classified as topological:", 
                                np.sum(predicted_labels_on_additional_evaluation_inputs) / num_of_additional_evaluation_samples)
    
    #applying the model to set B case 1 materials (i.e., the discovery space)
    decision_function_values_on_discovery_inputs = clf.decision_function(matrix_of_inputs_for_discovery)
    num_of_discovery_inputs_with_decision_function_geq_1 = np.sum(decision_function_values_on_discovery_inputs >= 1)
    num_of_discovery_inputs_with_decision_function_geq_0 = np.sum(decision_function_values_on_discovery_inputs >= 0)
    print("\nnumber of discovery space materials with g(M) ≥ 1.0:", num_of_discovery_inputs_with_decision_function_geq_1)
    print("number of discovery space materials with g(M) ≥ 0.0:", num_of_discovery_inputs_with_decision_function_geq_0)
    
    #create the txt file containing the discovery space materials with g(M) ≥ 1
    create_txt_file_containing_materials_with_decision_function_geq_thresh(list_of_set_B_case1_material_dicts,
                     decision_function_values_on_discovery_inputs, 1.0, "high_confidence_predictions_in_discovery_space.txt")
    
    #create a list of material dicts that is just the subset of the labeled data that has trivial labels
    list_of_material_dicts_with_trivial_label = []
    for material_dict in list_of_material_dicts_for_reretrain:
        if material_dict["label"] == 0:
            list_of_material_dicts_with_trivial_label.append(material_dict)
    
    #decision function values of the final model (which was fit using the entire labeled dataset) on the subset of the labeled
    #dataset that has trivial labels
    matrix_of_inputs_for_materials_with_trivial_label, _ = \
            material_representation.build_matrix_of_inputs_and_vector_of_labels(
                    list_of_material_dicts_with_trivial_label, list_of_atomic_numbers_for_featurization, atomic_number_to_drop)
    decision_function_values_on_materials_with_trivial_label = clf.decision_function(
                                                                    matrix_of_inputs_for_materials_with_trivial_label)
    
    #create a txt file containing the subset of the trivial label materials that have g(M) ≥ 1
    create_txt_file_containing_materials_with_decision_function_geq_thresh(list_of_material_dicts_with_trivial_label,
         decision_function_values_on_materials_with_trivial_label, 1.0, "trivial_label_materials_that_have_g_of_M_geq_1.txt")
    
    
    
if __name__ == '__main__':
    main()