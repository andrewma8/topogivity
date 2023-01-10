import numpy as np
import pickle
import matplotlib.pyplot as plt
from pymatgen.util.plotting import periodic_table_heatmap

import chemistry
import material_representation
import model
import metrics

from sklearn.linear_model import LogisticRegression

    
def main():
    
    
    # load data
    with open('processed_data/set_A_data.pkl', 'rb') as f:
        list_of_lists_of_material_dicts_for_cv = pickle.load(f)
    with open('processed_data/featurization_data.pkl', 'rb') as f:
        featurization_data = pickle.load(f)
    list_of_atomic_numbers_for_featurization = featurization_data["list_of_atomic_numbers_for_featurization"]
    atomic_number_to_drop = featurization_data["atomic_number_to_drop"]
    
    #convert set A data into matrix and vector forms
    list_of_material_dicts_for_retrain = []
    for i in range(len(list_of_lists_of_material_dicts_for_cv)):
        list_of_material_dicts_for_retrain.extend(list_of_lists_of_material_dicts_for_cv[i])
    matrix_of_inputs_for_retrain, vector_of_labels_for_retrain = \
                material_representation.build_matrix_of_inputs_and_vector_of_labels(
                        list_of_material_dicts_for_retrain, list_of_atomic_numbers_for_featurization, atomic_number_to_drop)
    
    #create the instance of logistic regression
    clf = LogisticRegression(penalty='none',tol=1e-5,max_iter=200)
    
    #fit on the entire labeled dataset
    clf.fit(matrix_of_inputs_for_retrain,vector_of_labels_for_retrain)
    
    #classify the entire labeled dataset
    predicted_labels_on_retrain_inputs = clf.predict(matrix_of_inputs_for_retrain)
    
    #compute retrain metrics and print them
    retrain_accuracy, retrain_recall, retrain_precision, retrain_F1 = metrics.compute_all_metrics(
                                                                vector_of_labels_for_retrain, predicted_labels_on_retrain_inputs)
    print("\nretrain_accuracy: ", retrain_accuracy)
    print("retrain_recall: ", retrain_recall)
    print("retrain_precision: ", retrain_precision)
    print("retrain_F1: ", retrain_F1)
    
    #map to topogivities from logistic regression
    tau_logistic_for_each_elt = {}
    for atomic_number in list_of_atomic_numbers_for_featurization:
        str_elt = chemistry.get_str_elt(atomic_number)
        vec_elt = material_representation.convert_element_to_vector_rep(
                                        str_elt,list_of_atomic_numbers_for_featurization,atomic_number_to_drop)
        tau_logistic_for_this_elt = clf.decision_function(vec_elt.reshape(-1,len(vec_elt)))
        tau_logistic_for_each_elt[str_elt] = tau_logistic_for_this_elt[0]
    
    # visualization on the periodic table
    periodic_table_heatmap(tau_logistic_for_each_elt, cbar_label="topogivity from logisitic regression",
                               show_plot=True, cmap="bwr", cmap_range=(-11.1,11.1), blank_color='gainsboro', value_format='%.3f')
    
    with open('results/learned_topogivities_svm.pkl', 'rb') as f:
        tau_svm_for_each_elt = pickle.load(f)
    
    #create vector of tau's for logistic and svm, where they are in order of ascending atomic number
    list_of_tau_logistic_in_order_of_atomic_number = []
    list_of_tau_svm_in_order_of_atomic_number = []
    for atomic_number in list_of_atomic_numbers_for_featurization:
        str_elt = chemistry.get_str_elt(atomic_number)
        list_of_tau_logistic_in_order_of_atomic_number.append(tau_logistic_for_each_elt[str_elt])
        list_of_tau_svm_in_order_of_atomic_number.append(tau_svm_for_each_elt[str_elt])
    vector_of_tau_logistic_in_order_of_atomic_number = np.array(list_of_tau_logistic_in_order_of_atomic_number)
    vector_of_tau_svm_in_order_of_atomic_number = np.array(list_of_tau_svm_in_order_of_atomic_number)
    
    #plot tau_logistic vs tau_svm
    plt.scatter(vector_of_tau_svm_in_order_of_atomic_number,vector_of_tau_logistic_in_order_of_atomic_number)
    plt.xlabel("topogivity learned using linear SVM")
    plt.ylabel("topogivity learned using logistic regression")
    plt.show()
    
    #determine |tau_E|_max for logistic regression tau_E's and for svm tau_E's; rename variables to reflect it's max magnitude
    tau_logistic_max = np.max(np.abs(vector_of_tau_logistic_in_order_of_atomic_number))
    tau_svm_max = np.max(np.abs(vector_of_tau_svm_in_order_of_atomic_number))
    print("tau_logistic_max: ", tau_logistic_max)
    print("tau_svm_max: ", tau_svm_max)
    
    # vector of tau_E'/|tau_E|_max for logistic and svm
    normalized_vector_of_tau_logistic_in_order_of_atomic_number =\
                                    vector_of_tau_logistic_in_order_of_atomic_number / tau_logistic_max
    normalized_vector_of_tau_svm_in_order_of_atomic_number =\
                                    vector_of_tau_svm_in_order_of_atomic_number / tau_svm_max
    plt.scatter(normalized_vector_of_tau_svm_in_order_of_atomic_number,
                                normalized_vector_of_tau_logistic_in_order_of_atomic_number,s=11)
    plt.xlabel("normalized topogivity from linear SVM")
    plt.ylabel("normalized topogivity from logistic regression")
    plt.grid()
    plt.show()
    
    
if __name__ == '__main__':
    main()