import pickle



def main():
    
    # load data
    with open('processed_data/set_A_data.pkl', 'rb') as f:
        list_of_lists_of_material_dicts_for_nested_cv = pickle.load(f)
    
    #put labeled dataset into single list of material dicts
    list_of_material_dicts_entire_labeled_dataset = []
    for i in range(len(list_of_lists_of_material_dicts_for_nested_cv)):
        list_of_material_dicts_entire_labeled_dataset.extend(list_of_lists_of_material_dicts_for_nested_cv[i])
    
    #create a set containing all of the reduced formulas that appear at least once in the labeled dataset
    set_of_reduced_formulas_that_appear_in_labeled_dataset = set()
    for material_dict in list_of_material_dicts_entire_labeled_dataset:
        set_of_reduced_formulas_that_appear_in_labeled_dataset.add(material_dict["reduced_formula"])
    
    #count the total number of errors that must be incurred
    total_number_of_errors_incurred = 0
    for reduced_formula in set_of_reduced_formulas_that_appear_in_labeled_dataset:
        number_of_times_this_reduced_formula_appears_as_topological = 0
        number_of_times_this_reduced_formula_appears_as_trivial = 0
        for material_dict in list_of_material_dicts_entire_labeled_dataset:
            if material_dict["reduced_formula"] == reduced_formula:
                if material_dict["label"] == 1:
                    number_of_times_this_reduced_formula_appears_as_topological += 1
                elif material_dict["label"] == 0:
                    number_of_times_this_reduced_formula_appears_as_trivial += 1
                else:
                    raise
        total_number_of_errors_incurred += min(number_of_times_this_reduced_formula_appears_as_topological,
                                                      number_of_times_this_reduced_formula_appears_as_trivial)
    print("total_number_of_errors_incurred:", total_number_of_errors_incurred)

    #upper bound on accuracy
    total_number_of_materials = len(list_of_material_dicts_entire_labeled_dataset)
    upper_bound_on_accuracy = (total_number_of_materials - total_number_of_errors_incurred) / total_number_of_materials
    print("upper_bound_on_accuracy:", upper_bound_on_accuracy)
    
    
    
    
if __name__ == '__main__':
    main()