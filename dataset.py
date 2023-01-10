import pickle
import random
import numpy as np
from pymatgen.util.plotting import periodic_table_heatmap

import chemistry


#TODO: unify the use of trivial, topological, case1, NAI, USI, etc.


def convert_raw_string_to_processed_tuple(raw_string):
#takes a raw string that will either be in the format '3, Ge1O5Pb3\n' or '3, Ge1O5Pb3'.  Returns a tuple where the first entry is the integer space group number and the second entry is the chemical formula as string (with \n removed if necessary)

    split_raw_string = raw_string.split(', ')
    space_group_number = int(split_raw_string[0])
    raw_chemical_formula_string = split_raw_string[1]
    processed_chemical_formula_string = raw_chemical_formula_string.replace("\n","")
    processed_tuple = (space_group_number, processed_chemical_formula_string)
    return processed_tuple



def process_raw_file(filename_with_path):
# takes as input a string indicating the filename with path. reads in the raw data for the corresponding ccmp txt file.  Processes this, checks that there are no redundant entries, and returns a set of processed tuples of the form (int space group number, str reduced formula)

    f = open(filename_with_path)
    list_of_raw_strings = f.readlines()
    f.close()
    
    list_of_processed_tuples = []
    for i in range(len(list_of_raw_strings)):
        raw_string = list_of_raw_strings[i]
        processed_tuple = convert_raw_string_to_processed_tuple(raw_string)
        list_of_processed_tuples.append(processed_tuple)
    
    set_of_processed_tuples = set(list_of_processed_tuples)
    if len(set_of_processed_tuples) != len(list_of_processed_tuples): #check that there were no redundancies
        raise
    
    return set_of_processed_tuples



def build_list_of_material_dicts(set_of_processed_tuples, ground_truth_label):
#takes as input a set of processed tuples (which all correspond to the same ground truth label) of the form (space group number, reduced formula), as well as a ground truth label.  Returns a list of material dicts of the format {"space_group": integer between 1-230, "reduced_formula": string that is the reduced chemical formula, "label": integer that is 0 for case1 and 1 for topological}

    list_of_material_dicts = []
    for processed_tuple in set_of_processed_tuples:
        material_dict = {}
        material_dict["space_group"] = processed_tuple[0]
        material_dict["reduced_formula"] = processed_tuple[1]
        material_dict["label"] = ground_truth_label
        list_of_material_dicts.append(material_dict)
    
    return list_of_material_dicts



def read_and_process_ccmp_lists():
#reads in the raw data ("case1_ccmp.txt", "case3_ccmp.txt", "tci_ccmp.txt", and "ti_ccmp.txt").  If the same (SG, reduced formula) combination appears in more than one of "case3_ccmp.txt", "tci_ccmp.txt", and "ti_ccmp.txt", then they are merged into the same entry when those three lists are lumped into a single topological category.  If the same (SG, reduced formula)  appears in "case1_ccmp.txt" and the lumped topological category, then that entry is simply discarded.  Returns two lists of dictionaries: one that corresponds to case 1 and one that corresponds to the lumped topological category.  Each dictionary is in the format {"space_group": integer between 1-230, "reduced_formula": string that is the reduced chemical formula, "label": integer that is 0 for case1 and 1 for topological}.  Note: it is assumed that the raw data is already in the form of reduced formulas, so we do not reduce them ourselves (this is indeed the case for the ccmp website).

    filename_with_path_case1 = "raw_data/case1_ccmp.txt"
    filename_with_path_case3 = "raw_data/case3_ccmp.txt"
    filename_with_path_tci = "raw_data/tci_ccmp.txt"
    filename_with_path_ti = "raw_data/ti_ccmp.txt"
    
    set_of_processed_tuples_case1 = process_raw_file(filename_with_path_case1)
    set_of_processed_tuples_case3 = process_raw_file(filename_with_path_case3)
    set_of_processed_tuples_tci = process_raw_file(filename_with_path_tci)
    set_of_processed_tuples_ti = process_raw_file(filename_with_path_ti)
    
    print("intersection of ccmp data for case3 and tci:", set_of_processed_tuples_case3.intersection(set_of_processed_tuples_tci))
    print("intersection of ccmp data for case3 and ti:", set_of_processed_tuples_case3.intersection(set_of_processed_tuples_ti))
    print("intersection of ccmp data for tci and ti:", set_of_processed_tuples_tci.intersection(set_of_processed_tuples_ti))
    
    set_of_processed_tuples_topological = set_of_processed_tuples_case3.union(set_of_processed_tuples_tci,
                                                                                      set_of_processed_tuples_ti)
    
    intersection_of_case1_and_topological = set_of_processed_tuples_case1.intersection(set_of_processed_tuples_topological)
    print("intersection of ccmp data for case1 and lumped topological:", intersection_of_case1_and_topological)
    
    set_of_processed_tuples_case1_without_conflicts = set_of_processed_tuples_case1.difference(
                                                                                    intersection_of_case1_and_topological)
    set_of_processed_tuples_topological_without_conflicts = set_of_processed_tuples_topological.difference(
                                                                                    intersection_of_case1_and_topological)
    
    list_of_material_dicts_case1 = build_list_of_material_dicts(set_of_processed_tuples_case1_without_conflicts, 0)
    list_of_material_dicts_topological = build_list_of_material_dicts(set_of_processed_tuples_topological_without_conflicts, 1)
    
    return list_of_material_dicts_case1, list_of_material_dicts_topological
    


def partition_list_of_material_dicts_by_space_group_set(list_of_material_dicts):
# takes as input a list of material dicts.  Partitions this into two lists of material dicts, one corresponding to set A space groups and one corresponding to set B space groups.  Each of these lists is shuffled, and then returned

    space_groups_set_B = {1, 3, 4, 5, 6, 7, 8, 9, 16, 17, 18, 
    19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 
    30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 
    41, 42, 43, 44, 45, 46, 75, 76, 77, 78, 79, 
    80, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 
    99, 100, 101, 102, 103, 104, 105, 106, 107, 
    108, 109, 110, 143, 144, 145, 146, 149, 150, 
    151, 152, 153, 154, 155, 156, 157, 158, 159, 
    160, 161, 168, 169, 170, 171, 172, 173, 177, 
    178, 179, 180, 181, 182, 183, 184, 185, 186, 
    195, 196, 197, 198, 199, 207, 208, 209, 210, 
    211, 212, 213, 214}
    
    list_of_material_dicts_set_A = []
    list_of_material_dicts_set_B = []
    
    for i in range(len(list_of_material_dicts)):
        material_dict = list_of_material_dicts[i]
        if material_dict["space_group"] in space_groups_set_B:
            list_of_material_dicts_set_B.append(material_dict)
        else:
            list_of_material_dicts_set_A.append(material_dict)
    
    random.shuffle(list_of_material_dicts_set_A)
    random.shuffle(list_of_material_dicts_set_B)
    
    return list_of_material_dicts_set_A, list_of_material_dicts_set_B



def elemental_presence_in_data(list_of_material_dicts):
#takes as input a list of material dicts.  For each element in the periodic table, determines the number of materials in this list of material dicts for which the element is present, and prints it.
    
    atomic_number_occurence_vector = chemistry.number_of_occurences_of_each_atomic_number_in_data(list_of_material_dicts)
    for i in range(len(atomic_number_occurence_vector)):
        number_of_occurences = atomic_number_occurence_vector[i]
        atomic_number = i + 1
        str_elt = chemistry.get_str_elt(atomic_number)
        print(atomic_number,str_elt,": ",number_of_occurences)

        

def get_list_of_common_atomic_numbers_in_data(list_of_material_dicts,cutoff):
# takes as input a list of material dicts.  Determines the atomic numbers that occur in at least cutoff number of materials.  Returns these common atomic numbers as a sorted list.
    
    atomic_number_occurence_vector = chemistry.number_of_occurences_of_each_atomic_number_in_data(list_of_material_dicts)
    list_of_common_atomic_numbers = []
    for i in range(len(atomic_number_occurence_vector)):
        number_of_occurences = atomic_number_occurence_vector[i]
        if number_of_occurences >= cutoff:
            atomic_number = i + 1
            list_of_common_atomic_numbers.append(atomic_number)
    
    return list_of_common_atomic_numbers



def remove_materials_containing_rare_elements(list_of_material_dicts,list_of_common_atomic_numbers):
# takes as input a list of material dicts and a list of atomic numbers that are considered to be common.  Returns a new list of material dicts for which all materials containing a rare element (defined as not being in list of common atomic numbers) have been removed

    set_of_common_atomic_numbers = set(list_of_common_atomic_numbers)
    
    list_of_material_dicts_excl_rare = []
    
    for i in range(len(list_of_material_dicts)):
        material_dict = list_of_material_dicts[i]
        set_of_atomic_numbers_in_this_material = chemistry.get_set_of_atomic_numbers_in_chemical_formula(
                                                                                material_dict["reduced_formula"])
        if set_of_atomic_numbers_in_this_material.issubset(set_of_common_atomic_numbers):
            list_of_material_dicts_excl_rare.append(material_dict)
    
    return list_of_material_dicts_excl_rare



def visualization_of_frequencies(list_of_case1_material_dicts,list_of_topological_material_dicts,list_of_common_atomic_numbers):
#makes a plot that visualizes the frequency of topological materials for each element.  I.e., for each element, we determine the fraction of topological materials among the set of materials that contain that element.  The visualization is made on a periodic table.
    
    occurence_vector_for_containing_given_atomic_number_and_being_topological =\
                                chemistry.number_of_occurences_of_each_atomic_number_in_data(list_of_topological_material_dicts)
    occurence_vector_for_containing_given_atomic_number = chemistry.number_of_occurences_of_each_atomic_number_in_data(
                                                                list_of_case1_material_dicts+list_of_topological_material_dicts)
    
    percentage_for_each_elt = {}
    for atomic_number in list_of_common_atomic_numbers:
        str_elt = chemistry.get_str_elt(atomic_number)
        num_containing_this_atomic_number_and_topological =\
                        occurence_vector_for_containing_given_atomic_number_and_being_topological[atomic_number - 1]
        num_containing_this_atomic_number = occurence_vector_for_containing_given_atomic_number[atomic_number - 1]
        frequency = num_containing_this_atomic_number_and_topological / num_containing_this_atomic_number
        percentage_for_each_elt[str_elt] = frequency * 100
    periodic_table_heatmap(percentage_for_each_elt, cbar_label="topological label percentage",
               show_plot=True, cmap="bwr", cmap_range=(0,100), blank_color='gainsboro', value_format='%.1f')



def create_nested_cv_splits(list_of_case1_material_dicts, list_of_topological_material_dicts, num_splits):
#takes as input a list of case1 material dicts and a list of topological material dicts, and makes num_splits differnet splits, where the splitting is stratified.  Returns these splits as a list of length num_splits, where each entry of the list is a list of dicts

    num_case1_per_split = round(len(list_of_case1_material_dicts) / num_splits)
    num_topological_per_split = round(len(list_of_topological_material_dicts) / num_splits)
    
    list_of_lists_of_material_dicts = []
    
    for i in range(num_splits):
        
        if i != (num_splits - 1):
            list_of_case1_material_dicts_for_this_split = \
                            list_of_case1_material_dicts[i*num_case1_per_split:(i+1)*num_case1_per_split]
            list_of_topological_material_dicts_for_this_split = \
                            list_of_topological_material_dicts[i*num_topological_per_split:(i+1)*num_topological_per_split]
        else:
            list_of_case1_material_dicts_for_this_split = list_of_case1_material_dicts[i*num_case1_per_split:]
            list_of_topological_material_dicts_for_this_split = list_of_topological_material_dicts[i*num_topological_per_split:]
            
        list_of_material_dicts_for_this_split = \
                            list_of_case1_material_dicts_for_this_split + list_of_topological_material_dicts_for_this_split
        random.shuffle(list_of_material_dicts_for_this_split) #for svm this shuffle strictly speaking is unnecessary
        list_of_lists_of_material_dicts.append(list_of_material_dicts_for_this_split)
    
    return list_of_lists_of_material_dicts



def breakdown_in_terms_of_number_of_distinct_elements_in_chemical_formula(list_of_material_dicts,compute_label_percentage=True):
# takes as input a list of material dicts.  Prints how many single element materials there are, how many binary compounds there are, how many ternary compounds there are, etc. (all the way up to the maximum number of distinct elements in this list of material dicts).
    
    #determine the maximum number of distinct elements 
    num_distinct_elts_max = 1
    for material_dict in list_of_material_dicts:
        num_distinct_elts = chemistry.number_of_distinct_elements_in_material_dict(material_dict)
        if num_distinct_elts > num_distinct_elts_max:
            num_distinct_elts_max = num_distinct_elts
    
    #iterate through all the material dicts and count up the occurences [might not need this at all given we have the "###" below
    num_distinct_elts_occurence_vector = np.zeros(num_distinct_elts_max,dtype=np.int64)
    for material_dict in list_of_material_dicts:
        num_distinct_elts = chemistry.number_of_distinct_elements_in_material_dict(material_dict)
        num_distinct_elts_occurence_vector[num_distinct_elts - 1] += 1
    
    print("")
    for i in range(num_distinct_elts_max):
        print("num materials with ", i+1, " distinct elements: ", num_distinct_elts_occurence_vector[i],
                     " (corresponding to ", num_distinct_elts_occurence_vector[i] / len(list_of_material_dicts) * 100,"%)")
    
    if compute_label_percentage:
    
        ###can make this more concise by merging with or simply encompassing the above
        print("")
        for i in range(num_distinct_elts_max):
            num_distinct_elts = i+1
            num_topological_with_this_num_of_distinct_elts = 0
            num_trivial_with_num_of_distinct_elts = 0
            for material_dict in list_of_material_dicts:
                if chemistry.number_of_distinct_elements_in_material_dict(material_dict) == num_distinct_elts:
                    if material_dict["label"] == 1:
                        num_topological_with_this_num_of_distinct_elts += 1
                    elif material_dict["label"] == 0:
                        num_trivial_with_num_of_distinct_elts += 1
                    else:
                        raise
            fraction_topological_among_materials_with_this_num_of_distinct_elts =\
                num_topological_with_this_num_of_distinct_elts /\
                    (num_topological_with_this_num_of_distinct_elts + num_trivial_with_num_of_distinct_elts)
            print("among materials with ", num_distinct_elts, " distinct elements:")
            print(num_topological_with_this_num_of_distinct_elts, " materials have topological label")
            print(num_trivial_with_num_of_distinct_elts, " materials have trivial label")
            print("fraction_topological_among_materials_with_this_num_of_distinct_elts = ",
                                 fraction_topological_among_materials_with_this_num_of_distinct_elts)

    
def main():
    
    # read ccmp files and process them into two lists of material dicts
    list_of_material_dicts_case1_incl_rare, list_of_material_dicts_topological_incl_rare = read_and_process_ccmp_lists()
    
    #partition each list of material dicts into set A and set B
    list_of_set_A_case1_material_dicts_incl_rare, list_of_set_B_case1_material_dicts_incl_rare = \
                                    partition_list_of_material_dicts_by_space_group_set(list_of_material_dicts_case1_incl_rare)
    list_of_set_A_topological_material_dicts_incl_rare, list_of_set_B_topological_material_dicts_incl_rare = \
                                partition_list_of_material_dicts_by_space_group_set(list_of_material_dicts_topological_incl_rare)
    
    #elemental presence prior to removing materials containing rare elements
    print("\nElemental presence results for set A materials prior to removal of materials containing rare elements:")
    elemental_presence_in_data(list_of_set_A_case1_material_dicts_incl_rare+list_of_set_A_topological_material_dicts_incl_rare)
    print("\nElemental presence results for set B materials prior to removal of materials containing rare elements:")
    elemental_presence_in_data(list_of_set_B_case1_material_dicts_incl_rare+list_of_set_B_topological_material_dicts_incl_rare)
    
    # determine the common atomic numbers based on set A materials
    cutoff = 25
    list_of_common_atomic_numbers_in_set_A = get_list_of_common_atomic_numbers_in_data(
                        list_of_set_A_case1_material_dicts_incl_rare+list_of_set_A_topological_material_dicts_incl_rare,cutoff)
    print("list_of_common_atomic_numbers_in_set_A:", list_of_common_atomic_numbers_in_set_A)
    
    #for each list, remove the materials containing rare elements
    list_of_set_A_case1_material_dicts_excl_rare = remove_materials_containing_rare_elements(
                                        list_of_set_A_case1_material_dicts_incl_rare,list_of_common_atomic_numbers_in_set_A)
    list_of_set_A_topological_material_dicts_excl_rare = remove_materials_containing_rare_elements(
                                        list_of_set_A_topological_material_dicts_incl_rare,list_of_common_atomic_numbers_in_set_A)
    list_of_set_B_case1_material_dicts_excl_rare = remove_materials_containing_rare_elements(
                                        list_of_set_B_case1_material_dicts_incl_rare,list_of_common_atomic_numbers_in_set_A)
    list_of_set_B_topological_material_dicts_excl_rare = remove_materials_containing_rare_elements(
                                        list_of_set_B_topological_material_dicts_incl_rare,list_of_common_atomic_numbers_in_set_A)
    
    #elemental presence after removing materials containing rare elements
    print("\nElemental presence results for set A materials after removal of materials containing rare elements:")
    elemental_presence_in_data(list_of_set_A_case1_material_dicts_excl_rare+list_of_set_A_topological_material_dicts_excl_rare)
    print("\nElemental presence results for set B materials after removal of materials containing rare elements:")
    elemental_presence_in_data(list_of_set_B_case1_material_dicts_excl_rare+list_of_set_B_topological_material_dicts_excl_rare)
    
    #most common element in set A materials
    atomic_number_occurence_vector_set_A_excl_rare = chemistry.number_of_occurences_of_each_atomic_number_in_data(
                                list_of_set_A_case1_material_dicts_excl_rare+list_of_set_A_topological_material_dicts_excl_rare)
    most_common_atomic_number_in_set_A_materials = np.argmax(atomic_number_occurence_vector_set_A_excl_rare) + 1
    print("most common atomic number in set A materials:", most_common_atomic_number_in_set_A_materials)
    
    least_number_of_occurences_excluding_zeros = np.inf
    for i in range(len(atomic_number_occurence_vector_set_A_excl_rare)):
        number_of_occurences = atomic_number_occurence_vector_set_A_excl_rare[i]
        if number_of_occurences != 0:
            if number_of_occurences < least_number_of_occurences_excluding_zeros:
                least_number_of_occurences_excluding_zeros = number_of_occurences
    print("number of occurences of least common atomic number in set A materials",least_number_of_occurences_excluding_zeros)
    
    #basic dataset statistics after removing materials containing rare elements
    num_of_set_A_case1_materials = len(list_of_set_A_case1_material_dicts_excl_rare)
    num_of_set_A_topological_materials = len(list_of_set_A_topological_material_dicts_excl_rare)
    num_of_set_B_case1_materials = len(list_of_set_B_case1_material_dicts_excl_rare)
    num_of_set_B_topological_materials = len(list_of_set_B_topological_material_dicts_excl_rare)
    print("total number of set A materials:", num_of_set_A_case1_materials+num_of_set_A_topological_materials)
    print("fraction of set A materials with lumped topological symmetry indicator diagnosis:", 
                      num_of_set_A_topological_materials / (num_of_set_A_topological_materials + num_of_set_A_case1_materials))
    print("total number of set B materials:", num_of_set_B_case1_materials+num_of_set_B_topological_materials)
    print("fraction of set B materials with case3 symmetry indicator diagnosis:",
                     num_of_set_B_topological_materials / (num_of_set_B_topological_materials+num_of_set_B_case1_materials))
    print("")
    print("number of negative labeled materials:", num_of_set_A_case1_materials)
    print("number of positive labeled materials:", num_of_set_A_topological_materials)
    print("number of discovery space materials:", num_of_set_B_case1_materials)
    print("number of sanity check materials:", num_of_set_B_topological_materials)
    
    #frequency visualization for set A materials after removing materials containing rare elements
    visualization_of_frequencies(list_of_set_A_case1_material_dicts_excl_rare,
                                     list_of_set_A_topological_material_dicts_excl_rare,list_of_common_atomic_numbers_in_set_A)
    
    #breakdown in terms of number of distinct elements
    print("\n\nBREAKDOWNS FOR LABELED DATASET:")
    breakdown_in_terms_of_number_of_distinct_elements_in_chemical_formula(
                        list_of_set_A_case1_material_dicts_excl_rare + list_of_set_A_topological_material_dicts_excl_rare)
    print("\n\nBREAKDOWNS FOR DISCOVERY SPACE:")
    breakdown_in_terms_of_number_of_distinct_elements_in_chemical_formula(list_of_set_B_case1_material_dicts_excl_rare,False)
    print("\n\nBREAKDOWNS FOR SANITY CHECK:")
    breakdown_in_terms_of_number_of_distinct_elements_in_chemical_formula(
                                                                    list_of_set_B_topological_material_dicts_excl_rare,False)
    
    
    # create nested cv splits
    num_splits = 11
    list_of_lists_of_material_dicts_for_nested_cv = create_nested_cv_splits(
                list_of_set_A_case1_material_dicts_excl_rare, list_of_set_A_topological_material_dicts_excl_rare, num_splits)
    
    #save results
    save_processed_data = True
    if save_processed_data:
        
        set_B_data = {}
        set_B_data["case1_material_dicts"] = list_of_set_B_case1_material_dicts_excl_rare
        set_B_data["case3_material_dicts"] = list_of_set_B_topological_material_dicts_excl_rare
        
        featurization_data = {}
        featurization_data["list_of_atomic_numbers_for_featurization"] = list_of_common_atomic_numbers_in_set_A
        featurization_data["atomic_number_to_drop"] = most_common_atomic_number_in_set_A_materials
        
        filename_with_path_for_set_A_data = "processed_data/set_A_data.pkl"
        output_for_set_A = open(filename_with_path_for_set_A_data, "wb")
        pickle.dump(list_of_lists_of_material_dicts_for_nested_cv, output_for_set_A, pickle.HIGHEST_PROTOCOL)
        output_for_set_A.close()
        
        filename_with_path_for_set_B_data = "processed_data/set_B_data.pkl"
        output_for_set_B = open(filename_with_path_for_set_B_data, "wb")
        pickle.dump(set_B_data, output_for_set_B, pickle.HIGHEST_PROTOCOL)
        output_for_set_B.close()
        
        filename_with_path_for_featurization_data = "processed_data/featurization_data.pkl"
        output_for_featurization = open(filename_with_path_for_featurization_data, "wb")
        pickle.dump(featurization_data, output_for_featurization, pickle.HIGHEST_PROTOCOL)
        output_for_featurization.close()
        
        print("\nProcessed data SAVED\n")
        
    else:
        print("\nProcessed data NOT SAVED\n")
    


if __name__ == '__main__':
    main()