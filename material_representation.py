import numpy as np
from pymatgen.core.composition import Composition
import chemistry



def convert_element_to_vector_rep(str_elt,list_of_atomic_numbers_for_featurization,atomic_number_to_drop):
#takes as input a string element symbol.  Forms a vector that is the one-hot atomic number (for the atomic numbers that actually occur in list_of_atomic_numbers_for_featurization).  Drops the category atomic_number_to_drop from this one-hot vector (i.e., that element is encoded as vector of all zeros), and then returns that vector.

    one_hot_atomic_number = np.zeros(len(list_of_atomic_numbers_for_featurization))

    atomic_number_corr_to_this_str_elt = chemistry.get_atomic_number(str_elt)
    
    one_hot_atomic_number[list_of_atomic_numbers_for_featurization.index(atomic_number_corr_to_this_str_elt)] = 1.0

    index_of_category_to_drop = list_of_atomic_numbers_for_featurization.index(atomic_number_to_drop)
    
    one_hot_atomic_number_after_dropping_category = np.delete(one_hot_atomic_number,index_of_category_to_drop)
    
    return one_hot_atomic_number_after_dropping_category



def convert_chemical_formula_to_vector_rep(str_chemical_formula,list_of_atomic_numbers_for_featurization,atomic_number_to_drop):
#Returns vector represesentation of the material, which is implemented as the weighted average of the element vectors.  The weighted average is with respect to the subscripts in the chemical formula.  Note that this is mathematically equivalent to the formulation described in the supplementary material of the paper.
    
    chemical_comp = Composition(str_chemical_formula)
    normalized_chemical_comp_el_amt_dict = chemical_comp.fractional_composition.get_el_amt_dict()
    rep_dim = len(convert_element_to_vector_rep("Li",list_of_atomic_numbers_for_featurization,atomic_number_to_drop))
    
    material_vector = np.zeros(rep_dim)
    for str_elt in normalized_chemical_comp_el_amt_dict:
        vector_for_this_elt = convert_element_to_vector_rep(str_elt,
                                                            list_of_atomic_numbers_for_featurization,atomic_number_to_drop)
        fractional_subscript_for_this_elt = normalized_chemical_comp_el_amt_dict[str_elt]
        material_vector += (fractional_subscript_for_this_elt * vector_for_this_elt)
    
    return material_vector


    
def build_matrix_of_inputs_and_vector_of_labels(list_of_material_dicts, list_of_atomic_numbers_for_featurization, atomic_number_to_drop):
#takes as input a list of n dictionaries, where each dictionary represents a material.  Each dictionary is in the format {"space_group": integer between 1-230, "reduced_formula": string that is the reduced chemical formula, "label": integer that is 0 for case1 and 1 for topological}.  Converts it to a n-by-dim(v_M) numpy array and a length n numpy vector.  For the matrix, each row is the vector representation v_M of the material.  For the vector, each entry is the ground truth label (integer 0 for case1, integer 1 for lumped topological).  The order is preserved (i.e., i-th row of array and i-th entry of vector correspond to i-th material dict in list_of_material_dicts)  Note: although list_of_material_dicts contains space group information, this featurization of the material does not encode the space group information.

    rep_dim = len(convert_chemical_formula_to_vector_rep("Li1",list_of_atomic_numbers_for_featurization,atomic_number_to_drop))
    num_of_materials = len(list_of_material_dicts)
    
    matrix_of_inputs = np.zeros((num_of_materials,rep_dim))
    vector_of_labels = np.zeros(num_of_materials,dtype=np.int64)
    
    for i in range(num_of_materials):
        material_vector = convert_chemical_formula_to_vector_rep(list_of_material_dicts[i]["reduced_formula"],
                                                                 list_of_atomic_numbers_for_featurization, atomic_number_to_drop)
        ground_truth_label = list_of_material_dicts[i]["label"]
        matrix_of_inputs[i] = material_vector
        vector_of_labels[i] = ground_truth_label
    
    return matrix_of_inputs, vector_of_labels
