import numpy as np
from pymatgen.core.composition import Composition
from pymatgen.core.periodic_table import Element


#Note: the old version of pymatgen only worked for elements with 1 ≤ atomic number ≤ 103, but the new version of pymatgen works for all elements.  This code is only compatible with the new version of pymatgen [TODO: put this in README istead].



def get_atomic_number(str_elt):
# takes as input a string element symbol.  Returns the atomic number.  

    elt = Element(str_elt)
    return elt.Z



def get_str_elt(atomic_number):
#takes as input an integer atomic number.  Returns the string element symbol
    
    elt = Element.from_Z(atomic_number)
    return elt.symbol



def get_set_of_atomic_numbers_in_chemical_formula(str_chemical_formula):
# takes as input a string chemical formula.  Returns the corresponding set of atomic numbers that appear in this chemical formula

    chemical_comp = Composition(str_chemical_formula)
    el_amt_dict = chemical_comp.get_el_amt_dict()
    set_of_atomic_numbers = set()
    for str_elt in el_amt_dict:
        atomic_number = get_atomic_number(str_elt)
        set_of_atomic_numbers.add(atomic_number)
        
    return set_of_atomic_numbers
    
    
    
def number_of_occurences_of_each_atomic_number_in_data(list_of_material_dicts):
# takes as input a list of material dicts.  Returns a vector of length 118, where entry at index i indicates the number of chemical formulas that contain the element with atomic number (i+1).

    num_of_atomic_numbers_in_periodic_table = 118 #this is for all possible elements, not just the ones that are in our dataset
    atomic_number_occurence_vector =  np.zeros(num_of_atomic_numbers_in_periodic_table,dtype=np.int64)
    
    for i in range(len(list_of_material_dicts)):
        str_chemical_formula = list_of_material_dicts[i]["reduced_formula"]
        set_of_atomic_numbers_in_this_chemical_formula = get_set_of_atomic_numbers_in_chemical_formula(str_chemical_formula)
        for atomic_number in set_of_atomic_numbers_in_this_chemical_formula:
            atomic_number_occurence_vector[atomic_number-1] += 1
            
    return atomic_number_occurence_vector



def number_of_distinct_elements_in_material_dict(material_dict):
#takes as input a material dict. returns the number of distinct elements in the chemical formula of the associated material.
    
    set_of_atomic_numbers = get_set_of_atomic_numbers_in_chemical_formula(material_dict["reduced_formula"])
    return len(set_of_atomic_numbers)



def material_dicts_with_given_number_of_distinct_elements(list_of_material_dicts, num_distinct_elements):
# takes as input a list of material dicts.  Returns another list of material dicts that contains the subset of materials that have the given number of distinct elements

    list_of_material_dicts_with_given_number_of_distinct_elements = []
    for material_dict in list_of_material_dicts:
        if number_of_distinct_elements_in_material_dict(material_dict) == num_distinct_elements:
            list_of_material_dicts_with_given_number_of_distinct_elements.append(material_dict)
    
    return list_of_material_dicts_with_given_number_of_distinct_elements