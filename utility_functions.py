###################################################################################################
'''
Code by : Adem Saglam and Syed Muhammad Hashaam Saeed


'''
###################################################################################################

import os

class UtilityFunctions:

  @staticmethod
  def read_patient_cfg(path_to_file):
    """
    Reads patient data in the cfg file and returns a dictionary
    """
    patient_info = {}
    with open(path_to_file) as f_in:
      for line in f_in:
        l = line.rstrip().split(": ")
        patient_info[l[0]] = l[1]
    return patient_info

    