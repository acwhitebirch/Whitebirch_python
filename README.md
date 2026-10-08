# Whitebirch_python
formerly Whitebirch_et_al._2026
Python code used for analysis of self-administration behavior (raw files from Med-PC) and fiber photometry (primarily for data acquired using Doric equipment and software, though prelminary code for TDT data as well), related to Whitebirch et al., 2026 (in preparation). 
All code was written in Jupyter Lab by A. C. Whitebirch, Ferguson Laboratory, University of Washington. 

Key sources and references include:

  Dr. Adam Gordon-Fennell (tidy_lab_tools) for some initial functions and guidance regarding raw photometry data extraction.
  
  Pre-processing logic inspired by Simpson et al., Neuron 2024, fiber photometry primer.
  
  Baseline correction using adaptive iteratively reweighted penalized least squares (airPLS) adapted from Yizeng Liang and Zhang Zhimin

  Self-adminisration analysis code was adapted from a version originally written by Dr. Aaron Garcia, Ferguson Laboratory, University of Washington
  

Files:

  fp_func_acw_051926.py 
  
  Contains the functions that will be imported and used in fp_preproc_051926.ipynb, fp_preproc_loop_051926.ipynb, and fp_analysis_plotting_051926.ipynb

  SA_analysis_ACW_051926.ipynb 
  
  Presents a self-contained analysis of Med-PC files containing raw data from rat drug self-administration. Lever presses, infusion counts, event timestamps,    
  and other data are extracted and saved as .csv files. Additional code allows for re-organization and synthesis of data across subjects. 
