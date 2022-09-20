The source code in this GitHub respiratory is based on the method proposed in:

"Improving the accuracy of fatty liver index to reflect liver fat content with predictive regression modelling (2022)" by Hykoush A. Asaturyan, Nicolas Basty, Marjola Thanaj, Brandon Whitcher, E. Louise Thomas and Jimmy D. Bell; Link: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0273171.

This GitHub respiratory contains the following files:

**(1) fatty_liver_plus_main.py:** main program.

**(2) fatty_liver_index_plus_ori.sav:** trained fatty liver index plus (FLI+) model.

**(3) sample_csv_file.csv:** a sample CSV file containing the headings and units of the main variables required to run (2). You can replace the units in this CSV file with appropriate data of multiple subjects.

The main program presented in this GitHub respiratory was built using Python 3.7.

**OPTION 1:** To train a FLI+ model (2) from scratch, you will need a multiple-subject dataset with the appropriate variables given in (3).

**OPTION 2:** To apply (2) to a dataset of multiple subjects, you can use (3) as a guide and then run the function "apply_fli_plus_csv" in (1).

**OPTION 3**: To apply (2) to a single test subject, edit the appropriate input variables in (1) and then run the function "apply_fli_plus_single" in (1).

**Note, that the FLI+ model requires the following unique variables**:

* age (years); 

* weight (kg) and height(cm) or BMI (kg/m^2);

* waist circumference (cm) and hip circumference (cm), or waist circumference/hip circumference;

* gamma-glutamyl transferase or GGT (U/L); triglycerides (mmol/L); urate or uric acid (umol/L); testosterone (nmol/L)

* aspartate aminotransferase or AST (U/L); alanine aminotransferase or ALT (U/L); platelet count (10^9/L).
