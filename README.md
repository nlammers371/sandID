# sandID
This repositiory contains scripts to replicate the analyses shown in Figure 7 of:
Sickmann et al. Fingerprinting construction sandsupply networks for traceable sourcing, 2023

To replicate, run the following scripts in order:
(1) train_googlenet.m (this script retrains a CNN classifier to identify the provenance of sand samples contained within the "data" folder)
(2) analyze_results.m (this script calculates the trained model's accuracy and calculates t-SNE projections used to generate the scatter plots in Fig 7)
(3) plot-results.m (this generates the actual confusion matrices and plots shown in the paper)

Additional notes:
  -By default, these scripts will run on sand samples sieved at 2mm. See the comments at the top of each script for instructions on how to switch to 500um
  -build_input_data.m was used to generate the sample snips contained in the "data" folder and is included for completeness. 
   Email nlammers@uw.edu if you wish to have access to the raw images used to generate these image snips.
  
