1. The file "Keras-UNet" contains the model and the trainiing parameters to train the model.Please DO NOT RUN it
   since it will overwrite the already saved "U-Net LV Segmentation Model" Folder containing all the weights
   for the network.

2.The file "model_predict" contains the functions required to run and load the model on test and train images,it
  can be run standalone or can be run with the APP.

3. ACDC_Dataset_PNG_192x192 folder contains the PNG images for train and test images with their masks.

4. ACDC_Dataset_Seperated_data_seperated contains the required .nii.gz files required to run the code.

5. In-order to run the code with GUI please open and run the App.py file and follow instructions in the report.

6. If for instance the app deosnt work(depending on your system lacking dependencies) or any other issues we have the 
   option to run the stand alone version in the file named "model_predict", just open that file and run it to get 
   results! change the parameter "ix" to change image set.

7. folder "dataset" contains data for the app to run.


