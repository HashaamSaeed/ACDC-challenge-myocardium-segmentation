# Project Description

The project aims to utilize data provided in the ACDC challenge for 100 patients. The data is in the .nii.gz format and includes MRI images of the heart's short axis with ED and ES images and corresponding ground truth values for RV, myocardium, and LV segmentation.

![Screenshot](capture.png)
![Screenshot](1.png)

## Objective

The project objective is to segment the LV, calculate dice coefficient, slice ED/ES volumes, and corresponding ejection fraction for those slices in the LV. The paper uses the U-Net architecture for semantic segmentation of images. The proposed method resulted in poor segmentation and required post-processing with morphology to connect the circle circling the LV. Therefore, the team tweaked the proposed model and methodology to achieve an average accuracy of >85% with this small dataset.

## Dataset Description

The provided grayscale 8-bit depth images and masks had three regions: LV in white, myocardium in grey, and RV outside myocardium in dull gray. The team thresholded the mask and used only the pixels marking the myocardium to simplify the segmentation model. After thresholding, the mask was converted to a numpy array of bools for use with the binary-cross entropy loss function. The .nii.gz files were converted to PNG images for easier handling and loading into the model.

## Model Description

The code loads the saved model and makes predictions on the ED and ES slices, calculating corresponding metrics. The results and segmentation are displayed in an easy-to-read figure, with accuracy ranging from 60% to 95% as calculated by the dice coefficient. The predicted images are probabilities for every pixel, with intensity values lost. Accuracy varies due to hard thresholding of the "predprob" variable, which only allows pixels with a probability above a set integer. Respectable results for dice coefficient and volumes can be derived from the model. Morphological gradients were used to generate green circles representing outer and inner walls of the LV, showing myocardium wall thickness. The gradient was generated from the predicted mask and masked onto the predicted image to outline inner and outer walls, which can be used to detect heart diseases.
