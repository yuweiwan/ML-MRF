# ML-MRF
use Markov Random Field for image segmentation. MRF will encourage pixels to be assigned to segments with a similar color and many adjacent pixels.  
This is a Potts model (a type of MRF): The Potts model connects each latent variable y_s to the latent variables corresponding to neighboring (adjacent) pixels. Assuming pixel s corresponds to the pixel in row i and column j of the image, the Potts model connects y_s to four other pixels:  
1. The pixel in row i + 1 and column j.  
2. The pixel in row i - 1 and column j.  
3. The pixel in row i and column j + 1.  
4. The pixel in row i and column j - 1.  
  
Use EM to train: The M-step will update the parameters of the Potts model theta given the most recent E-step's variational approxi-
mations to the posterior.  

command:  
python3 main.py --train-data image_1.jpg --model-file train.model --predictions-file train.predictions --algorithm mrf --edge-weight 1.2 --num-states 3 --visualize-predictions-file segmentation_view.png  
python3 compute_accuracy.py image_1.true_predictions image_1.predictions  

