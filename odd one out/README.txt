We hope that this repo is self-explanatory. If not, please reach out to us!

To make it easier for you to find relevant information:

* Where are the reference images? *
Many of the images used for the odd-one-out task come from the THINGS dataset. 
However, not all are included because we had collected them before then THINGS dataset 
was created and for a certain subset, we were not satisfied with the image quality. 
Some images are also not square but were padded with white pixels. The images included 
in the THINGS database that were also used for behavior are always the first image 
if it ends with a letter b (e.g., aardvark_01b.jpg). If you would like to access the 
reference images used, you can currently find them in the link below. By downloading 
you confirm that you only use these images for research purposes according to fair use:
https://martin-hebart.com/reference_images.zip
(if the link is dead, please reach out to us!)

* Is the order of THINGS alphabetical? *
Not necessarily! It depends on how your OS and your software sort. This is a very common 
mistake that people make. Use the variable unique_id.txt under the variables folder to be 
sure the order is correct.

* Where do I find the 66 dimensional embedding? *
Under data/spose_embedding_66d_sorted.txt

* Where do I find the raw data used for training? *
Under data/triplet_dataset. In there are what we call trainset.txt, validationset.txt,
and testset1.txt to testset 3.txt. The names are a little arbitrary. We used 
what is called validationset to evaluate performance of our model and the testsets only 
for estimating the noise ceiling.