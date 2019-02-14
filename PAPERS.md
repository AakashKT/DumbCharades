# Text2Action: Generative Adversarial Synthesis from Language to Action
- Paper: https://arxiv.org/pdf/1710.05298v2.pdf
- Input / Output -> Sentence -> Word2vec -> encoder -> attention -> generator(pose) -> discriminator  
- Code: None
- Publication: ICRA 2018
- Group: CPSLAB, Seoul, South Korea
- Dataset: MSR-VTT( 3052 Sentence,2211 Actions, 29770 Pairs)
- Network: Seq2seq + Attention, GAN based training
- Motivation
- Assumption
- Novelty 
- Description 	
	- Encoder: Converts Word2vec to its hidden states(ci), 
	- G: Takes Ci and random vextor Zi as hidden state, Xi-1 as input and creates action sequence Xi 
	- D: Takes Ci Real Actions as hidden state,Xi as input and gives a prediction of the sequence being real or fake(Many-one mapping)
- Training: Encoder is trained first as an auto encoder. To create setence to action space using the hidden states. Then we train G&D. Batch size 32, hidden dim = 256.  Lr = 1e-5, Z=16
- Results
- Collab: Added 	

# On human motion prediction using recurrent neural networks
- Paper:http://openaccess.thecvf.com/content_cvpr_2017/papers/Martinez_On_Human_Motion_CVPR_2017_paper.pdf
- Code: https://github.com/una-dinosauria/human-motion-prediction
- Publication: CVPR 2017
- Group:  Julieta Martinez∗1 , Michael J. Black2 , and Javier Romero3 1University of British Columbia, Vancouver, Canada 2MPI for Intelligent Systems, Tubingen, Germany ¨ 3Body Labs Inc., New York, NY
	Dataet: Human 3.6M
- Network: Seq2seq 
- Description:  
	- (a) an encoder that receives the inputs and generates an internal representation, 
	- (b), a decoder network, that takes the internal state and produces a maximum likelihood estimate for prediction. 
	- (c) Adding 15 label 1 hot encoding to the input for mulitple lables
	- (d) First frame is not predicted accurately leading to high error rate. 
	- (d) Hard to tune. 
	- (e) Paper has good insights for training and problems faced like error in short term motion 

# Discriminative Orderlet Mining For Real-time Recognition of Human-Object Interaction
- Input / Output
	- Video --> Skeleton --> Orderlets --> Classification
- Orderlet (Refer to section 3)
	- Feature that captures the ordinal (relative) information among a group of primitive features.
	- Capable of handling missing information (ex. missing joing positions)
	- Less sensitive to noise
- This paper defines and uses the 'orderlet' for action recognition task, since it captures inter-joint coordination patterns
- How is this paper different
	- Previous works use actionlet, which does not encode the spatial relationship between its joints. 
	- Also, the actionlet ensemble is a sequence-level representation that requires a segmented sequence for feature extraction and recognition.
		- This paper does not require segmented video
	- Other methods work on depth maps, instead of skeletal representations
	- Does not fix the orderlet size as opposed to previous approaches 
	- Uses data mining process to select discriminative orderlets, instead of randomly generating ordered features as in previous work.
- Method 
	- Use orderlets for both the skeleton and object features (object is important since, for example, the action of 'taking hand up to the face' can be either eating or talking on the phone)
	- Define a representation of each frame of the video, based on the response of each orderlet (refer section 3.3)
	- Start with size-2 orderlets, and iteratively increase to the orderlet size by 1, upto size-(l+1) orderlets.
	- Next, use ada-boost to further prune orderlets. (refer section 3.5)
	- The above steps are run over the dataset, to optimize an objective function.
	- Use a frame-level temporal score, for classification (refer section 4)

# Exploiting Language Models to Recognize Unseen Actions
- Input / Output
	- Image --> Action (Sentence describing the action)
- ASSUMPTION : Only human actions (Images with a human present, not arbit. actions)
- How is this paper different
	- Previous works required large amount of action datasets, which cannot cover all posible actions, and further is difficult to acquire.
	- This work exploits both language and visual models to handle unseen actions. 
	- The language model is built independently, unlike previous effors.
- Method
	- Train object recognizer, to localize and recognize object given an image.
	- Use language model to find a verb corresponding to the noun (object detected in the image)
	- <b> This disjoint training and inference provides flexibility. </b>
	- Finally, probability of action a:ij is predicted as
		- P( a:ij | I ) = (alpha) * P( o:i | I ) + (1-alpha) * P( v:j | o:i, phi)
		- a:ij, I, o:i, v:j --> Action 'ij' in image 'I', given object 'i' and verb 'j'

# Towards Universal Representation for Unseen Action Recognition
 


# Human Motion Reconstruction
- Paper: http://cs231n.stanford.edu/reports/2017/pdfs/945.pdf
- Code: https://github.com/alarmringing/text2motion
- Dataset: None
- Input / Output:  Label -> Pose

- Paper: http://cs231n.stanford.edu/reports/2017/pdfs/945.pdf
- Code: https://github.com/alarmringing/text2motion
- Dataset: None
- Input / Output:  Label -> Pose

# Generative Choreography using Deep Learning
- Paper: https://arxiv.org/pdf/1605.06921.pdf
- Code: https://github.com/LevinRoman/GC
- Dataset: (Have to contact)
- Input / Output: random -> dance sequence

# acLSTM_motion: Implementation of Auto-Conditioned Recurrent Networks for Extended Complex Human Motion Synthesis
- Paper:https://www.researchgate.net/publication/318527872_Auto-Conditioned_LSTM_Network_for_Extended_Complex_Human_Motion_Synthesis
- Code:https://github.com/papagina/Auto_Conditioned_RNN_motion.git
- Input / Output: Word -> word2vec -> encoder -> pose

# Conditional Video Generation Using Action-Appearance Captions:
- Paper: https://arxiv.org/pdf/1812.01261.pdf

# Deep Video Generation, Prediction and Completion of Human Action Sequences
- Paper: https://arxiv.org/pdf/1711.08682.pdf

# A Recurrent Variational Autoencoder for Human Motion Synthesis
- Paper: http://www.ipab.inf.ed.ac.uk/cgvu/0414.pdf

# Deep Video Generation, Prediction and Completion of Human Action Sequences
- Paper:https://arxiv.org/pdf/1711.08682.pdf

# Learning Human Motion Models for Long-term Predictions:
- Paper:https://arxiv.org/pdf/1704.02827.pdf

# MOCOGAN: https://github.com/sergeytulyakov/mocogan

# StoryGAN: A Sequential Conditional GAN for Story Visualization:
- Paper: https://arxiv.org/pdf/1812.02784.pdf

# Generating Single Subject Activity Videos as a Sequence of Actions Using 3D Convolutional Generative Adversarial Networks
-Paper: https://www.researchgate.net/publication/317416143_Generating_Single_Subject_Activity_Videos_as_a_Sequence_of_Actions_Using_3D_Convolutional_Generative_Adversarial_Networks



