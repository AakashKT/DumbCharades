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
	- Previous works required

