# Whale Identification Model

## Project Proposal

### A. Topic description
Nowadays, whale is really rare and protecting whale is necessary. Different species of whales have different features in their shape of tails and special markings. Thus, in many cases, scientists monitor whales’ activities by using photos of their tails. To help scientists confirm the species of different whales in huge number of photos and save their time, we aim to build a new machine learning model to do this instead of persons.


### B. Research of related work
#### B(1) FaceNet
A Unified Embedding for Face Recognition and Clustering. It uses a deep convolutional network trained to directly optimize the embedding itself, rather than an intermediate bottleneck layer as in previous deep learning approaches, directly learns a mapping from face images to a compact Euclidean space where distances directly correspond to a measure of face similarity. Since the way is really similar with the process for us to analyze the tail of different whales, we would use part of this model in our project.

#### B(2)    ALEXNET
An ImageNet Classification with Deep Convolutional Networks, widely used in neural learning area. In 2012, CNN used to achieve test error rate of 15.4%, nearly half of the best work before. This technology totally used ReLU for the nonlinearity functions and data augmentation technology consisting image translations, horizontal reflections, and patch extractions.

#### B(3)    YOLO
YOLO(You look only once) is a real-time object detection system, they apply the model to an image at multiple locations and scales. High scoring regions of the image are considered detections. It provided a pre-trained model for users to figure out the object from a photo in different size.


### C. Data sources
Most of datas comes from [Happy Whale](https://happywhale.com) and sorted by [Kaggle](https://www.kaggle.com), a platform which already use image process algorithms to category photos collected from public and various survey.


### D. Algorithms are being used and code sources
Since the competition just started, limited number of kernels are available. For most of public kernels, they just try to input data, resize photos and make color channels identical — even it means it may lose some information of colored photos.<br />
Some kernels made further research. For instance, some would use constructed [CNN model to finish the initial identification](https://www.kaggle.com/sunnybeta322/what-am-i-whale-let-me-tell-you). Other use self-developed [triplet model](https://www.kaggle.com/CVxTz/beating-the-baseline-keras-lb-0-38) and it performs better than general CNN model. They beat the baseline of the competition and reached 46.821% accuracy, which seems worth to make some further research.<br />

### E. Evaluating the success of the model
The success of the model will be evaluated based on the accuracy of the model could achieve. The host of the competition will provide one or more test set for participants to evaluate and improve the model. What we need to do is to construct, test and improve the model based on the result we get.


### F. Project timeline
2018.1.26 - 2018.2.9 Project preparation(Learning deep-learning and machine learning fundamental technology)<br />
2018.2.9 - 2018.3.11 Confirm the main method and create the main frame of thesis<br />
2018.3.11 - 2018.3.30 Test the project and improve the model according to the result<br />
2018.3.31 - due  improve model and thesis<br />

### G. References
1. [Python Naming Convention](http://visualgit.readthedocs.io/en/latest/pages/naming_convention.html)
2. https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
3. https://arxiv.org/abs/1512.03385
4. https://en.wikipedia.org/wiki/Humpback_whale
5. https://cs231n.github.io

----

#### Naming Conventions for This Project
##### [Reference](http://visualgit.readthedocs.io/en/latest/pages/naming_convention.html)

###### 1. General
* Avoid using names that are too general or too wordy. Strike a good balance between the two.<br />
* Bad: data_structure, my_list, info_map, dictionary_for_the_purpose_of_storing_data_representing_word_definitions<br />
* Good: user_profile, menu_options, word_definitions<br />
* Don’t be a jackass and name things “O”, “l”, or “I”<br />
* When using CamelCase names, capitalize all letters of an abbreviation (e.g. HTTPServer)<br />

###### 2. Packages
* Package names should be all lower case<br />
* When multiple words are needed, an underscore should separate them<br />
* It is usually preferable to stick to 1 word names<br />

###### 3. Modules
* Module names should be all lower case<br />
* When multiple words are needed, an underscore should separate them<br />
* It is usually preferable to stick to 1 word names<br />

###### 4. Classes
* Class names should follow the UpperCaseCamelCase convention<br />
* Python’s built-in classes, however are typically lowercase words<br />
* Exception classes should end in “Error”<br />

###### 5. Global (module-level) Variables
* Global variables should be all lowercase<br />
* Words in a global variable name should be separated by an underscore<br />

###### 6. Instance Variables
* Instance variable names should be all lower case<br />
* Words in an instance variable name should be separated by an underscore<br />
* Non-public instance variables should begin with a single underscore<br />
* If an instance name needs to be mangled, two underscores may begin its name<br />

###### 7. Methods
* Method names should be all lower case<br />
* Words in an method name should be separated by an underscore<br />
* Non-public method should begin with a single underscore<br />
* If a method name needs to be mangled, two underscores may begin its name<br />

###### 8. Method Arguments
* Instance methods should have their first argument named ‘self’.<br />
* Class methods should have their first argument named ‘cls’<br />

###### 9. Functions
* Function names should be all lower case<br />
* Words in a function name should be separated by an underscore<br />

###### 10. Constants
* Constant names must be fully capitalized<br />
* Words in a constant name should be separated by an underscore<br />
