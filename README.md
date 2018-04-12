# Whale Identification Model

## Project Proposal

### A. Topic description
Nowadays, whale is really rare and protecting whale is necessary. Different species of whales have different features in their shape of tails and special markings. Thus, in many cases, scientists monitor whales’ activities by using photos of their tails. To help scientists confirm the species of different whales in huge number of photos and save their time, we aim to build a new machine learning model to do this instead of persons.<br />

### B. Data sources
Most of datas comes from [Happy Whale](https://happywhale.com) and sorted by [Kaggle](https://www.kaggle.com), a platform which already use image process algorithms to category photos collected from public and various survey.<br />

### C. Algorithms are being used and best of public kernals
Since the competition has no award and participants have no responsibility to pubilc their code, limited number of kernels are available. For most of public kernels, they just try to input data, resize photos and make color channels identical — even it means it may lose some information of colored photos.<br />
Some kernels made further research. For instance, some would use constructed [CNN model to finish the initial identification](https://www.kaggle.com/sunnybeta322/what-am-i-whale-let-me-tell-you). Other use self-developed [triplet model](https://www.kaggle.com/CVxTz/beating-the-baseline-keras-lb-0-38) and it performs better than general CNN model. They beat the baseline of the competition and reached 46.821% accuracy, which seems worth to make some further research. Recently, another participant shared a [traidiional cnn model](https://www.kaggle.com/gimunu/data-augmentation-with-keras-into-cnn) with 32.875% accuracy, implement the CNN model which is different from us.<br />

### D. Evaluating the success of the model
The success of the model will be evaluated based on the accuracy of the model could achieve. The host of the competition will provide one or more test set for participants to evaluate and improve the model. What we need to do is to construct, test and improve the model based on the result we get.<br />

### E. Data preprocessing

### F. Main model of the project

### G. References
1. [Neural Networks and Deep Learnin](https://www.coursera.org/learn/neural-networks-deep-learning)
2. https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
3. https://arxiv.org/abs/1512.03385
4. https://en.wikipedia.org/wiki/Humpback_whale
5. https://cs231n.github.io
