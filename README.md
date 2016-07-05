
# Dynamic Bayes Classifier
Dynamic Bayes classifier based on Gaussian naive Bayes classifer.

----
## Generate Training and Testing Data
```
python generator.py
```

----
## Run Examples
```
python test.py
```

----
## Major States and Minor States
1. States in supervised offline learning are major states.
2. In the unsupervised online learning phase, if the data is not within three sigma of any major state, check if it's in any minor state. Create a new minor state for data not within three sigma of any major or minor state.
3. If the number of samples in a minor state exceeds a certain threshold, turn the minor state into a major state.

----
## Dynamic Online Incremental Learning
1. Offline: Initial supervised learning by Gaussian naive Bayes classifier.
see [Wikipedia](https://en.wikipedia.org/wiki/Naive_Bayes_classifier).
2. Online: Dynamic unsupervised incremental learning. Using Gaussian naive Bayes classifier to get the state of normal sample. Create new minor states for data deviating from known major states. If the number of samples in a minor states exceeds a certain threshold, turn the minor state into a major state.
