# Part-of-Speech Tagging

* Transition probability: It is defined as the occurrences of the part of speech relative to each other.
* Emission probability: The probability is defined as the occurrence of each word as a part of speech.

##### Models
* Simple: In this model, only the emission probabilities are considered so no dependency on past elements is considered.
* HMM using Viterbi: In this model, the posterior probabilities of the current state are calculated based on the previous state. This algorithm runs for 5000 iterations giving 95% word accuracy.
* Complex Model: In this model, we have used Gibbs sampling to get possible joint probabilities and then we sample based on the new probabilities.

* Accuracy Table

| Model         | Word-Accuracy     | Sentence-Accuracy  |
| ------------- |:-------------:| -----:|
| Simple      | 93.4% | 48.14% |
| HMM      | 95%      |   54.33% |
| Complex | 91.2%      |    38.02% |
