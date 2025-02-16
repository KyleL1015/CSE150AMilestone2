CSE150AMilestone2
Model: [CSE150AMilestone1.ipynb](https://raw.githubusercontent.com/KyleL1015/CSE150AMilestone2/refs/heads/main/CSE150AMilestone1.ipynb)

PEAS:
In terms of PEAS, the environment for our model would be all American news sources from the past to the present, especially ones where it's difficult to determine credibility, as they are a solid testing ground for how accurate our model is. The performance measure of our model is the accuracy of its guesses (fake or real news). Its actuators are the notebook outputs where we're able to gauge the accuracy of its guesses. The sensors for our model is the model itself taking in data from the tables.

Agent Type:
We first preprocessed our data to get it ready for computation. Initially we checked if and rows had NULL values or blank titles and got rid of any of those. Then we got rid of data we aren't currently considering (like the text in the article or the date the article was published) and classified it depending on whether the article is true or not. While preprocessing, we also got rid of very common words such as "the" or "and" as they shouldn't affect the probability of an article being real or fake.

Our agent is a probabilistic, passive, supervised learning agent. It is trained on the initial dataset and can be improved via retraining, but doesn't automatically train itself given data. By using the Naive Bayes assumption, all words in the title are assumed to be independent from each other. The probability for each word is given by

$$ 
P(w_i \mid C) = \frac{\text{count}(w_i, C) + α}{\sum_{j} \text{count}(w_j, C) + αV}
$$

where C is the class of the article(real or fake), α = 1, and V is the number of unique words in the training data. The α value (Laplace Smoothing) makes it so that inputted words that don't appear in the training data won't have a probability of 0 ruining our results. 

P(C | d) ∝ P(C) ∏(i=1 to n) P(W_i | C)
$$
P(C \mid d) ∝ P(C) \prod_{i=1}^{n} \ P(w_i \mid C)
$$