CSE150AMilestone2
Model: [CSE150AMilestone2.ipynb](https://github.com/KyleL1015/CSE150AMilestone2/blob/main/CSE150AMilestone2.ipynb)

PEAS:
In terms of PEAS, the environment for our model would be all American news sources from the past to the present, especially ones where it's difficult to determine credibility, as they are a solid testing ground for how accurate our model is. The performance measure of our model is the accuracy of its guesses (fake or real news). Its actuators are the notebook outputs where we're able to gauge the accuracy of its guesses. The sensors for our model is the model itself taking in data from the tables.

Agent Type:
We first preprocessed our data to get it ready for computation. Initially we checked if and rows had NULL values or blank titles and got rid of any of those. Then we got rid of data we aren't currently considering (like the text in the article or the date the article was published) and classified it depending on whether the article is true or not. While preprocessing, we also got rid of very common words such as "the" or "and" as they shouldn't affect the probability of an article being real or fake and converted our words into numbers using Tfidf in order to train our model.

Our agent is a probabilistic, passive, supervised learning agent. It is trained on the initial dataset and can be improved via retraining, but doesn't automatically train itself given data. By using the Naive Bayes assumption, all words in the title/text are assumed to be independent from each other. The probability for each word is given by

$$
P(w_i \mid C) = \frac{\text{count}(w_i, C) + α}{\sum_{j} \text{count}(w_j, C) + αV}
$$

where C is the class of the article(real or fake), α = 1, and V is the number of unique words in the training data. The α value (Laplace Smoothing) makes it so that inputted words that don't appear in the training data won't have a probability of 0 ruining our results. Then by multiplying all of our probabilities together and performing Bayes' Theorem, we can get a value proportional to P(C | e) where e is our evidence. The P(e) in the denominator of Bayes Theorem isn't necessary because the value of the denominator for fake/real news will be the same.

$$
P(C \mid e) \propto P(C) \prod_{i=1}^{n} P(w_i \mid C)
$$

Conclusion:
In conclusion, using multinomial Naive Bayes and our subset of testing data, we were able to achieve an overall accuracy of 95.01% (from accuracy_score) which is very good considering the Naive Bayes assumption (which is most likely untrue). From our classification report, we can see that our model is slightly better at detecting true negative news sources (recall = 0.944754 vs. 0.929698) but the overall accuracy for both classes is very close to 93.5% ~ 94.0% which is also very respectable. From our confusion matrix you can also see that we have a very minimal amount of False Negatives and False Positives compared to the much larger amount of True Negatives and True Positives which once again is a solid indication of the performance of our model. For future implementations of our model, we could include detection based on phrases rather than individual words as it's possible that certain phrases are more prevalent in fake/real news sources which could lead to higher accuracy for our model. We could also implement hyperparameter tuning to adjust and find a better alpha value that maximizes accuracy.