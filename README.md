# Text-Processing-using-NLTK-ML
The aim of the project was to develop 3 classification machine learning algorithms and check for their accuracy and suggesting the best suited algorithm for the given dataset.
The given dataset had various columns, many of them not being required for either the questions or for the prediction.
There was also a lot of cleaning involved in the given dataset, in columns containing numerical as well as categorical data.
First we cleaned the ‘text’ and ‘description’ column of the dataset. For these we imported a library called ‘re’, which was used to clean the data for each column. Here we removed words containing @ and # symbols. Then we also removed any hyperlinks if present. Then we removed all the other symbols except alphanumeric values. We also removed extra spaces between the words if there was any.
In the gender column we removed the rows containing gender=”unknown”. However we decided to include the rows having gender=”brand” because since it is a twitter dataset then there are many organizations that tweet and hence these fall under the “brand” category which is important to predict separately.
Then for the columns “fav_number”, “tweet_count”, “retweet_count”, “ gender:confidence” we plotted the boxplot to find the outliers to increase the accuracy of the models.
We removed the outliers and in the final dataset we had 18189 rows out of the original 20050 rows. 



ML MODEL:
Dependent Variable: gender.
Independent Variable: Text, Description
We decided that our machine learning model will take the tweet text and the description of the user to predict the gender of the user.
For that we imported CountVectorizer() from sklearn.feature_extraction.text library.
We provide the combined string of text and description for each user to this vectorizer. What this does is that it takes all the words of the string and makes a sparse matrix in which columns are these words and rows corresponding to each user. The value of each cell is the frequency of the words in the text and description of the respective users.
So it return a sparse matrix “X” which can be given to any model for training.
We apply label encoding to the gender column to prepare our dependent set “Y”.
Now we train the various ML classification algorithms on these X and Y inputs.
For the different algorithms we obtain different accuracy.
For ensemble learning we use VotingClassifier in which we gave LogisticRegssion, SVM and Random Forest.

Solving Questions:
For solving these questions we used the cleaned dataset.
Q1) What are the most common emotions/words used by Males and Females?
For this question, we created 2 lists namely: male_message and female_message which contain all the text in them.
Then we remove all the stopwords present in the string.  After removing the stopwords we iterate over the list and create a dictionary that contains the words as ‘keys’ and their frequency as ‘values’.
Then from this dictionary we create a dataframe and arrange them in decreasing order to find the maximum used words for male and female respectively.

Q2) Which gender makes more typos in their tweets?
In order to find the wrong spelling we used a library called SpellChecker.
For each string in the rows first we split the string and match the words with their correct spelling.
If the spelling is wrong then we increase the count of typos by 1.
Then simply we find the sum of typos for each gender separately and find the result.

