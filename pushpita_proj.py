#Input : The program takes review.json file as input dataset
#Output 1 : output.txt - Outputs the processed review text, sentiment analysis results, comparision against rating from dataset
#Output 2 : results.txt - Outputs the data table showing product id, ratings , sentiment analysis scores
#Output 3 : Graph showing the analysis of each product 
import json
import pandas as pd
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment import vader
import matplotlib.pyplot as plt
  

f = open('output.txt', 'w')

data=[]
inputFile = "reviews_Musical_Instruments_5.json"

#read the dateset in memory
file_data = pd.read_json(inputFile, lines=True)

#filter out the reviews and ratings
stars = file_data["overall"]
data = file_data["reviewText"]

output_data = file_data[["asin","overall"]]

#data pre processing 
stopwords_set = set(stopwords.words("english"))
#english_words = set(nltk.corpus.words.words())

positiveAnalysisCount = 0
negativeAnalysisCount = 0
correctAnalysisCount = 0
incorrectAnalysisCount = 0 

sentiments=[]
sid = SentimentIntensityAnalyzer() #scores from -1 to +1
for i, review in enumerate(data):
	processed_review=''
	tokens = word_tokenize(review)
	for word in tokens:
		if word not in stopwords_set:# and word in english_words:
			processed_review = processed_review + word + ' '
	processed_review = processed_review.strip()
	#print processed_review
	f.write(processed_review + '\n')
	#print ("rating :  ", stars[i]) 
	f.write("rating :  "+ str(stars[i]) + '\n')
	ss = sid.polarity_scores(processed_review)
	for k in ss:
		#print('{0}: {1}, '.format(k, ss[k]))
		f.write('{0}: {1}, '.format(k, ss[k]) + '\n')
		if k == "compound":
			sentiments.append(ss[k])
			if ss[k] >= 0 and stars[i] >= 3:
				#print "Verdict : Positive review, Correct analysis"
				f.write("Verdict : Positive review, Correct analysis" + '\n')
				positiveAnalysisCount = positiveAnalysisCount + 1
				correctAnalysisCount = correctAnalysisCount + 1
			elif ss[k] >= 0 and stars[i] < 3:
				#print "Verdict : Positive review, incorrect analysis"
				f.write("Verdict : Positive review, incorrect analysis" + '\n')
				positiveAnalysisCount = positiveAnalysisCount + 1
				incorrectAnalysisCount = incorrectAnalysisCount + 1
			elif ss[k] < 0 and stars[i] < 3:
				#print "Verdict : Negative Review, correct analysis"
				f.write("Verdict : Negative Review, correct analysis" + '\n')
				negativeAnalysisCount = negativeAnalysisCount + 1
				correctAnalysisCount = correctAnalysisCount + 1
			elif ss[k] <0 and stars[i] >= 3: 
				#print "Verdict : Negative review, incorrect analysis"
				f.write("Verdict : Negative review, incorrect analysis" + '\n')
				negativeAnalysisCount = negativeAnalysisCount + 1
				incorrectAnalysisCount = incorrectAnalysisCount + 1
	#print "---------------------------------------------------------------------------------------------------------------------------------------------"
	f.write("---------------------------------------------------------------------------------------------------------------------------------------------\n")
	sentiments_series = pd.Series(sentiments)
output_data['sentimentScores'] = pd.Series(sentiments) #sentiments_series.values
#print "positiveReviewCount : " , positiveReviewCount
#print "negativeReviewCount : " , negativeReviewCount
#print "correctAnalysisCount : " , correctAnalysisCount
#print "incorrectAnalysisCount : " , incorrectAnalysisCount
#print output_data
output_data.to_csv('results.txt', header=True, index=False, sep='\t', mode='a')
f2 = open('results.txt','a')
#f2.write(output_data)
f2.write('\n\nAnalysis results :\n')
f2.write("positiveAnalysisCount : " + str(positiveAnalysisCount) + '\n')
f2.write("negativeAnalysisCount : " + str(negativeAnalysisCount) + '\n')
f2.write("correctAnalysisCount : " + str(correctAnalysisCount) + '\n')
f2.write("incorrectAnalysisCount : " + str(incorrectAnalysisCount) + '\n')
output_data.groupby(by='asin').sentimentScores.plot.kde()
plt.show()
f.close()
f2.close()

#####################################################################################################
