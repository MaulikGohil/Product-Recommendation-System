import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

from surprise import accuracy
from surprise import Dataset
from surprise import Reader

from surprise.model_selection import train_test_split
from surprise.prediction_algorithms.baseline_only import BaselineOnly
from surprise import SVD, evaluate
from surprise.model_selection import cross_validate
from collections import defaultdict

def importdatafromJSON(): 
    #file_handler = open("ratings_Musical_Instruments.csv","r")
    #data = pd.read_csv(file_handler,sep = ",") 
    #file_handler.close()
    df = pd.read_json("Musical_Instruments_5.json", lines=True)
    print("------Columns Name-------")
    print(" ")
    print(df.columns)
    print(" ")
    print("------Columns Info-------")
    print(" ")
    print(df.info())
    print(" ")
    print("------Shape Of Data-------")
    print(" ")
    print(df.shape)
    print(" ")
    #print(df['overall'].min())
    #print(df['overall'].max())
    return df

def split(data): 
    # Seting the rating scale
    read = Reader(rating_scale=(1, 5))
    # Load data from the review data frame and create train and test sets
    dataset = Dataset.load_from_df(data[['reviewerID', 'asin', 'overall']], read)
    #spliting the data with 80% and 20% ratio
    trainset, testset = train_test_split(dataset, test_size=.20)
    return dataset, trainset, testset

def checkForDuplicateReviews(data):
    #checking for the IF a user has given tow reviews to a perticular prodict
    purchase_ids = ['asin', 'reviewerID']
    duplicates = data[data.duplicated(subset=purchase_ids, keep=False)].sort_values(purchase_ids)
    print("------Checking duplicated reviews-------")
    print(" ")
    duplicates.head(10)
    print("-------END---------")
    print(" ")
       
def MinMaxTimeReviewedProducts(data):  
    print("-------Single product reviewed MAX times-------")
    print(data.asin.value_counts().to_frame().head(5))
    print(" ")
    print("-------Single product reviewed MIN times-------")
    print(data.asin.value_counts().to_frame().tail(5))
    print(" ")

def ReviewCount_VS_Freq(data): 
    print("-----Matrix of Review Count VS Frequency-------")
    print(" ")
    print((data.reviewerID.value_counts().rename_axis('id').reset_index(name='frequency').frequency.value_counts(normalize=False).rename_axis('reviews').to_frame().head(20)))
    print(" ")
    print("------------------------------------------------------------------------------------")
    print(" ")
    
def Recommendation_baseline(trainset, testset):
    print("-----ACCURACY using Baseline------")
    bsl_options = {'method': 'als'} #with default parameters 'n_epochs': 10, 'reg_u': 15, 'reg_i': 10
    algo_baseline = BaselineOnly(bsl_options=bsl_options)
    fit_baseline = algo_baseline.fit(trainset)
    predictions_baseline = fit_baseline.test(testset)
    print("Baseline Prediction Accuracy : " , accuracy.rmse(predictions_baseline, verbose=False))
      
def Recommendation_MatrixFact_SVD(trainset, testset, dataset):
    algo_svd = SVD() #with default parameters 'n_epochs=20, lr_all=0.005, reg_all=0.02
    fit_svd = algo_svd.fit(trainset)
    predictions = fit_svd.test(testset)
    print("Matrix Factorization SVD Prediction Accuracy : " , accuracy.rmse(predictions, verbose=False))
    print(" ")
    print("-----ACCURACY using SVD------")
    print(" ")
    #print("Evaluate: ", evaluate(algo_svd, dataset, measures=['RMSE']))
    print(cross_validate(algo_svd, dataset, measures=['RMSE', 'MAE'], cv=5,  verbose=True))
    print(" ")
    return predictions
    
def get_recommendations(predictions,n):
    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, r_ui , est, _ in predictions:
        top_n[uid].append((iid, est))
    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    print(" ")    
    print("-------FOR SPECIFIC USER--------")
    print(" ")
    print(top_n['A1L7M2JXN4EZCR'])    
    print(" ")
    print("-------FOR ALL USER--------")
    print(" ")
    for uid, user_ratings in top_n.items():
        print(uid,"\t", [ iid for (iid, est) in user_ratings])
    print(" ")
    return top_n
    
def recommendation_list(user_list, user_predictions, item_list):
    recommendations = {}
    for i in range(100):
        user = user_list[i]
        if user in user_predictions:
            user_recs = [user_predictions[user][i][0] for i in range(len(user_predictions[user]))]
            if user_recs:
                num_items = len(user_recs)
            else:
                num_items = 0

            idx = 0
            while num_items < 10:
                product = item_list[idx]
                if product not in user_recs:
                    user_recs.append(product)
                    num_items = len(user_recs)
                idx += 1
            recommendations.update({user: user_recs})
    return recommendations

def main():   
    data = importdatafromJSON()

    checkForDuplicateReviews(data)
    MinMaxTimeReviewedProducts(data)
    ReviewCount_VS_Freq(data)
    dataset, trainset, testset = split(data)
    Recommendation_baseline(trainset, testset)
    predictions = Recommendation_MatrixFact_SVD(trainset, testset, dataset)    
    top_n = get_recommendations(predictions,n=10)
    
    review_count = data.asin.value_counts()
    review_count_ten = review_count[review_count >= 60]
    hundred_reviews = data[data.asin.isin(review_count_ten.index)]
    items = (hundred_reviews[['asin', 'overall']].groupby('asin').agg('mean').sort_values('overall', ascending=False).index)
    t = data.reviewerID.unique().tolist()
    #print("Number of Unique Customer/Reviewer In Test Set : ", len(t))
    print(" ")
    recs = recommendation_list(data.reviewerID.unique().tolist(), top_n, items)
    userRec=data.reviewerID.unique().tolist()[1]
    print("Recommendation for User ->", userRec)
    print(recs[userRec])
          
# Calling main function 
if __name__=="__main__": 
    main() 