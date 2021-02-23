import textdistance
import pandas as pd
import nltk
import textdistance
import pandas as pd
from fuzzywuzzy import fuzz


task1a_data = {"idAmazon":[],"idGoogleBase":[],"Score":[]}
google_df = pd.read_csv('google_small.csv')
amazon_df = pd.read_csv('amazon_small.csv')



#iterate through google data
for i in range(len(google_df.index)):
    
    #get google data
    google_id=google_df["idGoogleBase"][i]
    google_name = google_df['name'][i]
    
    # similarity threshold for products to be considered a match
    match_score = 55
    match_id=None
    
    # iterate through and get amazon data
    for j in range(len(amazon_df.index)):
        
        amazon_id=amazon_df['idAmazon'][j]
        amazon_title = amazon_df['title'][j]
           
        
        # get title score
        title_score =  fuzz.partial_ratio(amazon_title,google_name)
        
        
            
        #if amazon product is better match than last,disregard previous best
        if  title_score>=match_score:
            match_score = title_score
            match_id = amazon_id
            fin_title=title_score
#             fin_des=des_score
            fina_title = amazon_title
            
        else:
            continue
        
    # add match to data
    if match_id!=None :
        task1a_data["idAmazon"].append(match_id)
        task1a_data["idGoogleBase"].append(google_id)
        task1a_data["Score"].append(match_score)

        
df=pd.DataFrame(task1a_data)

#removes duplicate matches and picks highest scoring
df = df.sort_values(by='Score', ascending=False)
df = df.drop_duplicates(subset='idAmazon', keep="first")

df = df.drop(columns=['Score'])
df.to_csv('task1a.csv',index=False)
