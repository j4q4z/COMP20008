import requests
import json
import re
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import pandas as pd
import matplotlib.pyplot as plt
from numpy import arange
import numpy


base_url = "http://comp20008-jh.eng.unimelb.edu.au:9889/main/"
page = requests.get(base_url)
soup = BeautifulSoup(page.text,"html.parser")
with open('tennis.json') as file:
    tennis_data = json.load(file)

task_1_data = {'url':[],'headline':[]}
task_2_data ={'url':[],'headline':[],'player':[],'score':[]}
task_3_data = {'player': [],'avg_game_difference': []}
task_4_data = {}

def task_1(base_url,soup):
    while(1):
        # Get url for article
        url = urljoin(base_url,soup.find('a').get("href"))

        # visited all articles
        if url in task_1_data['url']:
            break
        
        task_1_data['url'].append(url)
        # scrape website
        page = requests.get(url)
        soup = BeautifulSoup(page.text,"html.parser")
        
        # Get headline for article
        headline = soup.find("h1",class_="headline").text
        task_1_data['headline'].append(headline)
        
    # dataframe with url and headline converted to csv file
    task_1_df = pd.DataFrame(task_1_data)
    task_1_df.to_csv('task1.csv',index=False)
    

def task_2(task_1_data):
    # get player names
    names = []
    for player in tennis_data:
        names.append(player['name'].split())

    for url in task_1_data['url']:
        #get website text
        page = requests.get(url)
        soup = BeautifulSoup(page.text,"html.parser")
        
        # Get headline for article
        headline = soup.find("h1",class_="headline").text
        
        article_text = headline
        # get article text for name and score search
        for body in soup.findAll('p',class_=None):
            para = body.text
            article_text+=' '+para
        
        #regex pattern used to identify the first complete match score in the article
        pattern = '(((\d)+(-|/)(\d)+ ?)(\((\d)+(-|/)(\d)+\) ?)?){2,5}'
        score = re.search(pattern,article_text)
        
        #split into uppercase for player matching
        article_text = article_text.upper().split()
        article_length = len(article_text)
        
        player_found = 0
        for i in range(0,article_length):
            # means that first player name has been found
            if player_found:
                break
            word = article_text[i]
            for player in names:
                # player name is found (first name and last name)
                if len(player)==2 and player[0]==word and player[1]==article_text[i+1]:
                    player_found = 1
                    break
                # player name is found (first name,middle name and last name)
                elif len(player)==3 and player[0]==word and player[1]==article_text[i+1] and player[2]==article_text[i+2]:
                    player_found = 1
                    break
        
        if player_found==1 and score:
            # group converts to string
            match_score = score.group(0)
            # removes brackets for complete score analysis
            no_brackets = re.sub("[\(].*?[\)]", "",match_score)
            score_list = re.findall("[\d]+",no_brackets)
            # iterate through each set to check if valid
            for first,second in zip(score_list[0::2], score_list[1::2]):
                first=int(first)
                second=int(second)
                diff = abs(first-second)
                if (first == 7 or second == 7) and diff in [1,2]:
                    pass
                elif (first > 7 or second >7) and diff == 2:
                    pass
                elif (first < 5 and second == 6 ) or (first==6 and second<5):
                    pass
               # game isnt complete 
                else:
                    score = None
                    break
                    
            if score:
                task_2_data['url'].append(url)
                task_2_data['headline'].append(headline)
                task_2_data['player'].append(' '.join(player))
                task_2_data['score'].append(match_score)
        
                    
    task_2_df = pd.DataFrame(task_2_data)
    task_2_df.to_csv('task2.csv',index=False)

def task_3(task_2_data):
    task_2_df= pd.DataFrame(task_2_data)
    task_3_df= pd.DataFrame(task_3_data)
    task_3_df['player']=task_2_data['player']

    for i in range(len(task_2_df.index)):
        #Get player and their score for that article
        player = task_2_df['player'][i]
        score = task_2_df['score'][i]

        # remove brackets
        no_brackets = re.sub("[\(].*?[\)]", "",score)
        score_list = re.findall("[\d]+",no_brackets)
        # iterate through each set
        game_diff = 0
        for first,second in zip(score_list[0::2], score_list[1::2]):
            first=int(first)
            second=int(second)
            game_diff+=first-second
        game_diff = abs(game_diff)
        task_3_df.loc[i,['avg_game_difference']] = game_diff
    task_3_df = task_3_df.groupby('player').mean().reset_index()
    task_3_df.to_csv('task3.csv',index=False)
    task_5(task_3_df)
    
def task_4(task_2_data):
    # get frequency count for each player
    for player in task_2_data['player']:
        if player not in task_4_data:
            task_4_data[player] = 1
        else:
            task_4_data[player]+=1
    # sorts the dict according to article frequency
    player_dict=sorted(task_4_data, key=task_4_data.get)
    first_5_players = player_dict[::-1][0:5]
    first_5_count = []
    #iterate through first 5, starting from lowest frequency
    for player in first_5_players:
        first_5_count.append(task_4_data[player])
    # create bar chart in ascending order of frequency
    plt.bar(arange(len(first_5_count)),first_5_count)
    plt.xticks(arange(len(first_5_players)),first_5_players,rotation = 15)
    plt.title("Top 5 Most Frequently Written about Players")
    plt.xlabel("Players")
    plt.ylabel("Number of Articles about Player")
    plt.tight_layout()
    plt.savefig('task4.png')
    plt.show()

def task_5(task_3_df):
    task_5_data = {'players': [],'avg_game_difference': [],'win_percentage':[]}
    #copy dataframe to get data for graph
    count=1.0
    for i in range(0,len(task_3_df)):
        player_name = task_3_df['player'][i]
        avg_game_diff = task_3_df['avg_game_difference'][i]
        for player in tennis_data:
            if player['name']==player_name:
                task_5_data['win_percentage'].append(float(player['wonPct'].strip("%"))) 
                task_5_data['players'].append(player_name)
                task_5_data['avg_game_difference'].append(avg_game_diff)

    
    task_5_df= pd.DataFrame(task_5_data)
    task_5_df = task_5_df.sort_values(by=['avg_game_difference'])
    c=['black','darkblue','green','teal','deepskyblue','mediumspringgreen','aqua','turquoise','darkgrey',
       'red','orange','magenta','darkolivegreen','brown','tan','blue','darkslategrey','pink','indigo']
    players=list(task_5_df['players'])
    
    count=0
    for x, y, z in zip(task_5_df['avg_game_difference'], task_5_df['win_percentage'], players):
        plt.scatter(x, y, c=c[count], label=z)
        count+=1
    
    plt.legend(bbox_to_anchor=(1, 1))
    plt.xlabel("Average Game Difference")
    plt.ylabel("Win Percentage (%)")
    plt.title("Average game Difference vs Win Percentage of Player")
    plt.tight_layout()
    plt.savefig('task5.png',bbox_inches='tight')
    plt.show()
    
    

    

    
         
task_1(base_url,soup)
task_2(task_1_data)
task_3(task_2_data)
task_4(task_2_data)