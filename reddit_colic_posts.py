#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project: Equine-Colic-Risk-Factors
File: reddit_colic_posts
Author: Claudia Leins
Description: Reddit colic posts dataset
Dataset of Reddit posts mentioning 'colic' in r/Horses, 
intended for analysis of contributing factors like weather,
season, and stable conditions."
"""
import praw
import pandas as pd
import os
from dotenv import load_dotenv

# .env laden
load_dotenv()

reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_ID"),
    client_secret=os.getenv("REDDIT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT")
)

posts = []
for submission in reddit.subreddit("Horses").search("colic", limit=None):
    posts.append({
        "title": submission.title,
        "text": submission.selftext,
        "date": pd.to_datetime(submission.created_utc, unit='s')
    })

# Als CSV speichern
import pandas as pd
df = pd.DataFrame(posts)
df.to_csv("reddit_colic_posts.csv")