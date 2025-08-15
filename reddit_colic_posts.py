#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project: Equine-Colic-Risk-Factors
File: reddit_colic_posts
Author: Claudia Leins
Description: Sentiment analysis
"""
import praw

reddit = praw.Reddit(
    client_id="oRwML2Tj49IgK29x9YPwjg",
    client_secret="JlaZCFFNvXcPy3jzhcshV8jfajm9cQ",
    user_agent="EquineColicResearch/1.0"
)

posts = []
for submission in reddit.subreddit("Horses").search("colic", limit=1000):
    posts.append({"title": submission.title, "text": submission.selftext, "date": submission.created_utc})

# Als CSV speichern
import pandas as pd
df = pd.DataFrame(posts)
df.to_csv("reddit_colic_posts.csv")