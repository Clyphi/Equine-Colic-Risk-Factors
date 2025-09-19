#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project: Equine-Colic-Risk-Factors
File: reddit_colic_posts
Author: Claudia Leins
Description: Fetch Reddit posts related to colic in horses
"""

import praw
import pandas as pd
from pathlib import Path
from datetime import datetime

class RedditFetcher:
    def __init__(self, client_id, client_secret, user_agent, subreddit="Horses"):
        self.client_id = client_id
        self.client_secret = client_secret
        self.user_agent = user_agent
        self.subreddit = subreddit
        self.limit = None
        self.reddit = praw.Reddit(
            client_id=self.client_id,
            client_secret=self.client_secret,
            user_agent=self.user_agent
        )
    
    def fetch_posts(self):
        """Fetches posts from the specified subreddit and returns a DataFrame."""
        posts = []
        try:
            submissions = list(self.reddit.subreddit(self.subreddit).new(limit=self.limit))
        except Exception as e:
            print(f"Fehler beim Abrufen der Posts: {e}")
            return pd.DataFrame()  # leeres DF zurückgeben bei Fehler

        for submission in submissions:
            posts.append({
                "title": submission.title,
                "text": submission.selftext,
                "date": datetime.fromtimestamp(submission.created_utc)
            })

        df = pd.DataFrame(posts)
        return df
