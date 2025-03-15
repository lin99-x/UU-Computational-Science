#!/usr/bin/env python3
"""A more advanced Mapper, using Python iterators and generators."""

import sys
import json
import re


def read_input(file):
    for line in file:
        # load line in tweets
        yield line


def main(separator='\t'):
    # input comes from STDIN (standard input)
    data = read_input(sys.stdin)
    unique = "unique tweets"
    for line in data:
        try:
            tweet = json.loads(line)
            # tweet_text = tweet["text"].lower()
            pronouns = ["han", "hon", "den", "det", "denna", "denne", "hen"]
            if not "retweeted_status" in tweet: # check if it is a retweeted tweets
                print('%s%s%d' % (unique, separator, 1)) # if not then mark it as unique tweets
                for pronoun in pronouns:
                    if re.search(r"\b" +pronoun+ r"\b", json.dumps(tweet), re.IGNORECASE): # search the pronouns in the tweet, ignor the case
                        print('%s%s%d' % (pronoun, separator, 1))
        except ValueError:
            pass



if __name__ == "__main__":
    main()
