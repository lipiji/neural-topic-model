# -*- coding: utf-8 -*-
#pylint: skip-file
import sys
import os
import numpy as np
import theano
import theano.tensor as T
import gzip
import cPickle as pickle
import json
import string
import argparse
import time
import datetime
import nltk
import operator
from nltk.tokenize import sent_tokenize
nltk.data.path.append("/misc/projdata12/info_fil/pjli/local/nltk/");


print 10/6

def load_amazon():

    LFW_T = 20

    amazon_all = "/misc/projdata12/info_fil/pjli/data/amazon_review/aggressive_dedup.json"
    amazon_mini = "/misc/projdata12/info_fil/pjli/data/amazon_review/mini.json"
    amazon_core5 = "/misc/projdata12/info_fil/pjli/data/amazon_review/kcore_5.json"
    amazon_movie = "/misc/projdata12/info_fil/pjli/data/amazon_review/item_cats/reviews_Movies_and_TV_5.json"
    amazon_data = amazon_movie

    size_corpus = 1697533. #41135700. #82836502.    

    dic = {}
    i2w = {}
    w2i = {}
    i2user = {}
    user2i = {}
    i2item = {}
    item2i = {}
    user2w = {}
    item2w = {}
    x_raw = []
    timestamps = []
    w2df = {}
 
    text_file = open("./data/movie.txt", "w")

    table = string.maketrans("","")
    with open(amazon_data) as f:
        i = 0.
        for line in f:
            try:
                line = line.strip("\n")
                a = json.loads(line)
                user_id = a["reviewerID"]
                item_id = a["asin"]
                review = a["reviewText"]
                rating = a["overall"]
                summary = a["summary"].lower()
                unix_time = a["unixReviewTime"]
                raw_time = a["reviewTime"]

                text_file.write("%s\n" % review)

            except KeyError:
                print "ops: " + line
            i += 1.
            if i == 10000:
                break
            print '\r{0}'.format(i / size_corpus) + " / 1 ",
    text_file.close()


def load_yelp():

    #map_tips =  634834
    #dic= 1061399 #user= 686556 #item= 85539
    #x_train= 2146716 #x_test= 212857

    LFW_T = 10

    yelp_root = "/misc/projdata12/info_fil/pjli/data/recsys/yelp/"
    yelp_tip = yelp_root + "yelp_academic_dataset_tip.json"
    yelp_review = yelp_root + "yelp_academic_dataset_review.json"

    size_corpus = 2685066.

    dic = {}
    i2w = {}
    w2i = {}
    i2user = {}
    user2i = {}
    i2item = {}
    item2i = {}
    user2w = {}
    item2w = {}
    x_raw = []
    timestamps = []
    w2df = {} 

    table = string.maketrans("","")

    text_file = open("./data/yelp.txt", "w")

    map_tips = {}
    with open(yelp_tip) as f:
        for line in f:
            try:
                line = line.strip("\n")
                a = json.loads(line)
                user_id = a["user_id"]
                item_id = a["business_id"]
                summary = a["text"]
                raw_time = a["date"]
            
                text_file.write("%s\n" % summary)
            
            except KeyError:
                print "ops: " + line
    print "#map_tips = ", len(map_tips)

    text_file.close()
    
    with open(yelp_review) as f:
        i = 0.
        for line in f:
            i += 1.
            try:
                line = line.strip("\n")
                a = json.loads(line)
                user_id = a["user_id"]
                item_id = a["business_id"]
                review = a["text"].lower()
                rating = a["stars"]
                raw_time = a["date"]
                unix_time = time.mktime(datetime.datetime.strptime(raw_time, "%Y-%m-%d").timetuple())
               
            except KeyError:
                print "ops: " + line
            print '\r{0}'.format(i / size_corpus) + " / 1 ",
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="which dataset will be processed")
    args = parser.parse_args()

    if args.data == "amazon":
        load_amazon()
    elif args.data == "yelp":
        load_yelp()
    elif args.data == "yelp5":
        load_yelp5()
    else:
        print "error: data"
