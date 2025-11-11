# PII Name Concepts for ECHR dataset
PII_NAME_CONCEPTS = [
    "Names of witnesses",
    "Full name of applicant or respondent", 
    # "Marital partner's name",
    # "Names of close relatives",
    # "Next of kin name",
    # "Mother's maiden name",
    # "Father's name",
    # "Full name of minor children",
]
example_name = {'SetFit/sst2': 'text', 'ag_news': 'text', 'yelp_polarity': 'text', 'dbpedia_14': 'content', 'custom_echr': 'text'}
concepts_from_labels = {'SetFit/sst2': ["negative","positive"], 'yelp_polarity': ["negative","positive"], 'ag_news': ["world", "sports", "business", "technology"], 'dbpedia_14': ["company","education","artist","athlete","office","transportation","building","natural","village","animal","plant","album","film","written"], 'custom_echr': PII_NAME_CONCEPTS}
class_num = {'SetFit/sst2': 2, 'ag_news': 4, 'yelp_polarity': 2, 'dbpedia_14': 14, 'custom_echr': 2}
epoch = {'SetFit/sst2': 3, 'ag_news': 1, 'yelp_polarity': 1, 'dbpedia_14': 1, 'custom_echr': 2}

unsup_dim = {'default': 4096, 'SetFit/sst2': 4096, 'custom_echr': 4096}