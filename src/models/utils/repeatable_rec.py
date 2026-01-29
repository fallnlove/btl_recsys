REPEATABLE_DATASETS = [
    "DeliveryHero-SE",
    "Foursquare-NYC1",
    "Foursquare-NYC2",
    "Foursquare-Tokyo",
    "Frappe",
    "Myket-Android",
    "Retailrocket",
    "Yoochoose",
]

def is_repeatable(dataset):
    return dataset.name in REPEATABLE_DATASETS
