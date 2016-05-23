# webclassification

The dependencies can be installed by running : ```python setup.py install --user```

webclassifier.py can be used to train the classifier on a given input training file and then predict the category for the given test file. Assumes that the country codes for each url are provided in the file data/data_country_codes.csv
```Usage : python webclassifier.py --train training_file --test testing_file --out output_filename --ngram ngram```
