'''

@author Negin Sobhani, Joshua Driscol, Karen Stengel


this script reads all of the SST data files, strips out unimportant* information,
and creates data sets with features based off of the lead prediction time given. all
new data should be stored based off of lead time. ie data with a 30 day prediction
should be in a separate file from data with a 45 day prediction time.



example directory organization?:
    10 day dataset -
                   - control; random labels
                   - train
                   - dev
                   - test
    30 day dataset -
                   - control; random labels
                   - train
                   - dev
                   - test
    45 day dataset -
                   - control; random labels
                   - train
                   - dev
                   - test
    60 day dataset -
                   - control; random labels
                   - train
                   - dev
                   - test
'''
