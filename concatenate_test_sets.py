#!/usr/bin/python3  
from os import listdir 
from os.path import isfile, join
import re


def write_german_files(files):
    de_count = 0
    for file_name in files:
        file_ = open(file_name)
        with open('combined_de.txt', 'a') as outfile:
            if 'de_' in file_name or 'de.lc' in file_name:
                de_count += 1
                outfile.write(file_.read())
            elif 'ref.de' in file_name:
                de_count += 1
                lines = file_.readlines()
                stripped_lines = []
                for line in lines:
                    if line[:4] == '<seg':
                        line = re.sub('<[^<]+>', '', line)
                        stripped_lines.append(line)
                outfile.write(''.join(stripped_lines))
        file_.close()
    return de_count
def write_english_files(files):
    en_count = 0
    for file_name in files:
        file_ = open(file_name)
        with open('combined_en.txt', 'a') as outfile_en:
            if file_name[4:9] == 'test.' or 'en.lc' in file_name:
                en_count += 1
                outfile_en.write(file_.read())
            elif 'src.en' in file_name:        
                en_count += 1
                lines = file_.readlines()
                stripped_lines = []
                for line in lines:
                    if line[:4] == '<seg':
                        line = re.sub('<[^<]+>', '', line)
                        stripped_lines.append(line)
                outfile_en.write(''.join(stripped_lines))
        file_.close()
    return en_count
if __name__ == "__main__":
    de_files = ['raw/newstest2015-ende-ref.de.sgm', 'raw/newstest2016-ende-ref.de.sgm', 
    'raw/newstest2017-ende-ref.de.sgm',  'raw/newstest2018-ende-ref.de.sgm', 'raw/test_2017_flickr.de.lc.norm.tok', 
    'raw/test_2017_mscoco.de.lc.norm.tok']

    en_files = ['raw/newstest2015-ende-src.en.sgm', 'raw/newstest2016-ende-src.en.sgm', 
    'raw/newstest2017-ende-src.en.sgm',  'raw/newstest2018-ende-src.en.sgm', 'raw/test_2017_flickr.en.lc.norm.tok', 
    'raw/test_2017_mscoco.en.lc.norm.tok']
    
    de_count = write_german_files(de_files)
    en_count = write_english_files(en_files)
    
    assert de_count == en_count