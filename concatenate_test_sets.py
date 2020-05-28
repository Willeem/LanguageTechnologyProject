#!/usr/bin/python3  
from os import listdir 
from os.path import isfile, join
import re


def write_german_files(files, test=False):
    outfile_name = 'test_de' if test else 'train_src_de'
    de_count = 0
    for file_name in files:
        file_ = open(file_name)
        with open(f'{outfile_name}.txt', 'a') as outfile:
            if 'de_' in file_name or 'de.lc' in file_name:
                de_count += 1
                outfile.write(file_.read())
            elif 'src.de' in file_name:
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


def write_english_files(files, test=False):
    outfile_name = 'test_en' if test else 'train_src_en'
    en_count = 0
    for file_name in files:
        file_ = open(file_name)
        with open(f'{outfile_name}.txt', 'a') as outfile_en:
            if file_name[4:9] == 'test.' or 'en.lc' in file_name:
                en_count += 1
                outfile_en.write(file_.read())
            elif 'ref.en' in file_name:        
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
    train_de_files = ['raw/newstest2014-deen-src.de.sgm', 'raw/newstest2015-deen-src.de.sgm', 
    'raw/newstest2016-deen-src.de.sgm', 
    'raw/newstest2017-deen-src.de.sgm',  'raw/newstest2018-deen-src.de.sgm']

    train_en_files = ['raw/newstest2014-deen-ref.en.sgm', 'raw/newstest2015-deen-ref.en.sgm', 
    'raw/newstest2016-deen-ref.en.sgm', 'raw/newstest2017-deen-ref.en.sgm', 
    'raw/newstest2018-deen-ref.en.sgm']
    de_count = write_german_files(train_de_files)
    en_count = write_english_files(train_en_files)
    assert de_count == en_count
    test_de_file = ['raw/newstest2019-deen-src.de.sgm']
    test_en_file = ['raw/newstest2019-deen-ref.en.sgm']
    write_german_files(test_de_file, test=True)
    write_english_files(test_en_file, test=True)