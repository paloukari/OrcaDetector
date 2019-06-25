# -*- coding: future_fstrings -*-

"""
Script to pull all data from the WHOI website (by permission of site owners)
https://cis.whoi.edu/science/B/whalesounds/fullCuts.cfm

W251 (Summer 2019) - Spyros Garyfallos, Ram Iyer, Mike Winton
"""

import multiprocessing
import os
import sys
import urllib
from urllib.request import urlopen

import pandas as pd
import requests
from bs4 import BeautifulSoup
from joblib import Parallel, delayed


def downloadSample(samplename, filename):
    sys.stdout.write('-')
    sys.stdout.flush()

    if not os.path.exists(filename):
        urllib.request.urlretrieve(
            'http://cis.whoi.edu/' + samplename, filename)


def downloadTable(url, name, year):
    # Scrape the HTML at the url
    r = requests.get(url)

    # Turn the HTML into a Beautiful Soup object
    soup = BeautifulSoup(r.text, 'lxml')

    # Create four variables to score the scraped data in
    location = []
    date = []

    # Create an object of the first object that is class=database
    table = soup.find(class_='database')

    downloadData = []
    for row in table.find_all('tr')[1:]:
        col = row.find_all('a', href=True)

        samplename = col[0]['href']
        samplename_parts = col[0]['href'].split('/')

        dir = 'data/' + name + '/' + year
        filename = dir + '/' + samplename_parts[-1:][0]

        if not os.path.exists(dir):
            os.makedirs(dir)
        downloadData.append([samplename, filename])

    num_cores = multiprocessing.cpu_count()

    results = Parallel(n_jobs=num_cores*2)(
        delayed(downloadSample)(data[0], data[1]) for data in downloadData)

    print('->')


def downloadAllAnimals(url):
    r = requests.get(url)

    soup = BeautifulSoup(r.text, 'lxml')

    # Loop over species
    list = soup.find(class_='large-4 medium-4 columns left')

    for species in list.find_all('option')[1:]:
        url_end = species['value']
        name = species.string.strip()

        print("Downloading " + name)

        name = name.replace(' ', '')
        name = name.replace('-', '_')
        name = name.replace(',', '_')

        # Loop over years
        ryears = requests.get(
            "http://cis.whoi.edu/science/B/whalesounds/" + url_end)

        soupYears = BeautifulSoup(ryears.text, 'lxml')

        listYears = soupYears.find(class_='large-4 medium-4 columns')

        for years in listYears.find_all('option')[1:]:
            urlFin = years['value']
            year = years.string.strip()

            print("         " + "\t" + year)

            downloadTable(
                "http://cis.whoi.edu/science/B/whalesounds/" + urlFin, name, year)


# url
#url = 'http://cis.whoi.edu/science/B/whalesounds/bestOf.cfm?code=BD15F'
#name = 'Atlantic'
#year = '1961'
#
#downloadTable(url, name, year)

url = 'http://cis.whoi.edu/science/B/whalesounds/fullCuts.cfm'
downloadAllAnimals(url)
