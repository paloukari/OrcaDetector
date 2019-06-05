# script to pull all data from the WHOI website

# Import required modules
import requests
from bs4 import BeautifulSoup
import pandas as pd
import urllib
import os
import sys
from urllib.request import urlopen


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

    # Find all the <tr> tag pairs, skip the first one, then for each.
    for row in table.find_all('tr')[1:]:
        # Create a variable of all the <td> tag pairs in each <tr> tag pair,
        col = row.find_all('a', href=True)

        filename = col[0]['href']
        filename_parts = col[0]['href'].split('/')

        dir = 'data/' + name + '/' + year

        if not os.path.exists(dir):
            os.makedirs(dir)

        sys.stdout.write('-')
        sys.stdout.flush()
        urllib.request.urlretrieve(
            'http://cis.whoi.edu/' + filename, dir + '/' + filename_parts[-1:][0])

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
