# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 11:16:20 2015

@author: XJL
"""

import httplib

httplib.HTTPConnection._http_vsn = 10
httplib.HTTPConnection._http_vsn_str = 'HTTP/1.0'

import urllib2,re,time
hdr = {'User-Agent': 'Mozilla/5.0'}

def pageload(url):
    req = urllib2.Request(url,headers=hdr)
    html = urllib2.urlopen(req)
    page = html.read()
    html.close()
    del req,html
    return page
    
links = open('movies_2011.txt','r')
movie_data_csv = open('movie_data_2011.csv','w')
movie_data_csv.write('NAME,DURATION,STAR_1,STAR_2,STAR_3,\
GENRE,YEAR,MONTH,DAY,CONTENT_RATING,DIRECTOR,COUNTRY_NUM,IS_USA,\
LANGUAGE_NUM,IS_ENGLISH,IS_3D,IS_IMAX,BUDGET,REVENUE\n')
movie_data_txt = open('movie_data_2011.txt','w')

file_num = 0

for link in links:
    link = link.strip()
    if len(link)>0:
        pagefile = pageload(link)
        name = re.search('<meta name="title" content="(.*?)"',pagefile).group(1)
        name = name.replace(',','').lower()
        
        if re.search('<time itemprop="duration" datetime="PT(.*?)M"',pagefile):
            duration = re.search('<time itemprop="duration" datetime="PT(.*?)M"',pagefile).group(1)
        else: duration = ''
        duration = duration.replace(',','').lower()
        
        find_star = re.search('<meta name="description".*?With(.*?),(.*?),(.*?),.*?/>',pagefile)
        if find_star:
            star_1 = find_star.group(1).strip()
            star_2 = find_star.group(2).strip()
            star_3 = find_star.group(3).strip()
        else:
            star_1 = ''
            star_2 = ''
            star_3 = ''
        star_1 = star_1.replace(',','').lower()
        star_2 = star_2.replace(',','').lower()
        star_3 = star_3.replace(',','').lower()
        
        if re.search('itemprop="genre">(.*?)<',pagefile):
            genre = re.search('itemprop="genre">(.*?)<',pagefile).group(1)
        else: genre = ''
        genre = genre.replace(',','').lower()
        
        if re.search('"datePublished" content="(.*?)"',pagefile):
            show_time = re.search('"datePublished" content="(.*?)"',pagefile).group(1).split('-')
            if len(show_time) == 1: year,month,day = show_time[0],'',''
            elif len(show_time) == 2: year,month,day = show_time[0],show_time[1],''
            else: year,month,day = show_time[0],show_time[1],show_time[2]
            del show_time
        else:
            year,month,day = '','',''
        
        if re.search('<meta itemprop="contentRating" content="(.*?)"',pagefile):
            content_rating = re.search('<meta itemprop="contentRating" content="(.*?)"',pagefile).group(1)
        else: content_rating = ''
        content_rating = content_rating.replace(',','').lower()
        
        if re.search('<meta name="description" content="Directed by (.*?).  With',pagefile):
            director = re.search('<meta name="description" content="Directed by (.*?).  With',pagefile).group(1)
        else: director = ''
        director = director.replace(',','').lower()
        
        if re.findall('<a href="/country/(.*?)?ref_=tt_dt_dt',pagefile):
            countries = re.findall('<a href="/country/(.*?)?ref_=tt_dt_dt',pagefile)
            country_num = str(len(countries))
            if 'us?' in countries:
                is_us = str(1)
            else: is_us = str(0)
            del countries
        else: country_num, is_us = '', ''
        
        if re.findall('<a href="/language/(.*?)?ref_=tt_dt_dt',pagefile):
            languages = re.findall('<a href="/language/(.*?)?ref_=tt_dt_dt',pagefile)
            language_num = str(len(languages))
            if 'en?' in languages:
                is_english = str(1)
            else: is_english = str(0)
            del languages
        else: language_num, is_english = '', ''
        
        newlink = link + 'technical?ref_=tt_dt_spec'
        newpagefile = pageload(newlink)
        if '3-D' in newpagefile:
            is_3d = str(1)
        else: is_3d = str(0)
        if 'IMAX' in newpagefile:
            is_imax = str(1)
        else: is_imax = str(0)    
                
        if re.search('<h4 class="inline">Budget:</h4>        (.*?)\n',pagefile):
            budget = re.search('<h4 class="inline">Budget:</h4>        (.*?)\n',pagefile).group(1)
            budget = ''.join(c for c in budget if c.isdigit())
        else: budget = ''
        
        if re.search('<h4 class="inline">Gross:</h4>        (.*?)\n',pagefile):
            revenue = re.search('<h4 class="inline">Gross:</h4>        (.*?)\n',pagefile).group(1)
            revenue = ''.join(c for c in revenue if c.isdigit())
        else: revenue = ''
        
        movie_data_csv.write(name+','+duration+','+star_1+','+star_2+','+star_3+','+genre+','+\
        year+','+month+','+day+','+content_rating+','+director+','+country_num+','+is_us+','+\
        language_num+','+is_english+','+is_3d+','+is_imax+','+budget+','+revenue+'\n')
        
        movie_data_txt.write(name+','+duration+','+star_1+','+star_2+','+star_3+','+genre+','+\
        year+','+month+','+day+','+content_rating+','+director+','+country_num+','+is_us+','+\
        language_num+','+is_english+','+is_3d+','+is_imax+','+budget+','+revenue+'\n')
        
        del name,duration,star_1,star_2,star_3,genre,year,month,day,content_rating,director,country_num,\
        is_us,language_num,is_english,is_3d,is_imax,budget,revenue,pagefile,newlink,newpagefile
        
        file_num +=1
        print file_num
        time.sleep(1)

links.close()
movie_data_csv.close()       
movie_data_txt.close()           