#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import logging

from urllib.parse import urljoin
from tqdm import tqdm

from requests import get
from requests.exceptions import RequestException
from contextlib import closing
from bs4 import BeautifulSoup


def is_good_response(resp):
    """
    Returns True if the response seems to be HTML, False otherwise.
    """
    content_type = resp.headers['Content-Type'].lower()
    return (resp.status_code == 200
            and content_type is not None
            and content_type.find('html') > -1)


class Clawer(object):
    def __init__(self, parser=BeautifulSoup, parser_feature='html.parser'):
        self.parser = parser
        self.parser_feature = parser_feature

    def craw(self, url, rule_fn):
        try:
            with closing(get(url, stream=True)) as resp:
                if is_good_response(resp):
                    response = resp.content
                else:
                    response = None

        except RequestException as e:
            logging.error('Error during requests to {0} : {1}'.format(url, str(e)))
            response = None

        parser_ = self.parser(response, self.parser_feature)
        outs = rule_fn(parser_)

        return outs


if __name__ == '__main__':
    # url = 'https://papers.nips.cc/'
    # crawler = Clawer()
    #
    #
    # def rule_href(parser):
    #     outs = []
    #     for li in parser.select('li'):
    #         href = li.find('a').get('href')
    #         if len(href) > 1:
    #             outs.append(href)
    #     return outs
    #
    #
    # print(crawler.craw(url, rule_href))
    #
    # hrefs = ['/book/advances-in-neural-information-processing-systems-32-2019']
    #
    #
    # def rule_title(parser):
    #     outs = []
    #     for li in parser.select('li'):
    #         for name in li.find('a').text.split('\n'):
    #             if len(name) > 5:
    #                 outs.append(name.strip())
    #     return outs
    #
    #
    # outs = crawler.craw(urljoin(url, hrefs[0]), rule_title)
    #
    # word_dict = {}
    # for out in outs:
    #     for word in out.split():
    #         try:
    #             word_dict[word] += 1
    #         except:
    #             word_dict[word] = 1
    # with open('word_dict.json', 'w+') as f:
    #     json.dump(word_dict, f, indent=4, sort_keys=True,
    #                ensure_ascii=False, separators=(',', ': '))
    """
    ----
    """
    with open('word_dict.json', 'r') as f:
        word_dict = json.load(f)
    word_dict_sort = {k: v for k, v in sorted(word_dict.items(), key=lambda item: item[1])}
    print(word_dict_sort)
    # import matplotlib.pyplot as plt
    # plt.bar(word_dict.keys(), word_dict)
    # plt.show()
