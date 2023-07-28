import re
import requests
from bs4 import BeautifulSoup
from lxml.html.clean import Cleaner

regexs = {
    'positive': re.compile('article|body|content|entry|hentry|main|page|pagination|post|text|'
                           'blog|story', re.I),
    'negative': re.compile('comment|combx|com|contact|meta|footer|foot|footnote|media|masthead|'
                           'outbrain|promo|realted|scroll|shoutbox|sidebar|sponsor|shopping|', re.I),
    'punctuation': re.compile('''[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]''', re.I),
    'etc': re.compile('print|archive|reply|login|sign', re.I),
    'whitespace': re.compile(r'\s+', re.I),
    'replace_brs': re.compile('<br */? *>[ \r\n\t]*<br */? *>', re.I),
}

class MainContentExtractor:
    def __init__(self, url):
        self.url = url
        self.html = self._get_html()
        
    def _get_html(self):
        try:
            response = requests.get(self.url)
            html = response.text
            return html
        except:
            print('Connection Error')
    
    def _normailize_whitespace(string):
        if not string:
            return ''
        return ' '.join(string.strip())

    def _clean_html(self, html):
    
        html = regexs['replace_brs'].sub('</p><p>', html)
        soup = BeautifulSoup(html, 'lxml')
        
        remove_tags = frozenset({'script', 'style', 'link', 'a', 'iframe', 'table', 'img', 
                       'embed', 'applet', 'object', 'form', 'header', 'footer'})
        for element in soup.find_all(remove_tags):
            element.extract()
        
        divs = soup.find_all('div')
        for div in divs:
            p = len(div.find_all('p'))
            img = len(div.find_all('img'))
            li = len(div.find_all('li')) - 100
            a = len(div.find_all('a'))
            embed = len(div.find_all('embed'))
            pre = len(div.find_all('pre'))
            code = len(div.find_all('code'))
            
            if (str(div.encode_contents()).count(',') < 10):
                if ((pre == 0) and (code == 0)):
                    if ((img > p) or (li > p) or (a > p) or (p == 0) or (embed > 0)):
                        div.extract()
                        
        paragraphs = soup.find_all('p')
        main_content = []
        for paragraph in paragraphs:
            parent = paragraph.parent
            grandparent = parent.parent
            text = paragraph.text
            if not parent or len(text) < 20:
                continue
                         
            main_content.append(paragraph.text.strip())
        
        return '\n'.join(main_content)

    
    def extract_main_content(self):
        main_content = self._clean_html(self.html)
        return main_content       

if __name__ == '__main__':
    # 추출하고 싶은 url
    url = "https://levelup.gitconnected.com/remove-whitespaces-from-strings-in-python-c5ee612ee9dc"
    
    extractor = MainContentExtractor(url)
    main_content = extractor.extract_main_content()
    
    print(main_content)