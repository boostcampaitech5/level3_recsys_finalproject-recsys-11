import re
import requests
import logging
from lxml.html.clean import Cleaner
from bs4 import BeautifulSoup

class MainContentExtractor:
    
    REGEXS = {
        'POSITIVE': re.compile('article|body|content|entry|hentry|main|page|pagination|post|text|blog|story', re.I),
        'NEGATIVE': re.compile('comment|combx|com|contact|meta|footer|foot|footnote|media|masthead|'
                           'outbrain|promo|realted|scroll|shoutbox|sidebar|sponsor|shopping|header|menu|sponsor|ad-break', re.I),
        'LIKELY': re.compile('and|column|shadow|entry', re.I),
        'UNLIKELY': re.compile('community|disqus|extra|remark|rss|'
                               'shoutbox|agegate|pagination|pager|perma|popup|'
                               'tweet|twitter|social|breadcrumb', re.I),
        'BR' :  re.compile('<br */? *>[ \r\n\t]*<br */? *>', re.I),
        'BREAKS': re.compile("(<br\s*/?>(\s|&nbsp;?)*)+",re.I),
        'DIV': re.compile("<(a|blockquote|dl|div|img|ol|p|pre|table|ul)", re.I),
        'PUNCTUATION': re.compile('''[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]''', re.I),
    }
    
    def __init__(self, url):
        self.url = url
        self.html = self._get_html()
        self.main_content = self.extract_main_content()
    
    @property   
    def logger(self):
        logger = logging.getLogger(self.__class__.__name__)
        return logging.LoggerAdapter(logger, {'main_content_extractor': self})
    
    def log(self, message, level=logging.DEBUG, **kwargs):
        self.logger.log(level, msg=message, **kwargs)

    def _get_html(self):
        try:
            response = requests.get(self.url)
            html = response.text
            return html
        except:
            print('ConnectionError')
    
    def __len__(self):
        return len(self.main_content)
    
    def word_count(self):
        return len(self.main_content.split())
    
    def _clean_html(self, html):
        options = {"scripts": True, "javascript": False, "comments": True, "style": True,
                   "links": False, "meta": False, "annoying_tags": False, "page_structure": False,
                   "frames": False, "forms": True, "kill_tags": ("head", "noscript",), 
                   "embedded": False, "safe_attrs_only": False, "processing_instructions": False}     
        cleaner = Cleaner(**options)
        html = cleaner.clean_html(html)
        html = self.REGEXS['BR'].sub('</p><p>', html)
        html = self.REGEXS['BREAKS'].sub('<br/>', html)
        
        return html
        
    def extract_main_content(self):
        html = self._clean_html(self.html)
        soup = BeautifulSoup(html, 'lxml')
        
        soup = self.remove_tags(soup, 'script', 'style', 'link', 'a', 'iframe', 'table', 'img',
                       'embed', 'applet', 'object', 'form', 'header', 'footer', 'h1', 'h2', 'h3')

        # remove unlikely nodes
        for element in soup.find_all(True):
            if self._is_unlkiely_node(element):
                element.extract()
            if element.tag == 'div':
                encoded_contents = element.encode_contents()
                if not self.REGEXS['DIV'].search(encoded_contents):
                    element.tag == 'p'
        
        soup = self._clean_divs(soup)
        paragraphs = soup.find_all('p')
        candidates = self._score_candidates(paragraphs)
        
        # top_candidate = None
        # for key in candidates:
        #     if (not top_candidate) or (candidates[key]['score'] > top_candidate['score']):
        #         top_candidate = candidates[key]
            
        # main_content = []
        # if top_candidate:
        #     main_content.append(top_candidate['node'].text.strip())
        
        min_score = 30
        main_content = ''
        for key in candidates:
            # print('============')
            # print(candidates[key]['node'])
            # print(candidates[key]['score'])
            if candidates[key]['score'] > min_score:
                # print(candidates[key]['node'].text.strip())
                # print('score', candidates[key]['score'])
                main_content += candidates[key]['node'].text.strip()
       
        sentences = re.findall(r'[^.!?]+[.!?]', main_content)
        postprocessed_sentences = [sentence.strip() for sentence in sentences if len(re.findall(r'[가-힣a-zA-Z]', sentence)) > 20]
        
        
        return ''.join(postprocessed_sentences)
       
        
    def _clean_divs(self, soup):
        divs = soup.find_all('div')
        for div in divs:
            p = len(div.find_all('p'))
            img = len(div.find_all('img')) * 50
            li = len(div.find_all('li')) - 100
            a = len(div.find_all('a'))
            embed = len(div.find_all('embed'))
            pre = len(div.find_all('pre'))
            code = len(div.find_all('code'))
            
            cond1 = (div.text.count(',') < 5) or (div.text.count('.') < 10)
            cond2 = (pre == 0) and (code == 0)
            cond3 = (img > p) or (li > p) or (a > p) or (p == 0) or (embed > 0)
            if all([cond1, cond2, cond3]):
                div.extract()
        
        return soup
    
    def remove_tags(self, soup, *tags):
        for element in soup.find_all(tags):
            element.extract()
        return soup
    
    def _check_node_attributues(self, regex, node, *attributes):
        '''check attributes (id, class, ...) of node'''
        for attr in attributes:
            attribute = node.get(attr)
            if attribute != None and regex.search(str(attribute)):
                return True
        return False
    
    def _is_unlkiely_node(self, node):
        unlikely = self._check_node_attributues(self.REGEXS['UNLIKELY'], node, 'class', 'id')
        likely = self._check_node_attributues(self.REGEXS['LIKELY'], node, 'class', 'id')
        is_unlikely: bool = (unlikely and not likely and node.tag != 'body')
        
        return is_unlikely
    
    def _score_node(self, node):
        score = 0
        
        if self._check_node_attributues(self.REGEXS['POSITIVE'], node, 'class'):
            score += 50
        if self._check_node_attributues(self.REGEXS['NEGATIVE'], node, 'class'):
            score -= 20
        
        if self._check_node_attributues(self.REGEXS['POSITIVE'], node, 'id'):
            score += 50
        if self._check_node_attributues(self.REGEXS['NEGATIVE'], node, 'id'):
            score -= 20
        
        return score
    
    def _initialize_node(self, node):
        score = 0
        
        if node.tag in ('div', 'article'):
            score += 10
        elif node.tag in ('blockquote', 'quote', 'pre', 'td'):
            score += 5
        elif node.tag in ('form', 'address', 'ol', 'ul', 'dl', 'dd', 'dt', 'li'):
            score -= 5
        elif node.tag in ('h1', 'h2', 'h3', 'h4', 'th'):
            score -= 10
        
        score += self._score_node(node)
        return {'node': node, 'score': score}
            
               
    def _score_candidates(self, nodes):
        candidates = {}

        for node in nodes:
            parent_node = node.parent
            grandparent_node = parent_node.parent
            text = node.text.strip()
            
            if not parent_node or len(text) < 20:
                continue
            
            if parent_node not in candidates:
                candidates[parent_node] = self._initialize_node(parent_node)
            
            if grandparent_node not in candidates:
                candidates[grandparent_node] = self._initialize_node(grandparent_node)
            
            if text:
                content_score = 10
                content_score += text.count(',')
                if len(text) > 100:
                    content_score += 100
                elif len(text) > 70:
                    content_score += 50
                
                if '.' not in text or '?' not in text or '!' not in text:
                    content_score -= 30
                
                content_score += text.count('.') * 5
            
            
            candidates[parent_node]['score'] += content_score
            
            if grandparent_node:
                candidates[grandparent_node]['score'] += content_score / 2
            
            if node not in candidates:
                candidates[node] = self._initialize_node(node)
            
            candidates[node]['score'] += content_score
            
            return candidates
        
 
if  __name__ == '__main__':
    url = 'https://scikitlearn.tistory.com/108'
   
    extractor = MainContentExtractor(url)
    main_content = extractor.extract_main_content()
    
    sentences = re.split(r'[.?!]', main_content)
    
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            print(sentence)