import ast
import gender_guesser.detector as gender
import json
import nltk
import numpy as np
import os
import pandas as pd
import pickle
import re
from bs4 import BeautifulSoup
from collections import Counter
from html.parser import HTMLParser
from langdetect import detect
from matplotlib import pyplot as plt
from scipy import sparse
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm_notebook
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

'''
Put train.json, test.json, train_log1p_recommends.csv and sample_submission.csv
files into PATH_TO_DATA folder.
  
Firstly the code of extracting features starts to work. The process lasts several hours. 
So you can skip it, download ready features files from here: 
https://goo.gl/oUF6ff - zip file 'datasets.zip' - and unpack it to PATH_TO_DATA folder.
Then you can start from 'Load all features, train model and predict' - 473 row of this file.

While creating vectorized bag of words by TfidfVectorizer up to 32Gb of 
RAM is needed (433 row). 

Current version of solution predicts with a little bit more public score then I recieved 
before competition deadline. That`s because of another way of creating tag features 
sparse matrix (411 row). The previous code of getting it was occasionaly erased, so when 
I was restored it - the better results I got. But the old sparse file exists - you can 
find it (full_tags_old.npz) in the folder at google drive (url above).

While features are being extracted text like "extract_content False: 34638" is printed
to show the progress. 

!!!!!! Important !!!!!
After getting "test_h1_h6 features.csv" from test.json - there is need to correct 
csv file manually. There is instruction above that code. I couldn`t find the way to make it
correct with Python. It seems like there is the problem with coding of text which causes
the breakage when saving .csv file. If this error isn`t corrected the final prediction will
fall down by about 0.15.

So you can wait until all features will be extracted, than stop the code: repair the file
and start the train and predict part. Or the better way to separate this file for two:
file to extract features and file to train and predict model.    
'''

PATH_TO_DATA = './datasets/'

# ------- default html transformation functions -------
class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)


def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()


def read_json_line(line=None):
    result = None
    try:
        result = json.loads(line)
    except Exception as e:
        # Find the offending character index:
        idx_to_replace = int(str(e).split(' ')[-1].replace(')',''))
        new_line = list(line)
        new_line[idx_to_replace] = ' '
        new_line = ''.json(new_line)
        return read_json_line(line=new_line)
    return result


# ------- function to extract information from title and meta-tag section of article -------
def extract_title_features(path_to_data, train=True):
    if train:
        inp_filename = 'train.json'
    else:
        inp_filename = 'test.json'

    with open(os.path.join(path_to_data, inp_filename), encoding='utf-8') as inp_json_file:
        count_lines = 0

        titles = []
        title_length = []
        published = []
        author = []
        meta_tags_desc = []
        meta_tags_reading_time = []
        type_of_words = []
        domain = []
        url = []

        for line in tqdm_notebook(inp_json_file):
            print('extract_title_features ' + str(train) + ': ' + str(count_lines))
            json_data = read_json_line(line)
            titles.append(re.sub(r'[^\w\s]', '', json_data['title'].lower()))
            published.append(pd.to_datetime(json_data['published']['$date']))
            author.append(json_data['author']['url'].split('@')[-1])
            meta_tags_desc.append(re.sub(r'[^\w\s]', '', json_data['meta_tags']['description'].lower()))
            meta_tags_reading_time.append(json_data['meta_tags']['twitter:data1'].split(' ')[0])
            domain.append(json_data['domain'])
            url.append(json_data['url'])
            count_lines += 1
            if titles[-1].split(' ')[-1] == domain[-1].split('.')[0]:
                titles[-1] = titles[-1].rsplit(' ', 1)[0]
            title_length.append(len(titles[-1]))
            title = nltk.word_tokenize(titles[-1])
            title = nltk.Text(title)
            title_tags = nltk.pos_tag(title)
            tags_count = Counter(tag for word, tag in title_tags)
            total_tags = sum(tags_count.values())
            tags_count = dict((word, np.round(float(count)/total_tags,2)) for word,count in tags_count.items())
            type_of_words.append(tags_count)

        export_data = pd.DataFrame({'titles': titles,
                                    'title_length': title_length,
                                    'title_type_of_words': type_of_words,
                                    'published': published,
                                    'author': author,
                                    'meta_tags_desc': meta_tags_desc,
                                    'meta_tags_reading_time': meta_tags_reading_time,
                                    'domain': domain,
                                    'url': url,
                                    })

        return export_data


# ------- function to extract information from tags inside articles -------
def extract_content_features(path_to_data, train=True):
    if train:
        inp_filename = 'train.json'
    else:
        inp_filename = 'test.json'

    with open(os.path.join(path_to_data, inp_filename), encoding='utf-8') as inp_json_file:
        count_lines = 0

        content_length, content_count = [], []
        content_type_words = []
        main_div_sections = []
        count_commas = []
        count_vosclicanie = []
        count_non_letters = []
        language = []
        count_img = []
        author_real_name = []
        site_dep = []

        for line in tqdm_notebook(inp_json_file):
            content = []
            print('extract_content_features '+str(train)+': ' + str(count_lines))
            count_lines += 1
            json_data = read_json_line(line)
            cur_url = json_data['url']
            text = ''

            try:
                site_dep.append(cur_url.split('/')[3])
            except:
                site_dep.append('unknown')

            try:
                cur_domain = cur_url.split('/')[2]
            except:
                cur_domain = 'unknown'
            if cur_domain == 'medium.com':
                soup = BeautifulSoup(json_data['content'], "lxml")
                count_img.append(soup.find_all('img'))
                author_name_a = soup.find_all('a', {'data-action': 'show-user-card'})
                if len(author_name_a) > 1:
                    author_real_name.append(author_name_a[1].get_text())
                else:
                    author_real_name.append('unknown')
                soup = soup.find_all('div', {"class": 'postArticle-content js-postField js-notesSource js-trackedPost'})[0]

            else:
                soup = BeautifulSoup(json_data['content'], "lxml")
                count_img.append(soup.find_all('img'))
                author_real_name.append('unknown')

            p_text = soup.find_all('p')
            content_count.append(len(p_text))
            for p in p_text:
                text = text + ' ' + strip_tags(str(p))
                content.append(re.sub(r'[^\w\s]', '', strip_tags(str(p)).lower()))
            count_commas.append(text.count(','))
            count_vosclicanie.append(text.count('!'))
            text = re.sub(r'[^\w\s]', '', text.lower())
            content_length.append(len(text.split(' ')))

            main_div_section = soup.find_all('section')
            main_div_sections.append(len(main_div_section))
            try:
                language.append(detect(text))
            except:
                language.append('unknown')
            content_words = nltk.word_tokenize(text)
            content_words = nltk.Text(content_words)
            content_tags = nltk.pos_tag(content_words)
            tags_count = Counter(tag for word, tag in content_tags)
            total_tags = sum(tags_count.values())
            tags_count = dict(
                (word, np.round(float(count) / total_tags, 2)) for word, count in tags_count.items())
            content_type_words.append(tags_count)
            if len(text) !=0:
                count_non_letters.append((len(re.sub(r'[^\w\s]', '', strip_tags(str(soup)).lower())) - len(text))/len(text))
            else:
                count_non_letters.append(len(re.sub(r'[^\w\s]', '', strip_tags(str(soup)).lower())))

        content_data_attr = pd.DataFrame({
                                    'site_dep': site_dep,
                                    'content_count': content_count,
                                    'content_length': content_length,
                                    'content_type_words': content_type_words,
                                    'main_div_sections': main_div_sections,
                                    'count_commas': count_commas,
                                    'count_vosclicanie': count_vosclicanie,
                                    'count_non_letters': count_non_letters,
                                    'language': language, 'author_real_name': author_real_name,
                                    })

        return content_data_attr


# ------- function to extract information from h1-h6 tags -------
def extract_h1_h6_features(path_to_data, train=True):

    def h1_extraction(hn_content, hn_length, tag, hn_count, hn_t):
        hn_text = soup.find_all(str(tag))
        hn_count.append(len(hn_text))
        for h1 in hn_text:
            hn_t = hn_t + ' ' + strip_tags(str(h1))
        hn_content.append(re.sub(r'[^\w\s]', '', hn_t.lower()))
        hn_length.append(len(hn_content[-1]))

    if train:
        inp_filename = 'train.json'
    else:
        inp_filename = 'test.json'

    with open(os.path.join(path_to_data, inp_filename), encoding='utf-8') as inp_json_file:
        count_lines = 0

        h1_content, h1_length, h1_count = [], [], []
        h2_content, h2_length, h2_count = [], [], []
        h3_content, h3_length, h3_count = [], [], []
        h4_content, h4_length, h4_count = [], [], []
        h5_content, h5_length, h5_count = [], [], []
        h6_content, h6_length, h6_count = [], [], []

        for line in tqdm_notebook(inp_json_file):
            print('extract_h1_h6_features ' + str(train) + ': ' + str(count_lines))
            count_lines += 1
            json_data = read_json_line(line)
            h1_t = ''
            h2_t = ''
            h3_t = ''
            h4_t = ''
            h5_t = ''
            h6_t = ''

            soup = BeautifulSoup(json_data['content'], "lxml")

            h1_extraction(h1_content, h1_length, 'h1', h1_count, h1_t)
            h1_extraction(h2_content, h2_length, 'h2', h2_count, h2_t)
            h1_extraction(h3_content, h3_length, 'h3', h3_count, h3_t)
            h1_extraction(h4_content, h4_length, 'h4', h4_count, h4_t)
            h1_extraction(h5_content, h5_length, 'h5', h5_count, h5_t)
            h1_extraction(h6_content, h6_length, 'h6', h6_count, h6_t)

        content_data_h1_h6 = pd.DataFrame({
                                           'h1_content': h1_content, 'h1_count': h1_count, 'h1_length': h1_length,
                                           'h2_content': h2_content, 'h2_count': h2_count, 'h2_length': h2_length,
                                           'h3_content': h3_content, 'h3_count': h3_count, 'h3_length': h3_length,
                                           'h4_content': h4_content, 'h4_count': h4_count, 'h4_length': h4_length,
                                           'h5_content': h5_content, 'h5_count': h5_count, 'h5_length': h5_length,
                                           'h6_content': h6_content, 'h6_count': h6_count, 'h6_length': h6_length,
                                          })

        return content_data_h1_h6


# ------- function to extract tags and image information from html code -------
def extract_tags_img_features(path_to_data, train=True):
    if train:
        inp_filename = 'train.json'
    else:
        inp_filename = 'test.json'

    with open(os.path.join(path_to_data, inp_filename), encoding='utf-8') as inp_json_file:
        count_lines = 0
        article_tags = []
        article_tags_length = []
        img_length = []

        for line in tqdm_notebook(inp_json_file):
            print('extract_tags_img_features ' + str(train) + ': ' + str(count_lines))
            count_lines += 1
            json_data = read_json_line(line)

            soup = BeautifulSoup(json_data['content'], "lxml")
            all_img = soup.find_all('img')
            img_length.append(len(all_img))
            tags = soup.find_all('div', {'class': 'col u-size12of12 js-postTags'})
            if len(tags) > 0:
                tags = tags[0].find_all('li')
                tags = [re.sub(r'[^\w\s]', '', strip_tags(str(cur_tag)).lower())
                        for cur_tag in tags]
                article_tags.append(tags)
                article_tags_length.append(len(tags))
            else:
                article_tags.append('unknown')
                article_tags_length.append(0)

        tags_img_attr = pd.DataFrame({
                                      'article_tags': article_tags,
                                      'article_tags_length': article_tags_length,
                                      'img_length': img_length
                                     })

        return tags_img_attr


# ------- function to extract text content from html -------
def extract_content(path_to_data, train=True):
    if train:
        inp_filename = 'train.json'
    else:
        inp_filename = 'test.json'
    content_list = []
    with open(os.path.join(path_to_data, inp_filename), encoding='utf-8') as inp_json_file:
        count_lines = 0
        for line in tqdm_notebook(inp_json_file):
            print('extract_content '+str(train)+': ' + str(count_lines))
            count_lines += 1
            json_data = read_json_line(line)
            content = json_data['content'].replace('\n', ' ').replace('\r', ' ').lower()
            content_no_html_tags = strip_tags(content)
            content_list.append(content_no_html_tags)

    return content_list


# ------- function to define length of content -------
def type_of_length(threshold_values, x):
    if x < threshold_values[0]:
        length_type = 'short'
    elif threshold_values[0] <= x < threshold_values[1]:
        length_type = 'middle1'
    elif threshold_values[1] <= x < threshold_values[2]:
        length_type = 'middle2'
    else:
        length_type = 'long'

    return length_type


# ------- create types of words datasets -------
def create_types_of_words(dataset):
    word_type_df = pd.DataFrame({})
    for i in range(len(dataset)):
        cur_dict = dataset['content_type_words'].iloc[i]
        data_dict = ast.literal_eval(cur_dict)
        if len(list(data_dict.keys())) > 0:
            cur_feat_df = pd.DataFrame(data_dict, index=[0])
            word_type_df = pd.concat([word_type_df, cur_feat_df])
        else:
            cur_feat_df = pd.DataFrame(np.zeros([1, word_type_df.shape[1]]), columns=word_type_df.columns.tolist())
            word_type_df = pd.concat([word_type_df, cur_feat_df])
    return word_type_df


# ------- extract content features from html files -------
train_content_attr = extract_content_features(PATH_TO_DATA, train=True)
train_content_attr.to_csv(PATH_TO_DATA+'train_content_attr.csv', index=False, encoding='utf-8')
test_content_attr = extract_content_features(PATH_TO_DATA, train=False)
test_content_attr.to_csv(PATH_TO_DATA+'test_content_attr.csv', index=False, encoding='utf-8')

# ------- extract titles features from html files -------
train_title_attr = extract_title_features(PATH_TO_DATA, train=True)
train_title_attr.to_csv(PATH_TO_DATA+'train_title_attr.csv', index=False, encoding='utf-8')
test_title_attr = extract_title_features(PATH_TO_DATA, train=False)
test_title_attr.to_csv(PATH_TO_DATA+'test_title_attr.csv', index=False, encoding='utf-8')

# ------- extract h1-h6 features from html files -------
train_h1_h6_attr = extract_h1_h6_features(PATH_TO_DATA, train=True)
train_h1_h6_attr.to_csv(PATH_TO_DATA+'train_h1-h6_attr.csv', index=False, encoding='utf-8')
# while extracting and saving test h1_h6 features the error occuries in dataset at 2307 row
# which we have to fix manually: to cut and paste values from 2308 to 2307 row
# starting with H column (replacing old content with that from 2308 row) or download ready
# dataframe "test_h1-h6_attr.csv" from here: https://goo.gl/oUF6ff
test_h1_h6_attr = extract_h1_h6_features(PATH_TO_DATA, train=False)
test_h1_h6_attr.to_csv(PATH_TO_DATA+'test_h1-h6_attr.csv', index=False, encoding='utf-8')

# ------- extract tags and img features from html files -------
train_tags_img_attr = extract_tags_img_features(PATH_TO_DATA, train=True)
train_tags_img_attr.to_csv(PATH_TO_DATA+'train_tags_attr.csv', index=False, encoding='utf-8')
test_tags_img_attr = extract_tags_img_features(PATH_TO_DATA, train=False)
test_tags_img_attr.to_csv(PATH_TO_DATA+'test_tags_attr.csv', index=False, encoding='utf-8')

# ------- create sparse feature matrix from tags -------
train_tags_img_attr = pd.read_csv(PATH_TO_DATA+'train_tags_attr.csv')
test_tags_img_attr = pd.read_csv(PATH_TO_DATA+'test_tags_attr.csv')
train_tags = train_tags_img_attr['article_tags'].fillna('unknown').tolist()
test_tags = test_tags_img_attr['article_tags'].fillna('unknown').tolist()
cv = TfidfVectorizer(ngram_range=(1,1), max_features=150000).fit(train_tags)
train_tags = cv.transform(train_tags)
test_tags = cv.transform(test_tags)
full_tags = sparse.vstack([train_tags, test_tags])
sparse.save_npz(PATH_TO_DATA+'full_tags.npz', full_tags)

# ------- extract text from html files -------
train_raw_content = extract_content(PATH_TO_DATA, train=True)
with open(PATH_TO_DATA+'train_content', 'wb') as fp:
    pickle.dump(train_raw_content, fp)

test_raw_content = extract_content(PATH_TO_DATA, train=False)
with open(PATH_TO_DATA+'test_content', 'wb') as fp:
    pickle.dump(test_raw_content, fp)

# ------- create sparse matrix with 1-3 ngrams -------
print('vectorize_train_content')
with open(PATH_TO_DATA+'train_content', 'rb') as fp:
     train_raw_content = pickle.load(fp)
vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=150000).fit(train_raw_content)
X_train = vectorizer.transform(train_raw_content)
sparse.save_npz(PATH_TO_DATA+'X_train.npz', X_train)

print('vectorize_test_content')
with open(PATH_TO_DATA+'test_content', 'rb') as fp:
    test_raw_content = pickle.load(fp)
X_test = vectorizer.transform(test_raw_content)
sparse.save_npz(PATH_TO_DATA+'X_test.npz', X_test)

# ------- create full content features table-------
train_content_attr = pd.read_csv(PATH_TO_DATA+'train_content_attr.csv')
# add additional feature based on 4 quintiles of length of content
train_content_length = train_content_attr['content_length'].describe()
train_threshold_values = train_content_length.iloc[4:7].tolist()
train_content_attr['type_of_length'] = train_content_attr['content_length'].\
    apply(lambda x: type_of_length(train_threshold_values, x))


test_content_attr = pd.read_csv(PATH_TO_DATA+'test_content_attr.csv')
# add additional feature based on 4 quintiles of length of content
test_content_length = test_content_attr['content_length'].describe()
test_threshold_values = test_content_length.iloc[4:7].tolist()
test_content_attr['type_of_length'] = test_content_attr['content_length'].\
    apply(lambda x: type_of_length(test_threshold_values, x))

full_cont_attr = pd.concat([train_content_attr, test_content_attr])

# ------- create full title features table-------
train_title_attr = pd.read_csv(PATH_TO_DATA+'train_title_attr.csv')
test_title_attr = pd.read_csv(PATH_TO_DATA+'test_title_attr.csv')
full_title_attr = pd.concat([train_title_attr, test_title_attr])

# ------- create full tag+img features table-------
train_tag_img_attr = pd.read_csv(PATH_TO_DATA+'train_tags_attr.csv')
test_tag_img_attr = pd.read_csv(PATH_TO_DATA+'test_tags_attr.csv')
full_tag_img_attr = pd.concat([train_tag_img_attr, test_tag_img_attr])

# ------- create all features table-------
full_attr = pd.concat([full_cont_attr, full_title_attr, full_tag_img_attr], axis=1)
full_attr.to_csv(PATH_TO_DATA+'full_attr.csv', index=False)

''' 
Load all features, train model and predict
'''

full_attr = pd.read_csv(PATH_TO_DATA+'full_attr.csv')

#  -------index to split train and test rows -------
idx_split = 62313

# ------- add gender feature to full attr -------
d = gender.Detector()
names = full_attr['author_real_name'].tolist()
gender = [[d.get_gender(str(x)) for x in name.split()] for name in names]
genders = []
for i in range(len(gender)):
    cur_list = gender[i]
    if 'female' in str(cur_list):
        genders.append('female')
    elif 'male' in str(cur_list):
        genders.append('male')
    else:
        genders.append('unknown')
full_attr['gender'] = genders

# ------- h1_h6 features aren`t added to full_attr to avoid conflicts -------
train_h1_h6_attr = pd.read_csv(PATH_TO_DATA+'train_h1-h6_attr.csv')
test_h1_h6_attr = pd.read_csv(PATH_TO_DATA+'test_h1-h6_attr.csv')

# ------- load tags matrix -------
full_tags = sparse.load_npz(PATH_TO_DATA+'full_tags.npz')
X_train_full_tags = full_tags[:idx_split, :]
X_test_full_tags = full_tags[idx_split:, :]

# ------- load main vectorized 150000 1-3 ngrams as sparse matrix -------
X_train = sparse.load_npz(PATH_TO_DATA+'X_train.npz')
X_test = sparse.load_npz(PATH_TO_DATA+'X_test.npz')

# ------- load train target values -------
train_target = pd.read_csv(os.path.join(PATH_TO_DATA, 'train_log1p_recommends.csv'),
                               index_col='id')
y_train = train_target['log_recommends'].values

# ------- split train data to train and valid -------
train_part_size = int(0.7 * train_target.shape[0])
X_train_part = X_train[:train_part_size, :]
y_train_part = y_train[:train_part_size]
X_valid = X_train[train_part_size:, :]
y_valid = y_train[train_part_size:]

# ------- stack tags -------
X_train_part = sparse.hstack([X_train_part, X_train_full_tags[:train_part_size, :]])
X_valid = sparse.hstack([X_valid, X_train_full_tags[train_part_size:, :]])
X_test = sparse.hstack([X_test, X_test_full_tags])

# ------- check shapes -------
print(X_train_part.shape, X_valid.shape, X_test.shape)

# -------add numerical features-------
print('add numerical feat')
# list of features to add
num_features = [
        'meta_tags_reading_time',
        'count_non_letters',
        'count_commas',
        # 'count_vosclicanie', # the result got worse
        'main_div_sections',
        'h3_count',
        'h4_count',
        'content_length',
        'img_length',
        'title_length',
        'h1_length',
        'content_count',
        # 'article_tags_length' # the result got worse
    ]

# loop to stack features.
for nummy in num_features:
    # stack h1_h6 features
    if nummy[0] == 'h':
        train_data = train_h1_h6_attr[nummy].fillna(0).apply(lambda x: 1 if x == 0 else x).apply(np.log).values
        test_data = test_h1_h6_attr[nummy].fillna(0).apply(lambda x: 1 if x == 0 else x).apply(np.log).values
        cur_nummies = StandardScaler().fit_transform(train_data.reshape(-1, 1))
        X_train_part = sparse.hstack([X_train_part, cur_nummies[:train_part_size, :]])
        X_valid = sparse.hstack([X_valid, cur_nummies[train_part_size:, :]])
        cur_test_nummies = StandardScaler().fit_transform(test_data.reshape(-1, 1))
        X_test = sparse.hstack([X_test, cur_test_nummies])
    # stack title and content features
    else:
        train_data = full_attr[nummy].fillna(0).apply(lambda x: 1 if x == 0 else x).apply(np.log).fillna(0).values[
                     :idx_split]
        test_data = full_attr[nummy].fillna(0).apply(lambda x: 1 if x == 0 else x).apply(np.log).fillna(0).values[
                    idx_split:]
        cur_nummies = StandardScaler().fit_transform(train_data.reshape(-1, 1))
        X_train_part = sparse.hstack([X_train_part, cur_nummies[:train_part_size, :]])
        X_valid = sparse.hstack([X_valid, cur_nummies[train_part_size:, :]])
        cur_test_nummies = StandardScaler().fit_transform(test_data.reshape(-1, 1))
        X_test = sparse.hstack([X_test, cur_test_nummies])

# ------- create and add dummies features -------
print('get_dummies')
dummies_features = [
    'site_dep',
    'author',
    'type_of_length',
    'gender',
    'author_real_name',
    'language',
]
for feat in dummies_features:
    cur_dummies = pd.get_dummies(full_attr[feat], prefix=feat)
    if 'gender_unknown' in str(cur_dummies.columns.tolist()):
        cur_dummies = cur_dummies.drop(['gender_unknown'], axis=1)
    cur_train_dummies = cur_dummies.values[:idx_split, :]
    cur_test_dummies = cur_dummies.values[idx_split:, :]
    X_train_part = sparse.hstack([X_train_part, cur_train_dummies[:train_part_size, :]])
    X_valid = sparse.hstack([X_valid, cur_train_dummies[train_part_size:, :]])
    X_test = sparse.hstack([X_test, cur_test_dummies])

# -------  add time_features -------
def day_period(x):
    period = 'unknown'
    if 6 <= x < 12:
        period = 'morning'
    elif 12 <= x < 18:
        period = 'day'
    elif 18 <= x <= 23:
        period = 'evening'
    else:
        period = 'night'

    return period


def get_time_features(data):
    train_time_features = [
                           pd.get_dummies(data['published'].apply(pd.to_datetime).dt.weekday, prefix='weekday'),
                           # pd.get_dummies(data['published'].apply(pd.to_datetime).dt.hour, prefix='weekday'),
                           # pd.get_dummies(data['published'].apply(pd.to_datetime).dt.month, prefix='month'),
                           pd.get_dummies(data['published'].apply(pd.to_datetime).apply(lambda x: x.isocalendar()[1]), prefix='week'),
                           # pd.get_dummies(data['published'].apply(pd.to_datetime).dt.year, prefix='year'),
                           pd.get_dummies(data['published'].apply(pd.to_datetime).dt.hour.apply(lambda x: day_period(x)), prefix='day_period')
                           ]

    return train_time_features


print('add time_feat')
full_data_time_features = get_time_features(full_attr)
for time_dummy in full_data_time_features:
    cur_train_dummies = time_dummy.values[:idx_split, :]
    cur_test_dummies = time_dummy.values[idx_split:, :]
    X_train_part = sparse.hstack([X_train_part, cur_train_dummies[:train_part_size, :]])
    X_valid = sparse.hstack([X_valid, cur_train_dummies[train_part_size:, :]])
    X_test = sparse.hstack([X_test, cur_test_dummies])

# ----- train and predict on partitial X_train data -----
print('Ridge')
ridge = Ridge(random_state=17, alpha=2.2)
ridge.fit(X_train_part, y_train_part)
ridge_pred = ridge.predict(X_valid)
print(ridge_pred)
valid_mae = mean_absolute_error(y_valid, ridge_pred)
print(valid_mae, np.expm1(valid_mae))
r2_score_lasso = r2_score(y_valid, ridge_pred)
print("r^2 on test data : %f" % r2_score_lasso)

# ----- train on whole X_train -----
# stack back X_train and X_valid
X_train = sparse.vstack([X_train_part, X_valid])
print('X_train and X_test shapes')
print(X_train.shape, X_test.shape)
# fit and predict
ridge.fit(X_train, y_train)
ridge_test_pred = ridge.predict(X_test)

# ----- write submission file -----
def write_submission_file(prediction, filename,
                          path_to_sample=os.path.join(PATH_TO_DATA, 'sample_submission.csv')):
    submission = pd.read_csv(path_to_sample, index_col='id')

    submission['log_recommends'] = prediction
    submission.to_csv(filename)


write_submission_file(prediction=ridge_test_pred+1, filename='second_ridge.csv')

# show hist y_valid vs y_predict
plt.hist(y_train, bins=30, alpha=.3, color='red', label='true', range=(0, 10))
plt.hist(ridge_pred, bins=30, alpha=.3, color='blue', label='test', range=(0, 10))
plt.legend()
plt.show()
