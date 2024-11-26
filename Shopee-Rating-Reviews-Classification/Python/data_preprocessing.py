import pandas as pd
import numpy as np
import malaya
import string as s
import langid
import contractions
import re
import stopwordsiso as stopwords
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from rake_nltk import Rake
from collections import OrderedDict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
model = malaya.pos.transformer(model='albert')


# import stopwords
malay = stopwords.stopwords("ms")
ms_stop_list = []
for i in malay:
    ms_stop_list.append(i)

eng = stopwords.stopwords("en")
en_stop_list = []
for i in eng:
    en_stop_list.append(i)

my_extra_en = ['while', 'guess', 'time', 'moving', 'miles', 'bought', 'wife']
my_extra_ms = ['lah', 'buah', 'aku', 'cun', 'mantap', 'haha', 'cuba', 'dia', 'kenapa']
en_stop_list.extend(my_extra_en)
ms_stop_list.extend(my_extra_ms)
exclude_words_ms = ['using', 'bank', 'barangan', 'lain', 'produk', 'bukan', 'kukuh', 'masalah', 'banyak', 'kali', 'terbaik', 'bagus','boleh', 'barang', 'tiada', 'tidak', 'tak', 'membantu', 'tk','tapi','naik','sama','hantar','dapat','baik','kecik']
exclude_words_en = ['hopefully','harap','better','yet','half','could','after','like','already','quickly','runs','run','down','together','use','all','bank','stop','long','worked','well','low','no','can','last','great', "wasn't",'very', 'not','cannot', 'useful', 'awfully', "don't", 'works', 'work','small','less','thin','did','not','poorly','good',"can't","didn't",'everything','ok','okay','length','bank','earphone','charger','cable', 'more','fully','full','do','working','does','longer']
final_stopwords_ms = [w for w in ms_stop_list if w not in exclude_words_ms]
final_stopwords_en = [w for w in en_stop_list if w not in exclude_words_en]

sentilexm_dict = {}

with open('Python/SentiLexM.txt', 'r') as f:
    for line in f:
        key, value = line.strip().split('\t')
        sentilexm_dict[key] = int(value)

remove = ['urgent', 'desak', 'tahan', 'cas', 'charged', 'power', 'nak', 'risau', 'worry', 'cancelling']

positive_words = []
negative_words = []
for key in remove:
    sentilexm_dict.pop(key, None)

for key, value in sentilexm_dict.items():
    if value > 0:
        positive_words.append(key)

    elif value < 0:
        negative_words.append(key)


def detect_language(text):
    langid.set_languages(['ms', 'en'])
    lang, _ = langid.classify(text)
    return lang


def clean_text(text):
    punct = s.punctuation
    punct = punct.replace('.', '')
    punct = punct.replace(',', '')
    punct = punct.replace('/', '')
    punct = punct.replace('-', '')
    punct = punct.replace('%', '')
    updated_punct = punct + '’‘'

    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)  # remove urls
    text = re.sub(r'[:]', ' ', text)  # remove semicolon with whitespace
    text = re.sub(r'\brm\s*\w+', '', text)  # remove rm
    text = re.sub('\n\n', ',', text)
    text = re.sub('\n', ',', text)
    text = re.sub(r"<.*?>", '', text)  # remove html tags
    text = re.sub("[^A-Za-z0-9 ,'.-/%,]+", ' ', text)  # remove unrecognizable alphabets
    text = re.sub('[(?!.*?\.\.).*?$]+', ' . ', text)  # remove consecutive dots with 1 dot
    text = re.sub(' +', ' ', text)  # remove extra whitespaces

    clean = "".join([i for i in text if i not in updated_punct])
    return clean


def remove_lengthy_words(text):
    filtered_list = []
    tokens = word_tokenize(text)
    for token in tokens:
        if len(token) < 13:
            filtered_list.append(token)

    return ' '.join(filtered_list).strip()


def correct_typos(text):
    custom_dict = {
        'nape': 'kenapa',
        'terai': 'try',
        'guana': 'macam mana',
        'wayer': 'wayar',
        'xleh': 'tak boleh',
        'kaler': 'warna',
        'ttp': 'tetap',
        'bosku': '',
        'boek': 'baik',
        'cinonet': 'kecik',
        'chas': 'cas',
        'jgak': 'jugak',
        'bgtau': 'bagitau',
        '²': '2',
        'xpress': 'express',
        'semlm': 'semalam',
        'pwettyyy': 'pretty',
        'pwettyy': 'pretty',
        'laen': 'lain',
        'xde': 'tiada',
        'jew': 'sahaja',
        'pn': 'pun',
        'selar': 'seller',
        'menit': 'minit',
        'dlam': 'dalam',
        'berpe': 'berapa',
        'shj': 'sahaja',
        'sgt': 'sangat',
        'mao': 'mahu',
        'mau': 'mahu',
        'slpas': 'selepas',
        'jga': 'juga',
        'mase': 'masa',
        'tggu': 'tunggu',
        'taktaw': 'tak tahu',
        'saler': 'seller',
        'selpas': 'selepas',
        'sngt': 'sangat',
        'dlm': 'dalam',
        'pawer': 'power',
        'hp': 'fon',
        'lg': 'lagi',
        'nie': 'ni',
        'sekli': 'sekali',
        'ble': 'boleh',
        'x': 'tak',
        'dh': 'dah',
        'ip': 'iphone',
        'n': 'and',
        'dun': 'do not',
        'coz': 'because',
        'cuz': 'because',
        'bcs': 'because',
        'jgn': 'jangan',
        'nnt': 'nanti',
        'nnti': 'nanti',
        'yg': 'yang',
        'dkt': 'dekat',
        'sbb': 'sebab',
        'dgn': 'dengan',
        'tlng': 'tolong',
        'tnx': 'thanks',
        'tdk': 'tidak',
        'srs': 'serious',
        'tq': 'thank you',
        'takde': 'tiada',
        'xda': 'tiada',
        'dye': 'dia',
        'die': 'dia',
        'lmbt': 'lambat',
        'msuk': 'masuk',
        'masok': 'masuk',
        'msok': 'masuk',
        'ckit': 'sikit',
        'tkde': 'tiada',
        'skit': 'sikit',
        'ori': 'original',
        'brg': 'barang',
        'barng': 'barang',
        'jer': 'sahaja',
        'mslh': 'masalah',
        'bgkus': 'bungkus',
        'blikan': 'belikan',
        'lak': 'pulak',
        'brp': 'berapa',
        'dak': 'ada',
        'tp': 'tetapi',
        'pape': 'apa-apa',
        'mmg': 'memang',
        'gmbr': 'gambar',
        'gbar': 'gambar',
        'oke': 'ok',
        'shoppee': 'shopee',
        'bile': 'bila',
        'bru': 'baru',
        'smpi': 'sampai',
        'mgu': 'minggu',
        'mnggu': 'minggu',
        'cpt': 'cepat',
        'mur': 'murah',
        'mcm': 'macam',
        'penghntaran': 'penghantaran',
        'penhantaran': 'penghantaran',
        'penhntaran': 'penghantaran',
        'saller': 'seller',
        'dhlh': 'dah lah',
        'muroh': 'murah',
        'jnji': 'janji',
        'thnks': 'thanks',
        'beteri': 'bateri',
        'packinging': 'packaging',
        'chrging': 'charging',
        'knpa': 'kenapa',
        'smlm': 'semalam',
        'jgk': 'jugak',
        'pnghntrn': 'penghantaran',
        'rekemen': 'recommend',
        'bole': 'boleh',
        'cube': 'cuba',
        'hntar': 'hantar',
        'brfungsi': 'berfungsi',
        'hantr': 'hantar',
        'bcoz': 'because',
        'bcos': 'because',
        'suke': 'suka',
        'naise': 'nice',
        'nais': 'nice',
        'blom': 'belum',
        'coba': 'cuba',
        'biashe': 'biasa',
        'pon': 'pun',
        'sy': 'saya',
        'shoppe': 'shopee',
        'satiesfied': 'satisfied',
        'dg': 'dengan',
        'okehh': 'okay',
        'okey': 'okay',
        'okeylah': 'okay',
        'puah': 'puas',
        'mmng': 'memang',
        'delevri': 'delivery',
        'mujar': 'mujur',
        'commend': 'comment',
        'obvi': 'obviously',
        'vid': 'video',
        'chrge': 'charge',
        'desc': 'description',
        'chrg': 'charge',
        'charg': 'charge',
        'soun': 'sound',
        'meterial': 'material',
        'betol': 'betul',
        'act': 'actually',
        'dpt': 'dapat',
        'noice': 'noise'

    }

    corrected = []
    tokens = word_tokenize(text)
    for token in tokens:
        if token in custom_dict:
            corrected.append(custom_dict[token])
        else:
            corrected.append(token)

    return ' '.join(corrected).strip()


def spell_checker(text):
    normalizer = malaya.normalize.normalizer()
    sentence = normalizer.normalize(text, normalize_entity=True, normalize_number=False)
    checked = sentence['normalize']
    return checked


def expand_contractions(text):
    corrected = []

    tokens = word_tokenize(text)

    for token in tokens:
        language = detect_language(token)

        if language == 'en':
            checked = contractions.fix(token)
            corrected.append(checked)
        else:
            corrected.append(token)

    return ' '.join(corrected).strip()


def remove_stopwords(text):
    corrected = []
    tokens = word_tokenize(text)
    for token in tokens:
        language = detect_language(token)

        if language == 'ms':
            if token not in final_stopwords_ms:
                corrected.append(token)

        elif language == 'en':
            if token not in final_stopwords_en:
                corrected.append(token)

    return ' '.join(corrected).strip()


def extract_phrase(text):
    language = detect_language(text)

    if language == 'en':
        r = Rake(
            stopwords=final_stopwords_en
        )
        r.extract_keywords_from_text(text)
        keyphrases = r.get_ranked_phrases()
        return keyphrases

    elif language == 'ms':
        keyphrases = malaya.keyword.extractive.rake(text, top_k=10, stopwords=final_stopwords_ms)
        keyphrases = [phrase for _, phrase in keyphrases]

        return keyphrases


def filtered(keyphrases):
    powerbank_aspects = ['power bank', 'magnet', 'power', 'type', 'jenis', 'usb', 'battery', 'bateri', 'charged',
                         'charging', 'caj', 'cas', 'penuh' ,'output', 'bank', 'size', 'saiz', 'micro', 'powerbank', 'polymer', 'input',
                         'lithium', 'cable', 'kabel', 'portable', 'phone', 'fon', 'charge', 'temperature', 'suhu',
                         'feature', 'mobile', 'interface', 'warranty', 'port', 'device', 'supply', 'capacity', 'length', 'fungsi',
                         '20kmah', '20000mah', '10000mah', '10kmah', '50000mah', '50kmah','protection', 'resistant', 'storage', 'electronic', 'function',
                         'security', 'material', 'height', 'energy', 'led', 'indicator']
    wireless_earphones_aspects = ['bluetooth', 'caj', 'charging', 'casing', 'battery', 'bateri', 'headphone', 'charged',
                                  'touch', 'type', 'frequency', 'response', 'ear', 'telinga', 'bunyi', 'sound', 'mic', 'microphone',
                                  'control', 'earphone', 'wireless', 'voice', 'suara', 'version', 'versi', 'deep', 'base',
                                  'mikrofon','bass', 'earbud', 'dual', 'cable', 'kabel', 'jam', 'hour', 'capacity', 'sensitivity',
                                  'maximum', 'power', 'connect', 'daya', 'pairing', 'led', 'longer', 'mode', 'function',
                                  'switch', 'noise cancelling', 'beat']
    cable_charger_aspects = ['cable', 'charging', 'usb', 'micro', 'support', 'charger', 'charged', 'device', 'phone',
                             'fon', 'braided', 'charge', 'type', 'jenis', 'current', 'material', 'nylon', 'warranty',
                             'interface', 'wire', 'digital', 'power', 'protection', 'kualiti', 'quality',
                             'transmission', 'premium', 'exterior', 'circuit', 'amp', 'pull', 'surge', 'length',
                             'fast charger', 'function', 'transfer', 'masuk']
    overall_aspects = []
    overall_aspects.extend(powerbank_aspects)
    overall_aspects.extend(wireless_earphones_aspects)
    overall_aspects.extend(cable_charger_aspects)
    overall_aspects.extend(['performance', 'durability'])

    # Initialize an empty list to store matching aspects
    filtered = []

    for phrase in keyphrases:
        # Check for matching aspects
        for aspect in overall_aspects:
            if re.search(r'\b' + re.escape(aspect) + r'\b', phrase, flags=re.IGNORECASE):
                if phrase not in filtered:
                    filtered.append(phrase)

    if filtered:
        return filtered
    else:
        return ['review unrelated']


def lemmatizer(phrases):

    lemmatizer = WordNetLemmatizer()
    text = ' '.join(phrases).strip()

    tagged_tokens = model.predict(text)

    tokens = word_tokenize(text)
    en_tagged_tokens = pos_tag(tokens)

    lemma = []

    for i, token in enumerate(tokens):

        foundtag = 'n'
        language = detect_language(token)

        if language == 'en':
            word, tag = en_tagged_tokens[i]
            if tag == 'NOUN':
                foundtag = 'n'

            elif tag == 'ADJ':
                foundtag = 'a'

            elif tag == 'VERB':
                foundtag = 'v'

            elif tag == 'ADV':
                foundtag = 'a'

        elif language == 'ms':
            word, tag = tagged_tokens[i]
            if tag == 'NOUN':
                foundtag = 'n'

            elif tag == 'ADJ':
                foundtag = 'a'

            elif tag == 'VERB':
                foundtag = 'v'

            elif tag == 'ADV':
                foundtag = 'a'

        lemma.append(lemmatizer.lemmatize(token, pos=foundtag))

    return ' '.join(lemma).strip()


def calculate_sentiment_score(text):
    list_pos = []
    list_neg = []
    neg_pos = []
    negated = []
    file = open('Python/negation-words.txt', 'r', encoding='cp1252')
    neg_words = file.read().split(',')
    neg_malay = ['tiada', 'tidak', 'tak', 'kurang', 'bukan']
    neg_words.extend(neg_malay)

    words = word_tokenize(text)
    for i, w in enumerate(words):
        if w in neg_words:
            try:
                if words[i + 1] in negative_words:
                    list_pos.append(w)
                    neg_pos.append(words[i + 1])
                elif words[i + 1] in positive_words:
                    list_neg.append(w)
                    negated.append(words[i + 1])
                elif words[i + 1] not in negative_words and words[i + 1] not in positive_words:
                    list_neg.append(w)
            except IndexError:
                pass
        elif w in negative_words and w not in list_pos and w not in list_neg and w not in neg_pos:
            list_neg.append(w)

        try:
            if w in positive_words and words[i + 1] in negative_words:
                list_neg.append(words[i + 1])
            elif w in positive_words and w not in list_pos and w not in negated:
                list_pos.append(w)

        except IndexError:
            if w in positive_words and w not in list_pos and w not in negated:
                list_pos.append(w)

    pos_count = len(list_pos)
    neg_count = len(list_neg)
    total_count = pos_count + neg_count

    if total_count != 0:
        polarity = round((pos_count - neg_count) / total_count, 2)
        return polarity
    else:
        return 0


def check_null(df):

    df.dropna(inplace=True)
    return df


def pre_processing(df):

    df = check_null(df)
    df['lower'] = df['review'].str.lower()
    print('lowered')
    df['cleaned'] = df['lower'].apply(lambda text: clean_text(text))
    print('cleaned')
    df['removed_lengthy_words'] = df['cleaned'].apply(lambda x: remove_lengthy_words(x))
    print('lengthy words removed')
    df['correct_typos'] = df["removed_lengthy_words"].apply(lambda x: correct_typos(x))
    print('typos corrected')
    df['correct_spelling'] = df["correct_typos"].apply(lambda x: spell_checker(x))
    print('correct spelling')
    df['expanded'] = df["correct_spelling"].apply(lambda x: expand_contractions(x))
    print('expanded')
    df['removed_stopwords'] = df["expanded"].apply(lambda x: remove_stopwords(x))
    print('remove stopwords')
    df['extracted'] = df['removed_stopwords'].apply(lambda x: extract_phrase(x))
    print('extracted')
    df['filtered'] = df['extracted'].apply(lambda x: filtered(x))
    print('filtered')
    df['finalized'] = df['filtered'].apply(lambda x: lemmatizer(x))
    print('lemmatized')

    # Calculate content or cosine similarity within text
    review_data = df
    res = OrderedDict()

    # Iterate over data and create groups of reviewers
    for row in review_data.iterrows():
        if row[1].username in res:
            res[row[1].username].append(row[1].cleaned)
        else:
            res[row[1].username] = [row[1].cleaned]

    individual_reviewer = [{'username': k, 'cleaned': v} for k, v in res.items()]

    df1 = dict()
    df1['username'] = pd.Series([])
    df1['duplicated_spam'] = pd.Series([])

    vector = TfidfVectorizer(min_df=0)
    count = -1
    for reviewer_data in individual_reviewer:
        count = count + 1
        try:
            tfidf = vector.fit_transform(reviewer_data['cleaned'])

        except:
            pass
        cosine = 1 - pairwise_distances(tfidf, metric='cosine')
        np.fill_diagonal(cosine, -np.inf)
        max_cos = cosine.max()

        # To handle reviewer with just 1 review
        if max_cos == -np.inf:
            max_cos = 0
        elif max_cos == 1:
            max_cos = 1
        else:
            max_cos = 0

        df1['username'][count] = reviewer_data['username']
        df1['duplicated_spam'][count] = max_cos

    df2 = pd.DataFrame(df1, columns=['username', 'duplicated_spam'])
    newdf = df.merge(df2, on=['username'])
    newdf['sentiment_score'] = newdf['finalized'].apply(lambda text: calculate_sentiment_score(text))
    newdf.drop(
        ['lower', 'cleaned', 'removed_lengthy_words', 'correct_typos', 'correct_spelling', 'expanded', 'extracted'],
        axis=1, inplace=True)

    newdf.rename(columns={'removed_stopwords': 'cleaned'}, inplace=True)

    return newdf

