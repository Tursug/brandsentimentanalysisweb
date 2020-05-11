import os
import io
import time
import re
import nltk
import glob
import requests
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import csv

from nltk.corpus import stopwords as sw
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, render_template, request, flash, redirect, send_file, send_from_directory
from werkzeug.utils import secure_filename
from textblob import TextBlob
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Embedding, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn import naive_bayes
from sklearn.metrics import roc_auc_score
from wordcloud import WordCloud, STOPWORDS
from keras import backend as K
from itertools import islice
from bs4 import BeautifulSoup

CSV_EXTENSION = {'csv'}
H5_EXTENSION = {'h5'}

UPLOAD_FOLDER = "static/uploads"

working_place = os.path.join("c:", os.sep, "Users", "dogacanbicer", "PyCharmProjects", "BrandSentimentAnalysis",
                             "static", "uploads")

os.chdir(working_place)
nltk.download('stopwords')

app = Flask(__name__)
app.secret_key = 'secret key'

app.config['MAX_CONTENT_LENGTH'] = 250 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def csv_extension(file_name):
    return '.' in file_name and \
           file_name.rsplit('.', 1)[1].lower() in CSV_EXTENSION


def h5_extension(file_name):
    return '.' in file_name and \
           file_name.rsplit('.', 1)[1].lower() in H5_EXTENSION


def pre_process(df, num):
    if num == 2:
        frame_new = pd.DataFrame(columns=['SentimentText'])
    elif num == 3:
        frame_new = pd.DataFrame(columns=['Sentiment', 'SentimentText'])

    content_array = []

    for index, row in islice(df.iterrows(), 0, None):

        content = str(row['SentimentText'])
        content = re.sub(
                r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''',
                " ", content)
        content = re.sub(r"http\S+", "", content)
        content = re.sub(r"@\S+", "", content)
        content = re.sub(r"#\S+", "", content)
        content = re.sub(r'\w*\d\w*', '', content).strip()
        content = re.sub(r'[^\w\s]', '', content)
        content = re.sub(' +', ' ', content)
        content = content.lower()

        if len(content) > 0:
            content_array.append(content)

            for i in range(0, len(content_array)):
                if num == 2:
                    frame_new.at[i] = content_array[i]
                elif num == 3:
                    frame_new.at[i, 'SentimentText'] = content_array[i]
                    frame_new.at[i, 'Sentiment'] = df.at[i, 'Sentiment']

    return frame_new


def represents_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def return_new_name(passed_name, passed_extension):

    file_list = []

    for file in glob.glob('*.'+passed_extension):
        file_list.append(file)

    returned_name = passed_name
    number_max = 0
    maxed = 0
    print(file_list)
    for i in range(0, len(file_list)):
        basic_file_name = file_list[i][0:int(len(file_list[i]) - 4)]
        if basic_file_name[int(len(basic_file_name) - 1)] == ")":
            j = int(len(basic_file_name) - 1)
            int_str = []
            j = j - 1
            while basic_file_name[j] != "(":
                if j > 0:
                    int_str.append(basic_file_name[j])
                else:
                    break
                j = j - 1
            int_str.reverse()
            int_str = "".join(int_str)
            if represents_int(int_str):
                value = int(int_str)
            else:
                value = 0
            its_name = basic_file_name[0:j]
            if its_name == returned_name:
                number_max = value
                maxed = 1
        elif basic_file_name == returned_name:
            number_max = 1
            break

    if maxed == 1:
        number_max += 1

    if number_max >= 1:
        returned_name = passed_name+'('+str(number_max)+')'
    else:
        returned_name = passed_name

    return returned_name


def test(site):
        regex = re.compile(
            r'^(?:http|ftp)s?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)

        if re.match(regex, site) is not None:
            return True
        else:
            return False

# INDEX
@app.route("/")
def index():
    return render_template("index.html")
# END OF INDEX

# SCRAP
@app.route("/scrap")
def scrap(filename=""):
    return render_template("scrap.html", filename=filename)


@app.route("/scrap", methods=["GET", "POST"])
def create_csv():
    if request.method == 'POST':

        site_name = request.form.get("sitename")
        tag_name = request.form.get("tagname")
        if tag_name == "":
            tag_name = "p"
        cls_name = request.form.get("classname")
        if cls_name == "":
            cls_name = ""
        csv_name = request.form.get("csvname")

        if not test(site_name):
            flash("Error in site name")
            return redirect(request.url)

        r = requests.get(site_name)
        soup = BeautifulSoup(r.content, features="lxml")
        text = ""
        for link in soup.find_all(tag_name, {'class': cls_name}):
            text = text + link.text

        text = text.split(".")
        df = pd.DataFrame(text, columns=['SentimentText'])
        df = pre_process(df, 2)

        extension = "csv"
        name = return_new_name(csv_name, extension)

        my_path = 'static/uploads'
        my_file = name + '.' + extension

        df.to_csv(os.path.join(working_place, my_file), sep=',', encoding='utf-8')

        return send_from_directory(my_path, my_file, as_attachment=True)
    else:
        return render_template("scrap.html")
# END OF SCRAP

# CREATE MODEL
@app.route("/createmodel")
def create_model(filename=""):
    return render_template("createmodel.html", filename=filename)


@app.route("/createmodel", methods=["GET", "POST"])
def create_model_values():
    if request.method == 'POST':

        name = request.form.get("name")
        number_of_words = int(request.form.get("numberofwords"))
        pad_length = int(request.form.get("padlength"))
        # hidden_layers = int(request.form.get("hiddenlayers"))
        epoch = int(request.form.get("epoch"))
        size_batch = int(request.form.get("batchsize"))

        my_path = ""
        my_file = ""

        if 'csvfile' in request.files:

            csv_file = request.files['csvfile']

            if not csv_extension(csv_file.filename):
                flash('Only csv files are supported')
                return redirect(request.url)

            # csv_file.save(os.path.join(app.config['UPLOAD_FOLDER'], csv_file.filename))

            my_path = 'static/uploads'
            my_file = csv_file.filename

            data = pd.read_csv(csv_file)

            '''PART WILL CHANGE'''
            try:
                if data.columns[0] != "Sentiment" and data.columns[1] != "Sentiment":
                    flash('One of the column names must be Sentiment!')
                    return redirect(request.url)
                elif data.columns[0] == "Sentiment":
                    if data.columns[1] != "SentimentText":
                        flash('After the Sentiment column, column name must be SentimentText!')
                        return redirect(request.url)
                elif data.columns[1] == "Sentiment":
                    if data.columns[2] != "SentimentText":
                        flash('After the Sentiment column, column name must be SentimentText!')
                        return redirect(request.url)
            except:
                flash('Error in Columns!')
                return redirect(request.url)
            '''PART WILL CHANGE'''

            unique = set(data['Sentiment'])
            output = len(list(unique))

            if output < 2 or output > 3:
                flash('Error in Sentiments!')
                return redirect(request.url)

            data = data.sample(frac=1).reset_index(drop=True)
            data = data[['Sentiment', 'SentimentText']]

            tokenizer = Tokenizer(num_words=5000, split=" ")
            tokenizer.fit_on_texts(data['SentimentText'].values)

            X = tokenizer.texts_to_sequences(data['SentimentText'].values)
            X = pad_sequences(X, maxlen=32)

            y = pd.get_dummies(data['Sentiment']).values

            K.clear_session()

            model = Sequential()

            model.add(Embedding(5000, 256, input_length=X.shape[1]))
            model.add(Dropout(0.3))
            model.add(LSTM(256, return_sequences=True, dropout=0.3, recurrent_dropout=0.2))
            model.add(LSTM(256, dropout=0.2, recurrent_dropout=0.2))
            model.add(Dense(output, activation='softmax'))

            model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
            model.fit(X, y, epochs=8, batch_size=32, verbose=2)

            extension = "h5"
            name = return_new_name(name, extension)

            my_path = 'static/uploads'
            my_file = name + '.' + extension

            model.save(os.path.join(working_place, my_file))

            K.clear_session()

        return send_from_directory(my_path, my_file, as_attachment=True)
    else:
        return render_template("createmodel.html")
# END OF CREATE MODEL

# NEURAL NETWORK
@app.route("/neuralnetwork")
def neural_network(filename="", image=""):
    return render_template("neuralnetwork.html", filename=filename, image=image)


@app.route('/neuralnetwork', methods=['GET', 'POST'])
def upload_neural_file():

    my_path = ''
    my_file = ''

    if request.method == 'POST':

        if 'modelfile' not in request.files:
            flash('File Not Found!')
            return redirect(request.url)

        model_file = request.files['modelfile']

        if not h5_extension(model_file.filename):
            flash('Only h5 files are supported')
            return redirect(request.url)

        if 'csvfile' not in request.files:
            flash('File Not Found!')
            return redirect(request.url)

        csv_file = request.files['csvfile']

        if not csv_extension(csv_file.filename):
            flash('Only csv files are supported')
            return redirect(request.url)

        name = request.form.get("name")
        data = pd.read_csv(csv_file)

        try:
            if data.columns[0] != "SentimentText":
                if data.columns[1] != "SentimentText":
                    if data.columns[2] != "SentimentText":
                        flash('One of the First three column name must be SentimentText!')
                        return redirect(request.url)
        except:
            flash('One of the column name must be SentimentText!')
            return redirect(request.url)

        data = data.sample(frac=1).reset_index(drop=True)
        # data = data[['Sentiment', 'SentimentText']]
        data = data[['SentimentText']]

        tokenizer = Tokenizer(num_words=5000, split=" ")
        tokenizer.fit_on_texts(data['SentimentText'].values)

        X = tokenizer.texts_to_sequences(data['SentimentText'].values)
        X = pad_sequences(X, maxlen=32)

        K.clear_session()

        model = load_model(model_file)
        predictions = model.predict(X)

        K.clear_session()

        output_dense = len(predictions[0])

        pos_count, neg_count, neu_count = 0, 0, 0

        if output_dense == 2:
            for i, prediction in enumerate(predictions):
                if np.argmax(prediction) == 0:
                    neg_count += 1
                else:
                    pos_count += 1

            exp_vals = [pos_count, neg_count]
            exp_labels = ["positive", "negative"]

        else:
            for i, prediction in enumerate(predictions):
                if np.argmax(prediction) == 2:
                    pos_count += 1
                elif np.argmax(prediction) == 1:
                    neu_count += 1
                else:
                    neg_count += 1

            exp_vals = [pos_count, neu_count, neg_count]
            exp_labels = ["positive", "neutral", "negative"]

        plt.pie(exp_vals, labels=exp_labels)

        extension = "png"
        name = return_new_name(name, extension)

        my_path = 'static/uploads'
        my_file = name + '.' + extension
        plt.savefig(os.path.join(working_place, my_file))
        plt.clf()

    return render_template("neuralnetwork.html", image=str(my_path+'/'+my_file))
# END OF NEURAL NETWORK

# NAIVE BAYES
@app.route("/naivebayes")
def bayes(filename="", image=""):
    return render_template("naivebayes.html", filename=filename, image=image)


@app.route('/naivebayes', methods=['GET', 'POST'])
def upload_bayes_file():

    my_path = ''
    my_file = ''

    if request.method == 'POST':

        if 'csvfile' not in request.files:
            flash('File Not Found!')
            return redirect(request.url)

        if 'trainingcsv' not in request.files:
            flash('File Not Found!')
            return redirect(request.url)

        csv_file = request.files['csvfile']

        if not csv_extension(csv_file.filename):
            flash('Only csv files are supported')
            return redirect(request.url)

        training_file = request.files['trainingcsv']

        if not csv_extension(training_file.filename):
            flash('Only csv files are supported')
            return redirect(request.url)

        name = request.form.get("name")

        df = pd.read_csv(csv_file)

        try:
            if df.columns[0] != "Sentiment" and df.columns[1] != "Sentiment":
                flash('One of the column names must be Sentiment!')
                return redirect(request.url)
            elif df.columns[0] == "Sentiment":
                if df.columns[1] != "SentimentText":
                    flash('After the Sentiment column, column name must be SentimentText!')
                    return redirect(request.url)
            elif df.columns[1] == "Sentiment":
                if df.columns[2] != "SentimentText":
                    flash('After the Sentiment column, column name must be SentimentText!')
                    return redirect(request.url)
        except:
            flash('Error in Columns!')
            return redirect(request.url)

        df_unique = set(df['Sentiment'])
        df_output = len(list(df_unique))

        if df_output < 1 or df_output > 2:
            flash('Error in Sentiments!')
            return redirect(request.url)

        df_one = pd.read_csv(training_file)

        try:
            if df_one.columns[0] != "SentimentText":
                if df_one.columns[1] != "SentimentText":
                    if df_one.columns[2] != "SentimentText":
                        flash('One of the First three column name must be SentimentText!')
                        return redirect(request.url)
        except:
            flash('One of the column name must be SentimentText!')
            return redirect(request.url)

        stopwords = set(sw.words('english'))
        vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii', stop_words=stopwords)

        y = pd.DataFrame(columns=['Sentiment'])

        for i in range(0, len(df_one)):
            if df_one.at[i, 'Sentiment'] == 'positive':
                y.loc[i] = int(1)
            else:
                y.loc[i] = int(0)

        X = vectorizer.fit_transform(df_one.SentimentText)
        y = y.astype('int')
        clf = naive_bayes.MultinomialNB()
        clf.fit(X, y.values.ravel())

        pos_count, neg_count = 0, 0

        for i in range(0, len(df)):

            sent_text = str(df.at[i, 'SentimentText'])

            movie_array = np.array([sent_text])
            movie_review_vector = vectorizer.transform(movie_array)

            if int(clf.predict(movie_review_vector)) == 0:
                neg_count += 1
            else:
                pos_count += 1

        exp_vals = [pos_count, neg_count]
        exp_labels = ["positive", "negative"]

        plt.pie(exp_vals, labels=exp_labels)

        extension = 'png'
        name = return_new_name(name, extension)

        my_path = 'static/uploads'
        my_file = name + '.' + extension

        plt.savefig(os.path.join(working_place, my_file))
        plt.clf()

    return render_template("naivebayes.html", image=str(my_path+'/'+my_file))
# END OF NAIVE BAYES

# DICTIONARY
@app.route("/dictionary")
def dictionary(filename="", image=""):
    return render_template("dictionary.html", filename=filename, image=image)


@app.route('/dictionary', methods=['GET', 'POST'])
def upload_dictionary_file():

    my_path = ""
    my_file = ""

    if request.method == 'POST':

        if 'csvfile' not in request.files:
            flash('File Not Found!')
            return redirect(request.url)

        file = request.files['csvfile']

        if not csv_extension(file.filename):
            flash('Only csv files are supported')
            return redirect(request.url)

        name = request.form.get("name")

        test_df = pd.read_csv(file)

        try:
            if test_df.columns[0] != "SentimentText":
                if test_df.columns[1] != "SentimentText":
                    if test_df.columns[2] != "SentimentText":
                        flash('One of the First three column name must be SentimentText!')
                        return redirect(request.url)
        except:
            flash('One of the column name must be SentimentText!')
            return redirect(request.url)

        '''
        negative_df = pd.read_csv('static/dictionary/negative.csv')
        positive_df = pd.read_csv('static/dictionary/positive.csv')
        '''
        negative_df = pd.read_csv(os.path.join("c:", os.sep, "Users", "dogacanbicer", "PyCharmProjects"
                                               , "BrandSentimentAnalysis", "static", "dictionary", "negative.csv"))
        positive_df = pd.read_csv(os.path.join("c:", os.sep, "Users", "dogacanbicer", "PyCharmProjects"
                                               , "BrandSentimentAnalysis", "static", "dictionary", "positive.csv"))

        positive_count = 0
        negative_count = 0
        neutral_count = 0

        for i in range(0, len(test_df)):

            string = test_df.at[i, 'SentimentText'].split(" ")

            pos_count = 0
            neg_count = 0

            for j in range(0, len(string)):

                word = string[j]
                word_index = string.index(string[j])

                for k in range(0, len(positive_df)):
                    positive_word = positive_df.at[k, 'Word']
                    if word == positive_word:
                        if word_index >= 1:
                            if string[word_index - 1] not in ["not", "isn't", "aren't", "wasn't", "weren't", "don't",
                                                              "didn't"]:
                                pos_count += 1
                            else:
                                neg_count += 1
                        else:
                            pos_count += 1

                for l in range(0, len(negative_df)):
                    negative_word = negative_df.at[l, 'Word']
                    if word == negative_word:
                        if word_index >= 1:
                            if string[word_index - 1] not in ["not", "isn't", "aren't", "wasn't", "weren't", "don't",
                                                              "didn't"]:
                                neg_count += 1
                            else:
                                pos_count += 1
                        else:
                            neg_count += 1

            if pos_count > neg_count:
                positive_count += 1
            elif neg_count > pos_count:
                negative_count += 1
            else:
                neutral_count += 1

        exp_vals = [positive_count, negative_count, neutral_count]
        exp_labels = ["positive", "negative", "neutral"]

        plt.pie(exp_vals, labels=exp_labels)

        extension = "png"

        name = return_new_name(name, extension)

        my_path = 'static/uploads'
        my_file = name + '.' + extension

        plt.savefig(os.path.join(working_place, my_file))
        plt.clf()

    return render_template("dictionary.html", image=str(my_path+'/'+my_file))
# END OF DICTIONARY

# WORD CLOUD
@app.route("/wordcloud")
def word_cloud(filename="", image=""):
    return render_template("wordcloud.html", filename=filename, image=image)


@app.route('/wordcloud', methods=['GET', 'POST'])
def upload_word_file():

    my_path = ''
    my_file = ''

    if request.method == 'POST':

        if 'csvfile' not in request.files:
            flash('File Not Found!')
            return redirect(request.url)

        file = request.files['csvfile']

        if not csv_extension(file.filename):
            flash('Only csv files are supported')
            return redirect(request.url)

        name = request.form.get("name")

        df = pd.read_csv(file)

        try:
            if df.columns[0] != "SentimentText":
                if df.columns[1] != "SentimentText":
                    if df.columns[2] != "SentimentText":
                        flash('One of the First three column name must be SentimentText!')
                        return redirect(request.url)
        except:
            flash('One of the column name must be SentimentText!')
            return redirect(request.url)

        text = ""

        for i in range(0, len(df)):
            text = text + " " + df.at[i, 'SentimentText']

        wordcloud = WordCloud(
            width=3000,
            height=2000,
            background_color='black',
            stopwords=STOPWORDS).generate(str(text))

        fig = plt.figure(
            figsize=(4, 3),
            facecolor='k',
            edgecolor='k')

        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout(pad=0)

        extension = "png"

        name = return_new_name(name, extension)

        my_path = 'static/uploads'
        my_file = name + '.' + extension
        print(my_file)

        plt.savefig(os.path.join(working_place, my_file))
        plt.clf()

    return render_template("wordcloud.html", image=str(my_path+'/'+my_file))
# END OF WORD CLOUD


if __name__ == "__main__":
    app.run(debug=True)
