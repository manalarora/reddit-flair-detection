from bs4 import BeautifulSoup
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import os
from werkzeug.utils import secure_filename
import pickle
import praw
import flask
import json
from flask import send_from_directory


replace_by_space = re.compile('[/(){}\[\]\|@,;]')
replace_symbol = re.compile('[^0-9a-z #+_]')
stopwords = set(stopwords.words('english'))

def clean_text(text):
    text = BeautifulSoup(text, "lxml").text # HTML decoding
    text = text.lower() # lowercase text
    text = replace_by_space.sub(' ', text) # replace certain symbols by space in text
    text = replace_symbol.sub('', text) # delete symbols from text
    text = ' '.join(word for word in text.split() if word not in stopwords) # remove STOPWORDS from text
    return text



# Use pickle to load in the pre-trained model
model = pickle.load(open('model/LR_data2.pkl','rb'))

reddit = praw.Reddit(client_id = "b5GZswE_-4JHvw",
                     client_secret = "weXodwTvLzJkPcnmfyP72DTs184",
                     user_agent = "Reddit Flare Detection",
                     username = "allergy21",
                     password = "##rkueCQf7ZGez!")

def prediction(url):
	submission = reddit.submission(url = url)
	data = {}
	data["title"] = str(submission.title)
	data["url"] = str(submission.url)
	data["body"] = str(submission.selftext)

	submission.comments.replace_more(limit=None)
	comment = ''
	count = 0
	for top_level_comment in submission.comments:
		comment = comment + ' ' + top_level_comment.body
		count+=1
		if(count > 10):
		 	break
		
	data["comment"] = str(comment)

	data['title'] = clean_text(str(data['title']))
	data['body'] = clean_text(str(data['body']))
	data['comment'] = clean_text(str(data['comment']))
    
	combined_features = data["title"] + data["comment"] + data["body"] + data["url"]

	return model.predict([combined_features])




# Initialise the Flask app
app = flask.Flask(__name__, template_folder='templates')
UPLOAD_FOLDER = 'upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Set up the main route
@app.route('/', methods=['GET', 'POST'])
def home():
	if flask.request.method == 'GET':
		return(flask.render_template('main.html'))
    
	if flask.request.method == 'POST':
		text = flask.request.form['url']
		flair = prediction(str(text))[0]
		return flask.render_template('result.html', result=flair,)

@app.route('/automated_testing', methods=['GET', 'POST'])
def upload_file():
	if flask.request.method == 'GET':
		return(flask.render_template('file.html'))

	if flask.request.method == 'POST':
		# check if the post request has the file part
		# if 'files' not in flask.request.files:
		# 	flash('No file part')
		# 	return flask.redirect(request.url)
		file = flask.request.files['upload_file']
		# if user does not select file, browser also
        # submit an empty part without filename
		obj = {}
		if file:
			filename = secure_filename(file.filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			f = open(os.path.join(app.config['UPLOAD_FOLDER'], filename), "r")
			for line in f.read().splitlines():
				obj[line]=prediction(line)[0]
				print(obj[line])
			# with open(os.path.join(app.config['UPLOAD_FOLDER'], 'resp.json'), 'w') as json_file:
			return json.dumps(obj)
			# return send_from_directory(app.config["UPLOAD_FOLDER"], filename='resp.json', as_attachment=True)
    
if __name__ == '__main__':
	app.run()