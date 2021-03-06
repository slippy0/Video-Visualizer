import os
from os import path

import flask
from flask import Flask, render_template

import data_loaders

app = Flask(__name__)
app.config.from_pyfile(os.environ['FLASK_CONFIG'])
data_loader = data_loaders.get_data_loader(app.config['DATA_LOADER'])(
    **app.config['DATA_LOADER_CONFIG'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_list')
def video_list():
    return flask.jsonify(data_loader.video_list())

@app.route('/video/<path:video_name>')
def video(video_name):
    if not video_name.endswith(app.config['VIDEO_EXTENSION']):
        video_name = video_name + '.' + app.config['VIDEO_EXTENSION']
    print video_name
    return flask.send_from_directory(data_loader.get_videos_dir(), video_name)


@app.route('/groundtruth/<video_name>')
def groundtruth(video_name):
    return flask.jsonify(data_loader.video_groundtruth(video_name))


@app.route('/predictions/<video_name>')
def predictions(video_name):
    return flask.jsonify(data_loader.video_predictions(video_name))

@app.route('/flow_predictions/<video_name>')
def video_flow_predictions(video_name):
    return flask.jsonify(data_loader.video_flow_predictions(video_name))

if __name__ == '__main__':
    app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024
    app.run(threaded=True)
