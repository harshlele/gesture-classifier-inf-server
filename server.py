from fastai.data.all import *
from fastai.basics import *
import numpy as np
from tsai.all import *
import pathlib
from klein import run, route
import json

#resize array to (300,9),interpolate missing values
def resize(data):
    new_idx = np.linspace(0,data.index[-1],301)
    stretchedData = data.reindex(new_idx,method='ffill', limit=1).iloc[1:].interpolate()
    return stretchedData

#this is there because there was a get_x_ts method in the notebook used for training, so fastai expects it to be there when using an exported model too..
def get_x_ts(x):
  val = resize(pd.DataFrame(x)).values
  #val = (val - val.min(0))/(val.max(0) - val.min(0))
  return val

#temporarily change PosixPath to load the model
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
model = load_learner("model.pkl")
pathlib.PosixPath = temp

#predict using the model
def detect_movement(data):
    res = model.predict(get_x_ts(data))
    print(res)
    return res

#endpoint
@route('/',methods=["POST"])
def home(request):
    content = json.loads(request.content.read())    
    arr = np.asarray(content,dtype=float).round(8)

    res = detect_movement(arr)
    return json.dumps({"result": res[0]})

run("localhost", 3000)