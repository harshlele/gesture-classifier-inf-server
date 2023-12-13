from fastai.data.all import *
from fastai.basics import *
import numpy as np
from tsai.all import *
import pathlib
from klein import run, route
import json

def resize(data):
    new_idx = np.linspace(0,data.index[-1],301)
    stretchedData = data.reindex(new_idx,method='ffill', limit=1).iloc[1:].interpolate()
    return stretchedData

def get_x_ts(x):
  val = resize(pd.DataFrame(x)).values
  #val = (val - val.min(0))/(val.max(0) - val.min(0))
  return val

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
model = load_learner("model.pkl")
pathlib.PosixPath = temp

pos_arr = np.empty((300,9))
sz = 0
inf_time = time.time()

def detect_movement(data):
    res = model.predict(get_x_ts(data))
    print(res)
    return res


@route('/',methods=["POST"])
def home(request):
    content = json.loads(request.content.read())    
    arr = np.asarray(content,dtype=float).round(8)

    res = detect_movement(arr)
    return json.dumps({"result": res[0]})

run("localhost", 3000)