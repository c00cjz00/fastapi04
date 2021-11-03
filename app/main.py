# 範例二 網頁執行預測
## 檔案路徑
import os
## 檔案上傳 Library
import shutil

## FASTAPI Library
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Request, Form
## FASTAPI Library 網頁框架 
from fastapi.templating import Jinja2Templates
## FASTAPI Library 外掛資料夾
from fastapi.staticfiles import StaticFiles

## FASTAI Library
from fastai.vision.all import *

## 服務器
#import nest_asyncio
#from pyngrok import ngrok
#import uvicorn


# 載入 FASTAPI
app = FastAPI()

# 網頁框架
templates = Jinja2Templates(directory='templates/')

# 掛載資料夾
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# 模型檔案
myPath='models'
myModel_01=myPath+'/mnist.pkl'
myModel_02=myPath+'/nodule.pkl'


## 首頁 
@app.get('/')
def form_post(request: Request):
    result = "Type a number"
    return templates.TemplateResponse('main.html', context={'request': request, 'result': result})

## GET 模式   
@app.get("/form")
def form_post(request: Request):
    result = "Type a number"
    return templates.TemplateResponse('main.html', context={'request': request, 'result': result})
   
## POST 模式   
@app.post("/form")
def form_post(request: Request, num: int = Form(...)):
    result = num
    return templates.TemplateResponse('main.html', context={'request': request, 'result': result})

# 上傳圖片1
@app.get('/imageUpload')
def form_get(request: Request):
    result = ''
    return templates.TemplateResponse('uploadFile.html', context={'request': request, 'result': result})

# 上傳圖片2
@app.post('/imageUpload')
def form_post(request: Request, file: UploadFile = File(...)):
    upload_image = 'uploads/'+file.filename
    web_upload_image = '/uploads/'+file.filename
    with open(upload_image, 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = '上傳圖片檔名: ' + file.filename
    return templates.TemplateResponse('uploadFile.html', context={'request': request, 'result': result, 'upload_image': web_upload_image})



# 檔案上傳 WEB01
@app.get('/mnist')
def form_get(request: Request):
    result = ''
    return templates.TemplateResponse('uploadFile_mnist.html', context={'request': request, 'result': result})

# 檔案預測 WEB02
@app.post('/mnist')
def form_post(request: Request, file: UploadFile = File(...)):
    upload_image = 'uploads/'+file.filename
    html_upload_image = '/uploads/'+file.filename

    with open(upload_image, 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    learn = load_learner(myModel_01)
    prediction = learn.predict(upload_image) 
    result = '圖片預測結果: ' + prediction[0]
    return templates.TemplateResponse('uploadFile_mnist.html', context={'request': request, 'result': result, 'upload_image': html_upload_image})
  

# 檔案上傳 WEB01
@app.get('/nodule')
def form_get(request: Request):
    result = ''
    return templates.TemplateResponse('uploadFile_nodule.html', context={'request': request, 'result': result})

# 檔案預測 WEB02
@app.post('/nodule')
def form_post(request: Request, file: UploadFile = File(...)):
    upload_image = 'uploads/'+file.filename
    html_upload_image = '/uploads/'+file.filename

    with open(upload_image, 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)

    learn = load_learner(myModel_02)
    prediction = learn.predict(upload_image) 
    result = '圖片預測結果: ' + prediction[0]
    return templates.TemplateResponse('uploadFile_nodule.html', context={'request': request, 'result': result, 'upload_image': html_upload_image})
  
