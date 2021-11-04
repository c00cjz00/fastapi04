# 範例二 網頁執行預測
# https://github.com/BoxOfCereal/FastAPI-Fastai2/blob/master/app/server.py
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
defaults.device = torch.device('cpu')

## 服務器
import asyncio
#import nest_asyncio
#from pyngrok import ngrok
#import uvicorn


# 載入 FASTAPI
app = FastAPI()

# 網頁框架
templates = Jinja2Templates(directory='templates/')

# 掛載資料夾
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")




# 設定初始啟動
learn1 = None
learn2 = None
@app.on_event("startup")
async def startup_event():
    """Setup the learner on server start"""
    global learn1, learn2
    #loop = asyncio.get_event_loop()  # get event loop
    #tasks = [asyncio.ensure_future(setup_learner())]  # assign some task
    #learn = (await asyncio.gather(*tasks))[0]  # get tasks

    # 模型檔案
    myPath='models'
    myModel_01=myPath+'/mnist.pkl'
    myModel_02=myPath+'/nodule.pkl'

    # 啟動學習
    learn1 = load_learner(myModel_01)
    learn1.dls.device = 'cpu'
    learn2 = load_learner(myModel_02)
    learn2.dls.device = 'cpu'


## 首頁 
@app.get('/')
async def form_post(request: Request):
    result = "Type a number"
    return templates.TemplateResponse('main.html', context={'request': request, 'result': result})

## GET 模式   
@app.get("/form")
async def form_post(request: Request):
    result = "Type a number"
    return templates.TemplateResponse('main.html', context={'request': request, 'result': result})
   
## POST 模式   
@app.post("/form")
async def form_post(request: Request, num: int = Form(...)):
    result = num
    return templates.TemplateResponse('main.html', context={'request': request, 'result': result})

# 上傳圖片1
@app.get('/imageUpload')
async def form_get(request: Request):
    result = ''
    return templates.TemplateResponse('uploadFile.html', context={'request': request, 'result': result})

# 上傳圖片2
@app.post('/imageUpload')
async def form_post(request: Request, file: UploadFile = File(...)):
    upload_image = 'uploads/'+file.filename
    web_upload_image = '/uploads/'+file.filename
    with open(upload_image, 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = '上傳圖片檔名: ' + file.filename
    return templates.TemplateResponse('uploadFile.html', context={'request': request, 'result': result, 'upload_image': web_upload_image})



# 檔案上傳 WEB01
@app.get('/mnist')
async def form_get(request: Request):
    result = ''
    return templates.TemplateResponse('uploadFile_mnist.html', context={'request': request, 'result': result})

# 檔案預測 WEB02
@app.post('/mnist')
async def form_post(request: Request, file: UploadFile = File(...)):
    upload_image = 'uploads/'+file.filename
    html_upload_image = '/uploads/'+file.filename

    with open(upload_image, 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    prediction = learn1.predict(upload_image) 
    result = '圖片預測結果: ' + prediction[0]
    return templates.TemplateResponse('uploadFile_mnist.html', context={'request': request, 'result': result, 'upload_image': html_upload_image})
  

# 檔案上傳 WEB01
@app.get('/nodule')
async def form_get(request: Request):
    result = ''
    return templates.TemplateResponse('uploadFile_nodule.html', context={'request': request, 'result': result})

# 檔案預測 WEB02
@app.post('/nodule')
async def form_post(request: Request, file: UploadFile = File(...)):
    upload_image = 'uploads/'+file.filename
    html_upload_image = '/uploads/'+file.filename

    with open(upload_image, 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)

    prediction = learn2.predict(upload_image) 
    result = '圖片預測結果: ' + prediction[0]
    return templates.TemplateResponse('uploadFile_nodule.html', context={'request': request, 'result': result, 'upload_image': html_upload_image})
  


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=80)
