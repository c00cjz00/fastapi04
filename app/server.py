from fastapi import FastAPI, File, Form, UploadFile, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastai.vision.all import *
import uvicorn
import asyncio
import aiohttp
import aiofiles
from fastapi.middleware.cors import CORSMiddleware


# 載入 FASTAPI
app = FastAPI()

# 網頁框架
templates = Jinja2Templates(directory='app/templates/')

# 掛載資料夾
app.mount("/uploads", StaticFiles(directory="app/uploads"), name="uploads")



origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


path = Path(__file__).parent
# REPLACE THIS WITH YOUR URL
export_url = "<your link to your model>"
export_file_name = 'export.pkl'


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
    myPath='app/models'
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
    upload_image = 'app/uploads/'+file.filename
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
    upload_image = 'app/uploads/'+file.filename
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
    upload_image = 'app/uploads/'+file.filename
    html_upload_image = '/uploads/'+file.filename

    with open(upload_image, 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)

    prediction = learn2.predict(upload_image) 
    result = '圖片預測結果: ' + prediction[0]
    return templates.TemplateResponse('uploadFile_nodule.html', context={'request': request, 'result': result, 'upload_image': html_upload_image})
  


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=80)
