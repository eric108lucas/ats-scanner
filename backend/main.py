from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import anthropic
import stripe
import json
import os
from typing import List

app = FastAPI(title="ATSCheck API")

origins = ["http://localhost:3000","http://localhost:8000","http://localhost:5500","http://127.0.0.1:5500","http://127.0.0.1:3000","http://127.0.0.1:8000"]

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
if not anthropic_api_key: raise ValueError("ANTHROPIC_API_KEY environment variable not set")
client = anthropic.Anthropic(api_key=anthropic_api_key)

stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
if not stripe.api_key: raise ValueError("STRIPE_SECRET_KEY environment variable not set")

SCAN_PRICE_CENTS = 299

class ScanRequest(BaseModel):
    resume_text: str
    job_description: str

class ScanResult(BaseModel):
    match_score: int
    missing_keywords: List[str]
    improvements: List[str]
    verdict: str
    summary: str

class CheckoutSessionRequest(BaseModel):
    success_url: str
    cancel_url: str

def parse_scan_response(response_text):
    try:
        jstart = response_text.find('{')
        jend = response_text.rfind('}') + 1
        if jstart != -1 and jend > jstart:
            data = json.loads(response_text[jstart:jend])
            return ScanResult(match_score=max(0,min(100,int(data.get('match_score',50)))),missing_keywords=data.get('missing_keywords',[])[:10],improvements=data.get('improvements',['','',''])[:3],verdict=data.get('verdict','Needs improvement'),summary=data.get('summary','Analysis complete.'))
    except: pass
    return ScanResult(match_score=50,missing_keywords=['Unable to parse keywords'],improvements=['Please try again'],verdict='Needs improvement',summary='Analysis could not be completed.')

@app.get('/health')
async def health_check(): return {'status':'ok'}

@app.post('/api/scan',response_model=ScanResult)
async def scan_resume(request:ScanRequest):
    if not request.resume_text or not request.job_description: raise HTTPException(status_code=400,detail='Resume and job description required')
    if len(request.resume_text)>50000 or len(request.job_description)>10000: raise HTTPException(status_code=400,detail='Input too long')
    try:
        prompt = f"You are an expert ATSanalyst. Analyze this resume against the job description.\n\nRESUME:\n{request.resume_text}\n\nJOB DESCRIPTION:\n{request.job_description}\n\nReturn JSON only: {{\"match_score\":0,\"missing_keywords\":[],\"improvements\":[\"\",\"\",\"\"],\"verdict\":\"\",\"summary\":\"\"}}"
        msg = client.messages.create(model='claude-haiku-4-5-20251001',max_tokens=1024,messages=[{'role':'user','content':prompt}])
        return parse_scan_response(msg.content[0].text)
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f'SCAN ERROR: {type(e).__name__}: {e}')
        raise HTTPException(status_code=503,detail=f'Error: {type(e).__name__}: {str(e)[:100]}')

@app.post('/api/create-checkout-session')
async def create_checkout_session(request:CheckoutSessionRequest):
    try:
        session=stripe.checkout.Session.create(payment_method_types=['card'],line_items=[{"price_data":{"currency":"usd","product_data":{"name":"ATS Resume Scan"},"unit_amount":SCAN_PRICE_CENTS},"quantity":1}],mode='payment',success_url=request.success_url+'?session_id={CHECKOUT_SESSION_ID}',cancel_url=request.cancel_url)
        return {"checkout_url":session.url,"session_id":session.id}
    except stripe.StripeError as e: raise HTTPException(status_code=400,detail=str(e))

@app.get('/api/verify-session')
async def verify_session(session_id:str=Query(...)):
    try:
        session=stripe.checkout.Session.retrieve(session_id)
        return {"paid":session.payment_status=="paid"}
    except stripe.StripeError: raise HTTPException(status_code=400,detail='Can not verify')

@app.options('/{path:path}')
async def options_handler(path:str): return {"status":"ok"}

static_dir=os.path.join(os.path.dirname(__file__),'static')
if os.path.exists(static_dir):
    app.mount('/static',StaticFiles(directory=static_dir), name='static')
    @app.get('/')
    async def serve_frontend(): return FileResponse(os.path.join(static_dir,'index.html'))

if __name__=='__main__':
    import uvicorn; uvicorn.run(app,host='0.0.0.0',port=8000)
