from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import anthropic
import stripe
import json
import os
from typing import List, Optional

app = FastAPI(title="ATSCheck API")

# CORS configuration
origins = [
    "http://localhost:3000",
    "http://localhost:8000",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8000",
    # Production domain - update this after deploying
    "https://YOUR_PRODUCTION_DOMAIN.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Anthropic client
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
if not anthropic_api_key:
    raise ValueError("ANTHROPIC_API_KEY environment variable not set")

client = anthropic.Anthropic(api_key=anthropic_api_key)

# Initialize Stripe
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
if not stripe.api_key:
    raise ValueError("STRIPE_SECRET_KEY environment variable not set")

SCAN_PRICE_CENTS = 299  # $2.99

# In-memory store of verified paid session IDs (single-use)
verified_sessions: set = set()


# Pydantic models
class ScanRequest(BaseModel):
    resume_text: str
    job_description: str = ""
    session_id: Optional[str] = None  # Required for paid scans


class ScanResult(BaseModel):
    match_score: int
    missing_keywords: List[str]
    improvements: List[str]
    verdict: str
    summary: str


class CheckoutSessionRequest(BaseModel):
    success_url: str
    cancel_url: str


# Helper function to parse Claude's response
def parse_scan_response(response_text: str) -> ScanResult:
    """Parse the Claude API response into structured data."""
    try:
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1

        if json_start != -1 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            data = json.loads(json_str)

            return ScanResult(
                match_score=max(0, min(100, int(data.get("match_score", 50)))),
                missing_keywords=data.get("missing_keywords", [])[:10],
                improvements=data.get("improvements", ["", "", ""])[:3],
                verdict=data.get("verdict", "Needs improvement"),
                summary=data.get("summary", "Analysis complete.")
            )
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        print(f"Error parsing response: {e}")

    return ScanResult(
        match_score=50,
        missing_keywords=["Unable to parse keywords"],
        improvements=["Please try again with a clearer resume format"],
        verdict="Needs improvement",
        summary="Analysis could not be completed. Please try again."
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/api/scan", response_model=ScanResult)
async def scan_resume(request: ScanRequest):
    """
    Scan a resume against a job description using Claude AI.
    Returns match score, missing keywords, improvements, and verdict.
    """
    if not request.resume_text:
        raise HTTPException(
            status_code=400,
            detail="Resume text is required"
        )

    # Validate paid session if provided (server-side gate for paid scans)
    if request.session_id:
        if request.session_id not in verified_sessions:
            raise HTTPException(
                status_code=402,
                detail="Invalid or already-used payment session. Please purchase a new scan."
            )
        verified_sessions.discard(request.session_id)

    if len(request.resume_text) > 50000 or len(request.job_description) > 10000:
        raise HTTPException(
            status_code=400,
            detail="Input text is too long. Please provide shorter versions."
        )

    try:
        if request.job_description.strip():
            prompt = f"""You are an expert ATS (Applicant Tracking System) analyst. Analyze the following resume against the job description and provide a detailed assessment.

RESUME:
{request.resume_text}

JOB DESCRIPTION:
{request.job_description}

Analyze this resume for ATS compatibility and relevance to the job. Provide your analysis in the following JSON format (no additional text, just the JSON):

{{
    "match_score": <integer 0-100, where 100 is perfect match>,
    "missing_keywords": [<list of 5-10 important keywords/phrases from the job description that are missing or underrepresented in the resume>],
    "improvements": [
        "<First specific, actionable improvement to make the resume better for this job>",
        "<Second specific, actionable improvement to make the resume better for this job>",
        "<Third specific, actionable improvement to make the resume better for this job>"
    ],
    "verdict": "<One of: 'Likely to pass ATS', 'Needs improvement', or 'Unlikely to pass ATS'>",
    "summary": "<A 1-2 sentence plain English summary of the overall match>"
}}

Be specific about missing keywords from the job description. Focus on technical skills, tools, certifications, and domain-specific terminology. Ensure the improvements are actionable and directly address gaps between the resume and job requirements."""
        else:
            prompt = f"""You are an expert ATS (Applicant Tracking System) analyst. Perform a general ATS compatibility audit on the following resume with no specific job description.

RESUME:
{request.resume_text}

Analyze this resume for general ATS compatibility — formatting, keyword density, structure, and common issues. Provide your analysis in the following JSON format (no additional text, just the JSON):

{{
    "match_score": <integer 0-100, where 100 means excellent general ATS compatibility>,
    "missing_keywords": [<list of 5-10 important keywords or skill categories that appear to be missing or thin based on the resume's apparent target role>],
    "improvements": [
        "<First specific, actionable improvement for better ATS compatibility>",
        "<Second specific, actionable improvement for better ATS compatibility>",
        "<Third specific, actionable improvement for better ATS compatibility>"
    ],
    "verdict": "<One of: 'Likely to pass ATS', 'Needs improvement', or 'Unlikely to pass ATS'>",
    "summary": "<A 1-2 sentence plain English summary of the resume's general ATS readiness>"
}}

Focus on formatting issues (tables, columns, headers), keyword density, action verbs, quantified achievements, and whether the resume structure is ATS-friendly."""

        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        response_text = message.content[0].text
        result = parse_scan_response(response_text)
        return result

    except anthropic.APIError as e:
        print(f"Anthropic API error: {e}")
        raise HTTPException(
            status_code=503,
            detail="AI service temporarily unavailable. Please try again."
        )
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred. Please try again."
        )


@app.post("/api/create-checkout-session")
async def create_checkout_session(request: CheckoutSessionRequest):
    """
    Create a Stripe Checkout session for a $2.99 scan purchase.
    Returns the Stripe-hosted checkout URL to redirect the user to.
    """
    try:
        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[{
                "price_data": {
                    "currency": "usd",
                    "product_data": {
                        "name": "ATS Resume Scan",
                        "description": "Instant ATS compatibility analysis for your resume",
                    },
                    "unit_amount": SCAN_PRICE_CENTS,
                },
                "quantity": 1,
            }],
            mode="payment",
            success_url=request.success_url + "?session_id={CHECKOUT_SESSION_ID}",
            cancel_url=request.cancel_url,
        )
        return {"checkout_url": session.url, "session_id": session.id}

    except stripe.StripeError as e:
        print(f"Stripe error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Unexpected error creating checkout session: {e}")
        raise HTTPException(status_code=500, detail="Could not create checkout session.")


@app.get("/api/verify-session")
async def verify_session(session_id: str = Query(...)):
    """
    Verify a completed Stripe Checkout session.
    Called after the user returns from Stripe with ?session_id=xxx.
    Returns {paid: true} if payment was successful.
    """
    try:
        session = stripe.checkout.Session.retrieve(session_id)
        paid = session.payment_status == "paid"
        if paid:
            verified_sessions.add(session_id)  # Register as single-use scan credit
        return {"paid": paid}

    except stripe.StripeError as e:
        print(f"Stripe verification error: {e}")
        raise HTTPException(status_code=400, detail="Could not verify payment.")
    except Exception as e:
        print(f"Unexpected error verifying session: {e}")
        raise HTTPException(status_code=500, detail="Payment verification failed.")


@app.options("/{path:path}")
async def options_handler(path: str):
    """Handle CORS preflight requests."""
    return {"status": "ok"}


# Serve frontend static files (must be after all API routes)
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    async def serve_frontend():
        return FileResponse(os.path.join(static_dir, "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
