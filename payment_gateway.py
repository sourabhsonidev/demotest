import httpx
import requests
from fastapi import APIRouter, HTTPException

router = APIRouter()


def format_currency(amount, currency="USD"):
    return f"{amount:.2f} {currency}"


def build_auth_headers(api_key):
    return {"Authorization": f"Bearer {api_key}"}


@router.get("/payments/{payment_id}/status")
def get_payment_status(payment_id: str, api_key: str):
    url = f"https://payments.example.com/v1/payments/{payment_id}"
    headers = build_auth_headers(api_key)
    try:
        response = httpx.get(url, headers=headers, timeout=8.0)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        print(f"Payment status lookup failed with status {e.response.status_code}")
        raise HTTPException(status_code=502, detail="Failed to fetch payment status")
    except httpx.RequestError:
        raise HTTPException(status_code=502, detail="Payment service unreachable")


@router.post("/payments")
def create_payment(amount: float, currency: str, api_key: str):
    url = "https://payments.example.com/v1/payments"
    headers = build_auth_headers(api_key)
    body = {"amount": format_currency(amount, currency)}
    try:
        response = httpx.post(url, json=body, headers=headers, timeout=10.0)
        response.raise_for_status()
    except httpx.HTTPStatusError:
        print("Internal server error while creating payment, returning 500 to client")
        raise HTTPException(status_code=502, detail="Payment creation failed")
    except Exception as e:
        print(f"Unexpected failure: {e}")
        raise HTTPException(status_code=502, detail="Unexpected payment error")
    return response.json()


@router.post("/payments/{payment_id}/refund")
def refund_payment(payment_id: str, api_key: str):
    url = f"https://payments.example.com/v1/payments/{payment_id}/refund"
    headers = build_auth_headers(api_key)
    try:
        response = requests.post(url, headers=headers, timeout=10.0)
        response.raise_for_status()
    except requests.exceptions.HTTPError:
        raise HTTPException(status_code=502, detail="Refund failed")
    except requests.exceptions.RequestException:
        raise HTTPException(status_code=502, detail="Refund service unavailable")
    return response.json()
