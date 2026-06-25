"""Auth endpoint tests via the TestClient."""

from __future__ import annotations

CREDS = {"email": "trader@example.com", "password": "supersecret123", "name": "Test Trader"}


def _set_cookie(resp) -> bool:
    return "access_token=" in resp.headers.get("set-cookie", "")


def test_signup_creates_user_and_sets_cookie(client):
    r = client.post("/auth/signup", json=CREDS)
    assert r.status_code == 201
    body = r.json()
    assert body["email"] == CREDS["email"]
    assert body["name"] == CREDS["name"]
    assert "id" in body
    assert _set_cookie(r)


def test_signup_duplicate_email_returns_409(client):
    assert client.post("/auth/signup", json=CREDS).status_code == 201
    r = client.post("/auth/signup", json=CREDS)
    assert r.status_code == 409


def test_login_correct_credentials_returns_200_and_cookie(client):
    client.post("/auth/signup", json=CREDS)
    client.cookies.clear()  # drop the signup session to test login in isolation
    r = client.post("/auth/login", json={"email": CREDS["email"], "password": CREDS["password"]})
    assert r.status_code == 200
    assert r.json()["email"] == CREDS["email"]
    assert _set_cookie(r)


def test_login_wrong_password_returns_401(client):
    client.post("/auth/signup", json=CREDS)
    r = client.post("/auth/login", json={"email": CREDS["email"], "password": "wrong-password"})
    assert r.status_code == 401


def test_me_without_cookie_returns_401(client):
    # Fresh client, never authenticated.
    assert client.get("/auth/me").status_code == 401


def test_me_with_valid_cookie_returns_user(client):
    client.post("/auth/signup", json=CREDS)  # client now holds the session cookie
    r = client.get("/auth/me")
    assert r.status_code == 200
    assert r.json()["email"] == CREDS["email"]
