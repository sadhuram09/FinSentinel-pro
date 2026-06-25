"""History endpoint tests: empty list and per-user ownership scoping."""

from __future__ import annotations

from backend.auth.security import hash_password
from backend.db_models import AnalysisHistory, User

USER_B = {"email": "user_b@example.com", "password": "supersecret123", "name": "User B"}


def test_history_empty_for_fresh_user(client):
    client.post("/auth/signup", json=USER_B)  # fresh account, no runs yet
    r = client.get("/history")
    assert r.status_code == 200
    assert r.json() == []


def test_history_detail_404_for_other_users_analysis(client, db_session):
    # User A owns an analysis (created directly in the DB).
    owner = User(email="owner_a@example.com", name="Owner A", hashed_password=hash_password("ownerpass1"))
    db_session.add(owner)
    db_session.commit()
    db_session.refresh(owner)

    record = AnalysisHistory(
        user_id=owner.id,
        ticker="AAPL",
        lookback_period="1y",
        prediction="up",
        verdict="Approved",
        result_json={"ticker": "AAPL", "id": None},
    )
    db_session.add(record)
    db_session.commit()
    db_session.refresh(record)

    # User B logs in and must NOT be able to read A's analysis.
    client.post("/auth/signup", json=USER_B)
    r = client.get(f"/history/{record.id}")
    assert r.status_code == 404
