if session["revision_id"] is not None:
    notice = call("read_notice", notice_id=session["revision_id"])
    if "error" in notice:
        return "insufficient", None, None, evidence, "I cannot verify the revision."
    evidence.append(notice["id"])
    if notice["status"] == "cancelled":
        return "cancelled", None, None, evidence, "The workshop is cancelled."
    room = notice.get("room_id", room)
    rate = notice.get("rate_cents", rate)
