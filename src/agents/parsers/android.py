"""Android system providers: SMS, contacts, call log, calendar, accounts.

Paths cover full-file-system extractions (``data/data/<pkg>/databases``), ``user_de`` / ``user/0``
layouts, chip-off partitions mounted at ``/data`` and Cellebrite dumps.
"""
from src.agents.parsers.base import Col, Lookup, TableSpec

_DIR = {1: "received from", 2: "sent to", 3: "draft to", 4: "outbox to", 5: "failed to", 6: "queued to"}
_CALL = {1: "incoming", 2: "outgoing", 3: "missed", 4: "voicemail", 5: "rejected", 6: "blocked"}

SPECS = [
    TableSpec("sms.message", r"(^|/)mmssms\.db$", ("sms",),
              (Col("address", ("address",)), Col("body", ("body",)), Col("date", ("date",), "unix_ms"),
               Col("date_sent", ("date_sent",), "unix_ms"), Col("type", ("type",)), Col("read", ("read",), cite=False),
               Col("thread_id", ("thread_id",), cite=False)),
              require=("body",), text=("body",), labels={"type": _DIR},
              claim="SMS {type_label} {address} at {date_utc}: {body}",
              description="Android telephony provider SMS (mmssms.db, table sms)"),
    TableSpec("sms.message", r"(^|/)bugle_db$", ("parts",),
              (Col("body", ("text",)), Col("timestamp", ("timestamp",), "unix_ms"),
               Col("message_id", ("message_id",), cite=False), Col("conversation_id", ("conversation_id",), cite=False)),
              require=("body", "message_id"), text=("body",),
              lookups=(Lookup("sender_id", ("messages",), ("_id",), ("sender_id",), "message_id"),
                       Lookup("sender", ("participants",), ("_id",), ("send_destination", "normalized_destination"),
                              "sender_id"),
                       Lookup("sender_name", ("participants",), ("_id",), ("full_name",), "sender_id")),
              claim="Google Messages: {sender} {sender_name} at {timestamp_utc}: {body}",
              description="Google Messages / Android Messages (bugle_db, parts + messages + participants)"),
    TableSpec("contact.entry", r"(^|/)contacts2?\.db$", ("data",),
              (Col("value", ("data1",)), Col("mimetype_id", ("mimetype_id",), cite=False),
               Col("raw_contact_id", ("raw_contact_id",), cite=False)),
              require=("value", "mimetype_id"), text=("value",),
              lookups=(Lookup("mimetype", ("mimetypes",), ("_id",), ("mimetype",), "mimetype_id"),),
              where="{col:mimetype_id} IN (SELECT _id FROM mimetypes WHERE mimetype IN ("
                    "'vnd.android.cursor.item/name','vnd.android.cursor.item/phone_v2',"
                    "'vnd.android.cursor.item/email_v2','vnd.android.cursor.item/organization',"
                    "'vnd.android.cursor.item/postal-address_v2','vnd.android.cursor.item/nickname'))",
              claim="Contact {mimetype}: {value} (raw contact {raw_contact_id})",
              description="Android contacts provider (contacts2.db, table data + mimetypes)"),
    TableSpec("call.log", r"(^|/)(contacts2?|calllog)\.db$", ("calls",),
              (Col("number", ("number",)), Col("date", ("date",), "unix_ms"), Col("duration", ("duration",)),
               Col("type", ("type",)), Col("name", ("name",)), Col("geocoded_location", ("geocoded_location",))),
              require=("number", "date"), text=("number",), labels={"type": _CALL},
              claim="Call {type_label} {number} ({name}) at {date_utc}, {duration}s",
              description="Android call log (calllog.db or contacts2.db, table calls)"),
    TableSpec("calendar.event", r"(^|/)calendar\.db$", ("Events",),
              (Col("title", ("title",)), Col("dtstart", ("dtstart",), "unix_ms"), Col("dtend", ("dtend",), "unix_ms"),
               Col("location", ("eventLocation",)), Col("description", ("description",))),
              require=("title",), text=("title",),
              claim="Calendar event '{title}' starting {dtstart_utc} at {location}",
              description="Android calendar provider (calendar.db, table Events)"),
    TableSpec("account.entry", r"(^|/)accounts(_ce|_de)?\.db$", ("accounts",),
              (Col("name", ("name",)), Col("type", ("type",))),
              require=("name", "type"), text=("name",),
              claim="Account on device: {name} ({type})",
              description="Android AccountManager (accounts.db / accounts_ce.db)"),
]
