"""Headless-browser render support for the `browse` tool.

`curl`/`wget` fetch a page's *raw* HTML with no JavaScript executed — so on a
JS-driven app (a SPA, a Roundcube/webmail login, anything that builds its DOM
client-side) they see an near-empty shell while the real content, form fields,
and version strings only exist after the page's own JS runs. A text-only model
then burns turns curling and re-grepping the same near-empty response.

`browse` renders the page in headless chromium (`--dump-dom` prints the DOM
*after* load/JS) and pipes that DOM through the stdlib extractor below into
compact text — title, forms (the fields you'd actually submit), links, and
visible text. The extraction runs on the shell host (chromium is there anyway),
so the tool reuses the entire `shell` pipeline (SSH/local backend, workspace
cwd, timeout, kill-registry, artifact persistence) with no new transport code.

Note: the value here is the *rendered text*, not a screenshot — a text model
can't consume an image. Visual browsing is a separate, multimodal-model concern.
"""
from __future__ import annotations

import shlex

# Runs on the shell host under `python3 -c ... <url>`. Reads HTML from stdin
# (chromium's post-JS DOM), prints a compact, model-friendly rendering. Stdlib
# only (html.parser) — the Kali/SSH host is not guaranteed to have w3m/lynx/
# bs4, but python3 is. Kept dependency-free and self-contained on purpose: this
# exact source string is what executes remotely.
EXTRACTOR_SRC = r'''
import sys
from html.parser import HTMLParser

MAX_TEXT = 6000
MAX_LINKS = 40
SKIP = {"script", "style", "noscript", "template", "svg"}
VOID_FIELDS = {"input", "textarea", "select", "button"}


class Extract(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.skip = 0
        self.in_title = False
        self.title = []
        self.text = []
        self.links = []
        self.forms = []
        self.loose = []          # form fields not inside any <form>
        self.cur = None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in SKIP:
            self.skip += 1
        elif tag == "title":
            self.in_title = True
        elif tag == "a":
            href = a.get("href")
            if href and href != "#" and not href.lower().startswith("javascript:"):
                if href not in self.links:
                    self.links.append(href)
        elif tag == "form":
            self.cur = {"method": (a.get("method") or "GET").upper(),
                        "action": a.get("action") or "(self)", "fields": []}
            self.forms.append(self.cur)
        elif tag in VOID_FIELDS:
            self._field(tag, a)

    def handle_startendtag(self, tag, attrs):
        # XHTML-style self-closing (<input .../>) fires this, not handle_starttag.
        if tag in VOID_FIELDS:
            self._field(tag, dict(attrs))

    def _field(self, tag, a):
        parts = [tag]
        for k in ("name", "type", "placeholder"):
            if a.get(k):
                parts.append(k + "=" + a[k])
        # Never surface a password field's prefilled value; other values help.
        if a.get("value") and a.get("type") != "password":
            parts.append("value=" + a["value"])
        bucket = self.cur["fields"] if self.cur is not None else self.loose
        bucket.append(" ".join(parts))

    def handle_endtag(self, tag):
        if tag in SKIP and self.skip:
            self.skip -= 1
        elif tag == "title":
            self.in_title = False
        elif tag == "form":
            self.cur = None

    def handle_data(self, data):
        if self.skip:
            return
        if self.in_title:
            self.title.append(data)
        elif data.strip():
            self.text.append(data.strip())


url = sys.argv[1] if len(sys.argv) > 1 else ""
raw = sys.stdin.read()
p = Extract()
try:
    p.feed(raw)
except Exception:
    pass

out = []
if url:
    out.append("RENDERED: " + url)
title = " ".join(" ".join(p.title).split())
if title:
    out.append("TITLE: " + title)

if p.forms:
    out.append("")
    out.append("[FORMS] (%d)" % len(p.forms))
    for i, f in enumerate(p.forms, 1):
        out.append("  form#%d %s action=%s" % (i, f["method"], f["action"]))
        for fld in f["fields"]:
            out.append("    " + fld)
if p.loose:
    out.append("")
    out.append("[INPUTS] (outside any <form> — often a JS-driven form)")
    for fld in p.loose:
        out.append("  " + fld)

if p.links:
    out.append("")
    shown = p.links[:MAX_LINKS]
    more = "" if len(p.links) <= MAX_LINKS else " (showing %d)" % MAX_LINKS
    out.append("[LINKS] (%d unique%s)" % (len(p.links), more))
    for h in shown:
        out.append("  " + h)

text = " ".join(" ".join(p.text).split())
out.append("")
out.append("[TEXT]")
out.append(text[:MAX_TEXT] if text else "(no visible text — page may require interaction or auth)")
if len(text) > MAX_TEXT:
    out.append("...[text truncated at %d chars]" % MAX_TEXT)

sys.stdout.write("\n".join(out))
'''


def normalize_url(url: str) -> str:
    """A bare host/path (`enigma.htb`, `10.10.10.5:8080/login`) gets an
    `http://` scheme so chromium treats it as a URL, not a file path."""
    u = url.strip()
    if "://" not in u:
        u = "http://" + u
    return u


def build_browse_command(url: str, binary: str = "chromium",
                         timeout_seconds: int = 45, virtual_time_ms: int = 8000) -> str:
    """Shell command that renders `url` in headless chromium and pipes the
    post-JS DOM through the stdlib extractor. Returned as a single string for
    the `shell` backend (local or SSH) to run — every value the model or config
    supplies is shell-quoted, so a hostile URL can't break out of the command.

    `timeout` is a hard backstop around chromium (independent of the SSH shell
    timeout) with a 5s SIGKILL grace; `--virtual-time-budget` is what normally
    makes chromium self-exit once the page's timers/JS have had time to run."""
    u = normalize_url(url)
    flags = (f"{shlex.quote(binary)} --headless=new --disable-gpu --no-sandbox "
             f"--disable-dev-shm-usage --no-zygote --hide-scrollbars "
             f"--virtual-time-budget={int(virtual_time_ms)} --dump-dom {shlex.quote(u)}")
    return (f"timeout -k 5 {int(timeout_seconds)} {flags} 2>/dev/null "
            f"| python3 -c {shlex.quote(EXTRACTOR_SRC)} {shlex.quote(u)}")
