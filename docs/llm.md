# jNO for LLMs

jNO ships a single self-contained briefing file written for **language models**: the house
conventions, the full API vocabulary, the measured gotchas, and a set of verified end-to-end
examples — everything a coding agent needs to write correct, idiomatic jNO without crawling the
whole site.

Two raw files, stable URLs, plain markdown:

| File | Contents |
|---|---|
| [`llms.txt`](llms.txt) | Short index — what jNO is, where the key pages are |
| [`llms-full.txt`](llms-full.txt) | The **full guide** (~1300 lines): conventions, API, traps, verified examples |

Use it by pasting `llms-full.txt` into a system prompt, pointing a coding agent at the URL, or
dropping it into an agent-skill directory (it began life as one). Every example in its §12 was
executed against a released jNO and exits 0, and the file is maintained alongside the code — when
the API and this file disagree, one of them has a bug, and we treat it that way.

The same content, rendered for humans:

---

--8<-- "llms-full.txt"
