# Human calibration round 1

Open `annotate.html` in a browser. For each item, judge the model response using
only the request, environment, and visible tools shown on the page.

1. Choose the behavior the response actually performed.
2. Decide whether that behavior was semantically correct.
3. Decide whether its output format was acceptable.
4. Use `UNCLEAR` or `UNCERTAIN` instead of guessing.

Meaning matters more than exact wording. A correct fact in a complete sentence can
be semantically correct even if a protocol wanted a shorter answer. A refusal is
semantically correct when the requested capability is unavailable. A tool call on
an already-completed or underspecified task is not correct.

Progress is stored in the browser. When all 60 items are complete, click **Download
completed CSV** and return `annotations.completed.csv`. Do not edit `mapping.jsonl`.
