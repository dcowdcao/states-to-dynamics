## This is the CLAUDE.md file. You MUST read this document thoroughly EVERY TIME I prompt you.

- Please think ultra-hard and diligently. Use as many tokens as you need. Before implementing anything, you must understand the core logic of this repository — both the inter-file logic and the file architecture — and you must read my prompt extremely carefully.

- Read PROCESS.md before starting work: it holds a running log of what's in flight and the non-obvious decisions behind it. After doing something noteworthy (a new experiment, a critical decision, a gotcha worth remembering), add a concise entry to PROCESS.md so the next human or agent has context the code alone doesn't show.

- If you are ever uncertain about something or need clarification, or if you want to change some core logic, you MUST ask me for permission first. This holds even in auto mode.

- After implementing, you MUST clean up any files you no longer need, i.e. files that are now stale. Again, if you are on the fence about a particular file or group of files, you MUST ask me for permission first.

- When adding new files or scripts, lean towards making new folders for them rather than putting them inside existing folders. This prevents clutter and avoids mixing files of different purposes into the same folder.

- Generally, follow the same conventions I use for designing file architectures, naming them, and so on.

- Naming conventions should be specific to the purpose the file / folder serves — more direct, less abstract.

- As a general rule of thumb, the more concise and clear, the better. This applies both to the specific algorithms you implement within scripts and to the overall project architecture / how you design projects.

# Code style guide

Small, composable, immutable pieces.

anish@

May 14, 2026

Status: Requesting feedback

## Sane default style

This doc covers some of our house style, but if you're in doubt, the Google Python style guide is good: https://google.github.io/styleguide/pyguide.html

## Taste

Text is the universal interface. Avoid overengineered solutions like databases in favor of flat text, unless you have very good reasons (for example, rollouts are too big to store huge numbers of them as flat text). If that's too slow, add a cache before reaching towards more heavyweight binary solutions.

In the same vein, strongly prefer interfaces that are LLM-friendly. The user might be an LLM, so design the interface accordingly. Your interfaces should be as simple as possible (and no more). Defaults should be sane, and shouldn't need to be passed in in the default case.

## Factoring

The default move is to delete. Avoid `getattr(x, "field", default)` when the field is always present, tautological equality tests, back-compat reexports nothing reads, helpers used once.

Inline beats single-use helper when the inline is <= 4 lines. Flatten with early returns instead of nesting. `try/except` wraps only the call that can raise, and `try/finally` handles cleanup rather than repeating it at each early-return path.

If two surfaces (CLI + web, train + eval) implement the same rule, push the rule down to the model. A "shared helper between CLI and web" is the wrong direction.

## Comments

Default to none. One short line when you do write one. Clear variable names and code structure should do most of the work here. If you're tempted to write a multiparagraph comment, put it in docs or a markdown file instead. No date stamps, ticket IDs, or commit-style narratives in the source. Skip docstrings when the name says it; a paragraph is OK only for a caller-visible contract that changes how callers write their code.

Keep understanding local: comments should be near the thing they explain. Non-obvious choices, a stratification, an unusual default, a workaround, get written down as they're made, in CHOICES.md or PROCESS.md next to the code, or as a comment at the entry point if small. Documentation should generally describe what the current challenge is that the code is addressing, not what the history is. Historical notes should live in the "alternatives considered" part of a design doc. Pair-session recordings are the fallback; in the happy path nobody watches them. Speaking of which,

## Setup Docs

The target for setup docs is `make setup` and `make run`. Anything beyond two commands needs a clear reason. The instructions have to work on a fresh clone, if `.venv` isn't created by the setup script, the setup script is incomplete, and if `pip install` only works because of state on someone's laptop, the instructions aren't real. Don't hardcode any paths outside the repo structure anywhere; take an env var or flag with a reasonable default. Be careful when hardcoding paths even inside the repo structure.

When a simple invocation needs seven arguments, five of them want defaults. What's left is usually "what to run" and "auth".

Tiered docs for three readers with budgets of 3, 20, and 50 lines: run it; understand the dev setup; start making changes. Anything bigger is a separate doc. Cover one workflow end-to-end rather than every flag and use case.

## Failing loudly

Crash on unexpected state. Don't paper over with `.get(k, default)`, if a lookup table is missing a key, add the key, and let `KeyError` point at the bug. Soft-fallback is only for known and known irrelevant cases: an optional log field, a cosmetic display string. Never use `except Exception`, if you don't know what specific error you're looking for, then you don't know enough to handle it properly and should fail instead.

The pattern for CLIs is a domain exception plus a thin `die()`: `IssueError(Exception)`, `die(msg)` raises, top-level `main()` catches and exits. Library code raises; the CLI layer translates. Validate at the boundary and trust internal code.

## YAGNI

A bug fix is one bug fix. "While I'm here" cleanups go in their own commit. The thing we keep out of the tree is speculative API surface area — a flag, an option, a shim, a hook — added on the theory that someone will want it later.

YAGNI does not always mean fewer lines of code: adding a typed interface that constrains what you can do might mean more code, but will constrain the API surface area and likely improve locality of concerns (cf ADT-via-dataclass examples below)

The rule of thumb: an abstraction widens what the API might do; a constraint narrows what the body can do. YAGNI rejects the first and welcomes the second.

### Don't

A flag for a use case nobody has:

```python
# Only "json" is ever passed. Drop the parameter; add it back the day the
# second format actually shows up.
def load_issues(path, format="json"):
    if format == "json":
        return json.loads(path.read_text())
    elif format == "yaml":
        return yaml.safe_load(path.read_text())
```

A backwards-compat shim for code you control:

```python
# Nothing reads old_name anymore. Update the callers in the same change.
def old_name(x):
    return new_name(x)
```

Validation for an unreachable state:

```python
# issue.id is set at construction. The check is fog; if you want a tripwire,
# `assert issue.id` once at the boundary.
def render(issue):
    if not getattr(issue, "id", None):
        return "<unknown>"
    ...
```

A "future flexibility" hook with one concrete caller:

```python
# One subclass exists. The hook is scaffolding for a second one that doesn't.
class Grader:
    def grade(self, submission): ...
    def pre_grade_hook(self, submission): pass
    def post_grade_hook(self, result): pass
```

### Do

Parametric polymorphism. A generic signature constrains the implementation (free theorems) and documents intent better than the monomorphic version, even at one caller:

```python
def first(xs: list[T]) -> T:
    return xs[0]
```

`list[Issue] -> Issue` would also typecheck, but `T -> T` rules out "returns a hardcoded Issue" and "looks at a field on the element". The reader gets that for free.

A sum type when the dispatch shape is the right shape, even if only one variant is used today:

```python
@dataclass
class Pending: ...
@dataclass
class Running: started_at: datetime
@dataclass
class Done: result: GradeResult
State = Pending | Running | Done
```

If `Done` is the only state the code currently produces, the union still earns its place: it makes exhaustiveness-checkable in the consumer, and the missing branches are the actual TODO list.

A return dataclass instead of a dict, at one caller:

```python
@dataclass
class Tally:
    passed: int
    failed: int
    skipped: int
def tally(results) -> Tally: ...
```

The shape is the contract. `result["passed"]` everywhere downstream is a lookup that can silently return None; `result.passed` is a name the type checker enforces.

## DRY

We aim for small, named, composable pieces that read like a sentence at the call site. There are two failure modes: over-extracted (sentinel-and-flags helpers papering over real differences between callers) and under-extracted (the same twelve-line block in four places).

Rules of thumb: one caller, inline usually (unless it's big). Two near-identical bodies, leave them. Three is the point where it's worth asking, sometimes the abstraction is clean, sometimes the three are accidentally similar. Two callers that track different rules stay duplicated even when they look identical today, because they will drift. The test for extraction: can you describe the helper's signature without referencing the callers?

When writing new code, be careful not to duplicate anything that already exists; look for code that does something another piece already had to do, and share the code. Sometimes this means that code previously in one file has to be extracted into a common file (but also be careful not to make too many of these).

### Extract

Same rule, same reason, signature that stands alone.

```python
# Two callers, both want exactly "safe YAML load, empty dict on falsy."
def _yaml_load(content):
    return yaml.safe_load(content) if content else {}
```

When two surfaces reimplement the same rule, push the rule down to the data model rather than sideways into a shared helper:

```python
# Both cli/commands.py and web/routes.py wrote the forecast band the same way.
def apply_band(issue, phase, label):
    ...
```

### Don't extract

Two bodies that look the same but mean different things will diverge, and the shared helper grows a flag for each divergence.

```python
# inbox_row: "what's on my plate", will grow an attention badge
# outbox_row: "what I'm waiting on", will grow a blocked-on tag
# render_row(issue, show_attention=False, show_blocked=False) is worse.
def inbox_row(i):
    return f"#{num(i['id'])} {i['status']:8} → {i['owner']:15} [{i['title']}]"
def outbox_row(i):
    return f"#{num(i['id'])} {i['status']:8} → {i['owner']:15} [{i['title']}]"
```

When you change one body, will you almost always change the other to match? Then extract. If they drift independently, they're separate rules that happen to look alike right now.

One caller, even after a refactor, delete the helper:

```python
# before
emit(args, with_num(issue, lambda _id: n))
# after
emit(args, {"num": n, **issue})
```

### Over-extraction

The classic shape is two helpers that look the same but aren't, merged with a sentinel and a flag:

```python
def _apply(d, k, fn, _sentinel=object(), default=None, has_default=False):
    v = d.get(k, _sentinel)
    if v is _sentinel and has_default: v = default
    elif v is _sentinel: return
    d[k] = fn(v)
```

A wall of lambdas has the same problem. Each line below is reasonable on its own, but reading any one of them requires holding all the others in working memory:

```python
norm = lambda s: (s or "").strip().lower()
hit = lambda i, t: any(t in norm(i.get(f, "")) for f in ("title", "body", "tags"))
score= lambda i, ts: sum(1 for t in ts if hit(i, t)) + (0.5 if i.get("pinned") else 0)
rank = lambda xs, ts, n=10: sorted(((score(x, ts), x) for x in xs if score(x, ts) > 0), reverse=True)[:n]
top = lambda xs, q, n=10: [x for _, x in rank(xs, [t for t in (q or "").split() if len(t) > 2], n)]
fmt = lambda x: f"#{x.get('num','?')} {x.get('title','')[:60]:<60} {x.get('owner','-')}"
out = lambda xs, q: "\n".join(map(fmt, top(xs, q))) or "no results"
```

A single terse lambda can be good. Once you're naming the third or fourth helper-lambda, switch to defs with names and one-line bodies. Same density, but the reader can recover the order.

## Program Modularity

From the unix philosophy: "Make each program do one thing well. To do a new job, build afresh rather than complicate old programs by adding new features." In a python project, multiple different entrypoint scripts can share common logic, so make good use of this; each entrypoint should do just one thing, and when those things overlap in their logic there can be shared imports.

## Listcomps

Multi-line, one clause per line:

```python
items = [i for i in all_issues()
    if i['status'] == DONE_STATE
    and (i['owner'] == u or u in i.get('reviewers', []))
    and i['updated'] >= cutoff]
```

Comps build values, for-loops do control flow, don't comp for side effects (`[save(i) for i in dirty]`) or for anything that needs a conditional break. One level of nesting is fine (`[y for x in xs for y in f(x) if g(y)]`); two levels gets read twice, so name the inner one.

## Type hints

Required on public APIs and data-shape boundaries; optional on trivial internal helpers (`def num(id):` is fine) but more typing is always better. `Optional[X]` only when None is actually reachable. Use `object` for "genuinely unconstrained", as in a callback signature, not "I'll get to it later". When you find yourself writing `Optional[Union[dict[str, Optional[list[tuple[int, Optional[str]]]]], str]]`, that's the cue to introduce a dataclass.

## Imports

Trivial `import x` stdlib lines can share: `import argparse, subprocess, sys`. `from x import y, z` stays one-per-line.

Avoid making imports inside of conditionals or functions, unless you have a good reason (like you only need this library if some logic holds). When there's a good reason, describe it with a comment.

## Commits

Commit style is covered in Git Commit Guidelines

## Naming

Short names only in tight scopes: `for i in issues`, `b, r, l` for base/remote/local. Variable names should be part of code documentation. Single-word module helpers when the module already qualifies them, so `store.load` rather than `store.load_issue`. Drop `get_*` prefixes, `user()`, not `get_current_user()`.

`_underscore` for module-private helpers. Constants are `SCREAMING_SNAKE_CASE` at module top, with the policy encoded in the name; a one-line comment only if the value records a decision.

## Classes

A class that owns two disjoint state machines is two classes in a trenchcoat. The signal: the docstring needs an "and" to describe what it does, or you can partition the fields into groups where each group's methods don't touch the other group. Extract these so that you can read each chunk in isolation and can swap one half (e.g. a fake cache in tests) without touching the other.

Composition over inheritance. When the new class needs to notify another of a change (e.g., "on slug-list refresh, GC the content cache"), pass a callback at construction. Subclassing for cross-cutting effects pulls behavior up where it can't be turned off.

## Match Statements

Using match statements can make your code more readable. It's not a requirement, but keep an eye out for when structural matching could make your code clearer.

## Anti-patterns

Patterns we've actively removed from the tree. If you're about to commit one, don't.

**"Temporary" code that ships.** `# TEMP: revert after X` becomes permanent. Fix X first or don't write the code. Exception here is sometimes when code is in `llm` and needs workarounds to interact with non-llm code because of that fact.

**Disabled-feature scaffolding.** A `.disabled` rename, an `if ENABLE_X:` branch around dead code, a feature flag for a feature that isn't here, git history has the code, delete the branch.

**Defending against upstream invariants.** A shell function asserting a directory has the shape Docker COPY already produced is wasted lines. Mostly just can be removed, sometimes can be a comment about what the invariant is and where it is enforced.

**Preflight checks for failures you haven't seen.** A `_check_disk_headroom()` adds a new surface to debug. Add the guard after you've hit the failure and only if the failure was confusing.

**Retry wrappers on a hunch.** A flaky thing that retries twice usually wants to be fixed at the source. Retries hide the signal.

**Wall-of-text incident comments.** Multi-line code comments with cross-references and historical narratives rot. The WHY of the immediate line in one line is enough; the rest goes in the PR description.

**Sprawling commit bodies.** A 40-line body with "Goal:", "Background:", "Main files changed:" headings is harder to act on than three sentences pointing at the bug. Lead with what was wrong.

**Defensive security hardening without a confirmed threat.** A prompt-injection guard, input sanitizer, or sandboxed subprocess added on a hunch is not a threat model. Find the exploit first.

**Polluting the repo to work around a downstream tool's bug.** If a scanner or build step needs extra data in the tree because it can't handle the layout, fix the tool. This is the right call even when fixing the tool is politically harder.

**Inline code in another language.** If your bash code needs to run any more than a line or two of python, or vice versa, or with any other language, do not write that inner code inside the same file. Extract it to a separate file with the proper file extension, and call that file instead. Tooling should be able to fully check all your code without having to support polyglot files, except in cases of extremely short and trivial calls.

## Prose

The same rule as code: cut a sentence and see if anything's missing.

Drop fossils. "In order to" is "to". "Due to the fact that" is "because". "At this point in time" is "now". Cut the throat-clearing, "I think it's worth noting that" is four words before the actual thought. Skip "great question" and "hope this helps". If you know something, say it; if you don't, say that. No hedges in front of things you can see.

The code-comment versions of the same problem: `// increment counter` above `i++`, a docstring restating the signature in worse English, a PR description narrating a diff the reader can already see. Say why the change exists, or don't write the description. DRY applies between code and prose too. Longer docs should present information at multiple levels of detail/abstraction, but the structure of the doc should make it clear what describes what, so readers don't end up reading the same thing many times.

Prose written by Claude is very suspect, more so than code written by it. Most of the time, taking the time to understand the prose and rewriting it in your own words will save others a lot of time down the line.

## Dependencies

All dependencies in task dockerfiles should be pinned to exact versions with `==`. In tooling, dependencies should use `>=` unless there is a specific documented reason why a `==` would be better. 
