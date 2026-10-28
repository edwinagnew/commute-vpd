#!/usr/bin/env python3
"""PreToolUse hook on Bash: no Python runs directly on the shared dev node.

Ported from mats-fleet (Will Mayner's fleet on the same dev node, 2026-10-26). Refuses a command that runs Python,
pytest, `uv run` or `make test`/`make lint` on this machine, even a heredoc or `python -c`, unless it goes through
Slurm (fleet/cpu-run, sbatch, srun). The harness's own tools in fleet/ and .claude/, and the stdlib-only
docs/research_log/build.py, still run here. On 19 October two of mats-fleet's agents ran experiments on the dev node
and hung it. Each agent's processes are also capped in memory by a cgroup scope, and all agents together by a slice
(fleet/up); this hook is the early, explained refusal, the caps the backstop. Inside a Slurm job commands already run
on the worker, so none is refused there. Unlike mats-fleet's, GPU requests to Slurm are allowed: we use the L40s.

It also refuses a CPU-only sbatch, srun or fleet/cpu-run when the job would load a pretrained language model: models
never run on CPUs. It reads the command, its --wrap, its here-documents, the batch file it names,
the Python scripts and modules those name (`x.py`, `-m pkg.mod`), but not the modules a script imports: experiments import helpers from
scripts that also load a model (A2's from Q2's coax_ioi.py), which would refuse jobs that never load one.
In Python it looks for loading calls: `X.from_pretrained(...)` (tokenizers, configs, lenses and SAEs do not count),
nnsight's `LanguageModel(...)` and the repository's loaders in REPO_LOADERS. In the shell text it also looks for
model ids (`Qwen/...`, `gpt2`, `pythia-...`); a model id inside a Python script, which may only label saved data,
does not count. The check is heuristic.
"""
import json
import os
import re
import shlex
import sys

SLURM_GPU = re.compile(r"\b(sbatch|srun|salloc)\b.*(--gres[= ]\S*gpu|--gpus?(-per-\w+)?[= ]|\s-G\s?\d)")

try:
    call = json.load(sys.stdin)
except ValueError:
    sys.exit(0)
if call.get("tool_name") != "Bash":
    sys.exit(0)
command = (call.get("tool_input") or {}).get("command", "")
def _asks_for_gpu(segment, cwd):
    """Whether a Slurm submission requests a GPU, on its command line or in the #SBATCH lines of its batch file."""
    if not re.search(r"\b(sbatch|srun|salloc)\b", segment):
        return False  # fleet/cpu-run never gets a GPU
    if SLURM_GPU.search(segment):
        return True
    for word in segment.split():
        if word.endswith((".sbatch", ".sh", ".slurm")):
            path = word if os.path.isabs(word) else os.path.join(cwd, word)
            try:
                text = open(path, errors="ignore").read()
            except OSError:
                continue
            if re.search(r"^#SBATCH\s+(--gres[= ]\S*gpu|--gpus?(-per-\w+)?[= ]|-G\s?\d)", text, re.M):
                return True
    return False


# Loading a pretrained language model in a CPU-only Slurm job (see the docstring).
REPO_LOADERS = ["load_component_model"]  # src/commute/vpd.py: the VPD component model
LOAD_CALL = re.compile(r"\b(\w+)\.from_pretrained\w*\s*\(|\bLanguageModel\s*\("
                       + "".join(rf"|\b{name}\s*\(" for name in REPO_LOADERS))
NOT_A_MODEL = re.compile(r"Tokenizer|Config|Processor|FeatureExtractor|Lens|SAE|Sae|Transcoder")
MODEL_ID = re.compile(r"(?<![\w./-])(Qwen/\S+|meta-llama/\S+|EleutherAI/\S+|google/gemma\S*|mistralai/\S+|"
                      r"gpt2(-(medium|large|xl))?(?![\w-])|pythia-\d+\w*)")
SUBMIT = re.compile(r"^\s*(\w+=\S*\s+)*(sbatch|srun|(\S*/)?fleet/cpu-run)\b")
SHELL_FILE = (".sbatch", ".sh", ".slurm", ".bash")
# Linters read the files they name and run none of them: `fleet/cpu-run make lint FILES="...qwen_vpd.py"` was refused
# because qwen_vpd.py loads a model (friction 13). Such a job is not searched for loading calls, unless it also runs Python.
STATIC_ONLY = re.compile(r"\b(make\s+(\S+\s+)*lint|ruff|pyright)\b")
RUNS_PYTHON = re.compile(r"\b(python[\d.]*|pytest|jupyter|papermill)\b|\buv\s+run\s+(?!ruff\b|pyright\b)")


def _read(path):
    try:
        return open(path, errors="ignore").read() if os.path.isfile(path) else None
    except OSError:
        return None


def _code(text):
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))


def _resolve(word, roots):
    """The file a path in shell text names, tried against each root; a leading $VAR/ or ${VAR}/ is dropped."""
    word = os.path.expanduser(word.strip("\"'"))
    word = re.sub(r"^.*(\$\{\w+\}|\$\w+)/", "", word)
    for root in ([""] if os.path.isabs(word) else roots):
        path = os.path.join(root, word)
        if os.path.isfile(path):
            return os.path.normpath(path)
    return None


def _language_model(segment, cwd):
    """Where a Slurm submission (one command of a shell line) loads a pretrained language model, or None."""
    top = cwd
    while top != os.path.dirname(top) and not os.path.exists(os.path.join(top, ".git")):
        top = os.path.dirname(top)
    roots = [cwd, top if os.path.exists(os.path.join(top, ".git")) else cwd]
    try:
        words = shlex.split(segment, comments=True)
    except ValueError:
        words = segment.split()
    shell = [("the command", segment)]
    for i, w in enumerate(words):
        if w.startswith("--wrap="):
            shell.append(("--wrap", w[7:]))
        elif w == "--wrap" and i + 1 < len(words):
            shell.append(("--wrap", words[i + 1]))
        elif w.endswith(SHELL_FILE) and (path := _resolve(w, roots)) and (text := _read(path)) is not None:
            shell.append((path, _code(text)))
            roots.append(os.path.dirname(path))
    scripts = []
    for where, text in shell:
        roots += [os.path.expanduser(d) for d in re.findall(r"(?:^|[;&\s])cd\s+(?:\"|')?([^\s;&\"']+)", text)]
        if m := MODEL_ID.search(text):
            return f"{where if where[0] != '/' else os.path.relpath(where, roots[1])} names {m[0]}"
        for w in re.findall(r"[^\s\"'=;&|()]+\.py\b", text):
            scripts.append(_resolve(w, roots))
        for mod in re.findall(r"(?:^|\s)-m\s+([\w.]+)", text):
            rel = mod.replace(".", "/")
            scripts += [_resolve(rel + ".py", roots), _resolve(rel + "/__main__.py", roots)]
    if STATIC_ONLY.search(segment) and not any(RUNS_PYTHON.search(t) for t in (command, *(t for _, t in shell))):
        return None
    for path in dict.fromkeys(p for p in scripts if p):
        if "tests" in os.path.relpath(path, roots[1]).split(os.sep)[:-1]:
            continue  # tests build tiny models; the runtime CPU guard still stops any module over 50M parameters
        code = re.sub(r'("""|\'\'\')[\s\S]*?\1', "", _code(_read(path) or ""))  # names in docstrings load nothing
        for m in LOAD_CALL.finditer(code):
            if not (m[1] and NOT_A_MODEL.search(m[1])):
                return f"{os.path.relpath(path, roots[1])} calls {m[0].rstrip('( ')}(...)"
    return None


HEREDOC = re.compile(r"(?<!<)<<(?!<)-?\s*(['\"]?)(\w+)\1")


def _heredocs(text):
    """(text without the bodies of its here-documents, the bodies). `ci/cpu-run python3 - <<'EOF'` is the usual way
    to run a short script on the worker, so a body can be a job's program; a file written with cat <<'EOF' runs nothing."""
    kept, bodies, ends = [], [], []
    for line in text.split("\n"):
        if ends:
            if line.lstrip("\t") == ends[0]:
                ends.pop(0)
            else:
                bodies[len(bodies) - len(ends)] += line + "\n"
            continue
        kept.append(line)
        new = [m[2] for m in HEREDOC.finditer(line)]
        ends += new
        bodies += [""] * len(new)
    return "\n".join(kept), bodies


cwd = call.get("cwd") or "."
found = None
for segment in re.split(r"&&|\|\||[;|\n]|\$\(", command):
    if m := re.fullmatch(r"\s*cd\s+(\S+)\s*", segment):  # `cd DIR && sbatch ...` submits from DIR
        cwd = os.path.join(cwd, os.path.expanduser(m[1].strip("\"'")))
    elif SUBMIT.search(segment) and not _asks_for_gpu(segment, cwd) and (found := _language_model(segment, cwd)):
        break
shell, bodies = _heredocs(command)
if not found and bodies and any(SUBMIT.search(s) for s in re.split(r"&&|\|\||[;|\n]|\$\(", shell)):
    for body in bodies:
        for m in LOAD_CALL.finditer(re.sub(r'("""|\'\'\')[\s\S]*?\1', "", _code(body))):
            if not (m[1] and NOT_A_MODEL.search(m[1])):
                found = found or f"its here-document calls {m[0].rstrip('( ')}(...)"
if found:
    print(f"Models never run on CPUs: {found}. Request a GPU (#SBATCH --gres=gpu:1) or use a RunPod pod, through "
          "/pod-runner (.claude/CLAUDE.md, Rules).", file=sys.stderr)
    sys.exit(2)

# No Python on this machine at all (Will, 25 October 2026; see the docstring). The command is split into simple
# commands and each one's program is found, past wrappers such as timeout or nice; Python, pytest and `uv run` are
# refused unless the program is Slurm or one of the fleet's tools. Text that only mentions Python (grep python, a
# commit message, a file written with cat <<'EOF') is not a program. `bash -c` strings and the shell scripts a
# command runs are read too. It is a heuristic, not the shell's grammar; the memory caps are the backstop.
PYTHON = re.compile(r"i?python(\d+(\.\d+)*)?|pytest|py\.test|jupyter")
EXEMPT_MODULES = {"py_compile", "compileall", "json.tool"}  # compiling and pretty-printing load nothing
WRAPPERS = {"time", "nice", "nohup", "env", "exec", "command", "builtin", "timeout", "stdbuf", "ionice",
            "if", "then", "else", "elif", "while", "until", "do", "!", "{"}  # the program follows these words
STARTS_ITS_ARGS = {"xargs", "parallel", "watch"}
TAKES_A_VALUE = {"-I", "-n", "-P", "-L", "-d", "-a", "-E", "-s", "-j"}  # options of those three
SHELLS = {"bash", "sh", "zsh", "dash"}
CONTROL = set(";&|()\n")


def _close(text, k):
    """The index just past the parenthesis that closes the one at text[k]."""
    depth, quote = 0, None
    for j in range(k, len(text)):
        c = text[j]
        if quote:
            quote = None if c == quote else quote
        elif c in "'\"":
            quote = c
        elif c in "()":
            depth += 1 if c == "(" else -1
            if depth == 0:
                return j + 1
    return len(text)


def _substitutions(text):
    """(text with each $(...) and `...` replaced by one word, each $((...)) by 0 and comments removed, the commands
    substituted). Substitutions run inside double quotes too, so "$(python3 -c ...)" counts. A # starts a comment
    only at the start of a word, as in the shell; shlex would also start one inside a word such as ${p##*:}."""
    out, inner, i, quote = [], [], 0, None
    while i < len(text):
        c = text[i]
        if quote == "'":
            quote = None if c == "'" else quote
        elif c == "#" and quote is None and (i == 0 or text[i - 1] in " \t\n;&|()"):
            i = text.find("\n", i) if "\n" in text[i:] else len(text)
            continue
        elif c == "\\":
            out.append(text[i:i + 2])
            i += 2
            continue
        elif c == "'" and quote is None:
            quote = "'"
        elif c == '"':
            quote = None if quote else '"'
        elif text.startswith("$((", i):
            out.append("0")
            i = _close(text, i + 1)
            continue
        elif text.startswith("$(", i) or c == "`":
            j = _close(text, i + 1) if c == "$" else (text.find("`", i + 1) + 1 or len(text))
            inner.append(text[i + 2:j - 1] if c == "$" else text[i + 1:j - 1])
            out.append("_")
            i = j
            continue
        out.append(c)
        i += 1
    return "".join(out), inner


def _commands(text):
    """The simple commands of shell text, each as its list of words without redirections, then those of its
    command substitutions."""
    shell, inner = _substitutions(_heredocs(text)[0].replace("\\\n", " "))
    lex = shlex.shlex(shell, posix=True, punctuation_chars=";&|()<>\n")
    lex.whitespace, lex.whitespace_split, lex.commenters = " \t\r", True, ""  # comments are gone already
    try:
        words = list(lex)
    except ValueError:  # an unclosed quote
        words = shell.split()
    command, target = [], False
    for w in words:
        if target:  # the file a redirection names
            target = False
        elif set(w) <= CONTROL or (set(w) <= CONTROL | set("<>") and set(w) & set("()")):  # also <( and >(
            if command:
                yield command
            command = []
        elif set(w) <= set("<>&|"):
            target = True
        else:
            command.append(w)
    if command:
        yield command
    for t in inner:
        yield from _commands(t)


def _program(words):
    """The words from the program on, past variable assignments and words such as `timeout 60`, `nice -n 19`, `if`."""
    i = 0
    while i < len(words):
        if words[i] == "command" and words[i + 1:i + 2] in (["-v"], ["-V"]):
            return []  # a lookup, not a run
        if re.fullmatch(r"\w+=.*", words[i]):
            i += 1
        elif os.path.basename(words[i]) in WRAPPERS:
            i += 1
            while i < len(words) and re.fullmatch(r"-.*|\w+=.*|\d+(\.\d+)?[smhd]?", words[i]):
                i += 1
        else:
            return words[i:]
    return []


def _top(path):
    """The repository that holds path, or path itself."""
    top = path
    while top != os.path.dirname(top) and not os.path.exists(os.path.join(top, ".git")):
        top = os.path.dirname(top)
    return top if os.path.exists(os.path.join(top, ".git")) else path


def _tool(word, roots):
    """Whether word names one of the harness's own tools: a file in the harness (~/commute-agents/fleet/, a repository
    of its own since 2026-10-27), in a repository's .claude/, or the research log's builder. cpu-run is among them."""
    path = _resolve(os.path.expandvars(word), roots)
    if not path:
        return False
    if os.path.dirname(path) == os.path.expanduser("~/commute-agents/fleet"):
        return True
    parts = os.path.relpath(path, _top(os.path.dirname(path))).split(os.sep)
    return (parts[0] == ".claude" and len(parts) > 1) or parts == ["docs", "research_log", "build.py"]


def _first_python(words):
    return next((j for j, w in enumerate(words) if PYTHON.fullmatch(os.path.basename(w)) or w.endswith(".py")), None)


def _runs_python(prog, roots, depth):
    """Why the command prog (its words from the program on) runs Python on this machine, or None."""
    if not prog:
        return None
    name, args = os.path.basename(prog[0]), prog[1:]
    shown = " ".join(w if len(w) <= 40 else w[:37] + "..." for w in prog[:3]).replace("\n", " ")
    if name in ("sbatch", "srun", "salloc") or _tool(prog[0], roots):
        return None
    if name in SHELLS:
        for i, a in enumerate(args[:-1]):
            if re.fullmatch(r"-[a-z]*c[a-z]*", a):
                return _python_here(args[i + 1], roots, depth + 1)
        if any(re.fullmatch(r"-[a-z]*n[a-z]*", a) for a in args):
            return None  # bash -n only checks the syntax
        script = next((a for a in args if not a.startswith("-")), None)
        return _script_runs_python(script, roots, depth) if script else None
    if name == "find":
        k = next((j for j, a in enumerate(args) if a in ("-exec", "-execdir", "-ok", "-okdir")), None)
        return None if k is None else _runs_python(args[k + 1:], roots, depth)
    if name in STARTS_ITS_ARGS:
        i = 0
        while i < len(args) and args[i].startswith("-"):
            i += 2 if args[i] in TAKES_A_VALUE else 1
        if name == "watch":  # watch 'python3 x.py' runs its argument through a shell
            return _python_here(" ".join(args[i:]), roots, depth + 1)
        return _runs_python(args[i:], roots, depth)
    if name == "uv" and args[:1] == ["run"]:
        rest = args[1:]
        if rest[:1] in (["-h"], ["--help"]):
            return None
        if (k := _first_python(rest)) is not None:
            return _runs_python(rest[k:], roots, depth)
        program = next((a for a in rest if not a.startswith("-")), "")
        return None if program and _tool(program, roots) else f"`{shown}` runs a program in the project's environment"
    if PYTHON.fullmatch(name):
        if name.startswith(("pytest", "py.test")):
            return f"`{shown}` runs tests"
        i = 0
        while i < len(args):
            a = args[i]
            if a in ("-V", "--version", "-h", "--help"):
                return None
            if a == "-m":
                module = args[i + 1] if i + 1 < len(args) else ""
                if module in EXEMPT_MODULES or module.split(".")[0] == "fleet":
                    return None
                return f"`{shown}` runs the module {module}"
            if a == "-c":
                return f"`{shown}` runs inline code"
            if a == "-":
                break
            if a in ("-W", "-X"):
                i += 2
            elif a.startswith("-"):
                i += 1
            else:
                return None if _tool(a, roots) else f"`{shown}` runs {a}"
        return f"`{shown}` reads its program from standard input"
    if name.endswith(".py"):
        return f"`{shown}` runs a Python script"
    if name == "make" and {"test", "lint"} & set(args):
        return f"`{shown}` runs `uv run` (Makefile)"
    if "/" in prog[0]:
        return _script_runs_python(prog[0], roots, depth)
    return None


def _script_runs_python(word, roots, depth):
    """Why running the file word (a Python script, or a shell script that runs Python) runs Python here, or None."""
    path = _resolve(os.path.expandvars(word), roots)
    if not path or depth > 2 or _tool(path, roots):
        return None
    text = _read(path) or ""
    first = text.split("\n", 1)[0]
    if first.startswith("#!") and "python" in first:
        return f"{word} is a Python script"
    if word.endswith(SHELL_FILE) or (first.startswith("#!") and re.search(r"\b(ba|z|da)?sh\b", first)):
        why = _python_here(_code(text), roots, depth + 1)
        return why and f"{word} runs Python: {why}"
    return None


def _python_here(text, roots, depth=0):
    """Why shell text would run Python on this machine, or None. A `cd DIR` moves where later paths are found."""
    for words in _commands(text):
        if words[0] == "cd" and len(words) == 2:
            roots = [os.path.join(roots[0], os.path.expandvars(os.path.expanduser(words[1])))] + roots[1:]
        elif why := _runs_python(_program(words), roots, depth):
            return why
    return None


here = call.get("cwd") or "."
if not os.environ.get("SLURM_JOB_ID") and (
        why := _python_here(command, [here, _top(here), os.environ.get("CLAUDE_PROJECT_DIR") or here])):
    print(f"No Python runs directly on the shared dev node, not even a heredoc that reads one file: {why}. To read "
          "JSON, use jq, which runs here. Otherwise put "
          "~/commute-agents/fleet/cpu-run in front of the command to run it on the cluster's worker through Slurm. It starts in "
          "about a second, passes standard input through (~/commute-agents/fleet/cpu-run python3 - <<'EOF' works) and asks for 1 CPU, "
          "2G of memory and 30 minutes unless you give -c, --mem or -t. The job cannot see /ephemeral, so keep its "
          "files in your clone or commute-data. GPU runs go through /pod-runner.",
          file=sys.stderr)
    sys.exit(2)
sys.exit(0)
