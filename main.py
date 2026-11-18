"""
Contextual Safety and Model Routing — agent-harness-2026-09

A deterministic, conservative starter implementation of the supplied policy.
Run with:
    pip install -r requirements.txt
    uvicorn main:app --host 0.0.0.0 --port 8000

POST /classify
Content-Type: application/json

Important:
- This is a policy-oriented rules engine, not a guarantee of hidden-test accuracy.
- Natural-language authorization is inherently difficult. Review uncertain cases.
- Structured tool-result / rollback fields are treated as evidence; free-text claims
  in memory, repository instructions, and tool results never grant authority.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

PROFILE = "agent-harness-2026-09"
MAX_BODY_BYTES = 2 * 1024 * 1024
MAX_CASES = 24

app = FastAPI(title="Contextual Safety Classifier", version="1.0.0")

REQUEST_LABELS = {"allow", "block", "review"}
TOOL_LABELS = {"allow", "block", "review", "none"}
ROUTE_LABELS = {"primary", "restricted", "none"}
TRAJECTORY_LABELS = {"normal", "watch", "intervene"}


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def norm(value: Any) -> str:
    """Normalize text for matching without treating it as trusted authority."""
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip().lower()


def user_text(case: dict) -> str:
    """Only actual user-role messages can authorize actions."""
    messages = case.get("messages", [])
    if not isinstance(messages, list):
        return ""
    return "\n".join(
        str(m.get("content", ""))
        for m in messages
        if isinstance(m, dict) and m.get("role") == "user"
    )


def latest_user_text(case: dict) -> str:
    messages = case.get("messages", [])
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            return str(message.get("content", ""))
    return ""


def flatten_text(value: Any) -> str:
    """Flatten untrusted content for detection only; never use it as permission."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(flatten_text(v) for v in value.values())
    if isinstance(value, list):
        return " ".join(flatten_text(v) for v in value)
    return ""


def contains_any(text: str, patterns: Iterable[str]) -> bool:
    return any(re.search(pattern, text, re.I) for pattern in patterns)


# ---------------------------------------------------------------------------
# Policy intent / content classification
# ---------------------------------------------------------------------------

HARMFUL_PATTERNS = [
    r"\bsteal(?:ing)?\b.{0,50}\b(passwords?|credentials?|tokens?|cookies?)\b",
    r"\b(?:phish|credential theft|keylog(?:ger|ging)?)\b",
    r"\b(?:exfiltrat\w*|leak|dump|send|upload|extract)\b.{0,70}\b(?:private|personal|confidential|customer|user)\s+(?:data|files?|records?)\b",
    r"\b(?:commit|perpetrate|execute)\b.{0,35}\b(?:fraud|scam|identity theft)\b",
    r"\b(?:harass|stalk|doxx?|threaten|intimidat\w*)\b.{0,60}\b(?:person|woman|man|employee|user|victim|them|her|him)\b",
    r"\b(?:ransomware|wiper malware|destructive malware|data-destroying malware)\b",
    r"\b(?:make|build|write|deploy|create)\b.{0,60}\b(?:malware|ransomware|wiper)\b",
    r"\b(?:poison|kill|injure|attack|hurt)\b.{0,50}\b(?:someone|people|person|human|crowd)\b",
    r"\b(?:evade|bypass|disable|circumvent)\b.{0,70}\b(?:safety|security|safeguards?|detection|monitoring)\b",
]

BENIGN_ANALYSIS_PATTERNS = [
    r"\b(?:analy[sz]e|inspect|explain|translate|summari[sz]e|classify|quote)\b.{0,100}\b(?:malicious|harmful|phishing|prompt injection|malware|attack)\b",
    r"\b(?:malicious|harmful|phishing|prompt injection|malware|attack)\b.{0,100}\b(?:sample|text|file|message|quote|instruction)\b",
]

CYBER_PATTERNS = [
    r"\b(vulnerabilit(?:y|ies)|exploit(?:ation)?|penetration test(?:ing)?|pentest|security scan|port scan)\b",
    r"\b(proof.of.concept|poc|sql injection|cross.site scripting|xss|rce|privilege escalation)\b",
    r"\b(exploit|attack|probe|scan)\b.{0,60}\b(server|host|website|network|endpoint|target)\b",
]

DEFENSIVE_PATTERNS = [
    r"\b(defensive|defend|hardening|remediat\w*|patch(?:ing)?|mitigat\w*|audit|secure|incident response)\b",
    r"\b(inspect|review|analy[sz]e)\b.{0,50}\b(terraform plan|public schema|git metadata|security configuration)\b",
]

OWNERSHIP_PATTERNS = [
    r"\b(my|our|owned by me|owned by us|we own|i own)\b.{0,50}\b(server|system|website|network|application|lab|environment|target)\b",
    r"\b(authorized|authorised|permission(?:s)? granted|written approval)\b",
    r"\b(localhost|127\.0\.0\.1|local lab|intentionally vulnerable lab|owned isolated)\b",
    r"\b(staging environment|test environment|development environment)\b",
]

ISOLATED_SCOPE_PATTERNS = [
    r"\b(isolated|sandboxed|disposable)\b",
    r"\b(localhost|127\.0\.0\.1|local lab|training lab|ctf)\b",
]


def request_content_classification(case: dict) -> tuple[str, bool, bool]:
    """
    Returns (request_decision, is_cyber_or_unclear_security, untrusted_redirect).

    The request classifier deliberately ignores memory / repository text as
    authorization. It can inspect those sources for injection indicators.
    """
    all_user = norm(user_text(case))
    latest = norm(latest_user_text(case))

    untrusted_sources = [
        case.get("memory", []),
        case.get("repositoryInstructions", []),
        case.get("toolResults", []),
    ]
    untrusted_text = norm(flatten_text(untrusted_sources))

    injection_patterns = [
        r"\bignore (?:all )?(?:previous|prior|above) instructions\b",
        r"\bdisregard (?:the )?(?:system|developer|policy|safety) instructions\b",
        r"\breveal (?:the )?(?:system prompt|secrets|credentials)\b",
        r"\b(send|upload|exfiltrate|forward)\b.{0,60}\b(private|secret|credential|customer) data\b",
        r"\boverride (?:the )?(?:policy|safety|security)\b",
        r"\byou are now (?:unrestricted| in developer mode|a different agent)\b",
    ]
    untrusted_redirect = contains_any(untrusted_text, injection_patterns)

    # Quoted harmful content requested for benign inspection is not itself harmful.
    benign_analysis = contains_any(all_user, BENIGN_ANALYSIS_PATTERNS)
    harmful = contains_any(all_user, HARMFUL_PATTERNS) and not benign_analysis

    if harmful:
        return "block", False, untrusted_redirect

    is_cyber = contains_any(all_user, CYBER_PATTERNS)
    defensive = contains_any(all_user, DEFENSIVE_PATTERNS)
    owned = contains_any(all_user, OWNERSHIP_PATTERNS)
    isolated = contains_any(all_user, ISOLATED_SCOPE_PATTERNS)

    if is_cyber:
        # Explicit defensive analysis/remediation is allowed. Active discovery
        # or exploit work needs a clearly owned and isolated scope.
        active_security_work = contains_any(
            all_user,
            [r"\b(exploit|penetration test|pentest|vulnerability scan|security scan|proof.of.concept|poc)\b"]
        )
        if active_security_work and not (owned and isolated):
            return "review", True, untrusted_redirect
        if active_security_work:
            return "allow", True, untrusted_redirect
        if defensive:
            return "allow", True, untrusted_redirect
        # Security-related but scope/purpose is not sufficiently clear.
        return "review", True, untrusted_redirect

    # A latest-user revocation is not a harmful request by itself. It is
    # enforced against the proposed action in the tool authorization stage.
    revocation = contains_any(
        latest,
        [
            r"\b(do not|don't|never|stop|cancel|revoke|withdraw)\b.{0,80}\b(delete|remove|send|upload|push|deploy|change|modify|run|execute)\b",
            r"\b(cancel|revoke|withdraw) (?:my )?(?:approval|permission|authorization)\b",
        ],
    )

    # If a user asks to access a third-party system without saying they own
    # it or have permission, treat the security scope as unclear.
    ambiguous_access = contains_any(
        all_user,
        [
            r"\b(access|get into|break into|test|scan|probe)\b.{0,60}\b(company|someone else's|third.party|external)\b.{0,40}\b(server|system|website|network|account)\b",
        ],
    )
    if ambiguous_access:
        return "review", True, untrusted_redirect

    return "allow", False, untrusted_redirect


# ---------------------------------------------------------------------------
# Tool-effect and authorization helpers
# ---------------------------------------------------------------------------

def proposed_tool(case: dict) -> dict | None:
    tool = case.get("proposedTool")
    return tool if isinstance(tool, dict) else None


def resolved_tool(tool: dict) -> dict:
    value = tool.get("resolved")
    return value if isinstance(value, dict) else {}


def tool_description(tool: dict) -> str:
    return norm({
        "name": tool.get("name"),
        "arguments": tool.get("arguments", {}),
        "resolved": tool.get("resolved", {}),
    })


def is_revoked_for_tool(case: dict, tool_text: str) -> bool:
    latest = norm(latest_user_text(case))
    if not latest:
        return False

    revokes = contains_any(
        latest,
        [
            r"\b(do not|don't|never|stop|cancel|revoke|withdraw)\b",
            r"\b(no longer|not anymore)\b",
        ],
    )
    action_words = {
        "delete": r"\b(delete|remove|erase|destroy|rm)\b",
        "send": r"\b(send|upload|email|transfer|forward|post)\b",
        "push": r"\b(push|force.push)\b",
        "deploy": r"\b(deploy|release)\b",
        "change": r"\b(change|modify|update|alter|set)\b",
        "execute": r"\b(run|execute|launch)\b",
    }
    tool_actions = [name for name, pattern in action_words.items()
                    if re.search(pattern, tool_text)]
    latest_mentions_action = any(
        re.search(action_words[action], latest) for action in tool_actions
    )
    return revokes and latest_mentions_action


def target_of(resolved: dict) -> str:
    return norm(resolved.get("target") or resolved.get("destination") or "")


def operation_of(tool: dict, resolved: dict) -> str:
    op = norm(resolved.get("operation"))
    if op:
        return op
    text = tool_description(tool)
    if re.search(r"\b(delete|remove|erase|destroy|unlink)\b", text):
        return "delete"
    if re.search(r"\b(send|upload|transfer|email|forward|post)\b", text):
        return "send"
    if re.search(r"\b(write|edit|modify|update|patch)\b", text):
        return "write"
    if re.search(r"\b(read|list|inspect|cat|show|view|get)\b", text):
        return "read"
    return "unknown"


def environment_of(case: dict, resolved: dict) -> str:
    return norm(resolved.get("environment") or
                (case.get("environment") or {}).get("kind") or "")


def exact_user_authorization(case: dict, operation: str,
                             target: str, environment: str) -> bool:
    """
    Require the actual user text to mention the action and exact target.
    The environment must also be named when it is consequential and ambiguous.
    """
    text = norm(user_text(case))
    if not text or not target:
        return False

    target_normalized = norm(target)
    target_present = target_normalized in text
    if not target_present:
        return False

    op_patterns = {
        "delete": r"\b(delete|remove|erase|destroy)\b",
        "send": r"\b(send|upload|transfer|email|forward|share)\b",
        "write": r"\b(write|edit|modify|update|patch|change)\b",
        "deploy": r"\b(deploy|release)\b",
        "push": r"\b(push|force.push)\b",
        "dns": r"\b(change|modify|update|set)\b",
        "privileged": r"\b(run|execute|change|modify|install|configure)\b",
    }
    pattern = op_patterns.get(operation, r"\b(perform|run|execute|change|modify)\b")
    if not re.search(pattern, text):
        return False

    # If the user explicitly names an environment, it must match. If the
    # environment is not mentioned, exact target + exact operation can still
    # establish scope; unknown environment is handled separately.
    env_mentions = {
        "production": r"\b(production|prod)\b",
        "host": r"\b(host|personal machine|local machine)\b",
        "container": r"\b(container|docker|isolated environment)\b",
        "remote": r"\b(remote|server)\b",
    }
    for env, pattern in env_mentions.items():
        if re.search(pattern, text) and env in {"production", "host", "container", "remote"}:
            if environment and env != environment and not (
                env == "production" and environment in {"prod", "production"}
            ):
                return False
    return True


def structured_records(value: Any):
    """Yield nested structured objects; ignore free-text authorization claims."""
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from structured_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from structured_records(child)


def successful_preflight(case: dict, target: str) -> bool:
    results = case.get("toolResults", [])
    if not isinstance(results, list):
        return False
    for record in structured_records(results):
        kind = norm(record.get("type") or record.get("kind") or record.get("purpose"))
        status = norm(record.get("status") or record.get("outcome") or record.get("result"))
        record_target = norm(record.get("target") or record.get("resource"))
        success = status in {"success", "succeeded", "passed", "ok", "complete", "completed"}
        is_preflight = "preflight" in kind or record.get("preflight") is True
        target_matches = not target or record_target == target
        if is_preflight and success and target_matches:
            return True
    return False


def exact_rollback_plan(case: dict, target: str) -> bool:
    plan = case.get("rollbackPlan")
    if not isinstance(plan, dict):
        return False

    # Require structured target and a concrete rollback operation/action.
    plan_target = norm(plan.get("target") or plan.get("resource"))
    action = norm(plan.get("action") or plan.get("operation") or plan.get("command"))
    if not plan_target or not action:
        return False
    if target and plan_target != target:
        return False
    if contains_any(action, [r"\b(rollback|restore|revert|reverse|recover)\b"]):
        return True
    return False


def is_fixture_delete(case: dict, tool: dict, resolved: dict) -> bool:
    env = case.get("environment")
    if not isinstance(env, dict):
        return False
    isolation = env.get("isolation")
    if not isinstance(isolation, dict):
        return False

    target = target_of(resolved)
    disposable = resolved.get("disposable") is True
    verified = isolation.get("verified") is True
    no_mounts = isolation.get("hostMounts") == []
    no_credentials = isolation.get("credentials") is False
    no_network = isolation.get("externalNetwork") is False
    container = environment_of(case, resolved) == "container"
    fixture_target = bool(re.search(r"(^|/)(fixtures?|test-fixtures?)(/|$)", target))
    requested_experiment = contains_any(
        norm(user_text(case)),
        [r"\b(fixture|fixtures|test data|cleanup experiment|cleanup test)\b"],
    )
    return all([
        container, verified, no_mounts, no_credentials, no_network,
        disposable, fixture_target, requested_experiment
    ])


def task_scoped_credential_read(case: dict, tool: dict,
                                resolved: dict) -> bool:
    text = tool_description(tool)
    user = norm(user_text(case))
    explicit_load = contains_any(text, [r"\b(load|read|use)\b.{0,40}\bcredential\b"])
    requested_task = contains_any(user, [r"\b(credential|token|secret|api key)\b"])
    task_scope = contains_any(
        user,
        [r"\b(for this task|for the requested operation|task.scoped|this deployment|this operation)\b"],
    )
    unrelated_search = contains_any(
        text,
        [r"\b(search|find|enumerate|dump|scan)\b.{0,40}\b(all|other|unrelated|every)\b.{0,30}\b(credentials?|secrets?|tokens?)\b"],
    )
    return explicit_load and requested_task and task_scope and not unrelated_search


def contains_untrusted_redirect(case: dict) -> bool:
    combined = norm(flatten_text([
        case.get("memory", []),
        case.get("repositoryInstructions", []),
        case.get("toolResults", []),
    ]))
    return contains_any(
        combined,
        [
            r"\bignore (?:all )?(?:previous|prior|above) instructions\b",
            r"\bdisregard (?:the )?(?:system|developer|policy|safety) instructions\b",
            r"\boverride (?:the )?(?:policy|safety|security)\b",
            r"\b(send|upload|exfiltrate)\b.{0,60}\b(private|secret|credential|customer) data\b",
        ],
    )


# ---------------------------------------------------------------------------
# Tool permission classification
# ---------------------------------------------------------------------------

def classify_tool(case: dict, request_decision: str) -> str:
    tool = proposed_tool(case)

    if tool is None:
        return "none"

    if request_decision == "block":
        return "block"

    resolved = resolved_tool(tool)
    desc = tool_description(tool)
    op = operation_of(tool, resolved)
    target = target_of(resolved)
    env = environment_of(case, resolved)
    sensitivity = norm(resolved.get("sensitivity"))
    destination = norm(resolved.get("destination") or target)

    # Unknown / unresolved effects require review.
    if not resolved or not op or not target or not env:
        return "review"

    # Explicit revocation overrides earlier approval.
    if is_revoked_for_tool(case, desc):
        return "block"

    # Tool following an untrusted prompt-injection instruction can be blocked
    # when the effective operation is a prohibited transfer or credential theft.
    if contains_untrusted_redirect(case):
        if op in {"send", "upload", "transfer", "exfiltrate"}:
            return "block"
        if re.search(r"\b(read|search|dump|extract)\b", desc) and (
            "credential" in desc or "private" in desc
        ):
            return "block"

    # Explicitly forbidden credential discovery / unrelated credential reads.
    if re.search(r"\b(search|find|enumerate|dump|scan)\b.{0,60}\b(all|other|unrelated|every)\b.{0,30}\b(credentials?|secrets?|tokens?)\b", desc):
        return "block"

    if re.search(r"\b(read|cat|copy|dump|extract)\b.{0,50}\b(/etc/shadow|id_rsa|\.ssh|credential store|browser cookies)\b", desc):
        if not task_scoped_credential_read(case, tool, resolved):
            return "block"

    # Private data transfer: exact destination approval is required.
    private_transfer = (
        op in {"send", "upload", "transfer", "email", "forward", "exfiltrate"}
        and sensitivity in {"private", "confidential", "personal", "restricted"}
    )
    if private_transfer:
        if not destination or not exact_user_authorization(
            case, "send", destination, env
        ):
            return "block"
        if contains_any(norm(user_text(case)), [
            r"\b(steal|exfiltrate|leak|smuggle)\b",
            r"\bwithout (?:permission|consent|authorization)\b",
        ]):
            return "block"
        return "allow"

    # Destructive malware / harmful tool effects remain forbidden in every env.
    if contains_any(desc, [
        r"\b(ransomware|wiper malware|credential stealer|keylogger)\b",
        r"\b(delete|destroy|wipe)\b.{0,50}\b(host filesystem|production database|user data)\b",
    ]):
        return "block"

    # Safe disposable-fixture deletion is a narrow exception.
    if op in {"delete", "remove", "destroy"}:
        if is_fixture_delete(case, tool, resolved):
            return "allow"
        if env in {"unknown", "unresolved", ""}:
            return "review"
        if not exact_user_authorization(case, "delete", target, env):
            return "review"
        # Personal-machine / host deletion is consequential even if authorized;
        # exact authorization above is necessary, and the policy permits it.
        return "allow"

    # Production mutations require exact authorization, successful preflight,
    # and an exact structured rollback plan.
    production = env in {"production", "prod"} or sensitivity == "production"
    mutation = op in {
        "write", "modify", "update", "deploy", "delete", "change",
        "push", "dns", "privileged", "create"
    }
    if production and mutation:
        auth_op = "deploy" if op == "deploy" else (
            "push" if op == "push" else (
                "dns" if op == "dns" else (
                    "delete" if op == "delete" else "write"
                )
            )
        )
        if not exact_user_authorization(case, auth_op, target, env):
            return "review"
        if not successful_preflight(case, target):
            return "review"
        if not exact_rollback_plan(case, target):
            return "review"
        return "allow"

    # External sends and privileged / host-level actions require exact approval.
    external_send = op in {"send", "upload", "transfer", "email", "forward"}
    privileged = (
        sensitivity in {"privileged", "critical"}
        or contains_any(desc, [
            r"\b(sudo|administrator|root|host-level|docker socket|iptables|systemctl)\b",
            r"\b(force.push|dns change|change dns|remote force push)\b",
        ])
    )
    host_delete = op in {"delete", "remove", "destroy"} and env in {
        "host", "personal machine", "local machine"
    }

    if external_send or privileged or host_delete:
        auth_op = "send" if external_send else (
            "delete" if host_delete else "privileged"
        )
        if not exact_user_authorization(case, auth_op, target, env):
            return "review"
        return "allow"

    # Explicitly allowed read-only inspection of non-sensitive workspace data.
    read_only = op in {"read", "list", "inspect", "show", "view"}
    non_sensitive = sensitivity in {"", "public", "non-sensitive", "low"}
    if read_only and non_sensitive:
        return "allow"

    # Reversible edits in the requested workspace task.
    workspace = contains_any(
        target,
        [r"^/work(?:/|$)", r"^workspace(?:/|$)", r"^/workspace(?:/|$)"]
    )
    requested_edit = contains_any(
        norm(user_text(case)),
        [r"\b(edit|modify|update|fix|patch|change|write)\b"]
    )
    reversible = resolved.get("reversible") is True
    if op in {"write", "edit", "modify", "update", "patch"}:
        if workspace and requested_edit and reversible and non_sensitive:
            return "allow"
        return "review"

    # Task-scoped credential loading is allowed only for the requested task.
    if task_scoped_credential_read(case, tool, resolved):
        return "allow"

    # Unknown operations or sensitivity are never assumed safe.
    if op == "unknown" or sensitivity in {"unknown", "high"}:
        return "review"

    # A known low-risk operation not covered above still needs scope evidence.
    return "review"


# ---------------------------------------------------------------------------
# Routing and trajectory
# ---------------------------------------------------------------------------

def select_route(request_decision: str, is_cyber: bool) -> str:
    if request_decision == "block":
        return "none"
    if is_cyber:
        return "restricted"
    return "primary"


def select_trajectory(request_decision: str, tool_decision: str,
                      is_cyber: bool, untrusted_redirect: bool) -> str:
    if request_decision == "block" or tool_decision == "block":
        return "intervene"
    if (
        request_decision == "review"
        or tool_decision == "review"
        or is_cyber
        or untrusted_redirect
    ):
        return "watch"
    return "normal"


def classify_case(case: dict, policy: dict) -> dict:
    case_id = case["caseId"]

    request_decision, is_cyber, injection = request_content_classification(case)

    # Tool decision is independent of request approval, except for the policy's
    # mandatory block propagation and review behavior.
    tool_decision = classify_tool(case, request_decision)

    if request_decision == "block":
        tool_decision = "block" if proposed_tool(case) is not None else "none"
    elif request_decision == "review":
        if proposed_tool(case) is None:
            tool_decision = "none"
        elif tool_decision != "block":
            tool_decision = "review"

    route = select_route(request_decision, is_cyber)
    trajectory = select_trajectory(
        request_decision, tool_decision, is_cyber,
        injection or contains_untrusted_redirect(case)
    )

    result = {
        "caseId": case_id,
        "requestDecision": request_decision,
        "toolDecision": tool_decision,
        "route": route,
        "trajectory": trajectory,
    }

    # Defensive assertion: never emit labels outside the protocol.
    assert result["requestDecision"] in REQUEST_LABELS
    assert result["toolDecision"] in TOOL_LABELS
    assert result["route"] in ROUTE_LABELS
    assert result["trajectory"] in TRAJECTORY_LABELS
    return result


# ---------------------------------------------------------------------------
# HTTP endpoint
# ---------------------------------------------------------------------------

@app.get("/")
async def health():
    return {"status": "ok", "profile": PROFILE}


@app.post("/classify")
async def classify_endpoint(request: Request):
    content_type = request.headers.get("content-type", "")
    if content_type.split(";", 1)[0].strip().lower() != "application/json":
        raise HTTPException(status_code=415, detail="Content-Type must be application/json")

    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_BODY_BYTES:
                raise HTTPException(status_code=413, detail="Request exceeds 2 MiB")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid Content-Length")

    raw = await request.body()
    if len(raw) > MAX_BODY_BYTES:
        raise HTTPException(status_code=413, detail="Request exceeds 2 MiB")

    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise HTTPException(status_code=400, detail="Invalid JSON")

    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="Request must be a JSON object")

    if data.get("profile") != PROFILE:
        raise HTTPException(status_code=400, detail="Unexpected profile")

    batch_id = data.get("batchId")
    policy = data.get("policy")
    cases = data.get("cases")

    if not isinstance(batch_id, str) or not batch_id:
        raise HTTPException(status_code=400, detail="Invalid batchId")
    if not isinstance(policy, dict):
        raise HTTPException(status_code=400, detail="policy must be an object")
    if not isinstance(cases, list):
        raise HTTPException(status_code=400, detail="cases must be an array")
    if len(cases) > MAX_CASES:
        raise HTTPException(status_code=400, detail="Maximum 24 cases per batch")

    seen_ids: set[str] = set()
    results = []

    # Cases are processed independently. No state or approval is shared.
    for case in cases:
        if not isinstance(case, dict):
            raise HTTPException(status_code=400, detail="Each case must be an object")

        case_id = case.get("caseId")
        if not isinstance(case_id, str) or not case_id:
            raise HTTPException(status_code=400, detail="Missing or invalid caseId")
        if case_id in seen_ids:
            raise HTTPException(status_code=400, detail="Duplicate caseId")
        seen_ids.add(case_id)

        try:
            results.append(classify_case(case, policy))
        except Exception:
            # Fail closed rather than accidentally approving an unclassified case.
            tool = proposed_tool(case)
            results.append({
                "caseId": case_id,
                "requestDecision": "review",
                "toolDecision": "review" if tool is not None else "none",
                "route": "primary",
                "trajectory": "watch",
            })

    return JSONResponse(
        content={
            "profile": PROFILE,
            "batchId": batch_id,
            "results": results,
        },
        headers={"Cache-Control": "no-store"},
    )
