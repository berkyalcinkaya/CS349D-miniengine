"""
Agentic workflow benchmark — Milestone 4.

Simulates N concurrent research sessions, each following a fan-out / fan-in
pipeline that mirrors the structure from Berk & Filip's CS244 project:

    orchestrator (plan)          ← 1 active call for this session
      → 5 parallel analysts      ← 5 active calls for this session
    → orchestrator (synthesis)   ← 1 active call
      → 3 parallel reviewers     ← 3 active calls
    → orchestrator (final)       ← 1 active call

Every request passes the session ID in the OpenAI ``user`` field.  With
``--mapreduce-scheduling`` enabled on the server, the scheduler uses
``session_active[session_id]`` to give higher admission priority to sessions
with fewer active concurrent calls — i.e., sessions sitting at a fan-in
convergence point get unblocked first.

Metrics reported are **per-session** (per-workload), not per-request, because
the scheduling policy optimises session completion latency.  Individual request
TTFT / TPOT are visible in the server logs; here we track what matters for the
agentic use-case: how long does a complete workflow take end-to-end?

Usage
-----
# FIFO baseline
python -m miniengine --model Qwen/Qwen3-8B --mode paged --prefill-chunk-size 512
python -m benchmark.bench_agentic --num-sessions 8 --stagger 1.0

# MapReduce scheduling
python -m miniengine --model Qwen/Qwen3-8B --mode paged --prefill-chunk-size 512 \\
    --mapreduce-scheduling
python -m benchmark.bench_agentic --num-sessions 8 --stagger 1.0
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
import uuid

from openai import AsyncOpenAI

# ── Prompt templates ──────────────────────────────────────────────────────

TOPICS = [
    "What are the latest advances in transformer architecture efficiency?",
    "How does retrieval-augmented generation improve factual accuracy?",
    "What are the tradeoffs between RLHF and constitutional AI alignment?",
    "How do mixture-of-experts models scale compared to dense transformers?",
    "What are the key challenges in multi-modal language model training?",
]

ANALYST_QUESTIONS = [
    "What are the core technical mechanisms involved?",
    "What empirical evidence supports the claimed benefits?",
    "What are the known limitations or failure modes?",
    "How does this compare to the leading alternative approaches?",
    "What open research questions remain unresolved?",
]

REVIEWER_ASPECTS = [
    "factual accuracy and any unsupported claims",
    "logical consistency and internal contradictions",
    "completeness and important missing points",
]

_ORCHESTRATOR_PLAN = (
    'You are a research orchestrator. Topic: "{topic}". '
    "Write a 2-sentence research plan identifying the key sub-questions. Be concise."
)
_ANALYST = (
    'You are a research analyst. Answer in 2 sentences: "{question}" '
    "Focus on factual accuracy."
)
_SYNTHESIS = (
    "You are a research orchestrator. Synthesize these findings into 3 sentences:\n"
    "{findings}\nHighlight the most important points."
)
_REVIEWER = (
    "You are a reviewer. In one sentence, assess this synthesis focusing on "
    "{aspect}:\n{synthesis}"
)
_FINAL = (
    "You are a research orchestrator. Write a 2-sentence final conclusion "
    "incorporating this feedback.\nSynthesis: {synthesis}\nFeedback: {reviews}"
)


# ── LLM call helper ───────────────────────────────────────────────────────

async def _call(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    session_id: str,
    max_tokens: int = 150,
) -> str:
    """Single chat completion.  ``user`` carries the session ID so the server
    can track per-session active request counts for MapReduce scheduling."""
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        user=session_id,
    )
    return resp.choices[0].message.content or ""


# ── Single session pipeline ───────────────────────────────────────────────

async def run_session(
    session_id: str,
    topic: str,
    client: AsyncOpenAI,
    model: str,
) -> dict:
    """Run one full agentic research session and return per-phase timings.

    Returns a dict with ``total_latency`` (end-to-end wall time for this
    session) and ``phase_latencies`` (breakdown per pipeline stage).
    """
    t0 = time.perf_counter()
    phase: dict[str, float] = {}

    # ── Step 1: orchestrator plan (1 active call) ─────────────────────
    t = time.perf_counter()
    plan = await _call(client, model, _ORCHESTRATOR_PLAN.format(topic=topic), session_id)
    phase["orchestrator_plan"] = time.perf_counter() - t

    # ── Step 2: 5 parallel analysts (5 active calls) ──────────────────
    t = time.perf_counter()
    findings_list = await asyncio.gather(*[
        _call(client, model, _ANALYST.format(question=q), session_id)
        for q in ANALYST_QUESTIONS
    ])
    phase["analysts_parallel"] = time.perf_counter() - t
    findings = "\n".join(f"- {f}" for f in findings_list)

    # ── Step 3: orchestrator synthesis (1 active call) ────────────────
    t = time.perf_counter()
    synthesis = await _call(
        client, model, _SYNTHESIS.format(findings=findings), session_id
    )
    phase["orchestrator_synthesis"] = time.perf_counter() - t

    # ── Step 4: 3 parallel reviewers (3 active calls) ─────────────────
    t = time.perf_counter()
    reviews_list = await asyncio.gather(*[
        _call(client, model, _REVIEWER.format(aspect=a, synthesis=synthesis), session_id)
        for a in REVIEWER_ASPECTS
    ])
    phase["reviewers_parallel"] = time.perf_counter() - t
    reviews = " | ".join(reviews_list)

    # ── Step 5: final orchestrator (1 active call) ────────────────────
    t = time.perf_counter()
    await _call(
        client, model,
        _FINAL.format(synthesis=synthesis, reviews=reviews),
        session_id,
    )
    phase["orchestrator_final"] = time.perf_counter() - t

    return {
        "session_id": session_id,
        "topic": topic,
        "total_latency": time.perf_counter() - t0,
        "phase_latencies": phase,
    }


# ── Workload runner ───────────────────────────────────────────────────────

async def run_workload(
    base_url: str,
    model: str,
    num_sessions: int,
    stagger: float,
) -> list[dict]:
    """Launch num_sessions sessions with staggered arrivals and collect results."""
    client = AsyncOpenAI(base_url=base_url, api_key="miniengine")

    async def _session(i: int) -> dict:
        await asyncio.sleep(i * stagger)
        sid = str(uuid.uuid4())[:8]
        topic = TOPICS[i % len(TOPICS)]
        return await run_session(sid, topic, client, model)

    return list(await asyncio.gather(*[asyncio.create_task(_session(i))
                                       for i in range(num_sessions)]))


# ── Reporting — per-session (per-workload) statistics ────────────────────

def report(results: list[dict]) -> None:
    """Print per-session latency statistics.

    The unit of analysis is the *session* (the full fan-out/fan-in workflow),
    not the individual LLM call.  This reflects what the MapReduce scheduler
    optimises: end-to-end session completion time.
    """
    latencies = sorted(r["total_latency"] for r in results)
    n = len(latencies)

    def pct(p: float) -> float:
        return latencies[min(int(n * p / 100), n - 1)]

    print(f"\n{'─' * 52}")
    print(f"  Sessions          : {n}")
    print(f"  Total wall time   : {max(latencies):.1f}s  "
          f"(first→last session complete)")
    print(f"  Session latency")
    print(f"    mean            : {statistics.mean(latencies):.1f}s")
    print(f"    p50             : {pct(50):.1f}s")
    print(f"    p75             : {pct(75):.1f}s")
    print(f"    p95             : {pct(95):.1f}s")
    print(f"    p99             : {pct(99):.1f}s")

    # Per-phase breakdown (mean across sessions)
    phases = [
        "orchestrator_plan",
        "analysts_parallel",
        "orchestrator_synthesis",
        "reviewers_parallel",
        "orchestrator_final",
    ]
    print(f"  Phase breakdown (mean across sessions):")
    for ph in phases:
        vals = [r["phase_latencies"].get(ph, 0.0) for r in results]
        print(f"    {ph:<28}: {statistics.mean(vals):.1f}s")
    print(f"{'─' * 52}\n")


# ── CLI ───────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Agentic workflow benchmark (fan-out/fan-in sessions)"
    )
    p.add_argument("--base-url", default="http://localhost:8000/v1",
                   help="miniengine server base URL")
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--num-sessions", type=int, default=8,
                   help="Number of concurrent agentic sessions")
    p.add_argument("--stagger", type=float, default=1.0,
                   help="Seconds between session arrivals (0 = fully simultaneous)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    print(
        f"\nAgentic benchmark  |  sessions={args.num_sessions}  "
        f"stagger={args.stagger}s  model={args.model}"
    )
    print(f"Server: {args.base_url}")
    print("Metrics: per-session end-to-end latency (not per-request)\n")

    results = asyncio.run(
        run_workload(args.base_url, args.model, args.num_sessions, args.stagger)
    )
    report(results)


if __name__ == "__main__":
    main()
