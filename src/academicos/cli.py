"""CLI entry point: `python -m academicos.cli <command>`.

Commands:
  ingest      discover + validate + register corpus files
  parse       run ensemble parse over registered sources
  extract     extract academic objects + chunk registered docs
  build-graph build the knowledge graph from extracted objects
  index       (re)build the retrieval index
  search      interactive/one-shot search over the index
  serve       run the FastAPI server
  stats       corpus/registry/graph stats
  learn       record/replay/stats learner interaction events
  sync-gcs    push local artifacts to GCS (when configured)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .algorithms.learner_model import Interaction
from .config import Config
from .graph.builder import GraphBuilder
from .graph.store import GraphStore
from .ingest.pipeline import safe_name
from .retrieval.hybrid import HybridRetriever
from .retrieval.index import ChunkIndex
from .storage.registry import SourceRegistry
from .storage.event_store import EventStore

log = logging.getLogger("academicos")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _dt(value) -> "object":
    from .models.enums import DocType

    try:
        return DocType(value) if value else DocType.UNKNOWN
    except ValueError:
        return DocType.UNKNOWN


def cmd_ingest(args: argparse.Namespace) -> int:
    from .ingest.pipeline import IngestionPipeline
    from .storage.base import LocalStore

    cfg = Config.load()
    registry = SourceRegistry(cfg.registry_db)
    store = LocalStore(cfg.documents_dir)
    pipe = IngestionPipeline(cfg.corpus_root, registry, store)
    result = pipe.run(max_files=args.limit, skip_validated=not args.reingest)
    print(f"accepted={len(result.accepted)} rejected={len(result.rejected)} duplicates={len(result.duplicates)}")
    for sf, problems in result.rejected[:20]:
        print(f"  REJECT {sf.path.name}: {problems}")
    registry.close()
    return 0


def cmd_parse(args: argparse.Namespace) -> int:
    from .parse.ensemble import EnsembleParser

    cfg = Config.load()
    registry = SourceRegistry(cfg.registry_db)
    # config.parse_priority was silently ignored here, so every run tried the
    # unlimited_ocr provider (and paid its connection timeout) even when the
    # config asked for pdf_native only.
    parser = EnsembleParser(priority=cfg.parse_priority, vlm_mode=args.vlm_mode,
                            vlm_url=args.vlm_url, vlm_device=cfg.vlm_device)
    store_dir = cfg.documents_dir
    sources = registry.all()
    if args.doc_id:
        sources = [s for s in sources if s["source_id"] == args.doc_id]
    done = 0
    failed = 0
    for src in sources:
        if src["status"] in ("parsed", "extracted") and not args.reparse:
            continue
        key = src["file_key"]
        local = store_dir / key
        if not local.exists():
            log.warning("missing artifact %s", key)
            continue
        try:
            out = parser.parse(local, src["source_id"], _dt(src.get("doc_type")), max_pages=cfg.max_pages_per_doc)
        except Exception as e:
            log.warning("parse failed %s: %s", src["source_id"], e)
            registry.update(src["source_id"], status="failed", notes=str(e)[:200])
            failed += 1
            continue
        dest = cfg.parse_dir / f"{safe_name(src['source_id'])}.json"
        dest.write_text(out.model_dump_json(indent=2), encoding="utf-8")
        registry.update(src["source_id"], status="parsed", quality_score=round(out.quality_score, 3))
        done += 1
        if done % 25 == 0:
            print(f"parsed {done}/{len(sources)}")
    print(f"parsed {done} documents ({failed} failed)")
    registry.close()
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    from .extract.academic import extract_marking_points, extract_questions
    from .extract.chunking import chunk_document
    from .models.document import ParsedDocument
    from .models.enums import DocType

    cfg = Config.load()
    registry = SourceRegistry(cfg.registry_db)
    index = ChunkIndex(cfg.index_db)
    sources = [s for s in registry.all() if s["status"] in ("parsed", "extracted")]
    if args.doc_id:
        sources = [s for s in sources if s["source_id"] == args.doc_id]
    for src in sources:
        parse_file = cfg.parse_dir / f"{safe_name(src['source_id'])}.json"
        if not parse_file.exists():
            continue
        doc = ParsedDocument.model_validate_json(parse_file.read_text(encoding="utf-8"))
        doc.doc_type = DocType(src["doc_type"]) if src["doc_type"] else DocType.UNKNOWN
        chunks = chunk_document(doc)
        index.add(chunks, replace_doc=True)
        payload = {"chunks": [c.__dict__ for c in chunks]}
        if src["doc_type"] == "question_paper":
            payload["questions"] = [q.model_dump(mode="json") for q in extract_questions(doc, doc.doc_type)]
        if src["doc_type"] == "marking_scheme":
            payload["marking_points"] = [m.model_dump(mode="json") for m in extract_marking_points(doc)]
        (cfg.extracted_dir / f"{safe_name(src['source_id'])}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        registry.update(src["source_id"], status="extracted")
    print(f"extracted {len(sources)} documents into index ({index.count()} chunks)")
    index.close()
    registry.close()
    return 0


def cmd_build_graph(args: argparse.Namespace) -> int:
    from .models.academic import AnswerScheme, MarkingPoint, Question, QuestionPaper
    from .models.enums import EdgeType

    cfg = Config.load()
    registry = SourceRegistry(cfg.registry_db)
    store = GraphStore(cfg.graph_db)
    builder = GraphBuilder(store)
    sources = registry.all()
    if args.doc_id:
        sources = [s for s in sources if s["source_id"] == args.doc_id]
    for src in sources:
        extract_file = cfg.extracted_dir / f"{safe_name(src['source_id'])}.json"
        if not extract_file.exists():
            continue
        payload = json.loads(extract_file.read_text(encoding="utf-8"))
        doc_node = builder.upsert_object(QuestionPaper(
            canonical_id=f"cbse:doc:{src['source_id']}",
            title=src["title"] or src["source_id"],
            academic_year=src["academic_year"],
            subject=src["subject"],
        ))
        for q in payload.get("questions", []):
            qobj = Question.model_validate(q)
            qobj.question_paper_id = doc_node
            qid = builder.build_question(qobj)
            builder.link(doc_node, qid, EdgeType.CONTAINS)
        for mp in payload.get("marking_points", []):
            scheme = AnswerScheme(
                canonical_id=f"cbse:scheme:{src['source_id']}",
                document_id=src["source_id"],
                title=f"Scheme {src['source_id']}",
                marking_points=[MarkingPoint.model_validate(mp)],
            )
            sid = builder.build_answer_scheme(scheme)
            builder.link(doc_node, sid, EdgeType.HAS_MARKING_POINT)
    n, e = store.count()
    snap = store.snapshot(notes=f"built from {len(sources)} sources")
    print(f"graph: {n} nodes, {e} edges, snapshot {snap.snapshot_id}")
    store.close()
    registry.close()
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    cfg = Config.load()
    index = ChunkIndex(cfg.index_db)
    graph = GraphStore(cfg.graph_db)
    retriever = HybridRetriever(index, graph)
    if args.query:
        queries = [args.query]
    else:
        queries = iter(sys.stdin.read().splitlines() or [])
    for q in queries:
        if not q.strip():
            continue
        hits = retriever.search(q, limit=args.limit)
        print(f"\n== {q} ==")
        for h in hits:
            print(f"[{h.score:.3f}] {h.chunk.document_id} p{h.chunk.page} <{h.chunk.heading}>")
            print(f"    {h.chunk.text[:200].replace(chr(10), ' ').encode('ascii', 'replace').decode()}")
    index.close()
    graph.close()
    return 0


def cmd_openie(args: argparse.Namespace) -> int:
    from .extract.local_ie import LocalIE
    from .extract.openie import OpenIE
    from .llm.sarvam import SarvamLLM

    cfg = Config.load()
    registry = SourceRegistry(cfg.registry_db)
    store = GraphStore(cfg.graph_db)
    builder = GraphBuilder(store)

    if args.extractor == "local":
        openie = LocalIE()
        print(f"local extractor: {openie._nlp_name} (deterministic, no API)")
    else:
        llm = SarvamLLM(api_key=cfg.llm_api_key, base_url=cfg.llm_base_url,
                        model=args.model or cfg.llm_model, timeout=cfg.llm_timeout)
        if not llm.available:
            print("no LLM API key (set SARVAM_API_KEY or ACOS_LLM_API_KEY)")
            registry.close(); store.close()
            return 1
        openie = OpenIE(llm, max_chars=cfg.openie_max_chunk_chars)

    sources = [s for s in registry.all() if s["status"] in ("parsed", "extracted")]
    if args.doc_id:
        sources = [s for s in sources if s["source_id"] == args.doc_id]
    if args.limit:
        sources = sources[: args.limit]

    total_nodes = total_edges = total_chunks = 0
    for src in sources:
        extract_file = cfg.extracted_dir / f"{safe_name(src['source_id'])}.json"
        if not extract_file.exists():
            continue
        payload = json.loads(extract_file.read_text(encoding="utf-8"))
        chunks = payload.get("chunks", [])[: args.max_chunks]
        doc_nodes = doc_edges = 0
        for chunk in chunks:
            text = (chunk.get("heading", "") + " " + chunk.get("text", "")).strip()
            if not text:
                continue
            result = openie.extract(text)
            if not result.triples:
                continue
            n, e = builder.build_triples(src["source_id"], result.triples, page=chunk.get("page", 0))
            doc_nodes += n
            doc_edges += e
            total_chunks += 1
        print(f"  {src['source_id'][:16]} triples: {doc_edges} (entities {doc_nodes}, chunks {len(chunks)})")
        total_nodes += doc_nodes
        total_edges += doc_edges
    if total_edges:
        snap = store.snapshot(notes=f"openie from {len(sources)} sources")
        print(f"openie: +{total_nodes} entity nodes, +{total_edges} edges from {total_chunks} chunks (snapshot {snap.snapshot_id})")
    store.close()
    registry.close()
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    """Study plan for a concept, optionally skipping mastered steps."""
    from .algorithms.learner_model import LearnerModel
    from .graph.query import prereq_plan
    from .models.enums import NodeType

    cfg = Config.load()
    store = GraphStore(cfg.graph_db)
    try:
        learner = None
        if args.learner:
            from .storage.event_store import EventStore
            es = EventStore(cfg.events_db)
            try:
                learner = es.replay(args.learner)
            finally:
                es.close()
        ntype = NodeType.LEARNING_OUTCOME if args.lo else NodeType.CONCEPT
        plan = prereq_plan(store, args.target, min_confidence=args.min_confidence,
                           max_depth=args.max_depth, ntype=ntype,
                           learner=learner, mastery_gate=args.mastery_gate)
        if plan["concept"] is None:
            print(f"no {ntype.value} node matching '{args.target}'")
            return 1
        print(f"plan for {plan['concept']} ({ntype.value})"
              + (f", learner {args.learner}" if learner else ""))
        for s in plan["steps"]:
            tag = " [done]" if s["mastered"] else ""
            print(f"  {s['label']}{tag}  diff={s['difficulty']} "
                  f"hours={s['learning_hours']} bloom={s['bloom']} "
                  f"conf={s['confidence']} spacing={s['spacing_days']}d")
    finally:
        store.close()
    return 0


def cmd_revise(args: argparse.Namespace) -> int:
    """What to revise today: concepts whose predicted retention is below
    the forgetting threshold, from the learner's event store (FSRS-backed)."""
    from .algorithms.forgetting import (ForgettingModel, ForgettingParams,
                                        RevisionScheduler)

    cfg = Config.load()
    store = EventStore(cfg.events_db)
    try:
        model = store.replay(args.learner)
        if not model.concepts:
            print(f"no interactions recorded for learner '{args.learner}'")
            return 1
        sched = RevisionScheduler(
            ForgettingModel(ForgettingParams(model=args.model)))
        plan = sched.plan(model)
        print(f"revision plan for {model.learner_id} "
              f"({plan.totals['concepts']} concepts tracked)")
        if plan.due_today:
            print(f"due today ({plan.totals['due_today']}):")
            for cid, retention, priority in plan.due_today:
                print(f"  {cid:<24} retention={retention:.2f} "
                      f"priority={priority:.3f} -> {sched.minutes_per_concept} min")
        else:
            print("nothing due today; next reviews:")
            for cid, due in plan.next_reviews:
                print(f"  {cid:<24} next={due}")
        if not plan.confident:
            print("(low evidence: keep logging answers for sharper predictions)")
    finally:
        store.close()
    return 0


def cmd_misconceptions(args: argparse.Namespace) -> int:
    """P1.10: known misconceptions for a concept, + answer diagnosis."""
    from .algorithms.misconceptions import (for_concept, generic_signals,
                                            match_answer)
    from .models.enums import NodeType

    cfg = Config.load()
    store = GraphStore(cfg.graph_db)
    try:
        if args.answer:
            hits = match_answer(args.concept, args.answer)
            print(f"diagnosis for '{args.concept}' answer '{args.answer}':")
            if hits:
                for m, s in hits:
                    print(f"  {s:.2f}  {m.id:<24} {m.belief}  [{m.source}]")
            sigs = generic_signals(args.answer)
            if sigs:
                print("  signals:", ", ".join(sigs))
            else:
                print("  no catalog misconception matched")
            return 0

        nodes = store.query_nodes(NodeType.CONCEPT, label_contains=args.concept, limit=3)
        if not nodes:
            print(f"no concept nodes matching '{args.concept}'")
            return 1
        for n in nodes[:2]:
            label = n.label
            a = n.attributes or {}
            profiled = a.get("misconceptions", [])
            catalog = for_concept(label)
            print(f"{label} ({len(profiled)} profiled, {len(catalog)} catalog):")
            for m in catalog:
                print(f"  [{m.id}] {m.belief}")
                print(f"       {m.source}")
            if not catalog and not profiled:
                print("  no known misconceptions")
    finally:
        store.close()
    return 0


def cmd_qmap(args: argparse.Namespace) -> int:
    """Map a question to knowledge-graph concepts (LLM-assisted, verified)."""
    from .qmap import QuestionMapper
    from .storage.question_map import QuestionMapStore

    cfg = Config.load()
    store = QuestionMapStore(cfg.question_map_db)
    if not args.question:
        maps = store.all_maps()
        print(f"question maps: {len(maps)} recorded at {cfg.question_map_db}")
        for m in maps[-10:]:
            names = ", ".join(c["name"] for c in m.get("concepts", []))
            print(f"  [{m.get('method','')}] {m['question']} -> {names}")
        return 0

    mapper = QuestionMapper(GraphStore(cfg.graph_db))
    qmap = mapper.map(args.question)
    store.append(qmap.to_dict())
    print(f"method: {qmap.method}")
    print(f"verified: {sum(1 for c in qmap.concepts if c.verified)}/"
          f"{len(qmap.concepts)} concepts")
    for c in qmap.concepts:
        flags = []
        if c.out_of_graph:
            flags.append("new")
        if c.bloom_conflict:
            flags.append("bloom-conflict")
        print(f"  {c.name:<22} bloom={c.bloom or '-':<8} "
              f"conf={c.confidence:.2f} "
              f"verified={c.verified} {' '.join(flags)}")
    return 0


def cmd_study_plan(args: argparse.Namespace) -> int:
    """A12: today's study session = prereqs in order, FSRS-due items first,
    bounded by a daily capacity window (default 15/25 min)."""
    from .algorithms.forgetting import ForgettingModel, ForgettingParams
    from .algorithms.study_planner import StudyPlanner

    cfg = Config.load()
    store = GraphStore(cfg.graph_db)
    try:
        learner = None
        if args.learner:
            es = EventStore(cfg.events_db)
            try:
                learner = es.replay(args.learner)
            finally:
                es.close()
        planner = StudyPlanner(
            store,
            ForgettingModel(ForgettingParams(model=args.model)),
            default_capacity_minutes=args.capacity_minutes or 15)
        plan = planner.plan(args.target, capacity_minutes=args.capacity_minutes,
                            learner=learner, lo=args.lo,
                            min_confidence=args.min_confidence,
                            mastery_gate=args.mastery_gate)
        if not plan.totals:
            print(f"no plan path for '{args.target}' (is it in the graph?)")
            return 1
        print(f"study plan for {plan.target} "
              f"({plan.totals['session_minutes']}/{args.capacity_minutes} min, "
              f"{plan.totals['steps']} steps)")
        if plan.session:
            print("study today:")
            for i in plan.session:
                print(f"  {i.label:<26} "
                      f"[{'retrieve' if i.action == 'retrieve' else 'learn'}] "
                      f"ret={i.retention} diff={i.difficulty}")
        if plan.upcoming:
            print(f"deferred ({plan.totals['deferred']}):")
            for i in plan.upcoming[:6]:
                print(f"  {i.label:<26} {i.action} -> {i.scheduled_for}")
            if len(plan.upcoming) > 6:
                print(f"  ... and {len(plan.upcoming) - 6} more")
    finally:
        store.close()
    return 0


def cmd_learn(args: argparse.Namespace) -> int:
    """Record learner interactions, replay a learner model, or show stats."""
    cfg = Config.load()
    store = EventStore(cfg.events_db)
    try:
        if args.action == "record":
            i = Interaction(
                concept_id=args.concept, kind=args.kind, outcome=args.outcome,
                bloom=args.bloom, affect=args.affect,
                duration_sec=args.duration_sec,
            )
            event_id, seq = store.append(args.learner, i)
            print(f"recorded {event_id} (seq {seq})")
        elif args.action == "replay":
            m = store.replay(args.learner)
            # mastery/confidence are computed here, not read off ConceptState.
            # Those attributes are caches that only forgetting.demo() ever
            # assigns: replay() -> observe() -> record() updates
            # attempts/correct/last_seen and leaves them at their defaults, so
            # printing them reported mastery=0.000 confidence=0.000 for every
            # learner no matter how well they had actually done. Retention is
            # deliberately not cached anywhere either -- it decays with wall
            # time, so it is only ever correct when computed at the moment of
            # use. See docs/compliance.md.
            from .algorithms.confidence_model import ConfidenceModel
            from .algorithms.mastery import KnowledgeMastery

            km, cm = KnowledgeMastery(), ConfidenceModel()
            print(f"learner {m.learner_id}: {len(m.concepts)} concepts tracked, "
                  f"{sum(c.attempts for c in m.concepts.values())} interactions")
            for cid, cs in sorted(m.concepts.items()):
                mastery = km.score(m, cid).mastery
                # ConceptState.confidence documents itself as "recency-weighted
                # accuracy", which is ConfidenceResult.accuracy (the observed
                # skill proxy), not .efficacy (the learner's self-belief).
                accuracy = cm.estimate(m, cid).accuracy
                print(f"  {cid}: attempts={cs.attempts} correct={cs.correct} "
                      f"mastery={mastery:.3f} confidence={accuracy:.3f} "
                      f"last_seen={cs.last_seen}")
        else:  # stats
            print(json.dumps(store.stats(args.learner), indent=2))
    finally:
        store.close()
    return 0


def cmd_confidence(args: argparse.Namespace) -> int:
    """A13: self-efficacy + calibration per concept (Bandura sources)."""
    from .algorithms.confidence_model import ConfidenceModel, \
        ConfidenceParams

    cfg = Config.load()
    es = EventStore(cfg.events_db)
    try:
        m = es.replay(args.learner)
        if not m.concepts:
            print(f"no interactions recorded for learner '{args.learner}'")
            return 1
        cm = ConfidenceModel(ConfidenceParams(social=args.social))
        tracks = {args.concept} if args.concept else set(m.concepts)
        print(f"self-efficacy for {m.learner_id} "
              f"({'one concept' if args.concept else str(len(tracks))} tracked):")
        for cid in sorted(tracks):
            r = cm.estimate(m, cid)
            mark = "*" if not r.confident else ""
            print(f"  {cid:<24} eff={r.efficacy:.2f} acc={r.accuracy:.2f} "
                  f"affect={r.affect:.2f} calibr={r.calibration:+.3f} "
                  f"{r.category}{mark}")
    finally:
        es.close()
    return 0


def cmd_lifelong(args: argparse.Namespace) -> int:
    """A14: cross-year (CBSE calendar) and cross-subject longitudinal view."""
    from .algorithms.lifelong_model import LifelongLearner
    from .models.enums import NodeType

    cfg = Config.load()
    try:
        es = EventStore(cfg.events_db)
        try:
            events = es.events(args.learner, limit=100000)
        finally:
            es.close()
        if not events:
            print(f"no interactions for learner '{args.learner}'")
            return 1

        # Optional subject attribution via the graph: concept label prefix
        # "sub:" not stored; caller passes --subject map? Skip for v1: keep
        # the caller-free aggregate (all under "mixed").
        ll = LifelongLearner(args.learner)
        r = ll.analyze(events)
        print(f"lifelong profile for {args.learner} ({len(events)} events, "
              f"{len(r.years)} academic year(s)):")
        for y in r.years:
            p = r.portraits[y]
            print(f"  {y}: {len(p.concepts)} concepts, {p.events} events, "
                  f"acc={p.accuracy:.2f}, days={len(p.active_days)}")
        if r.recurring:
            print("recurring concepts:", ", ".join(r.recurring))
        if r.continuity:
            print("continuity (accuracy by year):")
            for c in r.continuity[:5]:
                chain = " ".join(f"{y}={a:.2f}" for y, a in c.chain)
                print(f"  {c.concept:<24} {chain}  (Δ {c.improves:+.2f})")
        if r.growth:
            print(f"cross-year growth: {r.growth:+.2f}")
        if not r.confident:
            print("(low evidence: needs 2+ academic years, 5+ events)")
    finally:
        pass
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    cfg = Config.load()
    registry = SourceRegistry(cfg.registry_db)
    graph = GraphStore(cfg.graph_db)
    index = ChunkIndex(cfg.index_db)
    by_type: dict[str, int] = {}
    for s in registry.all():
        by_type[s["doc_type"] or "unknown"] = by_type.get(s["doc_type"] or "unknown", 0) + 1
    n, e = graph.count()
    print(json.dumps({
        "registered": registry.count(),
        "by_type": by_type,
        "graph_nodes": n,
        "graph_edges": e,
        "chunks": index.count(),
        "data_root": str(cfg.data_root),
        "corpus_root": str(cfg.corpus_root),
        "gcs": {"bucket": cfg.gcs_bucket, "enabled": cfg.sync_enabled},
    }, indent=2))
    registry.close(); graph.close(); index.close()
    return 0


def cmd_concept_spine(args: argparse.Namespace) -> int:
    """Compute per-concept stats (occurrences, source spread, degree, predicates)
    and store them as node attributes. This is the metadata spine the learning
    algorithms (mastery, forgetting, motivation) will consume."""
    import collections

    from .models.enums import NodeType
    from .graph.store import GraphStore
    from .graph.builder import GraphBuilder

    cfg = Config.load()
    store = GraphStore(cfg.graph_db)
    builder = GraphBuilder(store)

    concepts = store.query_nodes(ntype=NodeType.CONCEPT, status=None, limit=100000)
    stats: dict[str, dict] = {}
    for c in concepts:
        stats[c.id] = {"label": c.label, "occurrences": 0, "sources": set(),
                       "predicates": collections.Counter(), "degree": 0}
    rows = store.conn.execute(
        """SELECT source, target, attributes, provenance FROM edges WHERE type='RELATED_TO'"""
    ).fetchall()
    import json as _json
    for src, tgt, attrs, prov in rows:
        for nid in (src, tgt):
            s = stats.get(nid)
            if not s:
                continue
            s["degree"] += 1
            if attrs:
                try:
                    a = _json.loads(attrs)
                    s["occurrences"] += int(a.get("occurrences", 0))
                    for p in a.get("predicates", []):
                        s["predicates"][p] += 1
                except Exception:
                    pass
            if prov:
                try:
                    s["sources"].add(_json.loads(prov)["source"]["document_id"])
                except Exception:
                    pass
    n_updates = 0
    for nid, s in stats.items():
        attrs = {
            "occurrences": s["occurrences"],
            "source_docs": len(s["sources"]),
            "degree": s["degree"],
            "top_predicates": s["predicates"].most_common(5),
        }
        node = store.get_node(nid)
        if not node:
            continue
        merged = dict(node.attributes or {})
        merged.update(attrs)
        store.conn.execute(
            "UPDATE nodes SET attributes=? WHERE id=?",
            (_dumps_json(merged), nid))
        n_updates += 1
    store.conn.commit()
    ranked = sorted(stats.values(), key=lambda s: (s["occurrences"], s["degree"]), reverse=True)
    print(f"concept spine: {len(stats)} concepts updated")
    print(f"{'concept':<28} {'occ':>5} {'docs':>5} {'deg':>5} top predicates")
    for s in ranked[:args.top]:
        preds = ", ".join(f"{p}:{c}" for p, c in s["predicates"].most_common(3))
        print(f"{s['label'][:28]:<28} {s['occurrences']:>5} {len(s['sources']):>5} {s['degree']:>5} {preds}")
    store.close()
    return 0


def cmd_concept_profile(args: argparse.Namespace) -> int:
    """Enrich top concepts with difficulty, Bloom level, prerequisites,
    misconceptions and learning time.

    --method rule: deterministic, free, local (verb-list Bloom classifier,
    page-order prerequisite mining, heuristic difficulty). No API calls.
    --method llm:  batched LLM (Groq) enrichment.
    """
    from .models.enums import NodeType
    from .graph.store import GraphStore

    cfg = Config.load()
    store = GraphStore(cfg.graph_db)

    concepts = [n for n in store.query_nodes(ntype=NodeType.CONCEPT, status=None, limit=100000)
                if (n.attributes or {}).get("occurrences", 0) >= args.min_occ]
    concepts.sort(key=lambda n: (n.attributes or {}).get("occurrences", 0), reverse=True)
    concepts = concepts[: args.top]
    print(f"profiling {len(concepts)} concepts (min occurrences {args.min_occ}, method={args.method})...")

    if args.method == "rule":
        from .graph.curriculum import (CRITERION_WEIGHTS, JUNK_CONCEPTS,
                                       CurriculumEstimator, classify_bloom,
                                       estimate_difficulty,
                                       estimate_learning_hours,
                                       transitive_prereqs)
        est = CurriculumEstimator(cfg.index_db)
        est.graph = store
        registry = SourceRegistry(cfg.registry_db)
        textbooks = [(s["source_id"], s.get("title") or "") for s in registry.all()
                     if (s.get("title") or "").startswith("class")]
        textbook_ids = {sid for sid, _ in textbooks}
        registry.close()
        concepts = [n for n in concepts
                    if n.label.lower().strip() not in JUNK_CONCEPTS
                    and _clean_label(n.label)]
        labels = [n.label for n in concepts]
        print("mining prerequisites (multi-criteria voting: TemO, CMH, IOLR, BERTropy, Text; theta=0.28)...")
        title_by_id = dict(textbooks)
        # Subject groups for structural criteria (TemO/CMH/BERTropy). The
        # English reader (poorvi) is a storybook: page/grade order does not
        # follow conceptual difficulty, so it is excluded from structural
        # voting (the paper's method assumes structured courses). Text-based
        # triples from ALL subjects are still merged below via the graph.
        subject_groups: dict[str, set[str]] = {}
        for sid, title in textbooks:
            for t in ("curiosity", "ganita_prakash"):
                if t in title:
                    subject_groups.setdefault(t, set()).add(sid)
        prereqs: dict[str, list[tuple[str, float, int]]] = {}
        pair_detail: list = []
        prereq_candidates = {
            n.label for n in concepts
            if (n.attributes or {}).get("source_docs", 0) >= 2
        }
        for subj, sids in subject_groups.items():
            grade_map = {}
            for sid in sids:
                m = _grade_from_title(title_by_id.get(sid, ""))
                if m:
                    grade_map[sid] = m
            subj_votes = est.vote_prerequisites(labels, doc_ids=sids,
                                                doc_grades=grade_map, graph=store,
                                                min_criteria=3,
                                                prereq_candidates=prereq_candidates,
                                                strong_criteria=("CMH", "Text"),
                                                pair_detail=pair_detail)
            for k, v in subj_votes.items():
                prereqs.setdefault(k, []).extend(v)
        prereqs = transitive_prereqs(prereqs)
        for lb in prereqs:
            prereqs[lb].sort(key=lambda t: (-t[1], -t[2]))

        # Materialize voted pairs as PREREQUISITE_OF edges with estimator
        # properties (score, confidence, criteria breakdown, difficulty
        # jump, recommended review spacing) so retrieval/planning can
        # traverse prerequisites like any other relationship.
        difficulty_by_label: dict[str, int] = {}
        bloom_by_label: dict[str, str] = {}
        for n in concepts:
            occ = (n.attributes or {}).get("occurrences", 0)
            docs = (n.attributes or {}).get("source_docs", 0)
            if est.occurs_in(n.label, textbook_ids):
                ctx = est.contexts(n.label, max_chunks=10, doc_ids=textbook_ids)
                bloom = classify_bloom(ctx.verbs)
                bloom_by_label[n.label] = bloom
                difficulty_by_label[n.label] = estimate_difficulty(
                    bloom, occ, docs, len(prereqs.get(n.label, [])))
        label_to_id = {n.label: n.id for n in concepts}
        from .graph.prereq_edges import build_prerequisite_edges, write_prerequisite_edges
        prereq_edges = build_prerequisite_edges(
            prereqs, label_to_id, pair_detail=pair_detail,
            difficulty=difficulty_by_label, weights=dict(CRITERION_WEIGHTS))
        n_edges = write_prerequisite_edges(store, prereq_edges)

        # LO ontology layer: canonical LO statements per profiled concept
        # (Bloom template), MAPS_TO concept->LO, and PREREQUISITE_OF LO->LO
        # inherited from the concept-level pairs. LO-centric planning then
        # works through the same prereq_plan traversal on LO nodes.
        from .graph.learning_outcomes import build_lo_layer
        profiles = {
            n.label: {"bloom": bloom_by_label[n.label],
                      "difficulty": difficulty_by_label[n.label],
                      "learning_hours": estimate_learning_hours(difficulty_by_label[n.label])}
            for n in concepts if n.label in difficulty_by_label
        }
        n_lo, n_maps, n_lo_prereq = build_lo_layer(store, profiles, prereqs,
                                                   concept_ids=label_to_id)

        updated = 0
        for n in concepts:
            if n.label not in difficulty_by_label:
                continue
            bloom = bloom_by_label[n.label]
            occ = (n.attributes or {}).get("occurrences", 0)
            docs = (n.attributes or {}).get("source_docs", 0)
            difficulty = difficulty_by_label[n.label]
            merged = dict(n.attributes or {})
            merged.update({
                "bloom": bloom,
                "difficulty": difficulty,
                "learning_hours": estimate_learning_hours(difficulty),
                "prerequisites": [p for p, _, _ in prereqs.get(n.label, [])][:6],
                "prereq_evidence": prereqs.get(n.label, [])[:6],
                "profile_method": "rule",
            })
            from .algorithms.misconceptions import enrich_attributes
            merged = enrich_attributes(merged, n.label)
            store.conn.execute("UPDATE nodes SET attributes=? WHERE id=?", (_dumps_json(merged), n.id))
            n.attributes = merged
            updated += 1
        store.conn.commit()
        est.close()
        print(f"concept profile (rule): {updated} concepts enriched, {sum(len(v) for v in prereqs.values())} prereq pairs mined, {n_edges} PREREQUISITE_OF edges, LO layer: {n_lo} LOs, {n_maps} MAPS_TO, {n_lo_prereq} LO prereq edges")
        print(f"{'concept':<26} {'occ':>5} {'docs':>4} {'bloom':<10} {'diff':>4} {'hrs':>5}  prereqs")
        for n in concepts[: args.top]:
            a = n.attributes or {}
            prs = ", ".join(p for p, _, _ in prereqs.get(n.label, [])[:3])
            print(f"{n.label[:26]:<26} {a.get('occurrences',0):>5} {a.get('source_docs',0):>4} {a.get('bloom',''):<10} {a.get('difficulty',0):>4} {a.get('learning_hours',0):>5}  {prs}")
        store.close()
        return 0

    from .llm.sarvam import SarvamLLM

    llm = SarvamLLM(api_key=cfg.llm_api_key, base_url=cfg.llm_base_url,
                    model=args.model or cfg.llm_model, timeout=cfg.llm_timeout)
    if not llm.available:
        print("no LLM API key configured")
        store.close()
        return 1

    PROFILE_SYSTEM = (
        "You are building a curriculum knowledge graph. For each concept, respond with a JSON "
        "object of the form {\"profiles\": [{\"concept\": \"...\", \"difficulty\": 1|2|3|4|5, "
        "\"bloom\": \"remember|understand|apply|analyze|evaluate|create\", \"prerequisites\": [\"...\"], "
        "\"misconceptions\": [\"...\"], \"learning_hours\": number}]}. "
        "Answer only the JSON, no extra text."
    )
    BATCH = 8
    updated = 0
    for i in range(0, len(concepts), BATCH):
        batch = concepts[i:i + BATCH]
        names = [n.label for n in batch]
        try:
            out = llm.chat_json([
                {"role": "system", "content": PROFILE_SYSTEM},
                {"role": "user", "content": "Concepts: " + ", ".join(names)},
            ], temperature=0.0, max_tokens=4096)
        except Exception as e:
            print(f"  batch {i // BATCH} failed: {e}")
            continue
        profiles = out.get("profiles", []) if isinstance(out, dict) else (out or [])
        by_name = {str(p.get("concept", "")).strip().lower(): p for p in profiles if isinstance(p, dict)}
        for n in batch:
            p = by_name.get(n.label.strip().lower())
            if not p:
                continue
            merged = dict(n.attributes or {})
            merged.update({
                "difficulty": int(p.get("difficulty", 3)),
                "bloom": str(p.get("bloom", "understand")),
                "prerequisites": [str(x) for x in p.get("prerequisites", [])][:6],
                "misconceptions": [str(x) for x in p.get("misconceptions", [])][:4],
                "learning_hours": float(p.get("learning_hours", 1.0)),
            })
            store.conn.execute("UPDATE nodes SET attributes=? WHERE id=?", (_dumps_json(merged), n.id))
            updated += 1
        print(f"  batch {i // BATCH}: {len(profiles)} profiles returned, {updated} concepts updated")
    store.conn.commit()
    store.close()
    print(f"concept profile: {updated} concepts enriched")
    return 0


def _dumps_json(v) -> str:
    import json
    return json.dumps(v, ensure_ascii=False)


def _grade_from_title(title: str) -> int | None:
    import re
    m = re.search(r"class(\d+)", title or "")
    return int(m.group(1)) if m else None


def _clean_label(label: str) -> bool:
    """Reject extractor artifacts: mojibake, control chars, 1-char labels,
    pure symbols, possessive/contraction fragments (sun's -> 's), non-ASCII.
    Common units (cm, kg, m) survive the length check."""
    import re as _re
    if not label or len(label.strip()) < 2:
        return False
    if "\ufffd" in label or "\x00" in label:
        return False
    if "'" in label or "\u2019" in label or "\u2018" in label:
        return False
    if not _re.fullmatch(r"[A-Za-z0-9 .-]+", label):
        return False
    if _re.fullmatch(r"[\d.\s-]+", label):
        return False
    return True


def cmd_sync_gcs(args: argparse.Namespace) -> int:
    cfg = Config.load()
    if not cfg.gcs_bucket:
        print("GCS not configured (set gcs_bucket in config/config.toml or ACOS_GCS_BUCKET)")
        return 1
    from .storage.base import LocalStore
    from .storage.gcs import GcsStore

    local = LocalStore(cfg.data_root)
    remote = GcsStore(cfg.gcs_bucket, cfg.gcs_prefix)
    prefixes = args.prefixes.split(",")
    for prefix in prefixes:
        keys = local.keys(prefix)
        for i, key in enumerate(keys):
            remote.put(local._resolve(key), key)
            if i % 100 == 0:
                print(f"synced {prefix}: {i}/{len(keys)}")
        print(f"synced {prefix}: {len(keys)} files")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn
    from .api.main import init_runtime

    init_runtime()
    uvicorn.run("academicos.api.main:app", host=args.host, port=args.port, reload=False)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="academicos", description="AcademicOS CLI")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    pi = sub.add_parser("ingest"); pi.add_argument("--limit", type=int, default=None); pi.add_argument("--reingest", action="store_true"); pi.set_defaults(fn=cmd_ingest)
    pp = sub.add_parser("parse"); pp.add_argument("--doc-id"); pp.add_argument("--reparse", action="store_true"); pp.add_argument("--vlm-mode", default="service"); pp.add_argument("--vlm-url", default="http://127.0.0.1:8910"); pp.set_defaults(fn=cmd_parse)
    pe = sub.add_parser("extract"); pe.add_argument("--doc-id"); pe.set_defaults(fn=cmd_extract)
    pg = sub.add_parser("build-graph"); pg.add_argument("--doc-id"); pg.set_defaults(fn=cmd_build_graph)
    po = sub.add_parser("openie"); po.add_argument("--doc-id"); po.add_argument("--limit", type=int, default=None); po.add_argument("--max-chunks", type=int, default=20); po.add_argument("--model", default=None); po.add_argument("--extractor", choices=["llm", "local"], default="local"); po.set_defaults(fn=cmd_openie)
    ps = sub.add_parser("search"); ps.add_argument("query", nargs="?"); ps.add_argument("--limit", type=int, default=10); ps.set_defaults(fn=cmd_search)
    pst = sub.add_parser("stats"); pst.set_defaults(fn=cmd_stats)
    psp = sub.add_parser("concept-spine"); psp.add_argument("--top", type=int, default=25); psp.set_defaults(fn=cmd_concept_spine)
    pcp = sub.add_parser("concept-profile"); pcp.add_argument("--top", type=int, default=200); pcp.add_argument("--min-occ", type=int, default=3); pcp.add_argument("--model", default=None); pcp.add_argument("--method", choices=["rule", "llm"], default="rule"); pcp.set_defaults(fn=cmd_concept_profile)
    psv = sub.add_parser("serve"); psv.add_argument("--host", default="127.0.0.1"); psv.add_argument("--port", type=int, default=8000); psv.set_defaults(fn=cmd_serve)
    pln = sub.add_parser("learn")
    pln.add_argument("--learner", default="default")
    pln.add_argument("action", choices=["record", "replay", "stats"])
    pln.add_argument("--concept")
    pln.add_argument("--kind", default="answer")
    pln.add_argument("--outcome", type=float)
    pln.add_argument("--bloom")
    pln.add_argument("--affect")
    pln.add_argument("--duration-sec", type=float)
    pln.set_defaults(fn=cmd_learn)
    ppn = sub.add_parser("plan")
    ppn.add_argument("target")
    ppn.add_argument("--lo", action="store_true")
    ppn.add_argument("--min-confidence", type=float, default=0.0)
    ppn.add_argument("--max-depth", type=int, default=4)
    ppn.add_argument("--learner")
    ppn.add_argument("--mastery-gate", type=float, default=0.5)
    ppn.set_defaults(fn=cmd_plan)
    prv = sub.add_parser("revise")
    prv.add_argument("--learner", default="default")
    prv.add_argument("--model", choices=["fsrs", "exponential"], default="fsrs")
    prv.set_defaults(fn=cmd_revise)
    psp2 = sub.add_parser("study-plan")
    psp2.add_argument("target")
    psp2.add_argument("--learner", default=None)
    psp2.add_argument("--model", choices=["fsrs", "exponential"], default="fsrs")
    psp2.add_argument("--capacity-minutes", type=int, default=None)
    psp2.add_argument("--lo", action="store_true")
    psp2.add_argument("--min-confidence", type=float, default=0.0)
    psp2.add_argument("--mastery-gate", type=float, default=0.5)
    psp2.set_defaults(fn=cmd_study_plan)
    pqm = sub.add_parser("qmap")
    pqm.add_argument("question", nargs="?")
    pqm.set_defaults(fn=cmd_qmap)
    pms = sub.add_parser("misconceptions")
    pms.add_argument("concept")
    pms.add_argument("--answer")
    pms.set_defaults(fn=cmd_misconceptions)
    pcf = sub.add_parser("confidence")
    pcf.add_argument("--learner", default="default")
    pcf.add_argument("--concept")
    pcf.add_argument("--social", type=float, default=0.5, help="learner-level social support")
    pcf.set_defaults(fn=cmd_confidence)
    plt = sub.add_parser("lifelong")
    plt.add_argument("--learner", default="default")
    plt.set_defaults(fn=cmd_lifelong)
    psg = sub.add_parser("sync-gcs"); psg.add_argument("--prefixes", default="documents,extracted,parse,graph,index,registry"); psg.set_defaults(fn=cmd_sync_gcs)

    args = p.parse_args(argv)
    _setup_logging(args.verbose)
    try:
        return args.fn(args)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
