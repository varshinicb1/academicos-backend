"""Timetable generation (REQUIREMENTS SCH-3): a clash-free week for the whole
school from the allocations (M1.2), the bells (M1.3) and the rules below, by
constraint programming (OR-Tools CP-SAT).

Hard rules
- every (section, subject) allocation gets exactly its periods a week;
- a section holds one subject per period;
- a teacher is in one place at a time -- compared by CLOCK TIME, not by
  period number, so a junior wing on its own bell cannot hide a clash;
- a teacher teaches at most `max_per_day` periods a day and at most
  `max_consecutive` in a row -- also by clock time: periods that touch, or
  whose gap is shorter than a break, are in a row, whichever bell each one
  is on;
- a subject gets at most ceil(periods / days) periods in a section's day
  (spread across the week);
- a teacher's unavailable periods stay free;
- entries the principal locked stay where they are;
- SCH-8 co-teaching: an allocation's co-teacher is booked for every one of
  its periods, under the same one-place, per-day and in-a-row rules;
- SCH-2 labs: an allocation that needs a kind of room ("lab") gets one room
  of that kind for each period, no room holds two classes at one clock
  time, and a room's unavailable periods (a lab out of use) stay free;
- SCH-2 double periods: an allocation's `double_periods` come as two
  periods in a row with no break between them, in the same room;
- SCH-8 groups (audit N-3-20): an elective group or a combined class is one
  block -- each of its periods holds every member section and every lane's
  teacher and room at once.

Objective: keep the current timetable. A re-solve after a change (a new
allocation, a teacher leaving) moves the fewest periods (SCH-3: "re-solve
after a change touching the fewest periods"). Among equally short moves,
first and last periods are shared out between teachers.

Before any solving, `check_feasibility` names requests that cannot be met in
plain words ("10-A needs 50 periods a week; its bell gives 48"), because a
solver's bare INFEASIBLE tells a principal nothing.
"""
from __future__ import annotations

import logging
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from .school_model import WEEKDAY_NAMES, adjacent_pairs
from .teaching_groups import GroupSession

log = logging.getLogger(__name__)


@dataclass
class SolveOptions:
    max_per_day: int = 7
    max_consecutive: int = 4
    keep_existing: bool = True
    # The whole solve -- reading the school, building the model, searching --
    # returns within this (N-3-18), with the best week found by then.
    time_limit_seconds: float = 30.0
    section_ids: Optional[set[str]] = None   # solve only these; the rest are fixed
    num_workers: Optional[int] = None        # None: the CPUs this process may use (solver_workers)
    # Once a week is in hand, stop improving it when no better one has come
    # for this long: the objective left is fewer moves and fairer first and
    # last periods, and a principal should not wait a minute for a tie-break.
    stall_seconds: float = 10.0


def _read_text(path: str) -> str:
    with open(path, encoding="ascii") as f:
        return f.read()


def solver_workers() -> int:
    """CP-SAT search workers for this process: ACOS_SOLVER_WORKERS when set,
    else the CPUs the container may actually use -- its cgroup CPU quota
    (Cloud Run's --cpu=1 is a quota of one CPU on a many-core host, which
    os.cpu_count() does not see), else the CPUs it may run on -- at most 8."""
    env = os.environ.get("ACOS_SOLVER_WORKERS", "").strip()
    if env.isdigit() and int(env) > 0:
        return min(8, int(env))
    cpus: Optional[float] = None
    try:                                              # cgroup v2: "max 100000" or "100000 100000"
        quota, period = _read_text("/sys/fs/cgroup/cpu.max").split()[:2]
        if quota != "max":
            cpus = int(quota) / int(period)
    except (OSError, ValueError):
        try:                                          # cgroup v1
            quota = int(_read_text("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"))
            period = int(_read_text("/sys/fs/cgroup/cpu/cpu.cfs_period_us"))
            if quota > 0 and period > 0:
                cpus = quota / period
        except (OSError, ValueError) as e:
            # Not in a cgroup with a CPU quota (a laptop, a test): fall back
            # to the CPUs this process may run on.
            log.debug("no cgroup CPU quota: %s", e)
    if cpus is None:
        affinity = getattr(os, "sched_getaffinity", None)
        cpus = len(affinity(0)) if affinity else (os.cpu_count() or 1)
    return max(1, min(8, int(cpus)))


@dataclass
class ProposedEntry:
    section_id: str
    day_of_week: int
    period: int
    subject_id: str
    teacher_id: Optional[str]
    room_id: Optional[str] = None
    co_teacher_id: Optional[str] = None


@dataclass
class SolveResult:
    status: str                      # solved | infeasible | timeout | nothing_to_solve
    entries: list[ProposedEntry] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    kept: int = 0                    # current entries left where they were
    moved_or_added: int = 0          # entries that are new or moved
    removed: int = 0                 # current entries no longer present
    seconds: float = 0.0
    # The periods of the groups this solve placed (`group_ids`), N-3-20.
    group_sessions: list[GroupSession] = field(default_factory=list)
    group_ids: set[str] = field(default_factory=set)


def _minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


# Two of a teacher's periods closer than a break on the clock are in a row.
# A school whose bells declare no break: a gap this long is one.
DEFAULT_BREAK_MINUTES = 10


def break_minutes(bells) -> int:
    """The shortest break any of the year's bells declares (SCH-3): a gap
    between two periods shorter than this is no rest, so the periods count
    as in a row whichever bells they come from."""
    breaks = [_minutes(s.end) - _minutes(s.start) for b in bells for s in b.slots if s.kind == "break"]
    return min(breaks) if breaks else DEFAULT_BREAK_MINUTES


def runs_over(spans, longest: int, gap: int) -> list[tuple]:
    """Every chain of `longest + 1` spans (start, end) that follow each other
    on the clock, each starting at or after the last one ends and less than
    `gap` minutes later. Teaching all of one chain is more than `longest` in
    a row. Spans from two bells interleave, so a teacher's periods are
    compared by time, never by position in a list of start times (N-3-3)."""
    spans = sorted(set(spans))
    nxt = {a: [b for b in spans if 0 <= b[0] - a[1] < gap] for a in spans}
    chains: list[tuple] = []

    def extend(chain):
        if len(chain) == longest + 1:
            chains.append(tuple(chain))
            return
        for b in nxt[chain[-1]]:
            extend(chain + [b])
    for a in spans:
        extend([a])
    return chains


def _section_grid(store, section):
    """(days, {period: (start_min, end_min)}) from the section's bell."""
    bell = store.bell_for_section(section)
    if bell is None:
        return None
    periods = {s.period: (_minutes(s.start), _minutes(s.end)) for s in bell.slots if s.kind == "teaching"}
    return bell.days, periods


def check_feasibility(store, academic_year_id: str, options: SolveOptions) -> list[str]:
    """Requests no timetable can meet, as sentences. Empty when the obvious
    counts add up (the solver can still find a finer conflict)."""
    problems: list[str] = []
    sections = {s.id: s for s in store.sections_for_year(academic_year_id)}
    allocs = [a for a in store.allocations_for_year(academic_year_id) if a.section_id in sections]
    unavailable = store.teacher_unavailability_for_year(academic_year_id)
    grids = {sid: _section_grid(store, s) for sid, s in sections.items()}   # each bell read once
    need_by_section: dict[str, int] = defaultdict(int)
    need_by_teacher: dict[str, int] = defaultdict(int)
    for a in allocs:
        need_by_section[a.section_id] += a.periods_per_week
        for t in _teachers_of(a):
            need_by_teacher[t] += a.periods_per_week
    # SCH-8 (N-3-20): an elective or a combined class takes its periods from
    # every member section, and from every lane's teacher.
    for g in store.teaching_groups_for_year(academic_year_id):
        members = [sid for sid in g.section_ids if sid in sections]
        if len({repr(grids[sid]) for sid in members}) > 1:
            problems.append(f"{g.name} joins sections on different bells; give them one bell")
        for sid in members:
            need_by_section[sid] += g.periods_per_week
        for t in g.teacher_ids:
            need_by_teacher[t] += g.periods_per_week
    week_slots = 0
    for sid, need in need_by_section.items():
        grid = grids[sid]
        label = store._section_label(sections[sid])
        if grid is None:
            problems.append(f"{label} has no bell schedule: set up the year's bell first")
            continue
        days, periods = grid
        have = len(days) * len(periods)
        week_slots = max(week_slots, have)
        if need > have:
            problems.append(f"{label} needs {need} periods a week; its bell gives {have}")
    # Spread: a subject gets at most ceil(periods / days) a day, so it needs
    # its teacher free on enough of the section's days.
    for a in allocs:
        grid = grids[a.section_id]
        if grid is None or not _teachers_of(a):
            continue
        days, periods = grid
        blocked = set().union(*(unavailable.get(t, set()) for t in _teachers_of(a)))
        free_days = [d for d in days if any((d, p) not in blocked for p in periods)]
        cap = _day_cap(a, days)
        if a.periods_per_week > cap * len(free_days):
            who = "its teachers are" if a.co_teacher_id else "its teacher is"
            problems.append(
                f"{_cell(store, a, sections)} needs {a.periods_per_week} periods, at most {cap} a day, "
                f"but {who} free on only {len(free_days)} of the section's {len(days)} days")
    # Labs and other rooms of a kind (SCH-2), and double periods.
    rooms = store.rooms_for_school(store.get_section(allocs[0].section_id).school_id) if allocs else []
    closed = store.room_unavailability_for_year(academic_year_id)
    need_by_kind: dict[str, int] = defaultdict(int)
    for a in allocs:
        grid = grids[a.section_id]
        if a.room_kind:
            need_by_kind[a.room_kind] += a.periods_per_week
            of_kind = [r for r in rooms if r.kind == a.room_kind]
            if not of_kind:
                problems.append(f"{_cell(store, a, sections)} needs a {a.room_kind} and the school has "
                                f"none: add one under Rooms, or clear the subject's room kind")
            elif grid is not None:
                days, periods = grid
                open_days = [d for d in days if any((d, p) not in closed.get(r.id, set())
                                                    for r in of_kind for p in periods)]
                cap = _day_cap(a, days)
                if a.periods_per_week > cap * len(open_days):
                    problems.append(f"{_cell(store, a, sections)} needs {a.periods_per_week} periods, at most "
                                    f"{cap} a day, but a {a.room_kind} is open on only {len(open_days)} of the "
                                    f"section's {len(days)} days")
        if a.double_periods and grid is not None:
            bell = store.bell_for_section(sections[a.section_id])
            if not adjacent_pairs(bell):
                problems.append(f"{_cell(store, a, sections)} has {a.double_periods} double period(s), but "
                                f"{bell.name} has no two periods in a row without a break")
    for kind, need in need_by_kind.items():
        of_kind = [r for r in rooms if r.kind == kind]
        if not of_kind:
            continue
        slots_a_week = max((len(g[0]) * len(g[1]) for g in grids.values() if g), default=0)
        have = sum(slots_a_week - len(closed.get(r.id, ())) for r in of_kind)
        if need > have:
            problems.append(f"{need} periods a week need a {kind}; the school's {len(of_kind)} {kind}(s) "
                            f"are open for {have}")
    days_in_week = max((len(g[0]) for g in grids.values() if g), default=6)
    for tid, need in need_by_teacher.items():
        cap = options.max_per_day * days_in_week - len(unavailable.get(tid, ()))
        if need > cap:
            problems.append(f"a teacher ({tid}) is allocated {need} periods a week, more than "
                            f"{options.max_per_day} a day over {days_in_week} days allows "
                            f"(less their unavailable periods): {cap}")
    return problems


def _teachers_of(a) -> list[str]:
    """The allocation's teacher and co-teacher: both are in the room."""
    return [t for t in (a.teacher_id, getattr(a, "co_teacher_id", None)) if t]


def _day_cap(a, days) -> int:
    """Most periods of one subject in a section's day: its share of the
    week, and at least two when it has a double period."""
    cap = max(1, math.ceil(a.periods_per_week / len(days)))
    return max(cap, 2) if getattr(a, "double_periods", 0) else cap


def _cell(store, a, sections) -> str:
    subject = store.get_subject(a.subject_id)
    return f"{subject.name if subject else 'a subject'} for {store._section_label(sections[a.section_id])}"


def solve(store, academic_year_id: str, options: Optional[SolveOptions] = None) -> SolveResult:
    from ortools.sat.python import cp_model

    options = options or SolveOptions()
    started = time.monotonic()
    problems = check_feasibility(store, academic_year_id, options)
    if problems:
        return SolveResult(status="infeasible", problems=problems, seconds=time.monotonic() - started)

    sections = {s.id: s for s in store.sections_for_year(academic_year_id)}
    allocs = [a for a in store.allocations_for_year(academic_year_id) if a.section_id in sections]
    groups = [g for g in store.teaching_groups_for_year(academic_year_id)
              if g.section_ids and all(sid in sections for sid in g.section_ids)]
    if not allocs and not groups:
        return SolveResult(status="nothing_to_solve",
                           problems=["no allocations yet: give each section its subjects, teachers "
                                     "and periods a week first"])
    current = store.timetable_for_year(academic_year_id)
    solving = options.section_ids or set(sections)
    fixed = [e for e in current if e.section_id not in solving]
    unavailable = store.teacher_unavailability_for_year(academic_year_id)
    locked = {(e.section_id, e.day_of_week, e.period): e for e in store.locked_entries_for_year(academic_year_id)
              if e.section_id in solving}

    grids = {sid: _section_grid(store, sections[sid]) for sid in sections}
    school_id = next(iter(sections.values())).school_id
    rooms = store.rooms_for_school(school_id)
    closed = store.room_unavailability_for_year(academic_year_id)
    managed_kinds = {a.room_kind for a in allocs if a.room_kind}
    managed_rooms = {r.id for r in rooms if r.kind in managed_kinds} | {r for g in groups for r in g.room_ids}
    model = cp_model.CpModel()
    x: dict[tuple[str, int, int], cp_model.IntVar] = {}       # (alloc id, day, period)
    y: dict[tuple[str, int, int, str], cp_model.IntVar] = {}  # (alloc id, day, period, room id)
    alloc_by_id = {a.id: a for a in allocs if a.section_id in solving}
    # Indexes, so building the model is linear in its size (N-3-18): each
    # allocation's periods, and what each section's period could hold.
    x_of: dict[str, list[tuple[int, int, object]]] = defaultdict(list)
    by_section_slot: dict[tuple[str, int, int], list[tuple[str, object]]] = defaultdict(list)
    for a in alloc_by_id.values():
        days, periods = grids[a.section_id]
        blocked = set().union(*(unavailable.get(t, set()) for t in _teachers_of(a)))
        eligible = [r for r in rooms if r.kind == a.room_kind] if a.room_kind else []
        for d in days:
            for p in periods:
                if (d, p) in blocked:
                    continue
                open_rooms = [r for r in eligible if (d, p) not in closed.get(r.id, set())]
                if a.room_kind and not open_rooms:
                    continue
                v = model.NewBoolVar(f"x_{a.id}_{d}_{p}")
                x[(a.id, d, p)] = v
                x_of[a.id].append((d, p, v))
                by_section_slot[(a.section_id, d, p)].append((a.id, v))
                if a.room_kind:
                    ys = []
                    for r in open_rooms:
                        y[(a.id, d, p, r.id)] = model.NewBoolVar(f"y_{a.id}_{d}_{p}_{r.id}")
                        ys.append(y[(a.id, d, p, r.id)])
                    model.Add(sum(ys) == v)          # a period in a lab is in exactly one lab

    # SCH-8 (N-3-20): a group is placed when every one of its sections is
    # being solved; otherwise its periods stay where they are. A group
    # period takes its member sections' period (`None`: no allocation).
    solving_groups = {g.id: g for g in groups if set(g.section_ids) <= solving}
    placed_groups = [s for s in store.group_sessions_for_year(academic_year_id)
                     if any(g.id == s.group_id for g in groups)]
    group_of = {g.id: g for g in groups}
    gx: dict[tuple[str, int, int], object] = {}
    gx_of: dict[str, list[tuple[int, int, object]]] = defaultdict(list)
    for g in solving_groups.values():
        days, periods = grids[g.section_ids[0]]
        blocked = set().union(*(unavailable.get(t, set()) for t in g.teacher_ids))
        for d in days:
            for p in periods:
                if (d, p) in blocked or any((d, p) in closed.get(r, set()) for r in g.room_ids):
                    continue
                v = model.NewBoolVar(f"g_{g.id}_{d}_{p}")
                gx[(g.id, d, p)] = v
                gx_of[g.id].append((d, p, v))
                for sid in g.section_ids:
                    by_section_slot[(sid, d, p)].append((None, v))
        if not gx_of[g.id]:
            return SolveResult(status="infeasible", seconds=time.monotonic() - started, problems=[
                f"{g.name} has no period in which all its teachers and rooms are free"])
    fixed_group_periods = [(group_of[s.group_id], s) for s in placed_groups if s.group_id not in solving_groups]
    for g, s_ in fixed_group_periods:
        for sid in g.section_ids:
            if sid in solving:
                by_section_slot[(sid, s_.day_of_week, s_.period)].append((None, 1))

    # Every allocation gets exactly its periods, spread across the week.
    for a in alloc_by_id.values():
        days, _ = grids[a.section_id]
        model.Add(sum(v for _, _, v in x_of[a.id]) == a.periods_per_week)
        cap = _day_cap(a, days)
        on_day: dict[int, list] = defaultdict(list)
        for d, _, v in x_of[a.id]:
            on_day[d].append(v)
        for vs in on_day.values():
            if len(vs) > cap:
                model.Add(sum(vs) <= cap)

    for g in solving_groups.values():
        days, _ = grids[g.section_ids[0]]
        model.Add(sum(v for _, _, v in gx_of[g.id]) == g.periods_per_week)
        cap = max(1, math.ceil(g.periods_per_week / len(days)))
        on_day = defaultdict(list)
        for d, _, v in gx_of[g.id]:
            on_day[d].append(v)
        for vs in on_day.values():
            if len(vs) > cap:
                model.Add(sum(vs) <= cap)

    # Double periods (SCH-2): `double_periods` pairs of adjacent periods, no
    # period in two pairs, both halves in the same room.
    y_at: dict[tuple[str, int, int], dict[str, object]] = defaultdict(dict)
    for (aid, d, p, rid), yv in y.items():
        y_at[(aid, d, p)][rid] = yv
    for a in alloc_by_id.values():
        if not a.double_periods:
            continue
        days, _ = grids[a.section_id]
        pairs = adjacent_pairs(store.bell_for_section(sections[a.section_id]))
        z = {}
        for d in days:
            for p, q in pairs:
                if (a.id, d, p) in x and (a.id, d, q) in x:
                    zv = model.NewBoolVar(f"z_{a.id}_{d}_{p}")
                    model.Add(zv <= x[(a.id, d, p)])
                    model.Add(zv <= x[(a.id, d, q)])
                    z[(d, p, q)] = zv
                    first, second = y_at.get((a.id, d, p), {}), y_at.get((a.id, d, q), {})
                    for rid in set(first) | set(second):
                        if rid in first and rid in second:
                            model.Add(first[rid] - second[rid] <= 1 - zv)
                            model.Add(second[rid] - first[rid] <= 1 - zv)
                        else:
                            # The room is closed for one half: no double there.
                            # (Picked by key, never `a or b`: OR-Tools refuses
                            # to turn a literal into a bool.)
                            half = first[rid] if rid in first else second[rid]
                            model.Add(half <= 1 - zv)
        model.Add(sum(z.values()) == a.double_periods)
        for d in days:
            for p in {pp for _, pp, _ in z} | {qq for _, _, qq in z}:
                touching = [zv for (dd, pp, qq), zv in z.items() if dd == d and p in (pp, qq)]
                if len(touching) > 1:
                    model.Add(sum(touching) <= 1)

    # A section holds one subject per period (fixed sections are not solved).
    for key, held in by_section_slot.items():
        if key in locked or len(held) < 2 or all(isinstance(v, int) for _, v in held):
            continue
        model.Add(sum(v for _, v in held) <= 1)

    # Locked entries stay (and their slot holds nothing else).
    for key, e in locked.items():
        held = [(aid, v) for aid, v in by_section_slot.get(key, []) if not isinstance(v, int)]
        match = [v for aid, v in held if aid is not None and alloc_by_id[aid].subject_id == e.subject_id]
        if match:
            model.Add(match[0] == 1)
        for aid, v in held:
            if aid is None or alloc_by_id[aid].subject_id != e.subject_id:
                model.Add(v == 0)

    # A teacher is in one place at a time, by clock time. Interval-graph
    # cliques: at every period's start instant, at most one of the teacher's
    # periods that contain it.
    by_teacher: dict[str, list[tuple[int, int, int, object]]] = defaultdict(list)   # (day, start, end, var|1)
    for (aid, d, p), v in x.items():
        a = alloc_by_id[aid]
        start, end = grids[a.section_id][1][p]
        for t in _teachers_of(a):
            by_teacher[t].append((d, start, end, v))
    for e in fixed:
        if grids.get(e.section_id):
            start, end = grids[e.section_id][1].get(e.period, (None, None))
            if start is not None:
                for t in {e.teacher_id, e.co_teacher_id} - {None}:
                    by_teacher[t].append((e.day_of_week, start, end, 1))

    # A group's period holds every lane's teacher and room (N-3-20).
    by_room: dict[str, list[tuple[int, int, int, object]]] = defaultdict(list)
    group_periods = [(solving_groups[gid], d, p, v) for (gid, d, p), v in gx.items()] + \
        [(g, s_.day_of_week, s_.period, 1) for g, s_ in fixed_group_periods]
    for g, d, p, v in group_periods:
        start, end = grids[g.section_ids[0]][1].get(p, (None, None))
        if start is None:
            continue
        for t in g.teacher_ids:
            by_teacher[t].append((d, start, end, v))
        for r in g.room_ids:
            by_room[r].append((d, start, end, v))

    # A room holds one class at a time, by clock time, like a teacher.
    for (aid, d, p, rid), v in y.items():
        start, end = grids[alloc_by_id[aid].section_id][1][p]
        by_room[rid].append((d, start, end, v))
    for e in fixed:
        if e.room_id in managed_rooms and grids.get(e.section_id):
            start, end = grids[e.section_id][1].get(e.period, (None, None))
            if start is not None:
                by_room[e.room_id].append((e.day_of_week, start, end, 1))
    for rid, items in by_room.items():
        per_day: dict[int, list] = defaultdict(list)
        for d, s_, e_, v in items:
            per_day[d].append((s_, e_, v))
        for d, day_items in per_day.items():
            for s0, _, _ in day_items:
                overlapping = [v for s_, e_, v in day_items if s_ <= s0 < e_]
                if len(overlapping) > 1:
                    model.Add(sum(overlapping) <= 1)
    gap = break_minutes(store.bell_schedules_for_year(academic_year_id))
    for tid, items in by_teacher.items():
        by_day: dict[int, list] = defaultdict(list)
        for d, s, e_, v in items:
            by_day[d].append((s, e_, v))
        for d, day_items in by_day.items():
            for s0, _, _ in day_items:
                overlapping = [v for s, e_, v in day_items if s <= s0 < e_]
                if len(overlapping) > 1:
                    model.Add(sum(overlapping) <= 1)
            model.Add(sum(v for _, _, v in day_items) <= options.max_per_day)
            # At most `max_consecutive` in a row, by the clock: of any
            # max+1 periods that each start before a break's length has
            # passed since the last ended, not all are taught (N-3-3).
            at: dict[tuple[int, int], list] = defaultdict(list)
            for s, e_, v in day_items:
                at[(s, e_)].append(v)
            for chain in runs_over(at, options.max_consecutive, gap):
                in_chain = [v for span in chain for v in at[span]]
                free = [v for v in in_chain if not isinstance(v, int)]
                if free:
                    # A fixed section's periods already in the chain leave
                    # less room; a run they make alone is not this solve's.
                    fixed_here = len(in_chain) - len(free)
                    model.Add(sum(free) <= max(0, options.max_consecutive - fixed_here))

    # Objective: keep what is there; then share first/last periods fairly.
    objective = []
    alloc_of = {(a.section_id, a.subject_id): a for a in alloc_by_id.values()}
    if options.keep_existing:
        for e in current:
            a = alloc_of.get((e.section_id, e.subject_id)) if e.section_id in solving else None
            if a is None:
                continue
            v = x.get((a.id, e.day_of_week, e.period))
            if v is not None:
                objective.append(10 * (1 - v))
            yv = y.get((a.id, e.day_of_week, e.period, e.room_id)) if e.room_id else None
            if yv is not None:
                objective.append(1 - yv)       # and keep its lab, all else equal
        for s_ in placed_groups:
            v = gx.get((s_.group_id, s_.day_of_week, s_.period))
            if v is not None:
                objective.append(10 * (1 - v))
    edge_load: dict[str, list] = defaultdict(list)
    for (aid, d, p), v in x.items():
        a = alloc_by_id[aid]
        periods = grids[a.section_id][1]
        if a.teacher_id and (p == min(periods) or p == max(periods)):
            edge_load[a.teacher_id].append(v)
    # Fairness: the heaviest share of first and last periods any one teacher
    # carries (linear, so the solver stays fast at school scale).
    if edge_load:
        heaviest = model.NewIntVar(0, max(len(vs) for vs in edge_load.values()), "edge_max")
        for vs in edge_load.values():
            model.Add(heaviest >= sum(vs))
        objective.append(heaviest)
    if objective:
        model.Minimize(sum(objective))

    # Start from the current week (N-3-18): a re-solve after a small change
    # begins from a week that is nearly right, not from nothing.
    placed = {(e.section_id, e.day_of_week, e.period, e.subject_id): e.room_id for e in current
              if e.section_id in solving}
    placed_group_keys = {(s_.group_id, s_.day_of_week, s_.period) for s_ in placed_groups}
    if placed or placed_group_keys:
        for key, v in gx.items():
            model.AddHint(v, key in placed_group_keys)
        for (aid, d, p), v in x.items():
            a = alloc_by_id[aid]
            model.AddHint(v, (a.section_id, d, p, a.subject_id) in placed)
        for (aid, d, p, rid), yv in y.items():
            a = alloc_by_id[aid]
            model.AddHint(yv, placed.get((a.section_id, d, p, a.subject_id), "") == rid)

    solver = cp_model.CpSolver()
    workers = options.num_workers or solver_workers()
    solver.parameters.num_workers = workers
    if workers == 1:
        # One CPU, as Cloud Run's --cpu=1 gives (N-3-18). CP-SAT's default
        # single search found no week for a 40-section school in 60 s;
        # interleaved, its whole portfolio -- feasibility jump, local
        # search, the LP search -- takes turns on the one thread and a first
        # week comes in seconds. The result is deterministic, too.
        solver.parameters.interleave_search = True
    # The limit is for the whole solve: reading the school and building the
    # model have already used part of it.
    solver.parameters.max_time_in_seconds = max(1.0, options.time_limit_seconds - (time.monotonic() - started))
    status = _search(solver, model, options.stall_seconds)
    seconds = time.monotonic() - started
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        if status == cp_model.INFEASIBLE:
            return SolveResult(status="infeasible", seconds=seconds, problems=[
                "no timetable meets every rule: relax a teacher's maximum a day or in a row, "
                "free some unavailable periods, unlock entries, or lower an allocation"])
        return SolveResult(status="timeout", seconds=seconds, problems=[
            f"no timetable found within {options.time_limit_seconds:.0f} seconds; "
            "allow more time or relax a rule"])

    entries = [ProposedEntry(section_id=e.section_id, day_of_week=e.day_of_week, period=e.period,
                             subject_id=e.subject_id, teacher_id=e.teacher_id, room_id=e.room_id,
                             co_teacher_id=e.co_teacher_id)
               for e in fixed]
    # A subject with no room kind keeps the room it had in that slot, unless
    # that room is one the solver now books for labs (it could clash there).
    room_of = {(e.section_id, e.day_of_week, e.period, e.subject_id): e.room_id for e in current
               if e.room_id not in managed_rooms}
    lab_of = {(aid, d, p): rid for (aid, d, p, rid), v in y.items() if solver.Value(v)}
    for (aid, d, p), v in x.items():
        if solver.Value(v):
            a = alloc_by_id[aid]
            room = lab_of.get((aid, d, p)) if a.room_kind else room_of.get((a.section_id, d, p, a.subject_id))
            entries.append(ProposedEntry(section_id=a.section_id, day_of_week=d, period=p,
                                         subject_id=a.subject_id, teacher_id=a.teacher_id,
                                         room_id=room, co_teacher_id=a.co_teacher_id))
    group_out = [GroupSession(group_id=gid, day_of_week=d, period=p) for (gid, d, p), v in gx.items()
                 if solver.Value(v)]
    before = {(e.section_id, e.day_of_week, e.period, e.subject_id) for e in current
              if e.section_id in solving}
    before |= {("group", s_.group_id, s_.day_of_week, s_.period) for s_ in placed_groups
               if s_.group_id in solving_groups}
    after = {(e.section_id, e.day_of_week, e.period, e.subject_id) for e in entries
             if e.section_id in solving}
    after |= {("group", s_.group_id, s_.day_of_week, s_.period) for s_ in group_out}
    return SolveResult(status="solved", entries=entries, kept=len(before & after),
                       moved_or_added=len(after - before), removed=len(before - after), seconds=seconds,
                       group_sessions=group_out, group_ids=set(solving_groups))


def _search(solver, model, stall_seconds: float) -> int:
    """solver.Solve, stopped once a week is in hand and no better one has
    come for `stall_seconds` (N-3-18): what is left to gain then is a period
    or two kept in place, or first and last periods shared more evenly."""
    import threading

    from ortools.sat.python import cp_model

    class Progress(cp_model.CpSolverSolutionCallback):
        last: Optional[float] = None

        def on_solution_callback(self) -> None:
            self.last = time.monotonic()

    progress, done = Progress(), threading.Event()

    def watch() -> None:
        while not done.wait(0.2):
            if progress.last is not None and time.monotonic() - progress.last > stall_seconds:
                solver.stop_search()
                return
    watcher = threading.Thread(target=watch, name="timetable-stall", daemon=True)
    watcher.start()
    try:
        return solver.Solve(model, progress)
    finally:
        done.set()
        watcher.join()


def describe(entry: ProposedEntry) -> str:
    return f"{WEEKDAY_NAMES[entry.day_of_week]} period {entry.period}"
