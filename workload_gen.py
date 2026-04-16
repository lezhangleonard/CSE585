import json
import random
import argparse

# =========================
# SYNTHETIC CORRELATED
# =========================

SYN_HOT_SUBJECTS = [f"Sub_{i}" for i in range(5)]
SYN_HOT_PREDICATES = [f"Pred_{i}" for i in range(3)]
SYN_HOT_OBJECTS = [f"Obj_{i}" for i in range(5)]

SYN_COLD_SUBJECTS = [f"Sub_{i}" for i in range(5, 51)]
SYN_COLD_PREDICATES = [f"Pred_{i}" for i in range(3, 21)]
SYN_COLD_OBJECTS = [f"Obj_{i}" for i in range(5, 51)]

SYN_TEMPLATES = ["Demo : {s} {p} {o}."]

SYN_HOT_SUBJECT_PREDICATES = {
    "Sub_0": ["Pred_0", "Pred_1", "Pred_2"],
    "Sub_1": ["Pred_0", "Pred_1", "Pred_2"],
    "Sub_2": ["Pred_0", "Pred_1", "Pred_2"],
    "Sub_3": ["Pred_0", "Pred_1", "Pred_2"],
    "Sub_4": ["Pred_0", "Pred_1", "Pred_2"],
}

SYN_HOT_OBJECTS = {
    "Pred_0": ["Obj_0", "Obj_1", "Obj_2", "Obj_3", "Obj_4"],
    "Pred_1": ["Obj_0", "Obj_1", "Obj_2", "Obj_3", "Obj_4"],
    "Pred_2": ["Obj_0", "Obj_1", "Obj_2", "Obj_3", "Obj_4"],
}

def pick(hot_list, cold_list, ratio):
    return random.choice(hot_list) if random.random() < ratio else random.choice(cold_list)

def generate_synthetic_workload(n, hot_ratio, seed):
    random.seed(seed)
    workload = []
    for i in range(n):
        s = pick(SYN_HOT_SUBJECTS, SYN_COLD_SUBJECTS, hot_ratio)
        p = random.choice(SYN_HOT_SUBJECT_PREDICATES.get(s, SYN_HOT_PREDICATES)) if random.random() < hot_ratio else random.choice(SYN_COLD_PREDICATES)
        o = random.choice(SYN_HOT_OBJECTS.get(p, SYN_HOT_OBJECTS["Pred_0"])) if random.random() < hot_ratio else random.choice(SYN_COLD_OBJECTS)

        text = random.choice(SYN_TEMPLATES).format(s=s, p=p, o=o)
        workload.append({
            "id": i,
            "text": text,
            "metadata": {
                "subject": s,
                "predicate": p,
                "object": o,
                "subject_hot": s in SYN_HOT_SUBJECTS,
                "predicate_hot": p in SYN_HOT_PREDICATES,
                "object_hot": o in SYN_HOT_OBJECTS.get(p, []),
            }
        })
    return workload

# =========================
# REAL CORRELATED
# =========================

REAL_HOT_SUBJECTS = ["Jack", "Jane", "Alex", "Emma", "Noah"]

REAL_HOT_SUBJECT_PREDICATES = {
    "Jack": ["works_on", "collaborates_with", "uses_tool", "located_in", "prefers"],
    "Jane": ["works_on", "collaborates_with", "uses_tool", "located_in", "prefers"],
    "Alex": ["works_on", "collaborates_with", "uses_tool", "located_in", "prefers"],
    "Emma": ["works_on", "collaborates_with", "uses_tool", "located_in", "prefers"],
    "Noah": ["works_on", "collaborates_with", "uses_tool", "located_in", "prefers"],
}

REAL_HOT_OBJECTS = {
    "works_on": ["ProjectAtlas", "TaskAlpha", "Scheduler", "KnowledgeBase", "PipelineX"],
    "collaborates_with": ["Jack", "Jane", "Alex", "Emma", "Noah"],
    "uses_tool": ["PyTorch", "CUDA", "Jupyter", "Docker", "VSCode"],
    "located_in": ["Nevada", "Boston", "Seattle", "Austin", "Denver"],
    "prefers": ["Python", "Rust", "GPUCluster", "Graphs", "Automation"],
}

REAL_COLD_SUBJECTS = [f"User_{i:02d}" for i in range(1, 41)] + [f"Team_{i:02d}" for i in range(1, 11)]
REAL_COLD_PREDICATES = [
    "works_on", "collaborates_with", "uses_tool", "located_in", "prefers",
    "researches", "depends_on", "member_of"
]
REAL_COLD_OBJECTS = {
    "works_on": [f"Project_{i:02d}" for i in range(1, 21)] + [f"Task_{i:02d}" for i in range(1, 21)],
    "collaborates_with": [f"User_{i:02d}" for i in range(1, 41)],
    "uses_tool": ["Tool_01", "Tool_02", "Notebook", "Kubernetes", "ClusterA", "ClusterB", "TensorFlow"],
    "located_in": ["Chicago", "Phoenix", "Atlanta", "Miami", "Portland", "Dallas", "SanDiego"],
    "prefers": ["Java", "C++", "Databases", "Visualization", "Testing", "Optimization"],
    "researches": [f"Paper_{i:02d}" for i in range(1, 21)] + ["DistributedSystems", "AgentMemory", "Scheduling"],
    "depends_on": [f"Service_{i:02d}" for i in range(1, 16)] + ["Database", "Cache", "Queue", "Planner"],
    "member_of": [f"Group_{i:02d}" for i in range(1, 16)] + ["SystemsLab", "AITeam", "InfraGroup"],
}

REAL_TEMPLATES = {
    "works_on": [
        ("works_on_basic", "{s} works on {o}."),
        ("works_on_current", "{s} is currently working on {o}."),
        ("works_on_update", "Latest update: {s} now works on {o}."),
    ],
    "collaborates_with": [
        ("collab_basic", "{s} collaborates with {o}."),
        ("collab_progressive", "{s} is collaborating with {o}."),
        ("collab_note", "According to the latest note, {s} collaborates with {o}."),
    ],
    "uses_tool": [
        ("tool_basic", "{s} uses {o} as a tool."),
        ("tool_progressive", "{s} is using {o}."),
        ("tool_note", "The system notes that {s} uses {o}."),
    ],
    "located_in": [
        ("loc_basic", "{s} is located in {o}."),
        ("loc_based", "{s} is based in {o}."),
        ("loc_memory", "Current memory says that {s} is in {o}."),
    ],
    "prefers": [
        ("pref_basic", "{s} prefers {o}."),
        ("pref_noun", "{s} has a preference for {o}."),
        ("pref_update", "The latest preference update says that {s} prefers {o}."),
    ],
    "researches": [
        ("research_basic", "{s} researches {o}."),
        ("research_study", "{s} is studying {o}."),
        ("research_update", "A recent update indicates that {s} researches {o}."),
    ],
    "depends_on": [
        ("depends_basic", "{s} depends on {o}."),
        ("depends_state", "Current state: {s} depends on {o}."),
        ("depends_progressive", "{s} is dependent on {o}."),
    ],
    "member_of": [
        ("member_basic", "{s} is a member of {o}."),
        ("member_current", "{s} currently belongs to {o}."),
        ("member_record", "The record shows that {s} is part of {o}."),
    ],
}

def sample_real_subject(hot_ratio):
    return random.choice(REAL_HOT_SUBJECTS) if random.random() < hot_ratio else random.choice(REAL_COLD_SUBJECTS)

def sample_real_predicate(subject, hot_ratio):
    if subject in REAL_HOT_SUBJECT_PREDICATES and random.random() < hot_ratio:
        return random.choice(REAL_HOT_SUBJECT_PREDICATES[subject])
    return random.choice(REAL_COLD_PREDICATES)

def sample_real_object(predicate, subject, hot_ratio):
    hot_pool = REAL_HOT_OBJECTS.get(predicate, [])
    cold_pool = REAL_COLD_OBJECTS.get(predicate, [])
    if hot_pool and random.random() < hot_ratio:
        o = random.choice(hot_pool)
    else:
        o = random.choice(cold_pool) if cold_pool else random.choice(hot_pool)

    if predicate == "collaborates_with" and o == subject:
        alt = [x for x in (REAL_HOT_SUBJECTS + REAL_COLD_SUBJECTS) if x != subject]
        o = random.choice(alt)
    return o

def generate_real_workload(n, hot_ratio, seed):
    random.seed(seed)
    workload = []
    for i in range(n):
        s = sample_real_subject(hot_ratio)
        p = sample_real_predicate(s, hot_ratio)
        o = sample_real_object(p, s, hot_ratio)
        template_type, template = random.choice(REAL_TEMPLATES[p])
        workload.append({
            "id": i,
            "text": template.format(s=s, o=o),
            "metadata": {
                "subject": s,
                "predicate": p,
                "object": o,
                "template_type": template_type,
                "subject_hot": s in REAL_HOT_SUBJECTS,
                "predicate_hot": (s in REAL_HOT_SUBJECT_PREDICATES and p in REAL_HOT_SUBJECT_PREDICATES[s]),
                "object_hot": o in REAL_HOT_OBJECTS.get(p, []),
            }
        })
    return workload

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["synthetic", "real"], required=True)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--hot", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--out", type=str, default="workload.json")
    args = parser.parse_args()

    if args.mode == "synthetic":
        workload = generate_synthetic_workload(args.n, args.hot, args.seed)
    else:
        workload = generate_real_workload(args.n, args.hot, args.seed)

    with open(args.out, "w") as f:
        json.dump(workload, f, indent=2)

    print(f"Generated {args.mode} workload: n={args.n}, hot={args.hot}, seed={args.seed}")

if __name__ == "__main__":
    main()