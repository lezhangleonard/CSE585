import json
import random
import argparse

# --- POOL CONFIGURATION ---
HOT_SUBJECTS = [
    "Jack", "Jane", "Alex", "Emma", "Noah"
]
HOT_PREDICATES = [
    "works_on", "collaborates_with", "uses_tool", "located_in", "prefers"
]
HOT_OBJECTS = {
    "works_on": ["ProjectAtlas", "TaskAlpha", "Scheduler", "KnowledgeBase", "PipelineX"],
    "collaborates_with": ["Jack", "Jane", "Alex", "Emma", "Noah"],
    "uses_tool": ["PyTorch", "CUDA", "Jupyter", "Docker", "VSCode"],
    "located_in": ["Nevada", "Boston", "Seattle", "Austin", "Denver"],
    "prefers": ["Python", "Rust", "GPUCluster", "Graphs", "Automation"],
}

COLD_SUBJECTS = [f"User_{i:02d}" for i in range(1, 41)] + [f"Team_{i:02d}" for i in range(1, 11)]
COLD_PREDICATES = [
    "works_on", "collaborates_with", "uses_tool", "located_in", "prefers",
    "researches", "depends_on", "member_of"
]
COLD_OBJECTS = {
    "works_on": [f"Project_{i:02d}" for i in range(1, 21)] + [f"Task_{i:02d}" for i in range(1, 21)],
    "collaborates_with": [f"User_{i:02d}" for i in range(1, 41)],
    "uses_tool": ["Tool_01", "Tool_02", "Notebook", "Kubernetes", "ClusterA", "ClusterB", "TensorFlow"],
    "located_in": ["Chicago", "Phoenix", "Atlanta", "Miami", "Portland", "Dallas", "SanDiego"],
    "prefers": ["Java", "C++", "Databases", "Visualization", "Testing", "Optimization"],
    "researches": [f"Paper_{i:02d}" for i in range(1, 21)] + ["DistributedSystems", "AgentMemory", "Scheduling"],
    "depends_on": [f"Service_{i:02d}" for i in range(1, 16)] + ["Database", "Cache", "Queue", "Planner"],
    "member_of": [f"Group_{i:02d}" for i in range(1, 16)] + ["SystemsLab", "AITeam", "InfraGroup"],
}

TEMPLATES = {
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


def get_component(hot_list, cold_list, ratio):
    return random.choice(hot_list) if random.random() < ratio else random.choice(cold_list)


def get_predicate(hot_ratio):
    return get_component(HOT_PREDICATES, COLD_PREDICATES, hot_ratio)


def get_subject(hot_ratio):
    return get_component(HOT_SUBJECTS, COLD_SUBJECTS, hot_ratio)


def get_object(predicate, hot_ratio):
    hot_pool = HOT_OBJECTS.get(predicate, [])
    cold_pool = COLD_OBJECTS.get(predicate, [])
    if hot_pool and cold_pool:
        return get_component(hot_pool, cold_pool, hot_ratio)
    if hot_pool:
        return random.choice(hot_pool)
    return random.choice(cold_pool)


def generate_workload(n, hot_ratio):
    workload = []

    for i in range(n):
        s = get_subject(hot_ratio)
        p = get_predicate(hot_ratio)
        o = get_object(p, hot_ratio)

        if p == "collaborates_with" and s == o:
            alt_pool = HOT_SUBJECTS + COLD_SUBJECTS
            choices = [x for x in alt_pool if x != s]
            o = random.choice(choices)

        template_type, template = random.choice(TEMPLATES[p])
        text = template.format(s=s, o=o)

        workload.append({
            "id": i,
            "text": text,
            "metadata": {
                "subject": s,
                "predicate": p,
                "object": o,
                "template_type": template_type,
                "subject_hot": s in HOT_SUBJECTS,
                "predicate_hot": p in HOT_PREDICATES,
                "object_hot": o in HOT_OBJECTS.get(p, []),
            }
        })

    return workload


def main():
    parser = argparse.ArgumentParser(description="Generate natural-language KG update workloads.")
    parser.add_argument("--n", type=int, default=100, help="Number of updates")
    parser.add_argument("--hot", type=float, default=0.8, help="Probability of picking a hot item")
    parser.add_argument("--out", type=str, default="workload.json", help="Output file")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")

    args = parser.parse_args()

    random.seed(args.seed)
    workload = generate_workload(args.n, args.hot)

    with open(args.out, "w") as f:
        json.dump(workload, f, indent=2)

    print(f"Generated {args.n} updates (Hot Ratio: {args.hot}, Seed: {args.seed})!")


if __name__ == "__main__":
    main()