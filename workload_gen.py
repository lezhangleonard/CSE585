import json
import random
import argparse

# --- POOL CONFIGURATION ---
HOT_SUBJECTS = [f"Sub_{i}" for i in range(5)]
HOT_PREDICATES = [f"Pred_{i}" for i in range(3)]
HOT_OBJECTS = [f"Obj_{i}" for i in range(5)]

COLD_SUBJECTS = [f"Sub_{i}" for i in range(5,51,1)]
COLD_PREDICATES = [f"Pred_{i}" for i in range(3,21,1)]
COLD_OBJECTS = [f"Obj_{i}" for i in range(5,51,1)]

TEMPLATES = [
    "Demo: {s} {p} {o}."
]

def get_component(hot_list, cold_list, ratio):
    return random.choice(hot_list) if random.random() < ratio else random.choice(cold_list)

def generate_workload(n, hot_ratio):
    workload = []
    
    for i in range(n):
        s = get_component(HOT_SUBJECTS, COLD_SUBJECTS, hot_ratio)
        p = get_component(HOT_PREDICATES, COLD_PREDICATES, hot_ratio)
        o = get_component(HOT_OBJECTS, COLD_OBJECTS, hot_ratio)
        
        template = random.choice(TEMPLATES)
        text = template.format(s=s, p=p, o=o)
        text_type = text.split(':')[0]

        workload.append({
            "id": i,
            "text": text
            # "metadata": {
            #     "subject": s,
            #     "predicate": p,
            #     "object": o,
            #     "template_type": text_type.upper()
            # }
        })
        
    return workload

def main():
    parser = argparse.ArgumentParser(description="Generate formal KG update workloads.")
    parser.add_argument("--n", type=int, default=100, help="Number of updates")
    parser.add_argument("--hot", type=float, default=0.8, help="Probability of picking a hot item")
    parser.add_argument("--out", type=str, default="workload.json", help="Output file")
    
    args = parser.parse_args()

    random.seed(1337)
    
    workload = generate_workload(args.n, args.hot)
    
    with open(args.out, "w") as f:
        json.dump(workload, f, indent=2)
        
    print(f"Generated {args.n} updates (Hot Ratio: {args.hot})!")

if __name__ == "__main__":
    main()