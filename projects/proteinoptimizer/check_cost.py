import weave

weave.init("project_costs_single")
model_name = "gpt-5-2025-"
@weave.op()
def get_costs_for_project(project_name: str):
    total_cost = 0
    requests = 0

    client = weave.init(project_name)
    # Fetch all the calls in the project
    calls = list(
        client.get_calls(filter={"trace_roots_only": True}, include_costs=True)
    )

    for call in calls:
        # If the call has costs, we add them to the total cost
        if call.summary["weave"] is not None and call.summary["weave"].get("costs", None) is not None:
            
            for k, cost in call.summary["weave"]["costs"].items():
                if model_name in k:
                    requests += cost["requests"]
                    total_cost += cost["prompt_tokens_total_cost"]
                    total_cost += cost["completion_tokens_total_cost"]

    # We return the total cost, requests, and calls
    return {
        "total_cost": total_cost,
        "requests": requests,
        "calls": len(calls),
    }

# Since we decorated our function with @weave.op(),
# our totals are stored in weave for historic cost total calculations

# Check costs for both projects
projects = ["sde-harness-protein_multiobj", "sde-harness-protein_singleobj"]

print("=" * 60)
print("Cost Summary for Projects")
print("=" * 60)

for project_name in projects:
    print(f"\nProject: {project_name}")
    print("-" * 60)
    result = get_costs_for_project(project_name)
    print(f"Total Cost: ${result['total_cost']:.4f}")
    print(f"Total Requests: {result['requests']}")
    print(f"Total Calls: {result['calls']}")

print("\n" + "=" * 60)