"""
Nested JSON Benchmark

Tests agent's ability to navigate, query, and output nested JSON structures.
Validates correctness of deep traversal, aggregation, and structure matching.
"""

import json
from typing import List
from cave_agent.python_runtime import Variable, PythonRuntime
from core.validation import ValidatorResult
from core.types import BenchmarkTurn, ToolCall


# ============================================================================
# DATA STRUCTURE
# ============================================================================

company = {
    "name": "TechCorp",
    "founded": 2015,
    "departments": [
        {
            "id": "D1",
            "name": "Engineering",
            "budget": 500000,
            "teams": [
                {
                    "id": "T1",
                    "name": "Backend",
                    "members": [
                        {"id": "M1", "name": "Alice", "salary": 95000, "skills": ["Python", "Go", "SQL"]},
                        {"id": "M2", "name": "Bob", "salary": 87000, "skills": ["Python", "Rust"]},
                        {"id": "M3", "name": "Charlie", "salary": 92000, "skills": ["Go", "Kubernetes"]}
                    ]
                },
                {
                    "id": "T2",
                    "name": "Frontend",
                    "members": [
                        {"id": "M4", "name": "Diana", "salary": 88000, "skills": ["JavaScript", "React", "TypeScript"]},
                        {"id": "M5", "name": "Eve", "salary": 85000, "skills": ["JavaScript", "Vue"]}
                    ]
                }
            ]
        },
        {
            "id": "D2",
            "name": "Data Science",
            "budget": 350000,
            "teams": [
                {
                    "id": "T3",
                    "name": "ML",
                    "members": [
                        {"id": "M6", "name": "Frank", "salary": 105000, "skills": ["Python", "TensorFlow", "SQL"]},
                        {"id": "M7", "name": "Grace", "salary": 98000, "skills": ["Python", "PyTorch"]}
                    ]
                },
                {
                    "id": "T4",
                    "name": "Analytics",
                    "members": [
                        {"id": "M8", "name": "Henry", "salary": 82000, "skills": ["Python", "SQL", "Tableau"]}
                    ]
                }
            ]
        },
        {
            "id": "D3",
            "name": "Product",
            "budget": 200000,
            "teams": [
                {
                    "id": "T5",
                    "name": "Design",
                    "members": [
                        {"id": "M9", "name": "Iris", "salary": 78000, "skills": ["Figma", "CSS"]},
                        {"id": "M10", "name": "Jack", "salary": 80000, "skills": ["Figma", "Sketch"]}
                    ]
                }
            ]
        }
    ]
}


# ============================================================================
# EXPECTED VALUES
# ============================================================================

def _compute_expected():
    """Pre-compute all expected values."""

    # Task 1: Total company budget
    total_budget = sum(d["budget"] for d in company["departments"])

    # Task 2: Count all members
    member_count = sum(
        len(team["members"])
        for dept in company["departments"]
        for team in dept["teams"]
    )

    # Task 3: Find member M6
    member_m6 = None
    for dept in company["departments"]:
        for team in dept["teams"]:
            for member in team["members"]:
                if member["id"] == "M6":
                    member_m6 = member
                    break

    # Task 4: Total salary in Engineering
    eng_salary = sum(
        member["salary"]
        for dept in company["departments"]
        if dept["name"] == "Engineering"
        for team in dept["teams"]
        for member in team["members"]
    )

    # Task 5: All unique skills
    all_skills = set()
    for dept in company["departments"]:
        for team in dept["teams"]:
            for member in team["members"]:
                all_skills.update(member["skills"])
    unique_skills = sorted(list(all_skills))

    # Task 6: Members with Python skill
    python_members = []
    for dept in company["departments"]:
        for team in dept["teams"]:
            for member in team["members"]:
                if "Python" in member["skills"]:
                    python_members.append(member["name"])
    python_members = sorted(python_members)

    # Task 7: Avg salary per department
    avg_salary_by_dept = {}
    for dept in company["departments"]:
        salaries = [
            member["salary"]
            for team in dept["teams"]
            for member in team["members"]
        ]
        avg_salary_by_dept[dept["name"]] = sum(salaries) / len(salaries)

    # Task 8: Team with highest total salary
    max_team = None
    max_salary = 0
    for dept in company["departments"]:
        for team in dept["teams"]:
            total = sum(m["salary"] for m in team["members"])
            if total > max_salary:
                max_salary = total
                max_team = team["name"]

    # Task 9: Budget vs salaries per department
    budget_vs_salary = {}
    for dept in company["departments"]:
        total_salary = sum(
            member["salary"]
            for team in dept["teams"]
            for member in team["members"]
        )
        budget_vs_salary[dept["name"]] = {
            "budget": dept["budget"],
            "total_salary": total_salary,
            "ratio": round(dept["budget"] / total_salary, 2)
        }

    # Task 10: Org summary
    org_summary = {
        "company": company["name"],
        "total_departments": len(company["departments"]),
        "total_teams": sum(len(d["teams"]) for d in company["departments"]),
        "total_members": member_count,
        "total_budget": total_budget,
        "departments": [
            {
                "name": d["name"],
                "team_count": len(d["teams"]),
                "member_count": sum(len(t["members"]) for t in d["teams"])
            }
            for d in company["departments"]
        ]
    }

    return {
        "total_budget": total_budget,
        "member_count": member_count,
        "member_m6": member_m6,
        "eng_salary": eng_salary,
        "unique_skills": unique_skills,
        "python_members": python_members,
        "avg_salary_by_dept": avg_salary_by_dept,
        "max_salary_team": max_team,
        "max_salary_total": max_salary,
        "budget_vs_salary": budget_vs_salary,
        "org_summary": org_summary,
    }


EXPECTED = _compute_expected()


# ============================================================================
# VALIDATORS
# ============================================================================

def _extract_result(response: str):
    """Extract result from JSON response."""
    import re
    # Try to find {"result": ...}
    pattern = r'\{\s*"result"\s*:\s*(.+)\s*\}'
    match = re.search(pattern, response, re.DOTALL)
    if match:
        try:
            # Handle the full JSON
            full_match = re.search(r'\{[^{}]*"result"[^{}]*\}|\{"result"\s*:\s*\[.*?\]\}|\{"result"\s*:\s*\{.*?\}\}', response, re.DOTALL)
            if full_match:
                return json.loads(full_match.group())["result"]
        except:
            pass

    # Try to find just the value after "result":
    try:
        # Find JSON objects in response
        for match in re.finditer(r'\{[^{}]+\}|\{.*?\}', response, re.DOTALL):
            try:
                obj = json.loads(match.group())
                if "result" in obj:
                    return obj["result"]
            except:
                continue
    except:
        pass

    return None


def validate_total_budget(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    result = runtime.retrieve("total_budget")
    if result is None:
        result = _extract_result(response)

    if result is None:
        return ValidatorResult(False, "Could not find total_budget result")

    if result == EXPECTED["total_budget"]:
        return ValidatorResult(True, f"total_budget = {result}")
    return ValidatorResult(False, f"Got {result}, expected {EXPECTED['total_budget']}")


def validate_member_count(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    result = runtime.retrieve("member_count")
    if result is None:
        result = _extract_result(response)

    if result is None:
        return ValidatorResult(False, "Could not find member_count result")

    if result == EXPECTED["member_count"]:
        return ValidatorResult(True, f"member_count = {result}")
    return ValidatorResult(False, f"Got {result}, expected {EXPECTED['member_count']}")


def validate_member_m6(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    result = runtime.retrieve("member_m6")
    if result is None:
        result = _extract_result(response)

    if result is None:
        return ValidatorResult(False, "Could not find member_m6 result")

    expected = EXPECTED["member_m6"]
    if isinstance(result, dict):
        if result.get("id") == expected["id"] and result.get("name") == expected["name"]:
            return ValidatorResult(True, f"Found member M6: {result['name']}")
    return ValidatorResult(False, f"Got {result}, expected {expected}")


def validate_eng_salary(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    result = runtime.retrieve("eng_salary")
    if result is None:
        result = _extract_result(response)

    if result is None:
        return ValidatorResult(False, "Could not find eng_salary result")

    if result == EXPECTED["eng_salary"]:
        return ValidatorResult(True, f"eng_salary = {result}")
    return ValidatorResult(False, f"Got {result}, expected {EXPECTED['eng_salary']}")


def validate_unique_skills(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    result = runtime.retrieve("unique_skills")
    if result is None:
        result = _extract_result(response)

    if result is None:
        return ValidatorResult(False, "Could not find unique_skills result")

    expected = EXPECTED["unique_skills"]
    if isinstance(result, list):
        result_sorted = sorted(result)
        if result_sorted == expected:
            return ValidatorResult(True, f"Found {len(result)} unique skills")
    return ValidatorResult(False, f"Got {sorted(result) if isinstance(result, list) else result}, expected {expected}")


def validate_python_members(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    result = runtime.retrieve("python_members")
    if result is None:
        result = _extract_result(response)

    if result is None:
        return ValidatorResult(False, "Could not find python_members result")

    expected = EXPECTED["python_members"]
    if isinstance(result, list):
        result_sorted = sorted(result)
        if result_sorted == expected:
            return ValidatorResult(True, f"Found {len(result)} Python members: {result}")
    return ValidatorResult(False, f"Got {result}, expected {expected}")


def validate_avg_salary_by_dept(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    result = runtime.retrieve("avg_salary_by_dept")
    if result is None:
        result = _extract_result(response)

    if result is None:
        return ValidatorResult(False, "Could not find avg_salary_by_dept result")

    expected = EXPECTED["avg_salary_by_dept"]
    if isinstance(result, dict):
        # Check each department
        for dept, avg in expected.items():
            if dept not in result:
                return ValidatorResult(False, f"Missing department: {dept}")
            if abs(result[dept] - avg) > 0.01:
                return ValidatorResult(False, f"{dept}: got {result[dept]}, expected {avg}")
        return ValidatorResult(True, f"Avg salaries correct: {result}")
    return ValidatorResult(False, f"Got {result}, expected {expected}")


def validate_max_salary_team(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    result = runtime.retrieve("max_salary_team")
    if result is None:
        result = _extract_result(response)

    if result is None:
        return ValidatorResult(False, "Could not find max_salary_team result")

    expected = EXPECTED["max_salary_team"]
    if result == expected:
        return ValidatorResult(True, f"max_salary_team = {result} (${EXPECTED['max_salary_total']})")
    return ValidatorResult(False, f"Got {result}, expected {expected}")


def validate_budget_vs_salary(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    result = runtime.retrieve("budget_vs_salary")
    if result is None:
        result = _extract_result(response)

    if result is None:
        return ValidatorResult(False, "Could not find budget_vs_salary result")

    expected = EXPECTED["budget_vs_salary"]
    if isinstance(result, dict):
        for dept, data in expected.items():
            if dept not in result:
                return ValidatorResult(False, f"Missing department: {dept}")
            r = result[dept]
            if r.get("budget") != data["budget"]:
                return ValidatorResult(False, f"{dept} budget: got {r.get('budget')}, expected {data['budget']}")
            if r.get("total_salary") != data["total_salary"]:
                return ValidatorResult(False, f"{dept} salary: got {r.get('total_salary')}, expected {data['total_salary']}")
        return ValidatorResult(True, f"Budget vs salary correct for all departments")
    return ValidatorResult(False, f"Got {type(result)}, expected dict")


def validate_org_summary(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    result = runtime.retrieve("org_summary")
    if result is None:
        result = _extract_result(response)

    if result is None:
        return ValidatorResult(False, "Could not find org_summary result")

    expected = EXPECTED["org_summary"]
    if not isinstance(result, dict):
        return ValidatorResult(False, f"Expected dict, got {type(result)}")

    # Check top-level fields
    for key in ["company", "total_departments", "total_teams", "total_members", "total_budget"]:
        if result.get(key) != expected.get(key):
            return ValidatorResult(False, f"{key}: got {result.get(key)}, expected {expected.get(key)}")

    # Check departments array
    if "departments" not in result:
        return ValidatorResult(False, "Missing departments array")

    if len(result["departments"]) != len(expected["departments"]):
        return ValidatorResult(False, f"Department count: got {len(result['departments'])}, expected {len(expected['departments'])}")

    return ValidatorResult(True, f"Org summary structure verified: {len(result['departments'])} departments")


# ============================================================================
# EXPORTS
# ============================================================================

tools = []

variables = [
    Variable("company", company, "Company org data with nested departments, teams, and members."),
    Variable("total_budget", None, "Store total company budget here."),
    Variable("member_count", None, "Store total member count here."),
    Variable("member_m6", None, "Store member M6 data here."),
    Variable("eng_salary", None, "Store Engineering dept total salary here."),
    Variable("unique_skills", None, "Store list of unique skills here."),
    Variable("python_members", None, "Store list of Python-skilled member names here."),
    Variable("avg_salary_by_dept", None, "Store avg salary by department here."),
    Variable("max_salary_team", None, "Store team name with highest total salary here."),
    Variable("budget_vs_salary", None, "Store budget vs salary data here."),
    Variable("org_summary", None, "Store org summary JSON here."),
]

validators = {
    "validate_total_budget": validate_total_budget,
    "validate_member_count": validate_member_count,
    "validate_member_m6": validate_member_m6,
    "validate_eng_salary": validate_eng_salary,
    "validate_unique_skills": validate_unique_skills,
    "validate_python_members": validate_python_members,
    "validate_avg_salary_by_dept": validate_avg_salary_by_dept,
    "validate_max_salary_team": validate_max_salary_team,
    "validate_budget_vs_salary": validate_budget_vs_salary,
    "validate_org_summary": validate_org_summary,
}

hooks = {}
