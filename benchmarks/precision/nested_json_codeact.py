"""
Nested JSON Benchmark - CodeAct Version

Agent must output complex nested JSON structures in response.
Tests ability to accurately serialize large nested data.
"""

import json
import re
from typing import List
from cave_agent.python_runtime import Variable, PythonRuntime
from core.validation import ValidatorResult
from core.types import BenchmarkTurn, ToolCall


# ============================================================================
# LARGE DATA STRUCTURE
# ============================================================================

company = {
    "name": "TechCorp",
    "founded": 2015,
    "headquarters": {"city": "San Francisco", "country": "USA", "timezone": "PST"},
    "departments": [
        {
            "id": "D1",
            "name": "Engineering",
            "budget": 2500000,
            "location": "Building A",
            "teams": [
                {
                    "id": "T1",
                    "name": "Backend",
                    "focus": "API Development",
                    "members": [
                        {"id": "M1", "name": "Alice Chen", "salary": 145000, "level": "Senior", "skills": ["Python", "Go", "PostgreSQL", "Redis"], "projects": ["api-v2", "auth-service"]},
                        {"id": "M2", "name": "Bob Smith", "salary": 137000, "level": "Senior", "skills": ["Python", "Rust", "MongoDB"], "projects": ["data-pipeline"]},
                        {"id": "M3", "name": "Charlie Park", "salary": 122000, "level": "Mid", "skills": ["Go", "Kubernetes", "Docker"], "projects": ["api-v2", "infra"]},
                        {"id": "M4", "name": "Diana Lee", "salary": 118000, "level": "Mid", "skills": ["Python", "FastAPI", "PostgreSQL"], "projects": ["auth-service"]}
                    ]
                },
                {
                    "id": "T2",
                    "name": "Frontend",
                    "focus": "Web Applications",
                    "members": [
                        {"id": "M5", "name": "Eve Johnson", "salary": 138000, "level": "Senior", "skills": ["TypeScript", "React", "GraphQL", "CSS"], "projects": ["dashboard", "design-system"]},
                        {"id": "M6", "name": "Frank Wu", "salary": 125000, "level": "Mid", "skills": ["JavaScript", "Vue", "Tailwind"], "projects": ["customer-portal"]},
                        {"id": "M7", "name": "Grace Kim", "salary": 115000, "level": "Mid", "skills": ["TypeScript", "React", "Testing"], "projects": ["dashboard"]}
                    ]
                },
                {
                    "id": "T3",
                    "name": "Platform",
                    "focus": "Infrastructure",
                    "members": [
                        {"id": "M8", "name": "Henry Zhang", "salary": 155000, "level": "Staff", "skills": ["Kubernetes", "Terraform", "AWS", "Python"], "projects": ["cloud-migration", "monitoring"]},
                        {"id": "M9", "name": "Iris Wang", "salary": 142000, "level": "Senior", "skills": ["Docker", "CI/CD", "Linux", "Bash"], "projects": ["infra", "deployment"]}
                    ]
                }
            ]
        },
        {
            "id": "D2",
            "name": "Data Science",
            "budget": 1800000,
            "location": "Building B",
            "teams": [
                {
                    "id": "T4",
                    "name": "Machine Learning",
                    "focus": "Model Development",
                    "members": [
                        {"id": "M10", "name": "Jack Liu", "salary": 165000, "level": "Staff", "skills": ["Python", "TensorFlow", "PyTorch", "SQL"], "projects": ["recommendation-engine", "fraud-detection"]},
                        {"id": "M11", "name": "Karen Chen", "salary": 148000, "level": "Senior", "skills": ["Python", "PyTorch", "Computer Vision"], "projects": ["image-recognition"]},
                        {"id": "M12", "name": "Leo Brown", "salary": 135000, "level": "Mid", "skills": ["Python", "Scikit-learn", "Pandas"], "projects": ["recommendation-engine"]}
                    ]
                },
                {
                    "id": "T5",
                    "name": "Analytics",
                    "focus": "Business Intelligence",
                    "members": [
                        {"id": "M13", "name": "Mia Garcia", "salary": 128000, "level": "Senior", "skills": ["Python", "SQL", "Tableau", "Looker"], "projects": ["exec-dashboard", "metrics"]},
                        {"id": "M14", "name": "Noah Wilson", "salary": 112000, "level": "Mid", "skills": ["SQL", "Python", "dbt", "Airflow"], "projects": ["data-warehouse"]}
                    ]
                },
                {
                    "id": "T6",
                    "name": "NLP",
                    "focus": "Language Models",
                    "members": [
                        {"id": "M15", "name": "Olivia Taylor", "salary": 158000, "level": "Staff", "skills": ["Python", "Transformers", "NLP", "LangChain"], "projects": ["chatbot", "search"]},
                        {"id": "M16", "name": "Peter Martinez", "salary": 140000, "level": "Senior", "skills": ["Python", "BERT", "spaCy"], "projects": ["chatbot"]}
                    ]
                }
            ]
        },
        {
            "id": "D3",
            "name": "Product",
            "budget": 1200000,
            "location": "Building A",
            "teams": [
                {
                    "id": "T7",
                    "name": "Design",
                    "focus": "User Experience",
                    "members": [
                        {"id": "M17", "name": "Quinn Adams", "salary": 132000, "level": "Senior", "skills": ["Figma", "Sketch", "CSS", "User Research"], "projects": ["design-system", "mobile-app"]},
                        {"id": "M18", "name": "Rachel Green", "salary": 118000, "level": "Mid", "skills": ["Figma", "Prototyping", "Accessibility"], "projects": ["customer-portal"]}
                    ]
                },
                {
                    "id": "T8",
                    "name": "Product Management",
                    "focus": "Strategy",
                    "members": [
                        {"id": "M19", "name": "Sam Thompson", "salary": 155000, "level": "Staff", "skills": ["Roadmapping", "Analytics", "SQL", "Jira"], "projects": ["api-v2", "mobile-app"]},
                        {"id": "M20", "name": "Tina Harris", "salary": 138000, "level": "Senior", "skills": ["User Research", "A/B Testing", "SQL"], "projects": ["dashboard", "growth"]}
                    ]
                }
            ]
        },
        {
            "id": "D4",
            "name": "Security",
            "budget": 900000,
            "location": "Building C",
            "teams": [
                {
                    "id": "T9",
                    "name": "AppSec",
                    "focus": "Application Security",
                    "members": [
                        {"id": "M21", "name": "Uma Patel", "salary": 162000, "level": "Staff", "skills": ["Security", "Python", "Penetration Testing", "OWASP"], "projects": ["security-audit", "bug-bounty"]},
                        {"id": "M22", "name": "Victor Nguyen", "salary": 145000, "level": "Senior", "skills": ["Security", "Go", "Cryptography"], "projects": ["auth-service", "encryption"]}
                    ]
                },
                {
                    "id": "T10",
                    "name": "Infrastructure Security",
                    "focus": "Cloud Security",
                    "members": [
                        {"id": "M23", "name": "Wendy Clark", "salary": 152000, "level": "Senior", "skills": ["AWS Security", "Terraform", "Compliance", "SOC2"], "projects": ["cloud-security", "compliance"]},
                        {"id": "M24", "name": "Xavier Jones", "salary": 138000, "level": "Mid", "skills": ["Network Security", "Firewall", "Monitoring"], "projects": ["monitoring", "incident-response"]}
                    ]
                }
            ]
        }
    ]
}


# ============================================================================
# EXPECTED VALUES (complex nested structures)
# ============================================================================

def _compute_expected():
    """Pre-compute all expected complex outputs."""

    # Task 1: Full Engineering department with all nested data
    eng_dept = None
    for d in company["departments"]:
        if d["name"] == "Engineering":
            eng_dept = d
            break

    # Task 2: All staff-level members with full details
    staff_members = []
    for dept in company["departments"]:
        for team in dept["teams"]:
            for member in team["members"]:
                if member["level"] == "Staff":
                    staff_members.append({
                        "member": member,
                        "team": team["name"],
                        "department": dept["name"]
                    })

    # Task 3: All Python developers with team and project info
    python_devs = []
    for dept in company["departments"]:
        for team in dept["teams"]:
            for member in team["members"]:
                if "Python" in member["skills"]:
                    python_devs.append({
                        "id": member["id"],
                        "name": member["name"],
                        "salary": member["salary"],
                        "level": member["level"],
                        "skills": member["skills"],
                        "projects": member["projects"],
                        "team": team["name"],
                        "department": dept["name"]
                    })

    # Task 4: Department summaries with team details
    dept_summaries = []
    for dept in company["departments"]:
        teams_info = []
        total_salary = 0
        for team in dept["teams"]:
            team_salary = sum(m["salary"] for m in team["members"])
            total_salary += team_salary
            teams_info.append({
                "id": team["id"],
                "name": team["name"],
                "focus": team["focus"],
                "member_count": len(team["members"]),
                "total_salary": team_salary,
                "members": [{"id": m["id"], "name": m["name"], "level": m["level"]} for m in team["members"]]
            })
        dept_summaries.append({
            "id": dept["id"],
            "name": dept["name"],
            "budget": dept["budget"],
            "location": dept["location"],
            "team_count": len(dept["teams"]),
            "total_salary": total_salary,
            "budget_remaining": dept["budget"] - total_salary,
            "teams": teams_info
        })

    # Task 5: Project assignments (which members work on which projects)
    projects = {}
    for dept in company["departments"]:
        for team in dept["teams"]:
            for member in team["members"]:
                for proj in member["projects"]:
                    if proj not in projects:
                        projects[proj] = {"project": proj, "members": []}
                    projects[proj]["members"].append({
                        "id": member["id"],
                        "name": member["name"],
                        "team": team["name"],
                        "department": dept["name"]
                    })
    project_assignments = sorted(projects.values(), key=lambda x: x["project"])

    # Task 6: Skills matrix (all unique skills with who has them)
    skills_matrix = {}
    for dept in company["departments"]:
        for team in dept["teams"]:
            for member in team["members"]:
                for skill in member["skills"]:
                    if skill not in skills_matrix:
                        skills_matrix[skill] = []
                    skills_matrix[skill].append({
                        "id": member["id"],
                        "name": member["name"],
                        "level": member["level"]
                    })
    skills_matrix = {k: skills_matrix[k] for k in sorted(skills_matrix.keys())}

    # Task 7: Team comparison (all teams with stats)
    team_comparison = []
    for dept in company["departments"]:
        for team in dept["teams"]:
            salaries = [m["salary"] for m in team["members"]]
            team_comparison.append({
                "team_id": team["id"],
                "team_name": team["name"],
                "department": dept["name"],
                "focus": team["focus"],
                "member_count": len(team["members"]),
                "total_salary": sum(salaries),
                "avg_salary": round(sum(salaries) / len(salaries), 2),
                "min_salary": min(salaries),
                "max_salary": max(salaries)
            })

    # Task 8: High earners (salary > 150000) with full context
    high_earners = []
    for dept in company["departments"]:
        for team in dept["teams"]:
            for member in team["members"]:
                if member["salary"] > 150000:
                    high_earners.append({
                        "member": member,
                        "team": {"id": team["id"], "name": team["name"], "focus": team["focus"]},
                        "department": {"id": dept["id"], "name": dept["name"], "location": dept["location"]}
                    })
    high_earners.sort(key=lambda x: x["member"]["salary"], reverse=True)

    # Task 9: Cross-functional projects (projects with members from multiple departments)
    project_depts = {}
    for dept in company["departments"]:
        for team in dept["teams"]:
            for member in team["members"]:
                for proj in member["projects"]:
                    if proj not in project_depts:
                        project_depts[proj] = {"project": proj, "departments": set(), "members": []}
                    project_depts[proj]["departments"].add(dept["name"])
                    project_depts[proj]["members"].append({
                        "id": member["id"],
                        "name": member["name"],
                        "department": dept["name"],
                        "team": team["name"]
                    })
    cross_functional = []
    for proj, data in project_depts.items():
        if len(data["departments"]) > 1:
            cross_functional.append({
                "project": proj,
                "departments": sorted(list(data["departments"])),
                "member_count": len(data["members"]),
                "members": data["members"]
            })
    cross_functional.sort(key=lambda x: x["project"])

    # Task 10: Full org chart
    org_chart = {
        "company": company["name"],
        "founded": company["founded"],
        "headquarters": company["headquarters"],
        "total_departments": len(company["departments"]),
        "total_teams": sum(len(d["teams"]) for d in company["departments"]),
        "total_members": sum(len(t["members"]) for d in company["departments"] for t in d["teams"]),
        "total_budget": sum(d["budget"] for d in company["departments"]),
        "departments": [
            {
                "id": d["id"],
                "name": d["name"],
                "budget": d["budget"],
                "location": d["location"],
                "teams": [
                    {
                        "id": t["id"],
                        "name": t["name"],
                        "focus": t["focus"],
                        "members": [
                            {"id": m["id"], "name": m["name"], "level": m["level"], "salary": m["salary"]}
                            for m in t["members"]
                        ]
                    }
                    for t in d["teams"]
                ]
            }
            for d in company["departments"]
        ]
    }

    return {
        "eng_dept": eng_dept,
        "staff_members": staff_members,
        "python_devs": python_devs,
        "dept_summaries": dept_summaries,
        "project_assignments": project_assignments,
        "skills_matrix": skills_matrix,
        "team_comparison": team_comparison,
        "high_earners": high_earners,
        "cross_functional": cross_functional,
        "org_chart": org_chart,
    }


EXPECTED = _compute_expected()


# ============================================================================
# RESPONSE PARSING
# ============================================================================

def _extract_json(response: str):
    """Extract JSON object from response."""
    # Find the largest valid JSON object
    best_match = None
    best_len = 0

    # Pattern for nested JSON
    depth = 0
    start = -1
    for i, c in enumerate(response):
        if c == '{':
            if depth == 0:
                start = i
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    candidate = response[start:i+1]
                    obj = json.loads(candidate)
                    if "result" in obj and len(candidate) > best_len:
                        best_match = obj["result"]
                        best_len = len(candidate)
                except:
                    pass
                start = -1

    return best_match


# ============================================================================
# VALIDATORS
# ============================================================================

def _compare_nested(result, expected, path=""):
    """Deep compare two nested structures."""
    if type(result) != type(expected):
        return False, f"{path}: type mismatch {type(result).__name__} vs {type(expected).__name__}"

    if isinstance(expected, dict):
        for key in expected:
            if key not in result:
                return False, f"{path}.{key}: missing key"
            ok, msg = _compare_nested(result[key], expected[key], f"{path}.{key}")
            if not ok:
                return False, msg
        return True, "ok"

    elif isinstance(expected, list):
        if len(result) != len(expected):
            return False, f"{path}: length mismatch {len(result)} vs {len(expected)}"
        for i, (r, e) in enumerate(zip(result, expected)):
            ok, msg = _compare_nested(r, e, f"{path}[{i}]")
            if not ok:
                return False, msg
        return True, "ok"

    else:
        if result != expected:
            return False, f"{path}: value mismatch {result} vs {expected}"
        return True, "ok"


def validate_eng_dept(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    result = _extract_json(response)
    if result is None:
        return ValidatorResult(False, "Could not parse JSON from response")

    expected = EXPECTED["eng_dept"]
    ok, msg = _compare_nested(result, expected)
    if ok:
        return ValidatorResult(True, f"Engineering dept verified ({len(expected['teams'])} teams)")
    return ValidatorResult(False, msg)


def validate_staff_members(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    result = _extract_json(response)
    if result is None:
        return ValidatorResult(False, "Could not parse JSON from response")

    expected = EXPECTED["staff_members"]
    if not isinstance(result, list):
        return ValidatorResult(False, f"Expected list, got {type(result)}")
    if len(result) != len(expected):
        return ValidatorResult(False, f"Expected {len(expected)} staff members, got {len(result)}")

    # Check each staff member exists
    result_ids = {r.get("member", {}).get("id") for r in result}
    expected_ids = {e["member"]["id"] for e in expected}
    if result_ids != expected_ids:
        return ValidatorResult(False, f"Staff member IDs mismatch")

    return ValidatorResult(True, f"Verified {len(result)} staff members")


def validate_python_devs(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    result = _extract_json(response)
    if result is None:
        return ValidatorResult(False, "Could not parse JSON from response")

    expected = EXPECTED["python_devs"]
    if not isinstance(result, list):
        return ValidatorResult(False, f"Expected list, got {type(result)}")
    if len(result) != len(expected):
        return ValidatorResult(False, f"Expected {len(expected)} Python devs, got {len(result)}")

    result_ids = {r.get("id") for r in result}
    expected_ids = {e["id"] for e in expected}
    if result_ids != expected_ids:
        return ValidatorResult(False, f"Python dev IDs mismatch")

    return ValidatorResult(True, f"Verified {len(result)} Python developers")


def validate_dept_summaries(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    result = _extract_json(response)
    if result is None:
        return ValidatorResult(False, "Could not parse JSON from response")

    expected = EXPECTED["dept_summaries"]
    if not isinstance(result, list):
        return ValidatorResult(False, f"Expected list, got {type(result)}")
    if len(result) != len(expected):
        return ValidatorResult(False, f"Expected {len(expected)} departments, got {len(result)}")

    for exp_dept in expected:
        found = False
        for res_dept in result:
            if res_dept.get("id") == exp_dept["id"]:
                found = True
                if res_dept.get("total_salary") != exp_dept["total_salary"]:
                    return ValidatorResult(False, f"{exp_dept['name']}: salary mismatch")
                if len(res_dept.get("teams", [])) != len(exp_dept["teams"]):
                    return ValidatorResult(False, f"{exp_dept['name']}: team count mismatch")
                break
        if not found:
            return ValidatorResult(False, f"Missing department {exp_dept['id']}")

    return ValidatorResult(True, f"Verified {len(result)} department summaries")


def validate_project_assignments(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    result = _extract_json(response)
    if result is None:
        return ValidatorResult(False, "Could not parse JSON from response")

    expected = EXPECTED["project_assignments"]
    if not isinstance(result, list):
        return ValidatorResult(False, f"Expected list, got {type(result)}")
    if len(result) != len(expected):
        return ValidatorResult(False, f"Expected {len(expected)} projects, got {len(result)}")

    result_projects = {r.get("project") for r in result}
    expected_projects = {e["project"] for e in expected}
    if result_projects != expected_projects:
        return ValidatorResult(False, f"Project names mismatch")

    return ValidatorResult(True, f"Verified {len(result)} project assignments")


def validate_skills_matrix(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    result = _extract_json(response)
    if result is None:
        return ValidatorResult(False, "Could not parse JSON from response")

    expected = EXPECTED["skills_matrix"]
    if not isinstance(result, dict):
        return ValidatorResult(False, f"Expected dict, got {type(result)}")
    if len(result) != len(expected):
        return ValidatorResult(False, f"Expected {len(expected)} skills, got {len(result)}")

    for skill, members in expected.items():
        if skill not in result:
            return ValidatorResult(False, f"Missing skill: {skill}")
        if len(result[skill]) != len(members):
            return ValidatorResult(False, f"{skill}: member count mismatch")

    return ValidatorResult(True, f"Verified {len(result)} skills in matrix")


def validate_team_comparison(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    result = _extract_json(response)
    if result is None:
        return ValidatorResult(False, "Could not parse JSON from response")

    expected = EXPECTED["team_comparison"]
    if not isinstance(result, list):
        return ValidatorResult(False, f"Expected list, got {type(result)}")
    if len(result) != len(expected):
        return ValidatorResult(False, f"Expected {len(expected)} teams, got {len(result)}")

    for exp_team in expected:
        found = False
        for res_team in result:
            if res_team.get("team_id") == exp_team["team_id"]:
                found = True
                if res_team.get("total_salary") != exp_team["total_salary"]:
                    return ValidatorResult(False, f"{exp_team['team_name']}: salary mismatch")
                break
        if not found:
            return ValidatorResult(False, f"Missing team {exp_team['team_id']}")

    return ValidatorResult(True, f"Verified {len(result)} team comparisons")


def validate_high_earners(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    result = _extract_json(response)
    if result is None:
        return ValidatorResult(False, "Could not parse JSON from response")

    expected = EXPECTED["high_earners"]
    if not isinstance(result, list):
        return ValidatorResult(False, f"Expected list, got {type(result)}")
    if len(result) != len(expected):
        return ValidatorResult(False, f"Expected {len(expected)} high earners, got {len(result)}")

    result_ids = {r.get("member", {}).get("id") for r in result}
    expected_ids = {e["member"]["id"] for e in expected}
    if result_ids != expected_ids:
        return ValidatorResult(False, f"High earner IDs mismatch")

    return ValidatorResult(True, f"Verified {len(result)} high earners (>150k)")


def validate_cross_functional(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    result = _extract_json(response)
    if result is None:
        return ValidatorResult(False, "Could not parse JSON from response")

    expected = EXPECTED["cross_functional"]
    if not isinstance(result, list):
        return ValidatorResult(False, f"Expected list, got {type(result)}")
    if len(result) != len(expected):
        return ValidatorResult(False, f"Expected {len(expected)} cross-functional projects, got {len(result)}")

    result_projects = {r.get("project") for r in result}
    expected_projects = {e["project"] for e in expected}
    if result_projects != expected_projects:
        return ValidatorResult(False, f"Cross-functional project names mismatch")

    return ValidatorResult(True, f"Verified {len(result)} cross-functional projects")


def validate_org_chart(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    result = _extract_json(response)
    if result is None:
        return ValidatorResult(False, "Could not parse JSON from response")

    expected = EXPECTED["org_chart"]

    # Check top-level fields
    for key in ["company", "total_departments", "total_teams", "total_members", "total_budget"]:
        if result.get(key) != expected.get(key):
            return ValidatorResult(False, f"{key}: {result.get(key)} vs {expected.get(key)}")

    # Check departments exist
    if len(result.get("departments", [])) != len(expected["departments"]):
        return ValidatorResult(False, "Department count mismatch")

    # Check all members are present
    result_members = []
    for d in result.get("departments", []):
        for t in d.get("teams", []):
            result_members.extend([m.get("id") for m in t.get("members", [])])

    expected_members = []
    for d in expected["departments"]:
        for t in d["teams"]:
            expected_members.extend([m["id"] for m in t["members"]])

    if set(result_members) != set(expected_members):
        return ValidatorResult(False, f"Member IDs mismatch in org chart")

    return ValidatorResult(True, f"Verified org chart ({expected['total_members']} members)")


# ============================================================================
# EXPORTS
# ============================================================================

tools = []

variables = [
    Variable("company", company, "Company org data with departments, teams, and members."),
]

validators = {
    "validate_eng_dept": validate_eng_dept,
    "validate_staff_members": validate_staff_members,
    "validate_python_devs": validate_python_devs,
    "validate_dept_summaries": validate_dept_summaries,
    "validate_project_assignments": validate_project_assignments,
    "validate_skills_matrix": validate_skills_matrix,
    "validate_team_comparison": validate_team_comparison,
    "validate_high_earners": validate_high_earners,
    "validate_cross_functional": validate_cross_functional,
    "validate_org_chart": validate_org_chart,
}

hooks = {}
