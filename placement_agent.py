"""
Placement Assistant - multi-agent LangChain project.

Run in Colab or locally:
    pip install -q langchain langchain-google-genai

Modules:
    1. Check placement eligibility
    2. Analyze job description
    3. Match skills
    4. Generate interview questions
    5. Create preparation plan
    6. Check resume & placement readiness   <-- new module
"""

import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain.agents import create_agent


# --------------------------------------------------
# SETUP
# --------------------------------------------------

os.environ["GOOGLE_API_KEY"] = "MY_API_KEY"  # <-- replace with your real key

# NOTE: "gemini-3.5-flash-lite" is not a valid model id and will 404.
# Use a real one, e.g. "gemini-2.0-flash" (or whatever your key can access).
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash"
)


# --------------------------------------------------
# TOOL 1 + AGENT: ELIGIBILITY
# --------------------------------------------------

@tool
def check_eligibility(
    cgpa: float,
    required_cgpa: float,
    branch: str,
    eligible_branches: str,
    backlogs: int
) -> str:
    """
    Check whether a student is eligible for a placement opportunity.

    Args:
        cgpa: Student's current CGPA.
        required_cgpa: Minimum CGPA required by the company.
        branch: Student's academic branch.
        eligible_branches: Comma-separated branches accepted by the company.
        backlogs: Number of active backlogs.

    Returns:
        A simple eligibility result explaining whether the student is eligible.
    """

    branches = [
        branch_name.strip().lower()
        for branch_name in eligible_branches.split(",")
    ]

    if (
        cgpa >= required_cgpa
        and branch.lower() in branches
        and backlogs == 0
    ):
        return "Eligible for the placement opportunity."

    reasons = []

    if cgpa < required_cgpa:
        reasons.append("CGPA is below the required CGPA.")

    if branch.lower() not in branches:
        reasons.append("Branch is not eligible.")

    if backlogs > 0:
        reasons.append("Student has active backlogs.")

    return "Not eligible. " + " ".join(reasons)


eligibility_tools = [
    check_eligibility
]

eligibility_agent = create_agent(
    model=llm,
    tools=eligibility_tools,
    system_prompt="""
    You are a placement eligibility agent.

    Your job is to determine whether a student satisfies
    the eligibility requirements of a placement opportunity.

    Use the check_eligibility tool when eligibility information
    needs to be calculated.

    Give the final answer clearly and briefly.
    """
)


# --------------------------------------------------
# TOOL 2 + AGENT: JOB DESCRIPTION ANALYSIS
# --------------------------------------------------

@tool
def analyze_job_description(job_description: str) -> str:
    """
    Extract important placement requirements from a job description.

    Args:
        job_description: The complete job description provided by a company.

    Returns:
        The job description itself for the agent to analyze into
        role, skills, eligibility requirements, and responsibilities.
    """

    return job_description


job_analysis_tools = [
    analyze_job_description
]

job_analysis_agent = create_agent(
    model=llm,
    tools=job_analysis_tools,
    system_prompt="""
    You are a placement job description analysis agent.

    Your job is to analyze a company's job description.

    Identify:
    - Job role
    - Required technical skills
    - Preferred skills
    - Eligibility requirements
    - Main responsibilities

    Use the analyze_job_description tool when a job description
    needs to be processed.

    Keep the final answer structured and concise.
    """
)


# --------------------------------------------------
# TOOL 3 + AGENT: SKILL MATCHING
# --------------------------------------------------

@tool
def match_skills(
    student_skills: str,
    required_skills: str
) -> str:
    """
    Compare a student's skills with the skills required for a placement role.

    Args:
        student_skills: Comma-separated skills known by the student.
        required_skills: Comma-separated skills required by the company.

    Returns:
        A list of matched skills and missing skills.
    """

    student = {
        skill.strip().lower()
        for skill in student_skills.split(",")
    }

    required = {
        skill.strip().lower()
        for skill in required_skills.split(",")
    }

    matched = student.intersection(required)
    missing = required - student

    return (
        f"Matched skills: {', '.join(sorted(matched)) or 'None'}\n"
        f"Missing skills: {', '.join(sorted(missing)) or 'None'}"
    )


skill_matching_tools = [
    match_skills
]

skill_matching_agent = create_agent(
    model=llm,
    tools=skill_matching_tools,
    system_prompt="""
    You are a placement skill matching agent.

    Compare a student's skills with the skills required
    for a placement role.

    Use the match_skills tool to perform the comparison.

    Clearly show:
    - Matched skills
    - Missing skills

    Do not invent skills that the student has not provided.
    """
)


# --------------------------------------------------
# TOOL 4 + AGENT: INTERVIEW QUESTIONS
# --------------------------------------------------

@tool
def generate_interview_questions(
    role: str,
    skills: str
) -> str:
    """
    Prepare the information needed to generate placement interview questions.

    Args:
        role: The job role the student is applying for.
        skills: Comma-separated technical skills relevant to the role.

    Returns:
        The role and skills that should be used for interview preparation.
    """

    return f"Role: {role}\nSkills: {skills}"


interview_tools = [
    generate_interview_questions
]

interview_agent = create_agent(
    model=llm,
    tools=interview_tools,
    system_prompt="""
    You are a placement interview preparation agent.

    Generate useful interview questions for a given job role.

    Include:
    - Technical questions
    - Basic conceptual questions
    - Practical coding questions
    - A few behavioral questions

    Use the generate_interview_questions tool first
    when the user provides a role and skills.

    Keep the questions suitable for a student preparing
    for placement interviews.
    """
)


# --------------------------------------------------
# TOOL 5 + AGENT: PREPARATION PLAN
# --------------------------------------------------

@tool
def create_prep_plan(
    role: str,
    days: int,
    skills: str
) -> str:
    """
    Prepare the information needed to create a placement preparation plan.

    Args:
        role: The placement role the student is preparing for.
        days: Number of days available for preparation.
        skills: Comma-separated skills required for the role.

    Returns:
        The role, preparation duration, and required skills.
    """

    return (
        f"Role: {role}\n"
        f"Preparation days: {days}\n"
        f"Required skills: {skills}"
    )


prep_tools = [
    create_prep_plan
]

prep_agent = create_agent(
    model=llm,
    tools=prep_tools,
    system_prompt="""
    You are a placement preparation planning agent.

    Create a simple preparation plan for a student.

    Divide the available preparation time into useful areas
    such as:
    - DSA
    - Core technical skills
    - Coding practice
    - Projects
    - Resume preparation
    - Interview preparation

    Use the create_prep_plan tool to obtain the preparation
    information.

    Keep the plan realistic for a student.
    """
)


# ==================================================
# NEW MODULE: RESUME & PLACEMENT READINESS
# ==================================================

@tool
def resume_ats_score(
    resume_text: str,
    required_skills: str
) -> str:
    """
    Check how well a resume covers the skills required for a role (ATS-style).

    Args:
        resume_text: The full text of the student's resume.
        required_skills: Comma-separated skills the company requires.

    Returns:
        The ATS coverage percentage plus which required skills are
        present and which are missing from the resume.
    """

    resume_lower = resume_text.lower()

    required = [
        skill.strip()
        for skill in required_skills.split(",")
        if skill.strip()
    ]

    present = [
        skill for skill in required
        if skill.lower() in resume_lower
    ]

    missing = [
        skill for skill in required
        if skill.lower() not in resume_lower
    ]

    score = round((len(present) / len(required)) * 100) if required else 0

    return (
        f"ATS coverage score: {score}%\n"
        f"Skills found in resume: {', '.join(present) or 'None'}\n"
        f"Skills missing from resume: {', '.join(missing) or 'None'}"
    )


@tool
def placement_readiness_score(
    cgpa: float,
    matched_skills: int,
    total_required_skills: int,
    projects: int
) -> str:
    """
    Estimate a student's overall placement readiness as a simple score.

    Args:
        cgpa: Student's current CGPA (out of 10).
        matched_skills: Number of required skills the student already has.
        total_required_skills: Total number of skills the role requires.
        projects: Number of relevant projects the student has done.

    Returns:
        A readiness score out of 100 with a short readiness level.
    """

    cgpa_points = (cgpa / 10) * 40

    if total_required_skills > 0:
        skill_points = (matched_skills / total_required_skills) * 40
    else:
        skill_points = 0

    project_points = min(projects, 4) / 4 * 20

    total = round(cgpa_points + skill_points + project_points)

    if total >= 75:
        level = "Strong - almost placement ready."
    elif total >= 50:
        level = "Moderate - needs focused improvement."
    else:
        level = "Low - significant preparation required."

    return f"Placement readiness score: {total}/100\nReadiness level: {level}"


readiness_tools = [
    resume_ats_score,
    placement_readiness_score
]

readiness_agent = create_agent(
    model=llm,
    tools=readiness_tools,
    system_prompt="""
    You are a placement resume and readiness agent.

    Your job is to help a student understand:
    - How well their resume covers the required skills (ATS score)
    - Their overall placement readiness

    Use the resume_ats_score tool when a resume and required skills
    are provided.

    Use the placement_readiness_score tool when CGPA, skills, and
    project information are provided.

    Give clear, encouraging, and practical feedback.
    Suggest what the student should improve next.
    """
)


# --------------------------------------------------
# PREDEFINED PLACEMENT INFORMATION
# --------------------------------------------------

company = "ABC Technologies"

required_cgpa = 8.0
eligible_branches = "CSE, IT, ECE"

role = "Data Science Intern"

required_skills = "Python, SQL, Machine Learning, Statistics"


# --------------------------------------------------
# USER CHOICE
# --------------------------------------------------

choice = input("""
Choose what you want to do:

1. Check placement eligibility
2. Analyze job description
3. Match skills
4. Generate interview questions
5. Create preparation plan
6. Check resume & placement readiness

Enter your choice:
""")


# --------------------------------------------------
# 1. CHECK ELIGIBILITY
# --------------------------------------------------

if choice == "1":

    cgpa = input("Enter your CGPA: ")
    branch = input("Enter your branch: ")
    backlogs = input("Enter number of backlogs: ")

    response = eligibility_agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": f"""
                Check my eligibility for {company}.

                Company requirements:
                Required CGPA: {required_cgpa}
                Eligible branches: {eligible_branches}

                My details:
                CGPA: {cgpa}
                Branch: {branch}
                Backlogs: {backlogs}
                """
            }
        ]
    })

    print("\nResult:")
    print(response["messages"][-1].content)


# --------------------------------------------------
# 2. ANALYZE JOB DESCRIPTION
# --------------------------------------------------

elif choice == "2":

    job_description = input("""
Paste the job description:

""")

    response = job_analysis_agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": f"""
                Analyze the following job description:

                {job_description}
                """
            }
        ]
    })

    print("\nJob Analysis:")
    print(response["messages"][-1].content)


# --------------------------------------------------
# 3. MATCH SKILLS
# --------------------------------------------------

elif choice == "3":

    student_skills = input(
        "\nEnter your skills (comma-separated): "
    )

    response = skill_matching_agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": f"""
                Compare my skills with the skills required
                for the {role} role at {company}.

                Required skills:
                {required_skills}

                My skills:
                {student_skills}
                """
            }
        ]
    })

    print("\nSkill Matching Result:")
    print(response["messages"][-1].content)


# --------------------------------------------------
# 4. GENERATE INTERVIEW QUESTIONS
# --------------------------------------------------

elif choice == "4":

    response = interview_agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": f"""
                Generate interview questions for the following
                placement role.

                Company: {company}
                Role: {role}
                Required skills: {required_skills}

                Generate questions suitable for a student
                preparing for this placement.
                """
            }
        ]
    })

    print("\nInterview Questions:")
    print(response["messages"][-1].content)


# --------------------------------------------------
# 5. CREATE PREPARATION PLAN
# --------------------------------------------------

elif choice == "5":

    days = input(
        "\nHow many days do you have for preparation? "
    )

    response = prep_agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": f"""
                Create a preparation plan for this placement.

                Company: {company}
                Role: {role}
                Required skills: {required_skills}

                I have {days} days available for preparation.

                Create a realistic preparation plan.
                """
            }
        ]
    })

    print("\nPreparation Plan:")
    print(response["messages"][-1].content)


# --------------------------------------------------
# 6. RESUME & PLACEMENT READINESS  (new module)
# --------------------------------------------------

elif choice == "6":

    resume_text = input("\nPaste your resume text: ")
    cgpa = float(input("Enter your CGPA: "))
    matched = int(input("How many required skills do you have? "))
    total = int(input("How many skills does the role require? "))
    projects = int(input("How many relevant projects have you done? "))

    response = readiness_agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": f"""
                Evaluate my resume and placement readiness for the
                {role} role at {company}.

                Required skills: {required_skills}

                My resume:
                {resume_text}

                My CGPA: {cgpa}
                Skills I have that match: {matched}
                Total skills required: {total}
                Relevant projects: {projects}

                Give me my ATS resume score and my overall readiness score,
                then tell me what to improve.
                """
            }
        ]
    })

    print("\nResume & Readiness Result:")
    print(response["messages"][-1].content)


