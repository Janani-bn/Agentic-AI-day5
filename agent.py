"""
Student Info Agent using LangChain + Gemini
=============================================
This script:
1. Creates a SQLite database `students.db` with sample student data.
2. Defines 4 LangChain tools (@tool decorated functions).
3. Builds a LangChain agent (Gemini) that decides which tool(s) to call.
4. Runs the example questions from the problem statement.

Requirements:
    pip install langchain langchain-google-genai langchain-community python-dotenv

Set your Gemini API key as an environment variable before running:
    export GOOGLE_API_KEY="your_api_key_here"
"""

import os
import sqlite3
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

DB_NAME = "students.db"


# ---------------------------------------------------------------------------
# 1. DATABASE SETUP
# ---------------------------------------------------------------------------
def setup_database():
    """Creates students.db with the students table and sample data."""
    if os.path.exists(DB_NAME):
        os.remove(DB_NAME)  # fresh DB every run (comment this out if you want persistence)

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE students (
            student_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            department TEXT NOT NULL,
            python INTEGER,
            database INTEGER,
            ai INTEGER,
            web INTEGER
        )
    """)

    students_data = [
        ("22CS045", "Dhanushya", "Computer Science", 85, 72, 90, 78),
        ("22CS046", "Rahul", "Computer Science", 65, 70, 68, 72),
        ("22CS047", "Priya", "Information Technology", 92, 88, 95, 90),
        ("22CS048", "Arun", "Information Technology", 55, 60, 58, 62),
        ("22CS049", "Meena", "Computer Science", 78, 85, 80, 88),
    ]

    cursor.executemany("""
        INSERT INTO students (student_id, name, department, python, database, ai, web)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, students_data)

    conn.commit()
    conn.close()
    print(f"✅ Database '{DB_NAME}' created with {len(students_data)} students.\n")


# ---------------------------------------------------------------------------
# 2. TOOLS
# ---------------------------------------------------------------------------
@tool
def get_student_info(student_id: str) -> str:
    """
    Get a student's name and department using their student_id.
    Input: student_id (e.g. '22CS045')
    Output: A string containing the student's name and department.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name, department FROM students WHERE student_id = ?",
        (student_id,)
    )
    row = cursor.fetchone()
    conn.close()

    if row is None:
        return f"No student found with ID {student_id}."

    name, department = row
    return f"Student ID: {student_id}, Name: {name}, Department: {department}"


@tool
def get_student_marks(student_id: str) -> str:
    """
    Get a student's marks (python, database, ai, web) using their student_id.
    Input: student_id (e.g. '22CS045')
    Output: A string listing the marks for each subject.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT python, database, ai, web FROM students WHERE student_id = ?",
        (student_id,)
    )
    row = cursor.fetchone()
    conn.close()

    if row is None:
        return f"No student found with ID {student_id}."

    python_m, database_m, ai_m, web_m = row
    return (
        f"Marks for {student_id} -> "
        f"Python: {python_m}, Database: {database_m}, AI: {ai_m}, Web: {web_m}"
    )


@tool
def calculator(expression: str) -> str:
    """
    Evaluate a basic math expression. Use this to compute totals, sums,
    averages, percentages, or comparisons.
    Input: a valid Python math expression as a string, e.g. '(85+72+90+78)' or '(85+72+90+78)/4'
    Output: The numeric result of the expression as a string.
    """
    try:
        # restricted eval - only arithmetic allowed
        allowed_chars = "0123456789+-*/(). "
        if not all(c in allowed_chars for c in expression):
            return "Error: expression contains invalid characters."
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result of '{expression}' = {result}"
    except Exception as e:
        return f"Error evaluating expression: {e}"


@tool
def get_passing_rules() -> str:
    """
    Get the university's passing rules.
    Output: A string describing the minimum overall average and
    minimum mark required in each subject to pass.
    """
    return (
        "University Passing Rules:\n"
        "1. Minimum overall average required: 40%\n"
        "2. Minimum mark required in each individual subject: 35%"
    )


TOOLS = [get_student_info, get_student_marks, calculator, get_passing_rules]


# ---------------------------------------------------------------------------
# 3. AGENT SETUP
# ---------------------------------------------------------------------------
def build_agent():
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",   # or "gemini-1.5-flash"
        temperature=0,
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful student records assistant.
You have access to tools that let you look up student info, student marks,
perform calculations, and fetch university passing rules.

Rules for using tools:
- Only call the tools you actually need to answer the question.
- If the question needs marks, call get_student_marks.
- If it needs totals/averages, use calculator on the marks you retrieved.
- If it needs to check pass/fail, first get the marks, then get_passing_rules,
  then use calculator/reasoning to compare and decide.
- Always base your final answer only on tool outputs; do not guess numbers.
- Give a clear, well-formatted final answer to the user.
"""),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, TOOLS, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=TOOLS,
        verbose=True,          # shows which tools are being called, in order
        handle_parsing_errors=True,
    )
    return agent_executor


# ---------------------------------------------------------------------------
# 4. RUN EXAMPLE QUESTIONS
# ---------------------------------------------------------------------------
def main():
    setup_database()
    agent_executor = build_agent()

    questions = [
        "What is the name and department of student 22CS045?",
        "What are the marks of 22CS047?",
        "What is the total and average mark of 22CS045?",
        "Is 22CS045 eligible to pass according to the university rules?",
        ("I am 22CS045. Tell me my name, department, total marks, average marks, "
         "and whether I satisfy the university passing requirements."),
    ]

    for i, q in enumerate(questions, start=1):
        print(f"\n{'='*80}\nQ{i}: {q}\n{'='*80}")
        response = agent_executor.invoke({"input": q})
        print(f"\n📌 FINAL ANSWER:\n{response['output']}\n")


if __name__ == "__main__":
    main()
