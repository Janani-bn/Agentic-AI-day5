# 🤖 Agentic AI - Day 5

Hands-on exploration of **LangChain**, **Google Gemini**, and **Agentic AI** concepts — building intelligent chatbots and PDF-reading agents.

---

## 📋 Overview

This repository contains a Jupyter Notebook (`Agentic_AI_day5.ipynb`) covering the following topics:

| # | Topic | Description |
|---|-------|-------------|
| 1 | **LLM Setup** | Connecting to Google Gemini (`gemini-3.5-flash-lite`) using LangChain |
| 2 | **Model Parameters** | Configuring `temperature`, `timeout`, `max_tokens`, and `max_retries` |
| 3 | **Basic Q&A** | Invoking the LLM with user input and displaying responses |
| 4 | **Restaurant Chatbot** | A Gradio-based chatbot that answers questions from a restaurant PDF |
| 5 | **PDF Reader Agent** | An agentic AI tool that reads and answers questions from any PDF |

---

## 🛠️ Tech Stack

- **Python 3.13**
- **LangChain** — LLM orchestration framework
- **LangChain Google GenAI** — Google Gemini integration
- **Gradio** — Interactive chatbot UI
- **PyPDF** — PDF text extraction
- **Google Gemini 3.5 Flash Lite** — Large Language Model

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install langchain langchain-community langchain-google-genai gradio pypdf
```

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Janani-bn/Agentic-AI-day5.git
   cd Agentic-AI-day5
   ```

2. Set your Google API key:
   ```python
   import os
   os.environ["GOOGLE_API_KEY"] = "your-api-key-here"
   ```

3. Open and run the notebook:
   ```bash
   jupyter notebook Agentic_AI_day5.ipynb
   ```

---

## 📂 Project Structure

```
Agentic-AI-day5/
├── Agentic_AI_day5.ipynb   # Main notebook with all experiments
└── README.md               # Project documentation
```

---

## 📝 Key Concepts Covered

### 🔗 LangChain + Gemini Integration
Setting up and configuring a Google Gemini LLM through LangChain with various model parameters.

### 🍽️ Restaurant Support Chatbot
A Gradio-powered chatbot that extracts information from a restaurant PDF and answers customer queries using prompt engineering.

### 📄 PDF Reader Agent
An agentic AI system using LangChain's `@tool` decorator and `create_agent` to build an autonomous agent that reads PDFs and answers questions based on their content.

---

## 📌 Notes

- This notebook was originally run on **Google Colab**.
- Replace `"MY_API_KEY"` with your actual Google API key before running.
- The restaurant chatbot requires a `Resturaunt Q&A.pdf` file to be uploaded.

---

## 👩‍💻 Author

**Janani B N**

---

*Part of the Agentic AI learning series — Day 5*
