import os
import github
from github import Github
import subprocess
import json
import tempfile
import logging
import re
from crewai import Agent, Task, Crew
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import inquirer
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Environment variables with validation
def validate_env_var(var_name, var_value):
    if not var_value:
        logger.error(f"{var_name} not found in environment variables. Please set it in .env file.")
        sys.exit(1)
    return var_value

SEC_TOKEN = validate_env_var("SEC_TOKEN", os.getenv("SEC_TOKEN"))
GOOGLE_API_KEY = validate_env_var("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
REPO_NAME = os.getenv("REPO_NAME", "udaynex/SocialMedia-Nodejs")
try:
    PR_NUMBER = int(os.getenv("PR_NUMBER", "1"))
except ValueError:
    logger.error("PR_NUMBER must be a valid integer.")
    sys.exit(1)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# Log environment variables (partial for security)
logger.info(f"Loaded SEC_TOKEN: {SEC_TOKEN[:4]}...")
logger.info(f"Loaded GOOGLE_API_KEY: {GOOGLE_API_KEY[:4]}...")
logger.info(f"Loaded REPO_NAME: {REPO_NAME}")
logger.info(f"Loaded PR_NUMBER: {PR_NUMBER}")
logger.info(f"Loaded GEMINI_MODEL: {GEMINI_MODEL}")

# Interactive prompt for local runs
if os.getenv("GITHUB_ACTIONS") != "true":
    questions = [
        inquirer.Text("repo_name", message="Enter repository name", default=REPO_NAME),
        inquirer.Text("pr_number", message="Enter PR number", default=str(PR_NUMBER), validate=lambda _, x: x.isdigit())
    ]
    answers = inquirer.prompt(questions)
    if not answers:
        logger.error("No input provided for repository or PR number. Exiting.")
        sys.exit(1)
    REPO_NAME = answers["repo_name"]
    PR_NUMBER = int(answers["pr_number"])

# Initialize GitHub client
try:
    github_client = Github(SEC_TOKEN)
    repo = github_client.get_repo(REPO_NAME)
except github.GithubException as e:
    logger.error(f"Failed to initialize GitHub client: {str(e)}")
    sys.exit(1)

# Initialize LLM (Gemini AI)
try:
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.7,
    )
except Exception as e:
    logger.error(f"Failed to initialize Gemini AI LLM: {str(e)}")
    sys.exit(1)

# Define CrewAI Agents
code_reviewer = Agent(
    role="Code Reviewer",
    goal="Analyze JavaScript/TypeScript code in a GitHub PR, identify issues using ESLint, and generate clear review comments.",
    backstory="You are an experienced JavaScript developer specializing in React and Node.js, with expertise in code quality and best practices.",
    tools=[],
    llm=llm,
    verbose=True
)

code_fixer = Agent(
    role="Code Fixer",
    goal="Suggest and apply fixes for identified JavaScript/TypeScript issues, creating a new PR with the changes.",
    backstory="You are a skilled developer who automates fixes for common JavaScript issues in React and Node.js projects.",
    tools=[],
    llm=llm,
    verbose=True
)

def run_linter(file_content, file_path, selected_rules=None):
    """Run ESLint on the provided file content and return issues."""
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name
        cmd = ['npx', 'eslint', temp_file_path, '--format', 'json']
        if selected_rules:
            cmd.extend(['--rule', ' '.join([f"{rule}:error" for rule in selected_rules])])
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        os.unlink(temp_file_path)
        if not result.stdout:
            return []
        issues = json.loads(result.stdout)
        return [f"{issue['filePath']}:{issue['line']}:{issue['column']}: {issue['message']} ({issue['ruleId']})"
                for issue in issues[0].get('messages', [])]
    except subprocess.CalledProcessError as e:
        logger.error(f"ESLint failed on {file_path}: {e.stderr}")
        return []
    except json.JSONDecodeError:
        logger.error(f"Failed to parse ESLint output for {file_path}")
        return []
    except Exception as e:
        logger.error(f"Error running ESLint on {file_path}: {str(e)}")
        return []

def get_pr_files(pr_number):
    """Fetch JavaScript files from a GitHub PR."""
    try:
        pr = repo.get_pull(pr_number)
        files = pr.get_files()
        js_files = [f for f in files if f.filename.endswith(('.js', '.jsx', '.ts', '.tsx'))]
        if not js_files:
            logger.info("No JavaScript/React files found in PR.")
            return []
        if os.getenv("GITHUB_ACTIONS") != "true":
            questions = [
                inquirer.Checkbox(
                    "selected_files",
                    message="Select files to review",
                    choices=[f.filename for f in js_files]
                )
            ]
            answers = inquirer.prompt(questions)
            if not answers or not answers["selected_files"]:
                logger.info("No files selected for review. Exiting.")
                return []
            return [f for f in js_files if f.filename in answers["selected_files"]]
        return js_files
    except github.GithubException as e:
        logger.error(f"Error fetching PR #{pr_number}: {str(e)}")
        return []

def review_task(file_content, file_path):
    """Generate review comments for a file using ESLint and LLM."""
    try:
        selected_rules = ["no-unused-vars", "semi"]  # Default rules
        if os.getenv("GITHUB_ACTIONS") != "true":
            questions = [
                inquirer.Checkbox(
                    "rules",
                    message=f"Select ESLint rules for {file_path}",
                    choices=["no-unused-vars", "semi", "react/prop-types"],
                    default=selected_rules
                )
            ]
            answers = inquirer.prompt(questions)
            selected_rules = answers["rules"] if answers and answers["rules"] else selected_rules
        issues = run_linter(file_content, file_path, selected_rules)
        if not issues:
            return f"No issues found in {file_path}"
        prompt = (
            "You are a code review assistant for a React and Node.js project. Below is a list of issues found by ESLint:\n\n"
            f"{'\n'.join(issues)}\n\n"
            f"Code:\n```javascript\n{file_content}\n```\n\n"
            "For each issue, provide a clear, concise, and actionable comment explaining the problem and suggesting a fix. "
            "Return comments in a list format."
        )
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        logger.error(f"Error generating review comments for {file_path}: {str(e)}")
        return f"Failed to generate comments for {file_path}"

def fix_task(file_content, issues, file_path):
    """Suggest and apply fixes for identified issues."""
    try:
        lines = file_content.splitlines()
        fixed_lines = lines.copy()
        comments = []
        approved_issues = issues
        if os.getenv("GITHUB_ACTIONS") != "true":
            questions = [
                inquirer.Checkbox(
                    "approved_issues",
                    message=f"Select issues to fix in {file_path}",
                    choices=issues
                )
            ]
            answers = inquirer.prompt(questions)
            approved_issues = answers["approved_issues"] if answers and answers["approved_issues"] else []
        for issue in approved_issues:
            match = re.match(r'.*:(\d+):(\d+): (.*) \((.*)\)', issue)
            if match:
                line_num = int(match.group(1)) - 1
                issue_desc = match.group(3)
                rule_id = match.group(4)
                if rule_id == 'no-unused-vars' and 'is defined but never used' in issue_desc:
                    var_name = issue_desc.split("'")[1]
                    fixed_lines[line_num] = f"// Removed unused variable: {lines[line_num]}"
                    comments.append(f"Line {line_num + 1}: Removed unused variable '{var_name}'")
                elif rule_id == 'semi' and 'Missing semicolon' in issue_desc:
                    fixed_lines[line_num] = lines[line_num].rstrip() + ';'
                    comments.append(f"Line {line_num + 1}: Added missing semicolon")
        return {'fixed_content': '\n'.join(fixed_lines), 'comments': comments}
    except Exception as e:
        logger.error(f"Error suggesting fixes for {file_path}: {str(e)}")
        return {'fixed_content': file_content, 'comments': []}

def create_fix_pr(pr_number, file_path, fixed_content, comments):
    """Create a PR with suggested fixes."""
    if os.getenv("GITHUB_ACTIONS") != "true":
        questions = [inquirer.Confirm("create_pr", message=f"Create fix PR for {file_path}?", default=True)]
        answers = inquirer.prompt(questions)
        if not answers or not answers["create_pr"]:
            logger.info(f"Skipped creating fix PR for {file_path}")
            return None
    try:
        pr = repo.get_pull(pr_number)
        branch_name = f"fix-pr-{pr_number}-{file_path.replace('/', '-')}"
        repo.create_git_ref(
            ref=f"refs/heads/{branch_name}",
            sha=pr.base.sha  # Use PR's base branch SHA
        )
        try:
            file_sha = repo.get_contents(file_path, ref=pr.base.ref).sha
        except github.GithubException:
            file_sha = None  # File might be new
        repo.update_file(
            path=file_path,
            message=f"Automated fixes for PR #{pr_number}: {file_path}",
            content=fixed_content,
            sha=file_sha,
            branch=branch_name
        )
        fix_pr = repo.create_pull(
            title=f"Automated Fixes for PR #{pr_number}: {file_path}",
            body=f"Fixes for {file_path}:\n" + "\n".join(comments) + f"\n\nRelated to PR #{pr_number}",
            head=branch_name,
            base=pr.base.ref
        )
        logger.info(f"Created fix PR #{fix_pr.number}")
        return fix_pr.number
    except github.GithubException as e:
        logger.error(f"Error creating fix PR for {file_path}: {str(e)}")
        return None

def main():
    """Main function to orchestrate the code review process."""
    try:
        pr_files = get_pr_files(PR_NUMBER)
        if not pr_files:
            logger.info("No files to process. Exiting.")
            return
        for file in pr_files:
            logger.info(f"Processing {file.filename}")
            try:
                file_content = repo.get_contents(file.filename, ref=repo.get_pull(PR_NUMBER).head.sha).decoded_content.decode()
            except github.GithubException as e:
                logger.error(f"Error fetching content for {file.filename}: {str(e)}")
                continue
            # Define review task
            review = Task(
                description=f"Review the JavaScript file {file.filename} and generate comments for issues.",
                agent=code_reviewer,
                expected_output="List of review comments",
                context=[]  # Context handled via function args
            )
            # Execute review task
            review_comments = review_task(file_content, file.filename)
            try:
                repo.get_pull(PR_NUMBER).create_issue_comment(f"Review for {file.filename}:\n{review_comments}")
            except github.GithubException as e:
                logger.error(f"Error posting comment for {file.filename}: {str(e)}")
            # Run linter to get issues for fix task
            issues = run_linter(file_content, file.filename)
            if not issues:
                continue
            # Define fix task
            fix = Task(
                description=f"Suggest and apply fixes for issues in {file.filename}.",
                agent=code_fixer,
                expected_output="Dictionary with 'fixed_content' and 'comments'",
                context=[]  # Context handled via function args
            )
            # Execute fix task
            fix_result = fix_task(file_content, issues, file.filename)
            fixed_content = fix_result.get('fixed_content', file_content)
            fix_comments = fix_result.get('comments', [])
            if fix_comments:
                fix_pr_number = create_fix_pr(PR_NUMBER, file.filename, fixed_content, fix_comments)
                if fix_pr_number:
                    try:
                        repo.get_pull(PR_NUMBER).create_issue_comment(
                            f"Created fix PR #{fix_pr_number} with automated fixes for {file.filename}."
                        )
                    except github.GithubException as e:
                        logger.error(f"Error posting fix PR comment for {file.filename}: {str(e)}")
    except github.GithubException as e:
        logger.error(f"Error in main process: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()