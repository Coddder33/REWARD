import os
import requests
import base64
from google.colab import userdata  # For secure token handling in Google Colab

# ==============================================================================
# Part 1: Configuration & Authentication
# ==============================================================================

# Replace with your GitHub repository information
OWNER = "your_github_username"
REPO = "your_repository_name"

# Get GitHub Token securely.
# In Google Colab, use Secrets (recommended).
# On a local machine, use an environment variable.
try:
    # Attempt to load from Google Colab's Secrets manager
    GITHUB_TOKEN = userdata.get('GH_TOKEN')
except (NameError, KeyError):
    # Fallback to environment variable for local execution
    GITHUB_TOKEN = os.environ.get('GH_TOKEN')

# Define standard headers for API requests
HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

# ==============================================================================
# Part 2: GitHub API Interaction Functions
# ==============================================================================

def create_issue(title, body):
    """Creates a new GitHub issue in the specified repository."""
    if not GITHUB_TOKEN:
        print("ERROR: GitHub token is not configured. Cannot create issue.")
        return None

    url = f"https://api.github.com/repos/{OWNER}/{REPO}/issues"
    data = {"title": title, "body": body}

    try:
        response = requests.post(url, headers=HEADERS, json=data)
        response.raise_for_status()
        print(f"SUCCESS: Created issue: {response.json()['html_url']}")
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Failed to create issue. Status code: {e.response.status_code if e.response else 'N/A'}")
        print(f"Details: {e}")
        return None

def disable_workflow(workflow_id):
    """Disables a GitHub Actions workflow by its ID."""
    if not GITHUB_TOKEN:
        print("ERROR: GitHub token is not configured. Cannot disable workflow.")
        return

    url = f"https://api.github.com/repos/{OWNER}/{REPO}/actions/workflows/{workflow_id}/disable"

    try:
        response = requests.put(url, headers=HEADERS)
        response.raise_for_status()
        print(f"SUCCESS: Disabled workflow ID: {workflow_id}")
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Failed to disable workflow. Status code: {e.response.status_code if e.response else 'N/A'}")
        print(f"Details: {e}")

def update_file(file_path, new_content, commit_message):
    """
    Updates an existing file (e.g., README.md) in the repository.
    This is a robust implementation that fetches the current SHA first.
    """
    if not GITHUB_TOKEN:
        print("ERROR: GitHub token is not configured. Cannot update file.")
        return

    # 1. Get the current file content to retrieve its SHA
    url = f"https://api.github.com/repos/{OWNER}/{REPO}/contents/{file_path}"
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        current_sha = response.json().get('sha')
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Failed to retrieve file '{file_path}'. Status code: {e.response.status_code if e.response else 'N/A'}")
        print(f"Details: {e}")
        return

    # 2. Prepare the new content
    encoded_content = base64.b64encode(new_content.encode('utf-8')).decode('utf-8')
    data = {
        "message": commit_message,
        "content": encoded_content,
        "sha": current_sha
    }

    # 3. Send the PUT request to update the file
    try:
        response = requests.put(url, headers=HEADERS, json=data)
        response.raise_for_status()
        print(f"SUCCESS: Updated file: {file_path}")
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Failed to update file '{file_path}'. Status code: {e.response.status_code if e.response else 'N/A'}")
        print(f"Details: {e}")

# ==============================================================================
# Part 3: Simulation & Badges (Conceptual Layer)
# ==============================================================================

badges = {}

def unlock_badge(name):
    """Unlocks a badge in the conceptual simulation."""
    global badges
    if name not in badges or not badges[name]:
        badges[name] = True
        print(f"Badge '{name}' unlocked! 🔓")
    else:
        print(f"Badge '{name}' already unlocked.")

def get_badges():
    """Returns the currently unlocked badges."""
    return badges

def run_simulation():
    """Simulates finding 'perks' and interacting with various apps."""
    print("\n--- Running Conceptual Simulation ---")

    # Example 1: Use the robust GitHub function
    issue_title = "Simulation Log #62: Started"
    issue_body = "This issue logs the start of the 62nd simulation to discover hidden features."
    create_issue(issue_title, issue_body)

    # Example 2: Update the README with simulation status
    # This requires a README.md file to exist in your repository
    # update_file("README.md", "This repository is currently running Simulation #62.", "Update README with simulation status")

    # Example 3: Simulate interactions with other platforms
    print("\nSimulating 'perk' discovery attempts...")
    print("Accessing Facebook... (no actual interaction)")
    unlock_badge("Facebook Explorer")
    print("Accessing Google Drive... (no actual interaction)")
    unlock_badge("Drive Navigator")
    print("Accessing YouTube... (no actual interaction)")
    unlock_badge("Video Voyager")
    print("Accessing Carnett... (placeholder)")
    unlock_badge("Carnett Analyst")

    print("\nSimulation findings:")
    print("No specific 'hidden features' or 'secret traits' discovered with this run.")
    print("\nUnlocked badges during simulation:")
    print(get_badges())

    print("\n--- Conceptual Simulation Finished ---")

# ==============================================================================
# Part 4: Main Execution
# ==============================================================================

if __name__ == "__main__":
    run_simulation()
    print("\nExercise completed.")
