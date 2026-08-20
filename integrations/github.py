"""Real GitHub Integration using PyGithub

Replaces the mock logic with authenticated GitHub API requests.
"""

import os
from github import Github, GithubException, InputGitTreeElement


def push_to_github(state: dict) -> dict:
    """
    Submits generated code and tests as a Pull Request to a real GitHub repo.
    """
    repo_url = state.get("github_repo_url", "").strip()
    branch_name = state.get("github_branch", "").strip()
    token = state.get("github_token", "").strip()

    # Fallback to mock behavior if real info isn't provided,
    # or skip completely. Since user requested optional, if no token, skip gracefully.
    if not token or not repo_url or not branch_name:
        return {"error": "Missing GitHub token, repo URL, or branch name. GitHub PR skipped."}

    # Extract user/repo from github.com/user/repo or user/repo
    repo_path = repo_url
    if "github.com/" in repo_path:
        repo_path = repo_path.split("github.com/")[-1]
    if repo_path.endswith(".git"):
        repo_path = repo_path[:-4]

    try:
        g = Github(token)
        repo = g.get_repo(repo_path)
    except GithubException as e:
        return {"error": f"Failed to access repository: {str(e)}"}

    # Gather files
    files_to_commit = {}
    files_to_commit.update(state.get("generated_code", {}))
    files_to_commit.update(state.get("test_code", {}))
    
    doc = state.get("documentation", "")
    if doc:
        files_to_commit["README.md"] = doc

    if not files_to_commit:
        return {"error": "No files to commit."}

    try:
        # Get default branch
        default_branch = repo.default_branch
        ref = repo.get_git_ref(f"heads/{default_branch}")
        base_sha = ref.object.sha

        # Check if the branch we want to create already exists
        try:
            repo.get_git_ref(f"heads/{branch_name}")
            return {"error": f"Branch '{branch_name}' already exists in this repo."}
        except GithubException:
            pass  # Expected if branch is new

        # Create new branch
        repo.create_git_ref(ref=f"refs/heads/{branch_name}", sha=base_sha)

        # Create Tree
        element_list = []
        for filename, content in files_to_commit.items():
            blob = repo.create_git_blob(content, "utf-8")
            element = InputGitTreeElement(filename, '100644', 'blob', sha=blob.sha)
            element_list.append(element)

        base_tree = repo.get_git_tree(base_sha)
        tree = repo.create_git_tree(element_list, base_tree)
        
        # Create commit
        parent_commit = repo.get_git_commit(base_sha)
        commit_message = f"Auto-generated code for: {state.get('requirement', 'Task')[:50]}..."
        commit = repo.create_git_commit(commit_message, tree, [parent_commit])

        # Point branch to commit
        branch_ref = repo.get_git_ref(f"heads/{branch_name}")
        branch_ref.edit(commit.sha)

        # Create PR
        pr_title = "[Autonomous SDLC] Code implementation"
        pr_body = state.get("technical_spec", "Auto-generated PR from SDLC pipeline.")
        pr = repo.create_pull(
            title=pr_title,
            body=pr_body,
            head=branch_name,
            base=default_branch
        )

        return {
            "pr_url": pr.html_url,
            "pr_number": pr.number,
            "branch": branch_name,
            "status": "success",
            "files_committed": list(files_to_commit.keys())
        }

    except GithubException as e:
        return {"error": f"GitHub API error: {e.data.get('message', str(e))}"}
    except Exception as e:
        return {"error": f"Unexpected error pushing to GitHub: {str(e)}"}
