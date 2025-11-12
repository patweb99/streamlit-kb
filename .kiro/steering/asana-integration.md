---
inclusion: always
---

# Asana Integration

## Configuration

Asana project settings are stored in `.kiro/asana-config.json`. This file contains:
- Workspace ID
- Project ID and name

**Setup Instructions:**
1. Copy `.kiro/asana-config.example.json` to `.kiro/asana-config.json`
2. Update with your Asana workspace and project IDs

**To find your IDs:**
- Use `asana_list_workspaces` to get your workspace ID
- Use `asana_typeahead_search` with resource_type="project" to find your project

When the user requests task management or status updates:
1. Read the configuration from `.kiro/asana-config.json` to get workspace and project IDs
2. Use `asana_get_project_sections` to fetch current sections for the project
3. Find sections by name (e.g., "Backlog", "In Progress", "Completed")
4. Use the section IDs from the API response for task operations

Use the Asana tools to:
- Create tasks in the configured project
- Check task status and updates
- Search for existing tasks
- Update task information

## Kanban Task Creation Best Practices

When creating tasks for Kanban boards, follow these guidelines:

1. **User Story Format Titles**: Use the format "As an X, I want to do Y, in order to do Z"
   - Example: "As a developer, I want to implement user authentication, in order to secure the application"
   - Example: "As a user, I want to reset my password, in order to regain access to my account"

2. **Proper Sizing**: Break down large tasks into smaller, manageable subtasks that can be completed within a sprint

3. **Detailed Descriptions**: Include:
   - Acceptance criteria
   - Context and background
   - Links to related resources or documentation
   - Technical requirements or constraints

4. **Assignee and Due Dates**: Set clear ownership and realistic deadlines

5. **Dependencies**: Use task dependencies to show blocking relationships

6. **Tags and Custom Fields**: Apply relevant tags for categorization (e.g., bug, feature, technical-debt)

7. **Section Placement**: Place tasks in appropriate Kanban columns:
   - Backlog: Not yet prioritized
   - To Do: Ready to start
   - In Progress: Actively being worked on
   - Review: Awaiting review or testing
   - Done: Completed

8. **Followers**: Add relevant team members as followers for visibility

## Section Management

The project uses the following sections for task workflow:

- **Backlog**: New tasks that haven't been started
- **In Progress**: Tasks currently being worked on
- **Completed**: Finished tasks

### Section Workflow Rules

**At the start of any Asana workflow:**
1. Read `.kiro/asana-config.json` to get the project ID
2. Call `asana_get_project_sections` with the project ID
3. Search the returned sections by name to find the appropriate section ID
4. Use the found section ID for task operations

**Section names to search for:**
- "Backlog" or "backlog" - For new tasks
- "In Progress" or "in progress" or "In-Progress" - For active work
- "Completed" or "completed" or "Done" or "done" - For finished tasks

**Workflow:**
1. **When Creating a Task**: 
   - Get sections for the project
   - Find section with name matching "Backlog" (case-insensitive)
   - Place task in that section
   
2. **When Starting Work**: 
   - Get sections for the project
   - Find section with name matching "In Progress" (case-insensitive)
   - Move task to that section
   
3. **When Completing Work**: 
   - Get sections for the project
   - Find section with name matching "Completed" or "Done" (case-insensitive)
   - Move task to that section
   - Mark the task as completed (completed: true)
   - Add a comment summarizing what was accomplished

**Important**: Always fetch sections dynamically using `asana_get_project_sections` rather than using hardcoded section IDs. This makes the integration work across different projects with different section configurations.

## Task-Driven Workflow

When the user asks you to do work, follow this workflow:

1. **Check Asana First**: Search for tasks in Asana related to the user's request
2. **Review Task Details**: If a matching task exists, read its full details including:
   - Description and acceptance criteria
   - Notes and comments
   - Custom fields and requirements
3. **Move to In Progress**: Update the task to move it to the "In Progress" section
4. **Execute the Work**: Complete the work according to the task specifications
5. **Mark as Complete**: Once finished:
   - Add a comment documenting what was accomplished
   - Move the task to the "Completed" section
   - Mark the task as completed in Asana

This ensures all work is tracked and aligned with project management in Asana.
