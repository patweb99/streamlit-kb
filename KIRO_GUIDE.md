# 🤖 Kiro AI Development Guide

<div align="center">

| Kiro IDE |
|:--------:|
| ![Kiro QR Code](kiro-qr-code.png) |
| Scan to learn about Kiro |

</div>

---

This guide walks you through using Kiro's powerful features for AI-assisted development, including Specs, Vibecoding, MCP (Model Context Protocol), and Hooks.

## 📋 Table of Contents

- [What is Kiro?](#what-is-kiro)
- [Specs: Structured Feature Development](#specs-structured-feature-development)
- [Vibecoding: Natural Collaboration](#vibecoding-natural-collaboration)
- [MCP: Model Context Protocol](#mcp-model-context-protocol)
- [Hooks: Automated Workflows](#hooks-automated-workflows)
- [Asana MCP Integration Setup](#asana-mcp-integration-setup)

---

## What is Kiro?

Kiro is an AI-powered IDE assistant that helps you build software faster through natural language interaction, structured planning, and automated workflows. Think of it as pair programming with an AI that understands your entire codebase.

---

## Specs: Structured Feature Development

**Specs** are Kiro's way of breaking down complex features into manageable, trackable tasks. They formalize the design and implementation process.

### When to Use Specs

- Building complex features that span multiple files
- Need to iterate on requirements before implementation
- Want to track progress on multi-step implementations
- Working on features that require design decisions

### How Specs Work

1. **Create a Spec**: Define what you want to build
2. **Refine Requirements**: Iterate with Kiro on the design
3. **Break Down Tasks**: Kiro helps create implementation tasks
4. **Execute**: Work through tasks with Kiro's assistance
5. **Track Progress**: See what's done and what's remaining

### Example Spec Workflow

```
You: "Create a spec for adding user authentication to the app"

Kiro: Creates a spec document with:
- Requirements gathering
- Design considerations
- Implementation tasks
- Testing requirements

You: "Let's refine the password reset flow"

Kiro: Updates the spec with detailed password reset requirements

You: "Start implementing task 1"

Kiro: Begins implementation following the spec
```

### Spec File References

Specs support file references using `#[[file:path/to/file.ext]]` syntax:
- Include OpenAPI specs: `#[[file:api-spec.yaml]]`
- Reference GraphQL schemas: `#[[file:schema.graphql]]`
- Link to design docs: `#[[file:design.md]]`

This allows specs to pull in external documentation dynamically.

---

## Vibecoding: Natural Collaboration

**Vibecoding** is Kiro's natural, conversational development style. Instead of rigid commands, you describe what you want in plain language.

### Vibecoding Philosophy

- **Describe the outcome**, not the steps
- **Iterate naturally** like talking to a teammate
- **Trust Kiro** to figure out implementation details
- **Provide context** through conversation

### Vibecoding Examples

❌ **Traditional approach:**
```
1. Create a new file called UserService.ts
2. Add an interface for User
3. Implement getUserById method
4. Add error handling
5. Write unit tests
```

✅ **Vibecoding approach:**
```
"I need a user service that can fetch users by ID with proper error handling"
```

Kiro will:
- Create the necessary files
- Implement the service
- Add error handling
- Follow your project's patterns

### Tips for Effective Vibecoding

1. **Be specific about outcomes**: "Add tab navigation to the Streamlit app"
2. **Provide context**: "Following the existing code style in app.py"
3. **Iterate freely**: "Actually, let's use a config file instead"
4. **Ask questions**: "What's the best way to handle this?"

---

## MCP: Model Context Protocol

**MCP (Model Context Protocol)** extends Kiro's capabilities by connecting to external tools and services. Think of it as plugins for your AI assistant.

### What MCP Enables

- **External APIs**: Connect to services like Asana, GitHub, Slack
- **Custom Tools**: Build your own integrations
- **Data Sources**: Access databases, documentation, APIs
- **Automation**: Trigger actions in external systems

### MCP Configuration

MCP servers are configured in `.kiro/settings/mcp.json`:

```json
{
  "mcpServers": {
    "server-name": {
      "command": "npx",
      "args": ["mcp-package-name"],
      "disabled": false,
      "autoApprove": ["tool1", "tool2"]
    }
  }
}
```

### Configuration Levels

1. **User-level**: `~/.kiro/settings/mcp.json` (global)
2. **Workspace-level**: `.kiro/settings/mcp.json` (project-specific)

Workspace settings take precedence over user settings.

### Finding MCP Servers

Popular MCP servers:
- **Asana**: Task management integration
- **GitHub**: Repository operations
- **AWS**: Cloud service management (documentation, knowledge bases)
- **Filesystem**: Advanced file operations
- **Database**: SQL query execution
- **Slack**: Team communication
- **Puppeteer**: Browser automation

**Where to find MCP servers:**

1. **Official MCP Registry**: [github.com/modelcontextprotocol](https://github.com/modelcontextprotocol)
   - Community-maintained servers
   - Official protocol documentation
   - Server implementation examples

2. **AWS Labs MCP Collection**: [github.com/awslabs/mcp](https://github.com/awslabs/mcp/tree/main)
   - AWS-specific integrations
   - Bedrock AgentCore server
   - AWS documentation server
   - AWS knowledge base tools
   - Production-ready implementations

3. **NPM Registry**: Search for packages starting with `@modelcontextprotocol/` or `mcp-`

4. **GitHub**: Search for "mcp server" or "model context protocol"

### MCP Best Practices

1. **Auto-approve safe operations**: Add read-only tools to `autoApprove`
2. **Keep credentials secure**: Use environment variables
3. **Test incrementally**: Enable one server at a time
4. **Monitor usage**: Check MCP Server view in Kiro

---

## Hooks: Automated Workflows

**Hooks** are automated agent executions triggered by events or manual actions. They enable continuous automation in your development workflow.

### Hook Types

1. **Event-based**: Triggered by IDE events (file save, git commit)
2. **Manual**: Triggered by clicking a button
3. **Scheduled**: Run at specific times (future feature)

### Common Hook Use Cases

#### On File Save
```
When: User saves a test file
Action: Run tests and update coverage report
```

#### On Translation Update
```
When: User updates en.json translation file
Action: Update all other language files
```

#### Manual Spell Check
```
When: User clicks "Spell Check" button
Action: Review and fix grammar in README
```

#### On Code Change
```
When: User saves a Python file
Action: Run linter and auto-fix issues
```

### Creating Hooks

**Method 1: Command Palette**
1. Open Command Palette (Cmd/Ctrl + Shift + P)
2. Search for "Open Kiro Hook UI"
3. Configure your hook

**Method 2: Explorer View**
1. Open Kiro Explorer panel
2. Navigate to "Agent Hooks" section
3. Click "+" to create new hook

### Hook Configuration

Hooks are stored in `.kiro/hooks/` directory:

```yaml
name: "Run Tests on Save"
trigger: "onSave"
filePattern: "**/*.test.ts"
action: |
  Run the test file that was just saved and report results
```

---

## Asana MCP Integration Setup

This project includes pre-configured Asana integration for task-driven development. Here's how to set it up:

> **Note**: This is just one example of MCP integration. You can add many other MCP servers from the [MCP Registry](https://github.com/modelcontextprotocol) or [AWS Labs MCP Collection](https://github.com/awslabs/mcp/tree/main) to extend Kiro's capabilities with GitHub, Slack, AWS services, and more.

### Step 1: Install Prerequisites

The Asana MCP server requires Node.js and npx:

```bash
# Check if you have Node.js
node --version

# If not installed, install Node.js from https://nodejs.org
```

### Step 2: Configure Asana MCP Server

The MCP configuration is already set up in `.kiro/settings/mcp.json`:

```json
{
  "mcpServers": {
    "asana": {
      "command": "npx",
      "args": [
        "mcp-remote",
        "https://mcp.asana.com/sse"
      ],
      "disabled": false,
      "autoApprove": [
        "asana_get_projects",
        "asana_create_task",
        "asana_search_tasks",
        "asana_get_tasks",
        "asana_update_task",
        "asana_get_task",
        "asana_get_project_sections"
      ]
    }
  }
}
```

### Step 3: Authenticate with Asana

When you first use Asana tools, Kiro will prompt you to authenticate:

1. Click the authentication link
2. Log in to your Asana account
3. Grant permissions to the MCP server
4. Return to Kiro

### Step 4: Configure Your Project

Copy the example configuration:

```bash
cp .kiro/asana-config.example.json .kiro/asana-config.json
```

### Step 5: Find Your Asana IDs

Ask Kiro to help you find your IDs:

```
"List my Asana workspaces"
"Search for my project in Asana"
"Get sections for project ID 123456789"
```

### Step 6: Update Configuration

Edit `.kiro/asana-config.json` with your IDs:

```json
{
  "workspace_id": "YOUR_WORKSPACE_ID",
  "project_id": "YOUR_PROJECT_ID",
  "project_name": "Your Project Name"
}
```

**Note**: Section IDs are no longer needed in the config! Kiro automatically discovers sections by searching for common names like "Backlog", "In Progress", and "Completed" using the `asana_get_project_sections` API.

### Step 7: Test the Integration

Try these commands with Kiro:

```
"Create a task in Asana for adding dark mode"
"Show me my Asana tasks"
"Move task X to in progress"
"Mark task Y as complete"
```

### Asana Workflow

Once configured, Kiro follows this workflow:

1. **Task Creation**: New tasks go to Backlog section
2. **Starting Work**: Tasks move to In Progress when work begins
3. **Completion**: Tasks move to Completed and marked done
4. **Documentation**: Comments added with implementation details

### Task-Driven Development

Ask Kiro to work on Asana tasks:

```
"Work on the authentication task"
```

Kiro will:
1. Find the task in Asana
2. Read requirements and acceptance criteria
3. Move task to In Progress
4. Implement the feature
5. Move to Completed and mark done
6. Add completion comment

### Steering Documents

Customize Kiro's behavior with steering documents in `.kiro/steering/`:

- **Always included**: Default behavior
- **Conditional**: Included when specific files are open
- **Manual**: Included via `#` context key

Example steering document:

```markdown
---
inclusion: always
---

# Project Conventions

- Use TypeScript for all new files
- Follow Airbnb style guide
- Write tests for all new features
- Use functional components in React
```

---

## Resources

### Kiro & MCP
- **Kiro**: [kiro.dev](https://kiro.dev/)
- **MCP Specification**: [modelcontextprotocol.io](https://modelcontextprotocol.io)
- **MCP Registry**: [github.com/modelcontextprotocol](https://github.com/modelcontextprotocol)
- **AWS Labs MCP Servers**: [github.com/awslabs/mcp](https://github.com/awslabs/mcp/tree/main)

### Integrations
- **Asana MCP Server**: [developers.asana.com](https://developers.asana.com/docs/using-asanas-mcp-server)

### Community
- **Kiro Discord**: Join for support and discussions
- **MCP Community**: Contribute and discover new servers
