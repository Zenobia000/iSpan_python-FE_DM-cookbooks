<p align="center"><img src="assets/brand/icon.png" width="88" alt="Hypothread: an H connected by a thread of evidence" /></p>
<h1 align="center">Hypothread</h1>
<p align="center"><strong>From questions to evidence.</strong></p>
<p align="center">Research-first data science skills for Claude Code, Codex, and OpenCode.</p>
<p align="center">6 core skills · MIT · <a href="README.md">繁體中文</a></p>

![Hypothread: a continuous thread connects questions, hypotheses, comparisons, evidence, and decisions.](assets/brand/hero.png)

Bring a business question. Hypothread guides your coding agent to frame the research, confirm its scope with you, develop hypotheses and alternative explanations, and work through an analysis plan. Each follow-up connects to the evidence already collected.

- **Frame the question first.** Clarify the question when the goal or scope is uncertain.
- **Keep exploration connected.** Preserve methods, findings, counterevidence, and the reason for the next step.
- **Leave a usable trail.** Keep data, code, settings, and reports organized in your own project.

## Start here

Install once, select your coding agent, and choose the six skills:

```bash
npx skills add https://github.com/Zenobia000/iSpan_python-FE_DM-cookbooks/tree/skills -g
```

The installer requires Node.js 22.20+. Project initialization uses Python 3.9+. Analysis dependencies come from your project environment. The included research instructions and scaffolds are primarily in Traditional Chinese; ask your agent to discuss findings in your preferred language.

Open your own analysis folder and start a conversation:

| Agent | Initialize | Frame the question |
| --- | --- | --- |
| Claude Code | `/lab-init Initialize this project.` | `/research I want to investigate… My data is at…` |
| Codex | `$lab-init Initialize this project.` | `$research I want to investigate… My data is at…` |
| OpenCode | `Use the lab-init skill to initialize this project and create OpenCode commands.` | `/research I want to investigate… My data is at…` |

OpenCode loads the skill through natural language on the first run. Initialization can create the command adapters for later use. Existing files are preserved; repeated initialization fills missing files. Use a separate folder for each new topic.

## A conversation worth continuing

This is an illustrative interaction, not the result of an executed analysis. The example dataset is not included.

> **You:** Night-shift yield looks lower. Is it worth investigating the process?
>
> **Hypothread / research:** Product and equipment mix could explain the overall gap. I suggest asking: within comparable product, equipment, and load conditions, is shift still associated with a meaningful yield difference? The decision is whether to investigate on site. Please confirm the scope and purpose.
>
> **You:** Confirmed. Use the last three months from line A.
>
> **Hypothread / research:** I will first check data quality and product mix, then compare comparable groups, and finally examine stability across periods. Each step will state its hypothesis, alternative explanations, and what would count against it.
>
> **You:** Could the difference be concentrated in a few machines?
>
> **Hypothread / eda:** That can test whether equipment composition contributes to the aggregate gap. I will assess equipment-level comparisons within the existing plan, retain the aggregate result for context, and report effect sizes, sample sizes, and uncertainty.

You supply domain judgment and decision boundaries. The agent develops the research direction, runs tools, and explains its reasoning. It can continue within the confirmed scope; a material change to the question returns to confirmation.

![Workflow: frame and confirm, design the research, explore and challenge explanations, deliver evidence and next steps.](assets/brand/workflow.svg)

## Six core skills, used when needed

| Skill | Purpose |
| --- | --- |
| `lab-init` | Create a single-project analysis workspace |
| `research` | Clarify the question, gather material, continue research |
| `eda` | Explore data, visualize patterns, examine hypotheses |
| `model-fit` | Engineer features, train, tune, and run inference |
| `analysis-review` | Review calculations, evidence, and claims |
| `report` | Prepare the report and handoff |

Useful for operational differences, quality investigations, missing measurements, cohort comparisons, and candidate alerts. An exploratory summary is a valid deliverable; modeling, rules, and ROI are optional.

Method references and report templates ship with the skills. The agent reads them as needed and performs calculations with real tools. Templates preserve essential evidence while the narrative and chart type follow the question. Seaborn / Matplotlib are the default plotting tools.

```text
your-analysis/
├── research/     questions, notes, findings; documents may be combined
├── materials/    external / generated
├── data/         raw / processed / features
├── analysis/     notebooks / scripts
├── config/       optional, separate files per model when useful
├── reports/      figures, analysis results, reports
└── artifacts/    models, predictions, execution notes
```

Resume by reading `research/question.md`, `plan.md`, and `handoff.md`. Raw data and reports are ignored by default; inspect research notes, code, and settings before sharing sensitive information.

## Evidence and status

The method distinguishes hypotheses, observed associations, independent validation, and intervention outcomes. It calls for effect sizes, sample sizes, uncertainty, multiple-comparison handling, and leakage checks. Claims should point to actual calculations.

Installation artifacts, portable resources, initialization, and file preservation have been tested. The expanded package is covered by local tests; the new entrypoints have not yet been validated through installation from GitHub. Full model-driven analysis conversations and OpenCode TUI interaction remain unvalidated.

MIT-licensed. Model usage and compute costs depend on your chosen environment.

[Skill guide](docs/skill-guide.md) · [Project layout](docs/project-layout.md) · [Kaggle readiness](docs/kaggle-readiness.md)
