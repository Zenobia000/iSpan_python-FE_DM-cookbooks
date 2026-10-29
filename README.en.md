<p align="center"><img src="assets/brand/icon.png" width="88" alt="Hypothread: an H connected by a thread of evidence" /></p>
<h1 align="center">Hypothread</h1>
<p align="center"><strong>From questions to evidence.</strong></p>
<p align="center">Research-first data science skills for Claude Code, Codex, and OpenCode.</p>
<p align="center">9 skills · MIT · <a href="README.md">繁體中文</a></p>

![Hypothread: a continuous thread connects questions, hypotheses, comparisons, evidence, and decisions.](assets/brand/hero.png)

Bring a business question. Hypothread guides your coding agent to frame the research, confirm its scope with you, develop hypotheses and alternative explanations, and work through an analysis plan. Each follow-up connects to the evidence already collected.

- **Frame the question first.** Confirm the research question before the agent designs the analysis.
- **Keep exploration connected.** Preserve methods, findings, counterevidence, and the reason for the next step.
- **Leave a usable trail.** Keep data, code, settings, and reports organized in your own project.

## Start here

Install once, select your coding agent, and choose all nine skills:

```bash
npx skills add https://github.com/Zenobia000/iSpan_python-FE_DM-cookbooks/tree/skills -g
```

The installer requires Node.js 22.20+. Project initialization uses Python 3.9+. Analysis dependencies come from your project environment. The included research instructions and scaffolds are primarily in Traditional Chinese; ask your agent to discuss findings in your preferred language.

Open your own analysis folder and start a conversation:

| Agent | Initialize | Frame the question |
| --- | --- | --- |
| Claude Code | `/ds-init Initialize this project.` | `/ds-frame I want to investigate… My data is at…` |
| Codex | `$ds-init Initialize this project.` | `$ds-frame I want to investigate… My data is at…` |
| OpenCode | `Use the ds-init skill to initialize this project and create OpenCode commands.` | `/ds-frame I want to investigate… My data is at…` |

OpenCode loads the skill through natural language on the first run. Initialization can create the `/ds-*` command adapters for later use. Existing files are preserved; repeated initialization fills missing files. Use a separate folder for each new topic.

## A conversation worth continuing

This is an illustrative interaction, not the result of an executed analysis. The example dataset is not included.

> **You:** Night-shift yield looks lower. Is it worth investigating the process?
>
> **Hypothread / ds-frame:** Product and equipment mix could explain the overall gap. I suggest asking: within comparable product, equipment, and load conditions, is shift still associated with a meaningful yield difference? The decision is whether to investigate on site. Please confirm the scope and purpose.
>
> **You:** Confirmed. Use the last three months from line A.
>
> **Hypothread / ds-frame:** I will first check data quality and product mix, then compare comparable groups, and finally examine stability across periods. Each step will state its hypothesis, alternative explanations, and what would count against it.
>
> **You:** Could the difference be concentrated in a few machines?
>
> **Hypothread / ds-eda:** That can test whether equipment composition contributes to the aggregate gap. I will assess equipment-level comparisons within the existing plan, retain the aggregate result for context, and report effect sizes, sample sizes, and uncertainty.

You supply domain judgment and decision boundaries. The agent develops the research direction, runs tools, and explains its reasoning. It can continue within the confirmed scope; a material change to the question returns to confirmation.

![Workflow: frame and confirm, design the research, explore and challenge explanations, deliver evidence and next steps.](assets/brand/workflow.svg)

## Nine skills, used when needed

| Skill | Purpose |
| --- | --- |
| `ds-init` | Create a single-project analysis workspace |
| `decision-analysis` | Recommend the next role from existing evidence |
| `ds-frame` | Frame the question, obtain confirmation, design the research |
| `ds-proxy` | Propose testable proxy features |
| `ds-eda` | Explore data and examine hypotheses |
| `ds-rules` | Extract interpretable candidate rules |
| `ds-impact` | Compare cost and benefit scenarios |
| `ds-review` | Review calculations, evidence, and claims |
| `ds-deliver` | Prepare the report and handoff |

Useful for operational differences, quality investigations, missing measurements, cohort comparisons, and candidate alerts. An exploratory summary is a valid deliverable; modeling, rules, and ROI are optional.

Method references and report templates ship with the skills. The agent reads them as needed and performs calculations with real tools. Templates preserve essential evidence while the narrative and chart type follow the question. Seaborn / Matplotlib are the default plotting tools.

```text
your-analysis/
├── research/     question, confirmation, plan, hypotheses, findings, handoff
├── data/         raw / interim / processed
├── analysis/     Python, SQL, notebooks
├── config/       periods, groups, splits, cost assumptions
└── reports/      exploration rounds, figures, decision, review
```

Resume by reading `research/question.md`, `plan.md`, and `handoff.md`. Raw data and reports are ignored by default; inspect research notes, code, and settings before sharing sensitive information.

## Evidence and status

The method distinguishes hypotheses, observed associations, independent validation, and intervention outcomes. It calls for effect sizes, sample sizes, uncertainty, multiple-comparison handling, and leakage checks. Claims should point to actual calculations.

Installation artifacts, portable resources, initialization, and file preservation have been tested. The suite has 13 passing tests, and installation from GitHub has been exercised. Full model-driven analysis conversations and OpenCode TUI interaction remain unvalidated. See the [validation record](https://github.com/Zenobia000/iSpan_python-FE_DM-cookbooks/blob/feat/agentic-decision-analysis/docs/release-validation.md).

MIT-licensed. Model usage and compute costs depend on your chosen environment.

[Detailed installation](https://github.com/Zenobia000/iSpan_python-FE_DM-cookbooks/blob/feat/agentic-decision-analysis/docs/distribution-readme.md) · [Research workflow](https://github.com/Zenobia000/iSpan_python-FE_DM-cookbooks/blob/feat/agentic-decision-analysis/docs/research-workflow.md) · [Feedback](https://github.com/Zenobia000/iSpan_python-FE_DM-cookbooks/issues)

Share a question you want to investigate and where the analysis gets stuck. A star helps others discover Hypothread; concrete usage feedback helps improve it.
