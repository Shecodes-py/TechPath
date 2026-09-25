"""Human-readable HTML report for Key-Value Store (REPORT)."""

from __future__ import annotations

import html

from src.schemas import TechPathOutput


def _esc(value: object) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def build_html_report(output: TechPathOutput) -> str:
    profile = output.profile
    gaps_html = "".join(
        f"<li><strong>{_esc(g.skill)}</strong> ({_esc(g.category)}) — {_esc(g.explanation)}</li>"
        for g in output.skillGaps
    )

    stages_html = []
    for stage in output.roadmap:
        resources_html = "".join(
            f'<li><a href="{_esc(r.url)}" target="_blank" rel="noopener noreferrer">'
            f"{_esc(r.title)}</a> · {_esc(r.resourceType)}</li>"
            for r in stage.resources
        )
        quiz_html = "".join(
            f"<li>{_esc(q.question)}</li>" for q in stage.assessments
        )
        projects_html = "".join(
            f"<li><strong>{_esc(p.title)}</strong>: {_esc(p.problemStatement)}</li>"
            for p in stage.projects
        )
        stages_html.append(
            f"""
            <section class="stage">
              <h2>{_esc(stage.stageName)}</h2>
              <p>{_esc(stage.description)}</p>
              <p><b>Focus:</b> {_esc(", ".join(stage.focusTopics))}</p>
              <p><b>Resources</b></p>
              <ul>{resources_html or "<li>None listed</li>"}</ul>
              <p><b>Skill check</b></p>
              <ul>{quiz_html or "<li>None listed</li>"}</ul>
              <p><b>Projects</b></p>
              <ul>{projects_html or "<li>None listed</li>"}</ul>
            </section>
            """
        )

    cap = output.capstone
    opps_html = "".join(
        f'<li><a href="{_esc(o.applicationUrl)}" target="_blank" rel="noopener noreferrer">'
        f"{_esc(o.title)}</a> — {_esc(o.opportunityType)} (deadline {_esc(o.deadline)})</li>"
        for o in output.opportunities
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Your Learning Roadmap: {_esc(output.targetRole)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 760px; margin: 40px auto; padding: 0 20px; color: #222; line-height: 1.5; }}
    .stage {{ border-left: 4px solid #4A90D9; padding-left: 16px; margin-bottom: 30px; }}
    h1, h2 {{ color: #1a1a1a; }}
    a {{ color: #4A90D9; }}
    .meta {{ color: #555; font-size: 0.95rem; }}
  </style>
</head>
<body>
  <h1>Your Path to: {_esc(output.targetRole)}</h1>
  <p>{_esc(output.summaryPitch)}</p>
  <p class="meta">
    {_esc(profile.currentLevel)} · {_esc(profile.location)} ·
    {_esc(profile.hoursPerWeek)} hrs/week · generated {_esc(output.generatedAt)}
  </p>
  <h2>Skill gaps</h2>
  <ul>{gaps_html or "<li>None identified</li>"}</ul>
  {"".join(stages_html)}
  <h2>Capstone</h2>
  <p><strong>{_esc(cap.title)}</strong></p>
  <p>{_esc(cap.problemStatement)}</p>
  <h2>Opportunities</h2>
  <ul>{opps_html or "<li>None listed</li>"}</ul>
</body>
</html>"""
