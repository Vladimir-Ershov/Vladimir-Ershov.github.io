---
layout: post
title: "Schema-Guided Reasoning for AI Coding Agents"
date: 2025-10-30
categories: [llm-coding]
author: Vladimir Ershov
description: "Exploring structured approaches to building more capable AI coding assistants using schema-guided reasoning patterns"
tags: [AI, LLM, coding, agents, pydantic, schema]
---

# General

SGD: https://abdullin.com/schema-guided-reasoning/patterns
  Cascade ensures that LLM explicitly follows predefined reasoning steps while solving the problem. Each step - allocating thinking budget to take reasoning one step further

  ```python

  from pydantic import BaseModel  
  from typing import Literal, Annotated  
  from annotated_types import Ge, Le

  class CandidateEvaluation(BaseModel):
    brief_candidate_summary: str
    rate_skill_match:  Annotated[int, Ge(1), Le(10)]
    final_recommendation: Literal["hire", "reject", "hold"]
  ```

  Routing:
  ```python
  from pydantic import BaseModel
  from typing import Literal, Union

  class HardwareIssue(BaseModel):
      kind: Literal["hardware"]
      component: Literal["battery", "display", "keyboard"]

  class SoftwareIssue(BaseModel):
      kind: Literal["software"]
      software_name: str

  class UnknownIssue(BaseModel):
      kind: Literal["unknown"]
      category: str
      summary: str

  class SupportTriage(BaseModel):
      issue: Union[HardwareIssue, SoftwareIssue, UnknownIssue]
  ```

  Cycle
  ```python
  from pydantic import BaseModel
  from typing import List, Literal
  from annotated_types import MinLen, MaxLen

  class RiskFactor(BaseModel):
      explanation: str
      severity: Literal["low", "medium", "high"]

  class RiskAssessment(BaseModel):
      factors: Annotated[List[RiskFactor], MinLen(2), MaxLen(4)]
  ```