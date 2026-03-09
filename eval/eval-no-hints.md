# RAG Evaluation Report

**Source:** rag-files-20260302-091922/
**Date:** 2026-03-02 09:55
**Questions:** 25

## Overall Scores

| Metric | Score |
|--------|------:|
| Retrieval Hit Rate | 80.0% ████████████████░░░░ |
| Context Precision | 68.0% █████████████░░░░░░░ |
| Faithfulness | 99.8% ███████████████████░ |
| Answer Correctness | 57.0% ███████████░░░░░░░░░ |

**Composite Score: 76.2%**

## By Topic

| Topic | Hit Rate | Precision | Faithful | Correct | N |
|-------|---------|-----------|----------|---------|---|
| Assessment | 0% | 65% | 100% | 35% | 3 |
| Banking & Investing | 50% | 40% | 100% | 30% | 2 |
| Career & Education | 67% | 67% | 100% | 57% | 3 |
| Credit & Debt | 100% | 59% | 99% | 46% | 4 |
| Digital Financial Safety | 100% | 93% | 100% | 83% | 3 |
| Entrepreneurship | 100% | 90% | 100% | 80% | 1 |
| Goal Setting & Budgeting | 100% | 65% | 100% | 64% | 6 |
| Insurance & Risk | 100% | 70% | 100% | 60% | 1 |
| Saving | 100% | 80% | 100% | 65% | 2 |

## By Audience

| Audience | Hit Rate | Precision | Faithful | Correct | N |
|----------|---------|-----------|----------|---------|---|
| parent | 50% | 85% | 100% | 70% | 2 |
| student | 94% | 65% | 100% | 57% | 18 |
| teacher | 40% | 72% | 99% | 52% | 5 |

## Per-Question Results

| ID | Question | Hit | Prec | Faith | Correct | Top Source |
|----|----------|-----|------|-------|---------|------------|
| q01 | What is a SMART goal and how do I set one? | 100% | 90% | 100% | 95% | 4-5.FL.2 Anchor Chart - SMART  |
| q02 | How do I create a budget? | 100% | 80% | 100% | 70% | 4-5.FL.2 Anchor Chart - SMART  |
| q03 | What is the difference between saving for now and ... | 100% | 70% | 100% | 60% | 4-5.FL.5 Lesson - What’s the D |
| q04 | What is the difference between a credit card and a... | 100% | 30% | 100% | 0% | 4-5.FL.6 Rubric for Handout C. |
| q05 | What happens when you don't pay back borrowed mone... | 100% | 80% | 100% | 60% | 4-5.FL.7 Handout A. Responsibl |
| q06 | How can I protect my personal information online? | 100% | 90% | 100% | 80% | 4-5.FL.8 Handout A. Protect Yo |
| q07 | What is identity theft? | 100% | 100% | 100% | 90% | 4–5.FL Band Answer Key for Fin |
| q08 | What is interest and how does it help my savings g... | 0% | 10% | 100% | 0% | 4-5.FL.12 Handout D. Dream Job |
| q09 | What is investing and how is it different from sav... | 100% | 70% | 100% | 60% | 4-5.FL.9 Handout A. Growing Ou |
| q10 | What types of insurance are there and why do we ne... | 100% | 70% | 100% | 60% | 4-5.FL10 Answer Key For Handou |
| q11 | How can my hobbies and skills lead to a career? | 100% | 90% | 100% | 80% | 4-5.FL.11 Handout A. Exploring |
| q12 | What is an entrepreneur? | 100% | 90% | 100% | 80% | 4–5.FL Band Final Assessment S |
| q13 | What activities can I do at home with my child to ... | 100% | 90% | 100% | 70% | 4-5.FL.3 Handout A. Saving and |
| q14 | How should I teach lesson 6 about credit cards vs ... | 100% | 85% | 95% | 65% | 4-5.FL.6 Lesson - Credit Cards |
| q15 | What rubric should I use to grade the SMART goal a... | 0% | 70% | 100% | 10% | 4-5.FL.2 Anchor Chart - SMART  |
| q16 | What does prioritize mean in the context of budget... | 100% | 90% | 100% | 80% | 4–5.FL Band Final Assessment S |
| q17 | How do I help my child explore career options and ... | 0% | 80% | 100% | 70% | 4-5.FL.13 Handout A. Practicin |
| q18 | What is the final assessment for the financial lit... | 0% | 95% | 100% | 85% | 4–5.FL Band Final Assessment S |
| q19 | What is income and what are examples of it? | 100% | 40% | 100% | 50% | 4-5.FL.1 Anchor Chart - Goals  |
| q20 | What should I know about college before applying? | 100% | 30% | 100% | 20% | 4-5.FL.7 Handout B. Role Play  |
| q21 | What is the anchor chart for lesson 1 about? | 100% | 80% | 100% | 90% | 4-5.FL.5 Anchor Chart Saving f |
| q22 | How can I build a strong password? | 100% | 90% | 100% | 80% | 4-5.FL.8 Handout B. Let’s Buil |
| q23 | What is the difference between needs and wants? | 100% | 10% | 100% | 0% | 4-5.FL.6 Handout D. Borrowing  |
| q24 | How do I assess whether a student understands care... | 0% | 30% | 100% | 10% | 4-5.FL.13 Handout B. Practicin |
| q25 | What is a credit score and why does it matter? | 100% | 40% | 100% | 60% | 4-5.FL.7 Anchor Chart. Underst |

## Retrieval Misses

- **q08**: Expected `4-5.FL.3`, got: Career and Education Planning/4-5.FL.12 Handout D. Dream Job Presentation Checklist .md, Career and Education Planning/4-5.FL.13 Handout B. Practicing for My First College Application.md, Family Engagement Resources/4-5.FL.9 Handout A. Growing Our Money Tree Together Home Letter.md
- **q15**: Expected `4-5.FL.2 Rubric`, got: Financial Goals and Planning/Goal Setting and Budgeting Basics/4-5.FL.2 Anchor Chart - SMART Goals and Achieving Your Dreams.md, Financial Goals and Planning/Goal Setting and Budgeting Basics/4-5.FL.2 Handout C. SMART Goal Progress Tracker.md, Financial Goals and Planning/Goal Setting and Budgeting Basics/4-5.FL.2 Handout B. SMART Goal Brainstorm and Planning.md
- **q17**: Expected `4-5.FL.12 Handout A`, got: Family Engagement Resources/4-5.FL.13 Handout A. Practicing for My First College Application Home Letter.md, Family Engagement Resources/4-5.FL.11 Handout A. Exploring Careers Together Home Letter.md, Family Engagement Resources/4-5.FL.1 Handout A. Reaching Goals Together Home Letter.md
- **q18**: Expected `4-5.FL Band Final Assessment`, got: Assessment and Project Materials/4–5.FL Band Final Assessment Smart Money Decisions.md, Banking and Investment Concepts/4-5.FL.3 Anchor Chart Growing Savings and Banking Benefits.md, Assessment and Project Materials/4–5.FL Band Answer Key for Final Assessment Smart Money Decisions.md
- **q24**: Expected `4-5.FL.11 Rubric`, got: Career and Education Planning/4-5.FL.13 Handout B. Practicing for My First College Application.md, Financial Goals and Planning/Goal Setting and Budgeting Basics/4-5.FL.2 Anchor Chart - SMART Goals and Achieving Your Dreams.md, Risk Management and Insurance/4-5.FL.10 Anchor Chart How Insurance Saves the Day.md
