# RAG Evaluation Report

**Source:** rag-files-20260302-094034/
**Date:** 2026-03-02 09:55
**Questions:** 25

## Overall Scores

| Metric | Score |
|--------|------:|
| Retrieval Hit Rate | 76.0% ███████████████░░░░░ |
| Context Precision | 65.0% █████████████░░░░░░░ |
| Faithfulness | 99.0% ███████████████████░ |
| Answer Correctness | 49.2% █████████░░░░░░░░░░░ |

**Composite Score: 72.3%**

## By Topic

| Topic | Hit Rate | Precision | Faithful | Correct | N |
|-------|---------|-----------|----------|---------|---|
| Assessment | 0% | 65% | 97% | 35% | 3 |
| Banking & Investing | 50% | 35% | 100% | 20% | 2 |
| Career & Education | 67% | 68% | 98% | 58% | 3 |
| Credit & Debt | 100% | 58% | 98% | 32% | 4 |
| Digital Financial Safety | 100% | 95% | 100% | 85% | 3 |
| Entrepreneurship | 100% | 80% | 100% | 70% | 1 |
| Goal Setting & Budgeting | 83% | 52% | 100% | 44% | 6 |
| Insurance & Risk | 100% | 90% | 100% | 70% | 1 |
| Saving | 100% | 80% | 100% | 60% | 2 |

## By Audience

| Audience | Hit Rate | Precision | Faithful | Correct | N |
|----------|---------|-----------|----------|---------|---|
| parent | 50% | 88% | 98% | 68% | 2 |
| student | 94% | 65% | 99% | 53% | 18 |
| teacher | 20% | 55% | 98% | 27% | 5 |

## Per-Question Results

| ID | Question | Hit | Prec | Faith | Correct | Top Source |
|----|----------|-----|------|-------|---------|------------|
| q01 | What is a SMART goal and how do I set one? | 100% | 90% | 100% | 95% | 4-5.FL.2 Anchor Chart - SMART  |
| q02 | How do I create a budget? | 100% | 80% | 100% | 70% | 4-5.FL.1 Handout C. How I’ll S |
| q03 | What is the difference between saving for now and ... | 100% | 70% | 100% | 60% | 4-5.FL.5 Lesson - What’s the D |
| q04 | What is the difference between a credit card and a... | 100% | 30% | 100% | 0% | 4-5.FL.6 Rubric for Handout C. |
| q05 | What happens when you don't pay back borrowed mone... | 100% | 80% | 100% | 40% | 4-5.FL.7 Handout A. Responsibl |
| q06 | How can I protect my personal information online? | 100% | 95% | 100% | 85% | 4-5.FL.8 Handout A. Protect Yo |
| q07 | What is identity theft? | 100% | 100% | 100% | 90% | 4–5.FL Band Answer Key for Fin |
| q08 | What is interest and how does it help my savings g... | 0% | 10% | 100% | 0% | 4-5.FL.12 Handout D. Dream Job |
| q09 | What is investing and how is it different from sav... | 100% | 60% | 100% | 40% | 4-5.FL.9 Handout A. Growing Ou |
| q10 | What types of insurance are there and why do we ne... | 100% | 90% | 100% | 70% | 4-5.FL10 Answer Key For Handou |
| q11 | How can my hobbies and skills lead to a career? | 100% | 90% | 100% | 80% | 4-5.FL.11 Handout A. Exploring |
| q12 | What is an entrepreneur? | 100% | 80% | 100% | 70% | 4-5.FL.4 Anchor Chart Smart Sa |
| q13 | What activities can I do at home with my child to ... | 100% | 90% | 100% | 60% | 4-5.FL.3 Handout A. Saving and |
| q14 | How should I teach lesson 6 about credit cards vs ... | 100% | 80% | 100% | 30% | 4-5.FL.6 Lesson - Credit Cards |
| q15 | What rubric should I use to grade the SMART goal a... | 0% | 60% | 100% | 10% | 4-5.FL.2 Anchor Chart - SMART  |
| q16 | What does prioritize mean in the context of budget... | 100% | 100% | 100% | 90% | 4–5.FL Band Final Assessment S |
| q17 | How do I help my child explore career options and ... | 0% | 85% | 95% | 75% | 4-5.FL.13 Handout A. Practicin |
| q18 | What is the final assessment for the financial lit... | 0% | 95% | 100% | 85% | 4–5.FL Band Final Assessment S |
| q19 | What is income and what are examples of it? | 100% | 20% | 100% | 10% | 4-5.FL.4 Handout A. Creating a |
| q20 | What should I know about college before applying? | 100% | 30% | 100% | 20% | 4-5.FL.7 Handout B. Role Play  |
| q21 | What is the anchor chart for lesson 1 about? | 0% | 0% | 100% | 0% | 4-5.FL.7 Anchor Chart. Underst |
| q22 | How can I build a strong password? | 100% | 90% | 100% | 80% | 4-5.FL.8 Handout B. Let’s Buil |
| q23 | What is the difference between needs and wants? | 100% | 20% | 100% | 0% | 4-5.FL.6 Handout D. Borrowing  |
| q24 | How do I assess whether a student understands care... | 0% | 40% | 90% | 10% | 4-5.FL.12 Anchor Chart Connect |
| q25 | What is a credit score and why does it matter? | 100% | 40% | 90% | 60% | 4-5.FL.7 Anchor Chart. Underst |

## Retrieval Misses

- **q08**: Expected `4-5.FL.3`, got: Student Activities/Career Exploration/4-5.FL.12 Handout D. Dream Job Presentation Checklist .md, Student Activities/Career Exploration/4-5.FL.13 Handout B. Practicing for My First College Application.md, Family Engagement/Home Letters by Topic/4-5.FL.9 Handout A. Growing Our Money Tree Together Home Letter.md
- **q15**: Expected `4-5.FL.2 Rubric`, got: Teacher Resources/Lesson Plans & Anchor Charts/4-5.FL.2 Anchor Chart - SMART Goals and Achieving Your Dreams.md, Student Activities/Goal Setting & Budgeting/4-5.FL.2 Handout C. SMART Goal Progress Tracker.md, Student Activities/Goal Setting & Budgeting/4-5.FL.2 Handout B. SMART Goal Brainstorm and Planning.md
- **q17**: Expected `4-5.FL.12 Handout A`, got: Family Engagement/Home Letters by Topic/4-5.FL.13 Handout A. Practicing for My First College Application Home Letter.md, Family Engagement/Home Letters by Topic/4-5.FL.11 Handout A. Exploring Careers Together Home Letter.md, Family Engagement/Home Letters by Topic/4-5.FL.1 Handout A. Reaching Goals Together Home Letter.md
- **q18**: Expected `4-5.FL Band Final Assessment`, got: Teacher Resources/Assessment Materials/4–5.FL Band Final Assessment Smart Money Decisions.md, Teacher Resources/Lesson Plans & Anchor Charts/4-5.FL.3 Anchor Chart Growing Savings and Banking Benefits.md, Teacher Resources/Assessment Materials/4–5.FL Band Answer Key for Final Assessment Smart Money Decisions.md
- **q21**: Expected `4-5.FL.1 Anchor Chart`, got: Teacher Resources/Lesson Plans & Anchor Charts/4-5.FL.7 Anchor Chart. Understanding Credit, Debt, and Your Credit Score.md, Teacher Resources/Lesson Plans & Anchor Charts/4-5.FL.5 Anchor Chart Saving for Now vs. Saving for Later.md, Teacher Resources/Lesson Plans & Anchor Charts/4-5.FL.3 Anchor Chart Growing Savings and Banking Benefits.md
- **q24**: Expected `4-5.FL.11 Rubric`, got: Teacher Resources/Lesson Plans & Anchor Charts/4-5.FL.12 Anchor Chart Connecting Careers to Financial Goals.md, Student Activities/Career Exploration/4-5.FL.13 Handout B. Practicing for My First College Application.md, Teacher Resources/Lesson Plans & Anchor Charts/4-5.FL.2 Anchor Chart - SMART Goals and Achieving Your Dreams.md
