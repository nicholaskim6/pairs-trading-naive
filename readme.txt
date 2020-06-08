All files are ipynb, intended to be run directly on quantopian.com's hosted notebook servers.
(They have proprietary restrictions on access to their data pipelines)

To run, make an account on quantopian and plug the code into a notebook.

dataset - quantopian.com -> Createnotebook -> import with these commands
from quantopian.pipeline.data import morningstar
from quantopian.pipeline.filters.morningstar import Q500US, Q1500US, Q3000US
from quantopian.pipeline import Pipeline
from quantopian.research import run_pipeline

There is no explicit link to the dataset, as it is proprietary to the website
and only importable in quantopian notebooks


Python Files - (ipynb)
strategy1.py - data processing, testing, and evaluation for difference halving strategy
strategy2.py - data processing, testing, and evaluation for velocity reversal strategy
Note - the first half of each file (up to about line 450 in each) is from the cited Quantopian code
by Jonathan Larkin for the initial data processing into stock clusters