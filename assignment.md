Expected Content of the Progress Report
 
Your report should include the following sections.
 
1. Project Title, Group Members, and Problem Definition
 
State your project title and list all group members.
Briefly explain the NLP/NLU/NLG problem you are addressing. You should clearly describe:
What task you are working on;
Why the task is important;
What dataset, benchmark, or corpus you are using;
What the input and output of the system are;
Whether the task involves classification, generation, question answering, summarization, translation, retrieval, semantic parsing, evaluation, or another NLP task;
What makes the task technically challenging.
If your topic or title has changed since the proposal stage, briefly explain the change.


2. Dataset / Benchmark Status
Describe the dataset, benchmark, or corpus you are using and your current progress with it.
Please include:
Dataset or benchmark name;
Data source;
Number of samples, documents, prompts, questions, or instances;
Input and output format;
Train/validation/test split, if applicable;
Preprocessing steps completed;
Tokenization or prompt preparation steps, if applicable;
Any problems such as noisy data, limited data size, long inputs, hallucinated outputs, class imbalance, missing labels, or access/API limitations.
You should clearly state whether the dataset has already been downloaded, cleaned, and prepared for experiments.


3. Literature Review Progress
Briefly summarize your literature review progress.
You should mention:
The main papers, models, architectures, prompting strategies, or evaluation methods you have reviewed so far;
How these works are related to your project;
Which approaches are commonly used for this task;
Which methods, models, or prompts may be used as baselines;
What technical direction your project is likely to follow.
A simple list of paper titles is not sufficient. Please briefly explain how the reviewed works support your project.


4. Baseline Models or Prompting Strategies
Each project must include at least two baseline approaches in the final submission.
Depending on your topic, these may be:
Baseline NLP models;
Transformer-based models;
Prompting strategies;
Retrieval baselines;
Evaluation baselines;
Traditional machine learning baselines;
Zero-shot or few-shot LLM baselines.
In this section, report the current status of your baselines:
Which baselines you selected;
Why they are suitable for your project;
Which ones have already been implemented;
Whether you have obtained any initial results;
Any implementation, training, API, or computational problems you encountered.
At this checkpoint, you are expected to have started implementing or testing your baseline models or prompting strategies.


5. Initial Experimental Results
Include any preliminary results you have obtained so far.
Depending on your project, you may report metrics such as:
Accuracy;
Precision;
Recall;
F1-score;
Exact Match;
BLEU;
ROUGE;
METEOR;
BERTScore;
Perplexity;
Human or LLM-based evaluation scores;
Retrieval metrics such as Recall@K, MRR, or NDCG;
Runtime, memory usage, or API cost, if relevant.
Please include a small table if you already have results.
Example:
Model / Prompting Strategy	Metric 1	Metric 2	Notes
Baseline 1	-	-	Implemented / In progress
Baseline 2	-	-	Implemented / In progress
Improved version / current system	-	-	In progress
Do not only report numbers. Briefly interpret what the initial results indicate.


6. Planned Improvements and Technical Direction
At this stage, you are not required to present a fully finalized method. However, you should briefly explain the technical direction your group plans to follow in the next phase.
This may include:
Improving the best-performing baseline;
Trying a stronger Transformer or LLM model;
Revising prompts;
Adding few-shot examples;
Improving retrieval quality;
Fine-tuning or parameter-efficient fine-tuning;
Improving decoding settings;
Addressing hallucination or factuality issues;
Adding error filtering or post-processing;
Improving evaluation methodology;
Comparing alternative representations or model settings.
This section should explain how your group plans to improve or extend the current experimental pipeline.


7. Ablation or Prompt Sensitivity Plan
The final project must include an ablation study or prompt sensitivity analysis.
In the progress report, briefly explain what you plan to compare.
Depending on your project, this may include:
Different prompts;
Zero-shot vs. few-shot prompting;
Different decoding settings;
Different retrieval settings;
Different model sizes;
Different tokenization strategies;
With/without a specific module;
With/without fine-tuning;
Different hyperparameters.
You do not need to complete this analysis yet, but you should have a clear plan.


8. Error Analysis and Generation Quality Analysis
Your final report must include qualitative error analysis. In the progress report, briefly explain how you plan to analyze model errors.
You may discuss:
Incorrect predictions;
Hallucinated outputs;
Factual errors;
Incoherent generations;
Repetition problems;
Linguistic errors;
Bias-related outputs;
Failure cases of prompts or baseline models;
Examples where the model gives fluent but incorrect answers.
For generation tasks, you should also consider quality aspects such as coherence, factual accuracy, fluency, relevance, and linguistic quality.


9. Ethical Considerations and Bias
If your project involves generative models, LLMs, human-facing outputs, or sensitive data, include a brief note on ethical considerations.
You may discuss:
Hallucination risks;
Bias in generated outputs;
Toxicity or harmful content;
Privacy concerns;
Misuse scenarios;
Limitations of automatic evaluation;
Responsible use of LLM APIs.
This does not need to be long, but your group should show awareness of responsible AI issues.


10. GitHub and Reproducibility Status
Your project must be reproducible through GitHub.
Please include:
GitHub repository link;
Current repository structure;
Implemented scripts or notebooks;
Dataset preparation instructions;
README status;
Dependency file status, such as requirements.txt or environment.yml;
Prompt files or API configuration documentation, if LLM APIs are used;
Instructions for running current experiments;
Commit history status.
Please remember that project progress will also be monitored through GitHub commits. Consistent development is expected.


11. Current Challenges and Next Steps
Briefly explain your current challenges and your plan for the remaining weeks.
You may discuss:
Dataset problems;
Implementation difficulties;
Computational limitations;
API limitations or costs;
Weak baseline performance;
Evaluation difficulties;
Prompt instability;
Hallucination problems;
Coordination issues within the group.
Then summarize what your group plans to complete before the final submission.
Suggested Report Format
The progress report should be written in a clear and organized academic style.
Recommended length: 3–5 pages
Suggested structure:
Project Title, Group Members, and Problem Definition
Dataset / Benchmark Status
Literature Review Progress
Baseline Models or Prompting Strategies
Initial Experimental Results
Planned Improvements and Technical Direction
Ablation or Prompt Sensitivity Plan
Error Analysis and Generation Quality Analysis
Ethical Considerations and Bias
GitHub and Reproducibility Status
Current Challenges and Next Steps