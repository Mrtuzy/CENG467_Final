DECOMPOSE_PROMPT = """\
You are a task that decomposes an answer into atomic, factual claims.

An atomic claim is a single, indivisible fact that can be true or false.

Example 1:
Answer: "Paris is the capital of France and it is located in Europe."
Claims:
1. Paris is the capital of France.
2. Paris is located in Europe.

Example 2:
Answer: "Albert Einstein was a physicist born in Germany in 1879."
Claims:
1. Albert Einstein was a physicist.
2. Albert Einstein was born in Germany.
3. Albert Einstein was born in 1879.

Now decompose the following answer into atomic claims:

Answer: {answer}

Claims:"""


VERIFY_PROMPT = """\
You are a fact-checker. Given a claim and a context, determine whether the claim is fully supported by the context.

Answer "YES" if the claim is fully supported by the context.
Answer "NO" if the claim is NOT fully supported by the context, contradicted, or cannot be determined from the context.

Example 1:
Context: "Paris is the capital of France. It is located in Western Europe."
Claim: "Paris is the capital of France."
Answer: YES

Example 2:
Context: "Albert Einstein was a physicist who worked on relativity."
Claim: "Albert Einstein was born in Germany."
Answer: NO

Now answer the following:

Context: {context}
Claim: {claim}
Answer:"""
