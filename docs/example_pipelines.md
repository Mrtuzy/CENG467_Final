production-grade RAG context pruning pipeline

flowchart TD
    A[User Query] --> B[Query Understanding]

    B --> B1[Intent Detection]
    B --> B2[Entity / Keyword Extraction]
    B --> B3[Query Decomposition]
    B --> B4[Token Budget Estimation]

    B1 --> C[Query Planning]
    B2 --> C
    B3 --> C
    B4 --> C

    C --> D1[Vector Retrieval]
    C --> D2[Keyword / BM25 Retrieval]
    C --> D3[Metadata Filtering]
    C --> D4[Conversation / Memory Retrieval]

    D1 --> E[Candidate Context Pool]
    D2 --> E
    D3 --> E
    D4 --> E

    E --> F1[Deduplication]
    E --> F2[Chunk Quality Filtering]
    E --> F3[Source / Permission Filtering]
    E --> F4[Recency Filtering]

    F1 --> G[Clean Candidate Pool]
    F2 --> G
    F3 --> G
    F4 --> G

    G --> H1[Reranking]
    G --> H2[MMR / Diversity Selection]
    G --> H3[Clustering]
    G --> H4[Citation / Evidence Scoring]

    H1 --> I[Selected Evidence Set]
    H2 --> I
    H3 --> I
    H4 --> I

    I --> J1[Query-Focused Summarization]
    I --> J2[Prompt Compression]
    I --> J3[Table / Code / DOM Pruning]
    I --> J4[Entity-State Extraction]

    J1 --> K[Compressed Context]
    J2 --> K
    J3 --> K
    J4 --> K

    K --> L[Context Assembly]

    L --> L1[System Prompt]
    L --> L2[User Query]
    L --> L3[Compressed Evidence]
    L --> L4[Conversation Summary]
    L --> L5[Tool Schemas / Few-shot Examples]

    L1 --> M[Final Prompt]
    L2 --> M
    L3 --> M
    L4 --> M
    L5 --> M

    M --> N[LLM Generation]

    N --> O1[KV Cache Eviction]
    N --> O2[KV Cache Compression]
    N --> O3[Token / Feature Merging]
    N --> O4[Sliding Window / Sink Token Preservation]

    O1 --> P[Answer Draft]
    O2 --> P
    O3 --> P
    O4 --> P

    P --> Q1[Faithfulness Check]
    P --> Q2[Citation Check]
    P --> Q3[Missing Evidence Detection]
    P --> Q4[Answer Compression]

    Q1 --> R{Enough Evidence?}
    Q2 --> R
    Q3 --> R
    Q4 --> R

    R -- Yes --> S[Final Answer]
    R -- No --> T[Follow-up Retrieval / Agentic Paging]

    T --> D1
    T --> D2
    T --> D4




agentic paging + memory hierarchy

flowchart TD
    A[Agent Receives Task] --> B[Active Context Window]

    B --> C1[Current User Query]
    B --> C2[Recent Conversation]
    B --> C3[Current Tool Outputs]
    B --> C4[Working Plan]

    C1 --> D[Working Memory Manager]
    C2 --> D
    C3 --> D
    C4 --> D

    D --> E1[Keep in Active Window]
    D --> E2[Compress to Summary]
    D --> E3[Store as Episodic Memory]
    D --> E4[Store as Semantic Memory]
    D --> E5[Discard Low-Value Context]

    E1 --> F[LLM Reasoning Step]
    E2 --> G[Compressed Working State]
    E3 --> H[Episodic Memory Store]
    E4 --> I[Semantic Memory Store]

    G --> F

    F --> J{Need More Context?}

    J -- No --> K[Continue / Answer]
    J -- Yes --> L[Agentic Paging Request]

    L --> M1[Search Recent Conversation]
    L --> M2[Search Episodic Memory]
    L --> M3[Search Semantic Memory]
    L --> M4[Search External Documents]
    L --> M5[Search Tool Results]

    M1 --> N[Retrieved Memory Pages]
    M2 --> N
    M3 --> N
    M4 --> N
    M5 --> N

    N --> O1[Relevance Scoring]
    N --> O2[Deduplication]
    N --> O3[Compression]
    N --> O4[Token Budget Selection]

    O1 --> P[Loaded Context Page]
    O2 --> P
    O3 --> P
    O4 --> P

    P --> B


engineering implementation pipeline

flowchart LR
    A[Input Query] --> B[Query Analyzer]

    B --> C1[Retriever Worker]
    B --> C2[Memory Worker]
    B --> C3[Metadata Filter Worker]
    B --> C4[Budget Planner Worker]

    C1 --> D[Candidate Pool]
    C2 --> D
    C3 --> D
    C4 --> E[Token Budget]

    D --> F1[Dedup Worker]
    D --> F2[Rerank Worker]
    D --> F3[Cluster Worker]
    D --> F4[Quality Filter Worker]

    F1 --> G[Pruned Pool]
    F2 --> G
    F3 --> G
    F4 --> G

    G --> H1[Summarizer Worker]
    G --> H2[Compressor Worker]
    G --> H3[Entity Extractor Worker]
    G --> H4[Citation Selector Worker]

    E --> I[Context Assembler]
    H1 --> I
    H2 --> I
    H3 --> I
    H4 --> I

    I --> J[LLM Call]

    J --> K1[Answer]
    J --> K2[Updated KV Cache]
    J --> K3[Trace / Evidence Map]

    K2 --> L1[KV Eviction]
    K2 --> L2[KV Compression]
    K2 --> L3[Cache Offload]

    K1 --> M[Verifier]
    K3 --> M

    M --> N{Valid?}

    N -- Yes --> O[Return Final Answer]
    N -- No --> P[Trigger Extra Retrieval]

    P --> C1
    P --> C2



simple 

flowchart TD
    A[User Query] --> B[Query Decomposition]
    B --> C1[Vector Search]
    B --> C2[BM25 Search]
    B --> C3[Memory Search]

    C1 --> D[Candidate Chunks]
    C2 --> D
    C3 --> D

    D --> E[Deduplication]
    E --> F[Reranker]
    F --> G[Top Evidence Selection]
    G --> H[Query-Focused Compression]
    H --> I[Context Assembly]
    I --> J[LLM Answer]
    J --> K[Faithfulness / Citation Check]
    K --> L[Final Response]