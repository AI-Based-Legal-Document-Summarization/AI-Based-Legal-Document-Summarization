*****AI-Based-Legal-Document-Summarization***** 



 A machine learning model that can condense long legal documents into concise summaries while preserving key information, such as legal arguments, clauses, and decisions.

**Problem Statement:**
Legal professionals often spend significant time reading and analyzing lengthy legal documents, case laws, contracts, and court decisions. These documents are complex, full of technical jargon, and contain crucial information scattered throughout. Manually extracting key clauses, decisions, and legal arguments is time-consuming and prone to errors. This project seeks to address this issue by developing an AI-based solution that can automatically summarize long legal documents, focusing on the most important legal information.

**Project Objectives:**
1.	Develop an AI-based Summarization System:
        o	Build a machine learning model that can condense long legal documents into concise summaries while preserving key information, such as legal arguments, clauses, and decisions.
2.	Leverage Advanced NLP Techniques:
        o	Utilize transformer-based models like BERT or GPT to enhance the model's ability to understand and summarize complex legal text.
        o	Implement text summarization techniques to generate human-readable summaries.
3.	Enhance Information Extraction:
       o	Incorporate Named Entity Recognition (NER) to identify key legal entities, such as parties, legal statutes, and dates, to enrich the summaries and provide contextual relevance.

**Analysis Overview:**
**Data Collection:**
The project focused on summarizing legal documents from the Indian Supreme Court, specifically leveraging the IN-Abs dataset, which contains case documents and their corresponding abstractive summaries. The data was sourced from **Liiofindia.org**, a repository of legal cases. The collection process involved:

**Data Scraping:**
**Apache Nutch and Web Scraper.io** Automated sites were used to extract case documents from the website, ensuring that both the full text of the cases and their summaries were collected.
Data Structuring: The raw data was organized into a structured format for easy access and processing. This involved categorizing cases by relevant metadata such as case number, date, judges, and keywords.
Data Cleaning: The collected text underwent a cleaning process to remove any irrelevant information, formatting issues, and inconsistencies, ensuring high-quality input for training.

**Data Processing:**
Once the legal data was collected, it was processed as follows:

**Text Preprocessing:**
This involved tokenization, removal of stop words, and normalization of text (lowercasing, handling punctuation, etc.) to prepare the text for model training.
Segmentation: The legal documents were segmented into manageable chunks, particularly important for handling longer documents. This step ensured that the BART model could efficiently process the data without exceeding its input limits.

**Model Fine-tuning and Training:**
To generate effective summaries from the legal texts, we fine-tuned a BART (Bidirectional and Auto-Regressive Transformers) model. The process included:

**Model Selection:**
The BART model was chosen due to its effectiveness in text generation tasks, particularly in abstractive summarization.
Fine-tuning Process:

**Training Script:** The BART_fineTune.ipynb script was developed to fine-tune the BART model on the IN-Abs dataset. The model was trained to predict the summary given the case text.
Hyperparameter Tuning: Various hyperparameters such as learning rate, batch size, and number of epochs were experimented with to optimize performance.

**Summary Generation Methods:**
To generate summaries, several methods were implemented using different variations of the BART model:

**Chunking-based BART Method:**
Implemented in the generate_summaries_chunking_BART.ipynb, this method utilized chunking to feed sections of the legal documents into the model sequentially, allowing for efficient summarization of longer texts.

**Chunking-based BART with Relevance Ranking (BART_RR):** The generate_summaries_chunking_BART_RR.ipynb script incorporated relevance ranking to prioritize the most important chunks of text before summarization, enhancing the quality of the generated summaries.

**Chunking-based BERT-BART Hybrid Method:** The generate_summaries_chunking_BERT_BART.ipynb script combined BERT's contextual understanding with BART’s generative capabilities, leveraging the strengths of both models to produce more coherent and contextually accurate summaries.

**Conclusion:**
This project demonstrated the feasibility and effectiveness of using advanced NLP techniques, specifically the BART model, for summarizing complex legal documents. The structured approach to data collection, preprocessing, and model fine-tuning resulted in a robust system capable of producing high-quality abstractive summaries from Indian Supreme Court case documents. The use of chunking methods and hybrid models further enhanced the system's performance, paving the way for future applications in legal tech and automated document processing.

**Expected Outcomes:**
        •	Efficient Legal Document Summarization: The final model will produce concise, accurate, and relevant summaries of long legal documents, helping legal professionals save time and effort in document analysis.
        •	Reduction in Manual Effort: Legal teams will be able to quickly understand case details, legal arguments, and outcomes, reducing the need for manual reading and analysis.
        •	Scalable Solution: The system will be scalable, capable of handling various legal documents across different jurisdictions and languages.


