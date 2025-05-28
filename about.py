# in about.py

import gradio as gr

def render_about():
    with gr.Blocks() as about_page:
        gr.Markdown("""
        # About MIZAN: A Persian LLM Leaderboard

        MIZAN: A Persian LLM Leaderboard is designed to establish a standard and comprehensive benchmark for evaluating Large Language Models (LLMs) in the Persian language. This project combines existing datasets, translates and localizes globally recognized benchmarks, and incorporates newly developed, Persian-specific datasets. MIZAN aims to provide a multi-dimensional assessment of models' capabilities across various linguistic, knowledge-based, and reasoning tasks. Its primary goal is to offer researchers, developers, and enthusiasts a transparent and reliable view of LLM performance in the Persian language landscape.

        MIZAN provides a holistic view of models' strengths and weaknesses by assessing them across a suite of key tasks, contributing to the advancement of AI research for the Persian language.
        """)

        with gr.Accordion("MIZAN Benchmark Components Details", open=True): # Changed from PULL
            
            with gr.Accordion("1. PerCoR (Persian Commonsense Reasoning)", open=False):
                gr.Markdown("""
                PersCoR is the first large-scale Persian benchmark for evaluating models' ability in **commonsense reasoning** through multi-choice sentence completion. It includes over 106,000 samples from diverse domains such as news, religion, and lifestyle, extracted from more than 40 Persian websites. Innovative methods like "segmentation by conjunctions" were used to create coherent and diverse sentences and options, while the DRESS-AF technique helped generate challenging, human-solvable distractors.
                """)

            with gr.Accordion("2. IFEval-fa (Persian Instruction Following Evaluation)", open=False):
                gr.Markdown("""
                This dataset is a Persian-adapted and localized version of **IFEval**, assessing models' proficiency in **accurately executing complex, multi-step instructions (Instruction Following)**. The translation process involved a hybrid machine-human approach, with prompts unsuitable for the Persian language being rewritten or removed.
                """)

            with gr.Accordion("3. MMLU-Fa (Persian Massive Multitask Language Understanding)", open=False):
                gr.Markdown("""
                MMLU-Fa is an expanded and localized version of the renowned **MMLU** benchmark, designed to measure **general and specialized knowledge** of models in Persian. Tailored to cover knowledge at various levels and relevant to the Iranian cultural context, it comprises three main sub-datasets:
                <ul>
                    <li><strong>SPK (School Persian Knowledge):</strong> Contains 5,581 multiple-choice questions from the official Iranian school curriculum (grades 4-12) across 78 diverse subjects. Data was collected from the "Paadars" educational website and subsequently cleaned.</li>
                    <li><strong>UPK (University Persian Knowledge):</strong> Includes 7,793 multiple-choice questions from Master's and PhD entrance exams across 25 academic disciplines (e.g., medicine, engineering, humanities, arts). This data was extracted from exam booklets using OCR technology and cleaned by LLMs.</li>
                    <li><strong>GPK (General Persian Knowledge):</strong> Consists of 1,003 multiple-choice questions on 15 topics related to general knowledge specific to Iranian society (e.g., city souvenirs, religious edicts, national laws, famous personalities, cultural idioms). This data was generated using LLMs with specific prompts and reviewed by humans.</li>
                </ul>
                """)

            with gr.Accordion("4. Persian MT-Bench (Persian Multi-Turn Benchmark)", open=False):
                gr.Markdown("""
                This is a localized version of the **MT-Bench** benchmark, evaluating models on **multi-turn question-answering and dialogue-based tasks**. Questions involve multi-step requests or require creative responses. In the Persian version, all samples were translated and rewritten by humans, and some were expanded to 3 or 4 turns. Two new topics were also added:
                <ul>
                    <li><strong>Native Iranian Knowledge:</strong> Questions about cultural topics such as films, actors, and Iranian figures.</li>
                    <li><strong>Chat-Retrieval:</strong> Involves a multi-turn dialogue where the model must extract a relevant question and answer based on the user's needs.</li>
                </ul>
                """)

            with gr.Accordion("5. Persian NLU (Persian Natural Language Understanding)", open=False):
                gr.Markdown("""
                This section comprises a collection of existing Persian benchmarks for evaluating various aspects of **Natural Language Understanding**. Key tasks and datasets include:
                <ul>
                    <li><strong>Sentiment Analysis:</strong> DeepSentiPers</li>
                    <li><strong>Text Classification:</strong> Synthetic Persian Tone, SID</li>
                    <li><strong>Natural Language Inference (NLI):</strong> FarsTAIL</li>
                    <li><strong>Semantic Textual Similarity (STS):</strong> Synthetic Persian STS, FarSICK</li>
                    <li><strong>Named Entity Recognition (NER):</strong> Arman</li>
                    <li><strong>Paraphrase Detection:</strong> FarsiParaphraseDetection, ParsiNLU</li>
                    <li><strong>Extractive Question Answering (EQA):</strong> PQuAD</li>
                    <li><strong>Keyword Extraction:</strong> Synthetic Persian Keywords</li>
                </ul>
                """)

            with gr.Accordion("6. Persian NLG (Persian Natural Language Generation)", open=False):
                gr.Markdown("""
                This section focuses on **Natural Language Generation**, covering tasks such as:
                <ul>
                    <li><strong>Summarization:</strong> SamSUM-fa, PnSummary</li>
                    <li><strong>Machine Translation:</strong> TEP, MIZAN, EPOQUE</li>
                    <li><strong>Question Generation:</strong> PersianQA</li>
                </ul>
                The goal is to assess the generative capabilities of models.
                """)

            # with gr.Accordion("7. BoolQ-fa (Persian Boolean Question Answering)", open=False):
            #     gr.Markdown("""
            #     A Persian-adapted version of **BoolQ**, this benchmark evaluates a model's ability to answer **yes/no questions based on a given text**, testing common reasoning skills. Each instance includes a passage, a question about it, and a boolean answer.
            #     """)

            # with gr.Accordion("8. PIQA-fa (Persian Physical Interaction Question Answering)", open=False):
            #     gr.Markdown("""
            #     This is a Persian version of **PIQA**, focusing on **physical reasoning and commonsense understanding** of real-world interactions. Each instance presents a goal or question along with two potential solutions, requiring the model to choose the more physically plausible option.
            #     """)

        gr.Markdown("""
        ---
        MIZAN is a significant step towards the scientific and localized evaluation of language models for Persian, aiming to serve as a valuable assessment reference for researchers, developers, and anyone interested in practical language models.
        """)

    return about_page

# To test this function directly (if in a separate file):
# if __name__ == '__main__':
#     render_about().launch()