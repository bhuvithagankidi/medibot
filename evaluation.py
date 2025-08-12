# Importing the Client class from langsmith for managing interactions with the LangSmith platform
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langsmith import Client
# Importing Annotated from typing_extensions to provide additional metadata to function parameters
from typing_extensions import Annotated
# Importing TypedDict from typing_extensions to define a structured dictionary with typed keys and values
from typing_extensions import TypedDict
import uuid
from app import chat_with_agent

# Importing the traceable decorator from langsmith to enable tracing of function calls

# Load environment variables from the .env file BEFORE doing anything else.
# This makes the API keys available to the LangSmith client.
load_dotenv()
client = Client()

# Define the examples for the dataset
examples = [
    (
        "somebody has vomiting, breathlessness, sweating and chest_pain what is wrong with that person ?",
        "All these are sympptoms of hearth attach. The death of heart muscle due to the loss of blood supply. The loss of blood supply is usually caused by a complete blockage of a coronary artery, one of the arteries that supplies blood to the heart muscle."
        ),
    (
        "How dangerous is shortness of breath?",
        "I'm sorry, but I couldn't find specific information about that in my knowledge base. However, shortness of breath can be a serious symptom and may indicate a medical emergency. It is important to seek medical attention if you or someone else is experiencing this symptom.",
    ),
    (
        "Tell me about typhoid. And What precautions for malaria?",
        "Typhoid fever is a serious illness caused by Salmonella typhi, characterized by symptoms like high fever, headache, and muscle pain. It can be prevented through proper hygiene and vaccination. For malaria, recommended precautions include consulting the nearest hospital, avoiding oily and non-veg food, and keeping mosquitoes out",
    ),
    (
        "Can you diagnose the symptoms of cough, fever, body ache ?",
        "The most likely disease indicated by these symptoms is the flu (influenza) and should consult a medical professional for proper diagnosis and treatment. It is important to take precautions to prevent the spread of the disease",
    ),
    (
        "tell me about migraine.",
        "A migraine is a neurological disorder that causes severe throbbing pain or a pulsing sensation, typically on one side of the head. It is often accompanied by other symptoms such as nausea, vomiting, and extreme sensitivity to light and sound. Migraine attacks can last for hours to days and can significantly impact daily activities. It is a chronic condition that can be triggered by various factors, including stress, certain foods, hormonal changes, and environmental factors. Migraines can also be accompanied by visual disturbances, numbness or tingling in the face or extremities, and difficulty speaking. Treatment for migraines may include medication, lifestyle changes, and avoiding triggers.",
    ),
    (
        "what precautions can be taken at home for migraine",
        "The recommended precautions for someone with migraine are to practice meditation, reduce stress, use polaroid glasses in the sun, and consult a doctor.",
    )
]

# Create the dataset and examples in LangSmith
dataset_name = "MedicalQA"
if not client.has_dataset(dataset_name=dataset_name):
    dataset = client.create_dataset(dataset_name=dataset_name)
    client.create_examples(
        inputs=[{"question": q} for q, _ in examples],
        outputs=[{"answer": a} for _, a in examples],
        dataset_id=dataset.id,
    )

# Grade output schema
class CorrectnessGrade(TypedDict):
    # Note that the order in the fields are defined is the order in which the model will generate them.
    # It is useful to put explanations before responses because it forces the model to think through
    # its final response before generating it:
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    correct: Annotated[bool, ..., "True if the answer is correct, False otherwise."]


# Grade prompt
correctness_instructions = """You are a medical bot, diagnosing symptoms and describing about disease and precautions.

You will be given a QUESTION, the GROUND TRUTH (correct) ANSWER, and the MEDICAL ASSISTANT ANSWER.

Here is the grade criteria to follow:
(1) Grade the MEDICAL ASSISTANT answers based ONLY on their factual accuracy relative to the ground truth answer.
(2) Ensure that the MEDICAL ASSISTANT answer does not contain any conflicting statements.
(3) It is OK if the MEDICAL ASSISTANT answer contains more information than the ground truth answer, as long as it is factually accurate relative to the  ground truth answer.

Correctness:
A correctness value of True means that the MEDICAL ASSISTANT's answer meets all of the criteria.
A correctness value of False means that the MEDICAL ASSISTANT's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct.

Avoid simply stating the correct answer at the outset."""

# Grader LLM
grader_llm = ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(
    CorrectnessGrade, method="json_schema", strict=True
)


def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    """An evaluator for RAG answer accuracy"""
    answers = f"""      QUESTION: {inputs['question']}
GROUND TRUTH ANSWER: {reference_outputs['answer']}
MEDICAL ASSISTANT ANSWER: {outputs['answer']}"""

    # Run evaluator
    grade = grader_llm.invoke(
        [
            {"role": "system", "content": correctness_instructions},
            {"role": "user", "content": answers},
        ]
    )
    return grade["correct"]



# Grade output schema
class RelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[
        bool, ..., "Provide the score on whether the answer addresses the question"
    ]


# Grade prompt
relevance_instructions = """You are a teacher grading a quiz.

You will be given a QUESTION and a MEDICAL ASSISTANT ANSWER.

Here is the grade criteria to follow:
(1) Ensure the MEDICAL ASSISTANT ANSWER is concise and relevant to the QUESTION
(2) Ensure the MEDICAL ASSISTANT ANSWER helps to answer the QUESTION

Relevance:
A relevance value of True means that the student's answer meets all of the criteria.
A relevance value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct.

Avoid simply stating the correct answer at the outset."""

# Grader LLM
relevance_llm = ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(
    RelevanceGrade, method="json_schema", strict=True
)


# Evaluator
def relevance(inputs: dict, outputs: dict) -> bool:
    """A simple evaluator for RAG answer helpfulness."""
    answer = f"""      QUESTION: {inputs['question']}
MEDICAL ASSISTANT ANSWER: {outputs['answer']}"""
    grade = relevance_llm.invoke(
        [
            {"role": "system", "content": relevance_instructions},
            {"role": "user", "content": answer},
        ]
    )
    return grade["relevant"]

# Grade output schema
class GroundedGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    grounded: Annotated[
        bool, ..., "Provide the score on if the answer hallucinates from the documents"
    ]

# Grade prompt
grounded_instructions = """You are a teacher grading a quiz.

You will be given FACTS and a MEDICAL ASSISTANT ANSWER.

Here is the grade criteria to follow:
(1) Ensure the MEDICAL ASSISTANT ANSWER is grounded in the FACTS.
(2) Ensure the MEDICAL ASSISTANT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.

Grounded:
A grounded value of True means that the MEDICAL ASSISTANT's answer meets all of the criteria.
A grounded value of False means that the MEDICAL ASSISTANT's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct.

Avoid simply stating the correct answer at the outset."""

# Grader LLM
grounded_llm = ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(
    GroundedGrade, method="json_schema", strict=True
)


# Evaluator
def groundedness(inputs: dict, outputs: dict) -> bool:
    """A simple evaluator for RAG answer groundedness."""
    doc_string = "".join(doc.page_content for doc in outputs["documents"])
    answer = f"""      FACTS: {doc_string}
MEDICAL ASSISTANT ANSWER: {outputs['answer']}"""
    grade = grounded_llm.invoke(
        [
            {"role": "system", "content": grounded_instructions},
            {"role": "user", "content": answer},
        ]
    )
    return grade["grounded"]


# Grade output schema
class RetrievalRelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[
        bool,
        ...,
        "True if the retrieved documents are relevant to the question, False otherwise",
    ]


# Grade prompt
retrieval_relevance_instructions = """You are a teacher grading a quiz.

You will be given a QUESTION and a set of FACTS provided by the MEDICAL ASSISTANT.

Here is the grade criteria to follow:
(1) You goal is to identify FACTS that are completely unrelated to the QUESTION
(2) If the facts contain ANY keywords or semantic meaning related to the question, consider them relevant
(3) It is OK if the facts have SOME information that is unrelated to the question as long as (2) is met

Relevance:
A relevance value of True means that the FACTS contain ANY keywords or semantic meaning related to the QUESTION and are therefore relevant.
A relevance value of False means that the FACTS are completely unrelated to the QUESTION.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct.

Avoid simply stating the correct answer at the outset."""

# Grader LLM
retrieval_relevance_llm = ChatOpenAI(
    model="gpt-4o", temperature=0
).with_structured_output(RetrievalRelevanceGrade, method="json_schema", strict=True)


def retrieval_relevance(inputs: dict, outputs: dict) -> bool:
    """An evaluator for document relevance"""
    doc_string = "".join(doc.page_content for doc in outputs["documents"])
    answer = f"""      FACTS: {doc_string}
QUESTION: {inputs['question']}"""

    # Run evaluator
    grade = retrieval_relevance_llm.invoke(
        [
            {"role": "system", "content": retrieval_relevance_instructions},
            {"role": "user", "content": answer},
        ]
    )
    return grade["relevant"]

from app import get_response_for_evaluation # NEW

def target(inputs: dict) -> dict:
    """
    This target function now calls our new evaluation-specific function
    from app.py, which provides all the keys our evaluators need.
    """
    session_id = str(uuid.uuid4())
    # The 'inputs' dict from the dataset has a 'question' key.
    return get_response_for_evaluation(inputs["question"], session_id)

# This will now run as expected without starting Gradio and without failing on the evaluators.
print("Starting evaluation...")
experiment_results = client.evaluate(
    target,
    data=dataset_name,
    evaluators=[correctness, groundedness, relevance, retrieval_relevance],
    experiment_prefix="rag-doc-relevance-new",
    metadata={"version": "LCEL context, gpt-4o-mini"},
)

print("Evaluation complete! Results:")
print(experiment_results.to_pandas())



