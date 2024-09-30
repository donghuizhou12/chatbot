from datetime import datetime
from django.shortcuts import render,redirect
from .forms import QueryForm  # Assuming the form is already defined in forms.py
#from langchain_community.chat_models import ChatGemini
#from langchain.chat_models import ChatGemini
#from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
import os
from django.shortcuts import render
#from langchain_community.llms import OpenAI  # OpenAI model from langchain-community
from .forms import QueryForm  # Assuming you have a QueryForm for handling user input
from datetime import datetime

import openai


from .models import Specialist 

openai.api_key = os.environ["OPENAI_API_KEY"]
# Set your OpenAI API key
# Initialize the model for chat completions
# Define the prompt template to structure the input query for the model
prompt_template = """
I want you to answer me in English and for your answer to be limited to an ordered list of html.
List in order of probability the diseases that the patient may suffer with these data:
Age: {age}
Gender: {gender}
Symptoms: {symptoms}
Patient's medical history: {history}
Family medical history: {family}
Lifestyle: {lifestyle}
Medical Test Results: {tests}
Other relevant details: {other}
"""


# Define the LangChain process using LLMChain
def create_langchain_process():
    # LLMChain combines an LLM model with a prompt template
    return LLMChain(
        llm=model,
        prompt=prompt_template
    )
# Define the LangGraph workflow for managing states and invoking the model
def build_langgraph_workflow():
    # Initialize the state graph
    workflow = StateGraph(MessagesState)

    # Create a node for calling the model
    def call_model(state: MessagesState):
        query_data = state['messages'][-1].content  # Get the last message in the state
        llm_chain = create_langchain_process()  # Use the LLM chain to generate the response
        response = llm_chain.run(query_data)
        return {"messages": [{"content": response}]}

    # Define nodes and edges
    workflow.add_node("agent", call_model)
    workflow.add_edge(START, "agent")  # Start with the 'agent' node
    workflow.add_edge("agent", END)  # End after generating the response

    # Compile the workflow into a runnable app
    return workflow.compile()
    
#import openai
#from django.shortcuts import render
#from .forms import QueryForm
from .models import Specialist  # Assuming you have a Specialist model for matching specialists

# Set your OpenAI API key
#openai.api_key = os.getenv("OPENAI_API_KEY")

# Summarize the input and recommend a specialist


def find_specialist(summary):
    """
    Find the specialist based on the summarized patient data.
    Here you can match symptoms from the summary to specialists in your database.
    """
    # Example logic: match the most relevant specialist
    # Assuming you have a `Specialist` model that has `specialty` field
    specialists = Specialist.objects.all()

    # Search for a matching specialist based on the summary
    for specialist in specialists:
        if specialist.specialty in summary:
            return specialist

    return None  # No matching specialist found

from .utils import find_specialist
#def find_specialist(symptom):
    # Example: Find a specialist based on the symptom's specialty and availability
 #   specialist = Specialist.objects.filter(specialty=symptom.specialty, available_day='Monday')  # Adjust logic as needed
  #  return specialist

def handle_patient_submission(request):
    if request.method == "POST":
        form = PatientForm(request.POST)
        if form.is_valid():
            # Save patient data
            patient = form.save()

            # Assuming you have a symptom field in the form or patient model
            symptom = patient.symptoms

            # Find a matching specialist
            specialist = find_specialist(symptom)

            # Add logic to schedule an appointment with the found specialist
            # Or pass the specialist details to the template for user review
            return render(request, 'appointment_confirm.html', {'specialist': specialist, 'patient': patient})

    # For GET requests
    form = PatientForm()
    return render(request, 'patient_form.html', {'form': form})

def recommendation(request):
    # Here you can pass any context variables to the template if needed
    return render(request, 'main/recommendation.html')




def extract_specialist_from_summary(summary):
    # Split the summary by spaces and get the last word, assuming it's the specialist
    words = summary.split()
    if words:
        return words[-1]  # The last word in the summary
    return None  # Fallback in case something goes wrong

def abcome(request):
    form = QueryForm(request.POST or None)

    if request.method == "POST":
        if form.is_valid():
            query = form.save(commit=False)
            query.log_date = datetime.now()
            query.save()

            # Prepare the input message for the AI model
            patient_data = f"""
            Age: {query.age}
            Gender: {query.gender}
            Symptoms: {query.symptoms}
            Patient's medical history: {query.history}
            Family medical history: {query.family}
            Lifestyle: {query.lifestyle}
            Medical Test Results: {query.tests}
            Other relevant details: {query.other}
            """
            
            # AI model prompt with instruction to include a specialist at the end
            message = (
                "I want you to answer me in English and limit your answer to an ordered list of html. "
                "List in order of probability the diseases that the patient may suffer with this data and "
                "determine the most appropriate medical specialist to refer the patient to based on the information. "
                "At the end of your response, always finish with a sentence like 'Specialist to refer to: [specialist name]' "
                "where [specialist name] is the most appropriate specialist for the patient's condition."
            )

            message += f"\nAge: {query.age}"
            message += f"\nGender: {query.gender}"
            message += f"\nSymptoms: {query.symptoms}"
            message += f"\nPatient's medical history: {query.history}"
            message += f"\nFamily medical history: {query.family}"
            message += f"\nLifestyle: {query.lifestyle}"
            message += f"\nMedical Test Results: {query.tests}"
            message += f"\nOther relevant details: {query.other}"

            try:
                # Generate the response using the OpenAI ChatCompletion model
                response = openai.ChatCompletion.create(
                    model="gpt-4-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful medical assistant."},
                        {"role": "user", "content": message}
                    ],
                    temperature=0.7
                )

                # Extract AI-generated summary
                summary = response['choices'][0]['message']['content'].strip()

                # Extract the specialist from the summary
                specialist = extract_specialist_from_summary(summary)

                # Query the doctors based on the extracted specialist
                doctors = Specialist.objects.filter(specialty__icontains=specialist)

                # Render the response and the list of doctors in the template
                return render(request, "main/home.html", {
                    "form": form,
                    "summary": summary,
                    "specialist": specialist,
                    "doctors": doctors  # Pass the matched doctors to the template
                })

            except openai.error.OpenAIError as e:
                # Handle errors from the AI model
                return render(request, "main/home.html", {
                    "form": form,
                    "error": str(e)
                })

    # Render the form if the request method is GET or the form is invalid
    return render(request, "main/home.html", {"form": form})
def recommend_specialist(request):
    form = QueryForm(request.POST or None)

    if request.method == "POST":
        if form.is_valid():
            query = form.save(commit=False)
            query.save()

            # Prepare input for the AI model
            patient_data = f"""
            Age: {query.age}
            Gender: {query.gender}
            Symptoms: {query.symptoms}
            History: {query.history}
            Family History: {query.family}
            Lifestyle: {query.lifestyle}
            Test Results: {query.tests}
            Other: {query.other}
            """

            # AI model generates summary
            response = openai.ChatCompletion.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are a medical assistant."},
                    {"role": "user", "content": patient_data}
                ],
                temperature=0.7
            )

            # Extract AI-generated summary
            summary = response['choices'][0]['message']['content']

            # Match the summary with a specialist
            matched_specialist = find_specialist(summary)

            return render(request, 'main/recommendation.html', {
                'form': form,
                'summary': summary,
                'specialist': matched_specialist
            })

    return render(request, "main/form.html", {"form": form})
















def recommend_specialist(request):
    form = QueryForm(request.POST or None)

    if request.method == "POST":
        if form.is_valid():
            query = form.save(commit=False)
            query.log_date = datetime.now()
            query.save()

            # Prepare the input message for the AI model
            patient_data = f"""
            Age: {query.age}
            Gender: {query.gender}
            Symptoms: {query.symptoms}
            Patient's medical history: {query.history}
            Family medical history: {query.family}
            Lifestyle: {query.lifestyle}
            Medical Test Results: {query.tests}
            Other relevant details: {query.other}
            """
            
            # AI model prompt with instruction to include a specialist at the end
            message = (
                "I want you to answer me in English and limit your answer to an ordered list of html. "
                "List in order of probability the diseases that the patient may suffer with this data and "
                "determine the most appropriate medical specialist to refer the patient to based on the information. "
                "At the end of your response, always finish with a sentence like 'Specialist to refer to: [specialist name]' "
                "where [specialist name] is the most appropriate specialist for the patient's condition."
            )

            message += f"\nAge: {query.age}"
            message += f"\nGender: {query.gender}"
            message += f"\nSymptoms: {query.symptoms}"
            message += f"\nPatient's medical history: {query.history}"
            message += f"\nFamily medical history: {query.family}"
            message += f"\nLifestyle: {query.lifestyle}"
            message += f"\nMedical Test Results: {query.tests}"
            message += f"\nOther relevant details: {query.other}"

            try:
                # Generate the response using the OpenAI ChatCompletion model
                response = openai.ChatCompletion.create(
                    model="gpt-4-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful medical assistant."},
                        {"role": "user", "content": message}
                    ],
                    temperature=0.7
                )

                # Extract AI-generated summary
                summary = response['choices'][0]['message']['content'].strip()

                # Extract the specialist from the summary
                specialist = extract_specialist_from_summary(summary)

                # Query the doctors based on the extracted specialist
                doctors = Specialist.objects.filter(specialty__icontains=specialist)

                # Render the response and the list of doctors in the template
                return render(request, "main/recommendation.html", {
                    "form": form,
                    "summary": summary,
                    "specialist": specialist,
                    "doctors": doctors  # Pass the matched doctors to the template
                })

            except openai.error.OpenAIError as e:
                # Handle errors from the AI model
                return render(request, "main/home.html", {
                    "form": form,
                    "error": str(e)
                })

    # Render the form if the request method is GET or the form is invalid
    return render(request, "main/home.html", {"form": form})

def extract_specialist_from_summary(summary):
    # Split the summary by spaces and get the last word, assuming it's the specialist
    words = summary.split()
    if words:
        return words[-1]  # The last word in the summary
    return None  # Fallback in case something goes wrong

    
def recommendation(request):
    # Here you can pass any context variables to the template if needed
    return render(request, 'main/recommendation.html')



def home(request):
    form = QueryForm(request.POST or None)

    if request.method == "POST":
        if form.is_valid():
            query = form.save(commit=False)
            query.log_date = datetime.now()
            query.save()

            # Prepare the input message for the AI model
            patient_data = f"""
            Age: {query.age}
            Gender: {query.gender}
            Symptoms: {query.symptoms}
            Patient's medical history: {query.history}
            Family medical history: {query.family}
            Lifestyle: {query.lifestyle}
            Medical Test Results: {query.tests}
            Other relevant details: {query.other}
            """

            # AI model prompt with instruction to include a specialist at the end
            message = (
                "I want you to answer me in English and limit your answer to an ordered list of html. "
                "List in order of probability the diseases that the patient may suffer with this data and "
                "determine the most appropriate medical specialist to refer the patient to based on the information. "
                "At the end of your response, always finish with a sentence like 'Specialist to refer to: [specialist name]' "
                "where [specialist name] is the most appropriate specialist for the patient's condition."
            )

            message += f"\nAge: {query.age}"
            message += f"\nGender: {query.gender}"
            message += f"\nSymptoms: {query.symptoms}"
            message += f"\nPatient's medical history: {query.history}"
            message += f"\nFamily medical history: {query.family}"
            message += f"\nLifestyle: {query.lifestyle}"
            message += f"\nMedical Test Results: {query.tests}"
            message += f"\nOther relevant details: {query.other}"

            try:
                # Generate the response using the OpenAI ChatCompletion model
                response = openai.ChatCompletion.create(
                    model="gpt-4-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful medical assistant."},
                        {"role": "user", "content": message}
                    ],
                    temperature=0.7
                )

                # Extract AI-generated summary
                summary = response['choices'][0]['message']['content'].strip()

                # Extract the specialist from the summary
                specialist = extract_specialist_from_summary(summary)

                # Query the doctors based on the extracted specialist
                doctors = Specialist.objects.filter(specialty__icontains=specialist)

                # Render the form, summary, specialist, and doctors in the same template
                return render(request, "main/home.html", {
                    "form": form,
                    "summary": summary,
                    "specialist": specialist,
                    "doctors": doctors  # Pass the matched doctors to the template
                })

            except openai.error.OpenAIError as e:
                # Handle errors from the AI model
                return render(request, "main/home.html", {
                    "form": form,
                    "error": str(e)
                })

    # Render the form if the request method is GET or the form is invalid
    return render(request, "main/home.html", {"form": form})
