from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool

load_dotenv()  #Load environment variables (like API keys) from .env file

#Define the structure of the expected LLM output using Pydantic
class ResearchResponse(BaseModel):
    topic : str
    summary : str
    sources : list[str]
    tools_used : list[str]

#Initialize the OpenAI language model (GPT-3.5-turbo)
llm = ChatOpenAI(model="gpt-3.5-turbo")

#Set up a parser to convert LLM output into the ResearchResponse model
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

#Create a prompt template for the agent, including instructions and output format
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", 
         """
         You are a helpful research assistant that will help generate a research paper.
         Answer the user query using the following format and provide no other text:
         {format_instructions}
         For example, if the user asks "What is the capital of France?", your response should look like:
         {{
           "topic": "Capital of France",
           "summary": "The capital of France is Paris.",
           "sources": [],
           "tools_used": []
         }}
         """
        ),
        ("placeholder", "{chat_history}"),  #Placeholder for conversation history
        ("human", "{query}"),  #Placeholder for the user's question
        ("placeholder", "{agent_scratchpad}" )  #Placeholder for agent's internal state
    ]
).partial(format_instructions=parser.get_format_instructions())  # Insert format instructions into the prompt

#List of tools the agent can use (search and Wikipedia)
tools = [search_tool, wiki_tool]

#Create the agent with the LLM, prompt, and tools
agent = create_tool_calling_agent(
    llm = llm, 
    prompt = prompt,
    tools = tools
)

#Create an executor to run the agent and handle tool calls
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)  

#Prompt the user for a research question
query = input("What can I help you research? ")   

#Invoke the agent with the user's query and get the raw response
raw_response = agent_executor.invoke({"query" : query})       

print (raw_response)   #Print the raw response for debugging

#Try to parse the agent's output into the structured ResearchResponse model
try:
    structured_response = parser.parse(raw_response.get("output"))
    print(structured_response)  #Print the structured response
except Exception as e:  
    #Print an error message and the raw response if parsing fails
    print("Error parsing response:", e, "Raw Response - ", raw_response)