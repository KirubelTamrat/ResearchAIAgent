from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool 
from datetime import datetime

#Saves research output to a text file with a timestamp
def save_to_txt(data: str, filename: str = "research_output.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Get current time
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"  # Format the output

    #Append the formatted text to the specified file
    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)
    
    return f"Data successfully saved to {filename}"  # Return confirmation message

#Tool wrapper for saving research output to a file
save_tool = Tool(
    name="save_text_to_file",  #Tool name (must be valid for OpenAI API)
    func=save_to_txt,  #Function to call
    description="Saves structured research data to a text file.",  #Description for the agent
)

#Initialize DuckDuckGo search tool for web queries
search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="Search",  #Tool name (must be valid for OpenAI API, use lowercase and underscores)
    func=search.run,  #Function to call for search
    description="Search the web for information",  #Description for the agent
) 

#Initialize Wikipedia tool for querying Wikipedia articles
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)  #Limit to 1 result, 100 chars max
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)  #Create the Wikipedia tool