# langchain -- 1. components 2. chains 3. agents (install resource you're using i.e. wikipedia)
from langchain.llms import OpenAI
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType

load_dotenv()

# generate few names

def genearate_pet_names(animal_type,pet_color):
    llm=OpenAI(temperature=0.7)

    prompt_template = PromptTemplate(
        input_variables=['animal_type','pet_color'],
        template="""I have a {animal_type} pet,it is {pet_color} color and I want a cool name for it, suggest five cool names
                  for my pet"""
    )
    # name = llm("""I have a parrot pet and I want a cool name for it, suggest five cool names
    #            for my pet""")
    name = LLMChain(llm=llm,prompt = prompt_template,output_key='pet_name')
    response = name.chain({'animal_type': animal_type,'pet_color':pet_color})
    return response
    
def langchain_agent():
    llm=OpenAI(temperature=0.5)
    tools = load_tools(["wikipedia","llm-math"],llm=llm)
    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    result = agent.run(
        "What is the average age of dog? Multiply the age by 3."
    )

    print(result)

if __name__=="__main__":
    langchain_agent()
    # print(genearate_pet_names("parrot","red"))
# if __name__=="__main__":
#     print(genearate_pet_names("parrot","red"))
