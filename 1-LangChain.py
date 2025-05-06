import os
from dotenv import load_dotenv
from getpass import getpass
from langchain_openai import ChatOpenAI
from help import article
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from pydantic import BaseModel, Field
from skimage import io
import matplotlib.pyplot as plt
from langchain_core.runnables import RunnableLambda
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain_core.prompts import PromptTemplate
load_dotenv()


# Set your OpenAI API key as an environment variable
os.environ["OPENAI_API_KEY"] = os.environ.get('OPENAI_API_KEY')

# Correct model name
openai_model = "gpt-4o-mini"

# For normal accurate responses
llm = ChatOpenAI(temperature=0.0, model=openai_model)
# For unique creative responses
creative_llm = ChatOpenAI(temperature=0.9, model=openai_model)





# Defining the system prompt (how the AI should act)
system_prompt = SystemMessagePromptTemplate.from_template(
    "You are an AI assistant {name} that helps generate article titles.",
    input_variables=["name"]
)

# the user prompt is provided by the user, in this case however the only dynamic
# input is the article
user_prompt = HumanMessagePromptTemplate.from_template(
    """You are tasked with creating a name for a article.
The article is here for you to examine {article}

The name should be based of the context of the article.
Be creative, but make sure the names are clear, catchy,
and relevant to the theme of the article.

Only output the article name, no other explanation or
text can be provided.""",
    input_variables=["article"]
)

first_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])


# we define our inputs with the first dictionary segment (ie {"article": lambda x: x["article"]}) and 
# then we use the pipe operator (|) to say that the output from the left of the pipe will be 
# fed into the input to the right of the pipe.

chain_one=(
    {
        "name":lambda x: x["name"],
        "article":lambda x: x["article"]
    }
    | first_prompt 
    | creative_llm 
    | {"article_title": lambda x: x.content}
)

article_title = chain_one.invoke({"article":article, "name":"Joe"})
print(article_title)


# second_user_prompt = HumanMessagePromptTemplate.from_template(
#     """You are tasked with creating a description for
# the article. Maximum 100 words The article is here for you to examine:

# ---

# {article}

# ---

# Here is the article title '{article_title}'.

# Output the SEO friendly article description. Do not output
# anything other than the description.""",
#     input_variables=["article", "article_title"]
# )

# second_prompt = ChatPromptTemplate.from_messages([
#     system_prompt,
#     second_user_prompt
# ])

# chain_two = (
#     {
#         "article": lambda x: x["article"],
#         "article_title": lambda x: x["article_title"],
#         "name": lambda x: x["name"]
#     }
#     | second_prompt
#     | llm
#     | {"summary": lambda x: x.content}
# )

# article_description_msg = chain_two.invoke({
#     "article": article,
#     "name":"Joe",
#     "article_title": article_title["article_title"]
# })
# print(article_description_msg)


# third_user_prompt = HumanMessagePromptTemplate.from_template(
#     """You are tasked with creating a new paragraph for the
# article. The article is here for you to examine:

# ---

# {article}

# ---

# Choose one paragraph to review and edit. During your edit
# ensure you provide constructive feedback to the user so they
# can learn where to improve their own writing.""",
#     input_variables=["article"]
# )

# # prompt template 3: creating a new paragraph for the article
# third_prompt = ChatPromptTemplate.from_messages([
#     system_prompt,
#     third_user_prompt
# ])


# class Paragraph(BaseModel):
#     original_paragraph: str = Field(description="The original paragraph")
#     edited_paragraph: str = Field(description="The improved edited paragraph")
#     feedback: str = Field(description=(
#         "Constructive feedback on the original paragraph"
#     ))

# structured_llm = creative_llm.with_structured_output(Paragraph)


# chain_three = (
#     {
#         "article": lambda x: x["article"],
#         "name": lambda x: x["name"]
#     }
#     | third_prompt
#     | structured_llm
#     | {
#         "original_paragraph": lambda x: x.original_paragraph,
#         "edited_paragraph": lambda x: x.edited_paragraph,
#         "feedback": lambda x: x.feedback
#     }
# )
# out = chain_three.invoke({"article": article, "name":"Joe"})
# print(out)



# image_prompt = PromptTemplate(
#     input_variables=["article"],
#     template=(
#         "Generate a prompt with less then 500 characters to generate an image "
#         "based on the following article: {article}"
#     )
# )

# def generate_and_display_image(image_prompt):
#     print(image_prompt)
#     # Create DALL-E wrapper with proper configuration
#     dalle = DallEAPIWrapper(
#         model="dall-e-3",
#         size="1024x1024",
#         n=1
#     )
#     image_url = dalle.run(image_prompt)
#     image_data = io.imread(image_url)

#     # Display the image
#     plt.figure(figsize=(10, 10))
#     plt.imshow(image_data)
#     plt.axis('off')
#     plt.show()

# # we wrap this in a RunnableLambda for use with LCEL
# image_gen_runnable = RunnableLambda(generate_and_display_image)


# # chain 4: inputs: article, article_para / outputs: new_suggestion_article
# chain_four = (
#     {"article": lambda x: x["article"],
#     }
#     | image_prompt
#     | llm
#     | (lambda x: x.content)
#     | image_gen_runnable
# )

# out = chain_four.invoke({"article": article})
# print(out)