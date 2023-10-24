import streamlit as st
import openai
import langchain
from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster
from dotenv import dotenv_values
from audio_recorder_streamlit import audio_recorder
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

from langchain.memory import CassandraChatMessageHistory
from langchain.schema import SystemMessage
from gtts import gTTS
from io import BytesIO
import os

config = dotenv_values('.env')
openai_key = config['OPENAI_API_KEY']

langchain.debug = True

from astraretriver import TotalRevenueReaderTool, ClientSimilarityTool, GetClientInformationTool
tools = [TotalRevenueReaderTool(), ClientSimilarityTool(), GetClientInformationTool()]

SECURE_CONNECT_BUNDLE_PATH = config['SECURE_CONNECT_BUNDLE_PATH']
ASTRA_CLIENT_ID = config['ASTRA_CLIENT_ID']
ASTRA_CLIENT_SECRET = config['ASTRA_CLIENT_SECRET']
ASTRA_KEYSPACE_NAME = config['ASTRA_KEYSPACE_NAME']

# Open a connection to the Astra database
cloud_config = {
    'secure_connect_bundle': SECURE_CONNECT_BUNDLE_PATH
}
auth_provider = PlainTextAuthProvider(ASTRA_CLIENT_ID, ASTRA_CLIENT_SECRET)
cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
astraSession = cluster.connect()

conversational_memory = CassandraChatMessageHistory(
    session_id='finbot-conversation-555',
    session=astraSession,
    keyspace=ASTRA_KEYSPACE_NAME,
    ttl_seconds=3600,
)
conversational_memory.clear()

system_message = SystemMessage(content="You are a FinTech Bot named FinBot, a sophisticated bank assistant, specialized in credit scores and "
                                       "currency transactions. With expert knowledge and precision, you are here to "
                                       "provide accurate information and solutions to banking queries. When calling the tool, include as much detail as possible " 
                                       "and translate arguments to English. All the responses should be the same language as the user used.")

llm = ChatOpenAI(
    openai_api_key=openai_key,
    temperature=0,
    streaming=True,
    model_name="gpt-4"
)

agent = initialize_agent(
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    tools=tools,
    llm=llm,
    max_iterations=3,
    verbose=False,
    chat_memory=conversational_memory,
    handle_parsing_errors=True,
    early_stopping_method='generate',
    agent_kwargs={
        "system_message": system_message.content
    }
)

# set title
st.title('Fin Bot')

# set header
st.header("Welcome to our bank Fin Bot! You can speak to this bot")

user_question = st.text_input('Ask a question here:')

#audio = audiorecorder("Speak to me!", "Recording...")
audio = audio_recorder()

# if the question has more than 5 characters, run the agent
if len(user_question) > 5:
    with st.spinner(text="In progress..."):
        conversational_memory.add_user_message(user_question)
        response = agent.run(input=user_question, chat_history=conversational_memory.messages)
        st.write(response)
elif audio:
    # To play audio in frontend:
    #st.audio(audio.tobytes())
    st.audio(audio, format="audio/mp3")
    wav_file = open("test.mp3", "wb")
    wav_file.write(audio)

    audio_file= open("test.mp3", "rb")
    user_question = openai.Audio.transcribe("whisper-1", audio_file)
    st.write(user_question.text)
    if user_question and user_question != "":
        sound_file = BytesIO()
        with st.spinner(text="In progress..."):
            conversational_memory.add_user_message(user_question.text)
            response = agent.run(input=user_question.text, chat_history=conversational_memory.messages)
            #response = agent.run('{}, {}'.format(user_question, user_question))
            
            st.write(response)
            # Initialize gTTS with the text to convert
            speech = gTTS(response,tld='us',slow=False)

            # Save the audio file to a temporary file
            speech_file = 'speech.mp3'
            speech.save(speech_file)

            # Play the audio file
            os.system('afplay ' + speech_file)